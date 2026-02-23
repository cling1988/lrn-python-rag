"""
pipeline.py – The RAG pipeline: ties all four steps together.

How RAG works (the big picture)
---------------------------------
Traditional LLMs answer questions from their training data alone – they can
be out-of-date and may hallucinate facts. RAG solves this by:

  1. **Retrieve** – search a vector store for the most relevant text chunks
     from *your* documents.
  2. **Augment** – prepend those chunks to the prompt so the model can read
     them before answering.
  3. **Generate** – ask the model to answer the question *using only the
     provided context*.

This gives you answers grounded in your own up-to-date documents, with
citations/sources you can trace.

Pipeline steps in this file
----------------------------
  ingest()  → Step 1 (load) → Step 2 (split) → Step 3 (embed & store)
  query()   → Step 4 (retrieve) → Step 5 (generate with Gemini)
  retrieve() → Step 4 only (useful for debugging retrieval quality)
"""

import os
from typing import List, Optional, Tuple

from google import genai

from rag.document_loader import Document, load_documents
from rag.text_splitter import split_documents
from rag.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# The prompt is the bridge between retrieved context and the LLM.
# We explicitly instruct the model to:
#   a) use ONLY the provided context (prevents hallucination)
#   b) say "I don't know" when context is insufficient (reduces confabulation)
_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the information \
provided in the context below. If the answer is not contained in the context, \
say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline powered by Google GenAI.

    Quick-start example::

        from rag import RAGPipeline

        pipeline = RAGPipeline()
        pipeline.ingest("my_document.pdf")
        answer = pipeline.query("What is the document about?")
        print(answer)

    Args:
        model:             Gemini model name for generation (default: ``gemini-2.0-flash``).
        embedding_model:   Google GenAI embedding model (default: ``models/text-embedding-004``).
        chunk_size:        Max characters per text chunk (default: 1000).
        chunk_overlap:     Overlap characters between chunks (default: 200).
        k:                 Number of chunks to retrieve per query (default: 4).
        persist_directory: Directory to persist the vector store on disk.
                           ``None`` keeps everything in memory.
        api_key:           Google AI API key.  Defaults to ``GOOGLE_API_KEY``
                           environment variable.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        chunk_size: int = 200,
        chunk_overlap: int = 20,
        k: int = 4,
        persist_directory: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.persist_directory = persist_directory

        # Resolve API key: constructor argument takes priority over env var
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        # The Google GenAI client is shared between embedding and generation
        self._genai_client = genai.Client(api_key=self._api_key)

        # _vector_store is None until ingest() or load() is called
        self._vector_store: Optional[VectorStore] = None

    # ------------------------------------------------------------------
    # Step 1-3: Ingestion
    # ------------------------------------------------------------------

    def ingest(self, source: str) -> int:
        """Load, split, embed and store documents from *source*.

        This method runs Steps 1, 2 and 3 of the RAG pipeline:

        * **Step 1** – :func:`~rag.document_loader.load_documents` reads the
          file / directory / URL and produces raw Document objects.
        * **Step 2** – :func:`~rag.text_splitter.split_documents` breaks each
          Document into overlapping chunks sized for the embedding model.
        * **Step 3** – :class:`~rag.vector_store.VectorStore` embeds each
          chunk and stores the vectors in ChromaDB.

        Calling ``ingest()`` multiple times on the same pipeline adds more
        documents to the *existing* vector store (no data is overwritten).

        Args:
            source: File path (PDF or text), directory path, or HTTP(S) URL.

        Returns:
            Number of chunks that were embedded and stored.
        """
        # --- Step 1: load raw documents ---
        print(f"[RAG] Loading documents from: {source}")
        documents = load_documents(source)
        print(f"[RAG] Loaded {len(documents)} document(s)")

        # --- Step 2: split into chunks ---
        chunks = split_documents(documents, self.chunk_size, self.chunk_overlap)
        print(f"[RAG] Split into {len(chunks)} chunk(s)")

        # --- Step 3: embed and store ---
        # Create the vector store on the first ingest; reuse it on subsequent calls
        if self._vector_store is None:
            self._vector_store = VectorStore(
                api_key=self._api_key,
                persist_directory=self.persist_directory,
            )

        self._vector_store.add_documents(chunks)
        print(f"[RAG] Embedded and stored {len(chunks)} chunk(s)")

        return len(chunks)

    def load(self, persist_directory: Optional[str] = None) -> None:
        """Load a previously persisted vector store (skip re-embedding).

        Use this when you have already called ``ingest()`` in a previous
        session and saved the store to disk with ``persist_directory``.

        Args:
            persist_directory: Path to the persisted store.  Falls back to
                               the value passed to the constructor.

        Raises:
            ValueError: If no directory is specified.
        """
        directory = persist_directory or self.persist_directory
        if directory is None:
            raise ValueError(
                "persist_directory must be provided either at construction "
                "time or when calling load()."
            )
        self._vector_store = VectorStore(
            api_key=self._api_key,
            persist_directory=directory,
        )
        print(f"[RAG] Loaded vector store from: {directory}")

    # ------------------------------------------------------------------
    # Step 4: Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> List[Tuple[Document, float]]:
        """Find the most relevant document chunks for *question* (no generation).

        This is Step 4 of the RAG pipeline in isolation. It is useful for:
        * Debugging – check whether the right chunks are being retrieved.
        * Custom pipelines – you can do your own generation after retrieval.

        Args:
            question: Natural-language question.

        Returns:
            List of ``(Document, distance)`` tuples sorted by relevance
            (lowest cosine distance = most relevant).

        Raises:
            RuntimeError: If no documents have been ingested yet.
        """
        self._require_vector_store()
        return self._vector_store.similarity_search(question, k=self.k)

    # ------------------------------------------------------------------
    # Step 4-5: Full RAG query
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        """Answer *question* using the ingested documents.

        This runs Steps 4 and 5:

        * **Step 4** – retrieve the *k* most relevant chunks from the vector store.
        * **Step 5** – build a prompt that includes the retrieved context and
          send it to Gemini for generation.

        Args:
            question: Natural-language question to answer.

        Returns:
            Generated answer string grounded in the retrieved context.

        Raises:
            RuntimeError: If no documents have been ingested yet.
        """
        self._require_vector_store()

        # --- Step 4: retrieve relevant chunks ---
        results = self._vector_store.similarity_search(question, k=self.k)

        # Concatenate chunk texts into a single context string.
        # We separate chunks with a blank line so the model can tell them apart.
        context = "\n\n".join(doc.page_content for doc, _ in results)

        # --- Step 5: build the augmented prompt and generate an answer ---
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

        # Call the Gemini model through the Google GenAI SDK
        response = self._genai_client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        # response.text is the generated answer as a plain string
        return response.text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_vector_store(self) -> None:
        """Raise RuntimeError if no vector store has been initialised."""
        if self._vector_store is None:
            raise RuntimeError(
                "No documents have been ingested. Call ingest() or load() first."
            )

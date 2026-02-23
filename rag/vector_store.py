"""
vector_store.py – Step 3 of the RAG pipeline: embed chunks and store them.

What is a vector store?
-----------------------
Text cannot be compared mathematically as-is. An *embedding model* converts
each chunk into a dense numerical vector (a list of ~768 floats) that
captures its *semantic meaning*. Two chunks about similar topics will have
vectors that are geometrically close to each other.

A *vector store* (here: ChromaDB) saves those vectors and can quickly find
the N vectors closest to a query vector – this is the "retrieval" in RAG.

Google GenAI embedding model
-----------------------------
We use ``models/text-embedding-004`` – Google's general-purpose text
embedding model. It produces 768-dimensional vectors and supports both
document ingestion (task_type="RETRIEVAL_DOCUMENT") and query encoding
(task_type="RETRIEVAL_QUERY") to maximise retrieval quality.

ChromaDB
--------
ChromaDB is a lightweight, embedded vector database written in Python.
It requires no external server – data lives in-process or on disk.
"""

import os
import uuid
from typing import Dict, List, Optional, Tuple

import chromadb
from google import genai
from google.genai import types as genai_types

from rag.document_loader import Document


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _get_embedding(
    client: genai.Client,
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> List[float]:
    """Convert a single text string into an embedding vector.

    Args:
        client:    Authenticated google.genai.Client instance.
        text:      Text to embed.
        task_type: ``"RETRIEVAL_DOCUMENT"`` when embedding corpus chunks;
                   ``"RETRIEVAL_QUERY"`` when embedding the user's question.
                   Using the correct task type improves retrieval accuracy.

    Returns:
        A list of floats representing the embedding vector.
    """
    # The embed_content method accepts a single string or a list of strings.
    # Using task_type tells the model *how* the embedding will be used, which
    # lets it optimise the vector representation accordingly.
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    # response.embeddings is a list; we embedded one string so take index 0
    return response.embeddings[0].values


def _get_embeddings_batch(
    client: genai.Client,
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> List[List[float]]:
    """Embed a list of texts in a single API call (more efficient than looping).

    Args:
        client:    Authenticated google.genai.Client instance.
        texts:     List of text strings to embed.
        task_type: Embedding task type (see :func:`_get_embedding`).

    Returns:
        List of embedding vectors, one per input text, in the same order.
    """
    print(f"Embedding {len(texts)} chunks with task_type={task_type}...")
    for model in client.models.list():
        print(model)
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return [emb.values for emb in response.embeddings]


# ---------------------------------------------------------------------------
# VectorStore class
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around a ChromaDB collection with Google GenAI embeddings.

    This class encapsulates:
    * Creating or loading a ChromaDB collection.
    * Embedding documents with Google GenAI and adding them to the collection.
    * Embedding a query and retrieving the most similar document chunks.

    Args:
        api_key:          Google AI API key. Falls back to the
                          ``GOOGLE_API_KEY`` environment variable when ``None``.
        collection_name:  Name of the ChromaDB collection to use.
        persist_directory: Directory to persist data on disk.  When ``None``
                           the collection lives only in memory (useful for tests).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        collection_name: str = "rag_collection",
        persist_directory: Optional[str] = None,
    ) -> None:
        # --- Initialise Google GenAI client ---
        # The client reads GOOGLE_API_KEY from the environment if api_key is None
        self._genai_client = genai.Client(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY")
        )

        # --- Initialise ChromaDB ---
        # PersistentClient saves to disk; EphemeralClient stays in memory
        if persist_directory:
            self._chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._chroma_client = chromadb.EphemeralClient()

        # get_or_create_collection: reuses existing data if the collection
        # already exists (important when loading a previously built store)
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            # cosine distance is better than Euclidean for text embeddings
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Adding documents
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> None:
        """Embed and store a list of document chunks.

        Each chunk gets:
        * A unique ID (UUID4) so ChromaDB can track it.
        * An embedding vector from Google GenAI.
        * Its raw text stored alongside the vector for later retrieval.
        * Its metadata (source, page, chunk_index, …).

        Args:
            documents: Chunks produced by the text splitter (Step 2).
        """
        if not documents:
            return

        # Extract plain text to embed (we batch for efficiency)
        texts = [doc.page_content for doc in documents]

        # --- Step 3a: embed all chunks in one API call ---
        embeddings = _get_embeddings_batch(
            self._genai_client, texts, task_type="RETRIEVAL_DOCUMENT"
        )

        # --- Step 3b: build ChromaDB input lists ---
        # ChromaDB requires parallel lists of ids / embeddings / documents / metadatas
        ids = [str(uuid.uuid4()) for _ in documents]
        metadatas = [
            # ChromaDB metadata values must be str, int, float, or bool
            {k: str(v) for k, v in doc.metadata.items()}
            for doc in documents
        ]

        # --- Step 3c: upsert into the collection ---
        # upsert = insert new, update existing (idempotent)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Find the *k* document chunks most semantically similar to *query*.

        Process
        -------
        1. Embed the query with task_type="RETRIEVAL_QUERY".
        2. Ask ChromaDB for the *k* nearest vectors (cosine similarity).
        3. Wrap results back into Document objects and return with distances.

        Args:
            query: The user's natural-language question.
            k:     Number of chunks to return.

        Returns:
            List of (Document, distance) tuples sorted by relevance
            (lowest distance = most similar).
        """
        # --- Step 4a: embed the query ---
        # Note: task_type="RETRIEVAL_QUERY" – different from document embedding!
        query_embedding = _get_embedding(
            self._genai_client, query, task_type="RETRIEVAL_QUERY"
        )

        # --- Step 4b: search the vector store ---
        # Request k results; ChromaDB raises if k > collection size, so we
        # catch that and retry with the actual count.
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Fewer documents in the collection than k – retrieve all
            actual_count = self._collection.count()
            if actual_count == 0:
                return []
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_count,
                include=["documents", "metadatas", "distances"],
            )

        # --- Step 4c: unpack ChromaDB results ---
        # results["documents"] is a list-of-lists because we sent one query;
        # take index 0 to get the flat list for our single query.
        retrieved: List[Tuple[Document, float]] = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            doc = Document(page_content=text, metadata=meta)
            retrieved.append((doc, distance))

        return retrieved

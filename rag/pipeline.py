"""RAG pipeline: orchestrates loading, splitting, embedding, retrieval and generation."""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag.document_loader import load_documents
from rag.text_splitter import split_documents
from rag.vector_store import build_vector_store, load_vector_store

_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following retrieved context to answer the question.
If you don't know the answer based on the context, say that you don't know.

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Usage example::

        pipeline = RAGPipeline()
        pipeline.ingest("path/to/document.pdf")
        answer = pipeline.query("What is the document about?")
        print(answer)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4,
        persist_directory: Optional[str] = None,
    ) -> None:
        """Initialise the pipeline.

        Args:
            model: OpenAI chat model name.
            chunk_size: Maximum characters per text chunk.
            chunk_overlap: Character overlap between consecutive chunks.
            k: Number of documents to retrieve per query.
            persist_directory: Optional path to persist the vector store.
        """
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.persist_directory = persist_directory
        self._vector_store = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, source: str) -> int:
        """Load, split and embed documents from *source*.

        Args:
            source: File path (PDF or text), directory path, or HTTP(S) URL.

        Returns:
            Number of chunks added to the vector store.
        """
        documents = load_documents(source)
        chunks = split_documents(documents, self.chunk_size, self.chunk_overlap)

        if self._vector_store is None:
            self._vector_store = build_vector_store(
                chunks,
                persist_directory=self.persist_directory,
            )
        else:
            self._vector_store.add_documents(chunks)

        return len(chunks)

    def load(self, persist_directory: Optional[str] = None) -> None:
        """Load a previously persisted vector store.

        Args:
            persist_directory: Path to the persisted vector store.  Falls
                back to the value set during construction.
        """
        directory = persist_directory or self.persist_directory
        if directory is None:
            raise ValueError(
                "persist_directory must be provided either at construction "
                "time or when calling load()."
            )
        self._vector_store = load_vector_store(directory)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        """Answer a question using the ingested documents.

        Args:
            question: Natural-language question.

        Returns:
            Generated answer string.
        """
        if self._vector_store is None:
            raise RuntimeError(
                "No documents have been ingested. Call ingest() or load() first."
            )

        retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )

        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
        llm = ChatOpenAI(model=self.model, temperature=0)

        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain.invoke(question)

    def retrieve(self, question: str) -> List[Document]:
        """Retrieve relevant document chunks without generating an answer.

        Args:
            question: Natural-language question.

        Returns:
            List of relevant Document chunks.
        """
        if self._vector_store is None:
            raise RuntimeError(
                "No documents have been ingested. Call ingest() or load() first."
            )

        retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )
        return retriever.invoke(question)

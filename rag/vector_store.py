"""Vector store utilities for the RAG pipeline."""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def build_vector_store(
    documents: List[Document],
    persist_directory: Optional[str] = None,
    collection_name: str = "rag_collection",
) -> Chroma:
    """Create a Chroma vector store from a list of documents.

    Args:
        documents: List of LangChain Document objects to embed and store.
        persist_directory: Directory to persist the vector store.  When
            ``None`` the store lives only in memory.
        collection_name: Name of the Chroma collection.

    Returns:
        A populated ``Chroma`` vector store instance.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def load_vector_store(
    persist_directory: str,
    collection_name: str = "rag_collection",
) -> Chroma:
    """Load a previously persisted Chroma vector store.

    Args:
        persist_directory: Directory where the vector store was persisted.
        collection_name: Name of the Chroma collection.

    Returns:
        A ``Chroma`` vector store instance backed by the persisted data.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

"""Text splitting utilities for the RAG pipeline."""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: List of LangChain Document objects to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)

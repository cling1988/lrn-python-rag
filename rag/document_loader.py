"""Document loading utilities for the RAG pipeline."""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)


def load_pdf(file_path: str) -> List[Document]:
    """Load documents from a PDF file."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_text(file_path: str) -> List[Document]:
    """Load documents from a plain-text file."""
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_directory(directory_path: str, glob: str = "**/*.txt") -> List[Document]:
    """Load all matching documents from a directory.

    Args:
        directory_path: Path to the directory.
        glob: Glob pattern for file matching (default: all .txt files).
    """
    loader = DirectoryLoader(
        directory_path,
        glob=glob,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def load_web(urls: List[str]) -> List[Document]:
    """Load documents from web URLs."""
    loader = WebBaseLoader(urls)
    return loader.load()


def load_documents(source: str) -> List[Document]:
    """Load documents from a file path or URL.

    Dispatches to the appropriate loader based on the source type:
    - URL  → WebBaseLoader
    - .pdf → PyPDFLoader
    - directory → DirectoryLoader (txt files)
    - otherwise → TextLoader

    Args:
        source: File path, directory path, or HTTP(S) URL.

    Returns:
        List of LangChain Document objects.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_web([source])
    if os.path.isdir(source):
        return load_directory(source)
    if source.lower().endswith(".pdf"):
        return load_pdf(source)
    return load_text(source)

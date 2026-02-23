"""
document_loader.py – Step 1 of the RAG pipeline: load raw text from a source.

A RAG system needs documents to search over. This module handles reading
content from different source types and normalising everything into a simple
Document dataclass so the rest of the pipeline doesn't need to know where
the text came from.

Supported sources
-----------------
* Plain-text files  (.txt, .md, …)
* PDF files         (.pdf)  – embedded text extracted with pypdf; image-based
                             (scanned) pages fall back to OCR via pdf2image +
                             pytesseract (requires Tesseract to be installed)
* Directories       – loads every .txt file found recursively
* Web URLs          – fetches the page and strips HTML tags
"""

import glob
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment,misc]

try:
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover
    convert_from_path = None  # type: ignore[assignment]

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single piece of text together with optional metadata.

    Attributes:
        page_content: The raw text of this document or chunk.
        metadata:     Arbitrary key/value pairs (e.g. source file, page number).
    """

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_text_file(file_path: str) -> List[Document]:
    """Load a plain-text file and return it as a single Document.

    We read the entire file at once; the text splitter (Step 2) will break
    it into smaller, overlapping chunks later.

    Args:
        file_path: Absolute or relative path to the text file.

    Returns:
        A one-element list containing the file's content as a Document.
    """
    with open(file_path, encoding="utf-8") as fh:
        text = fh.read()

    # Store the source path in metadata so we can trace chunks back later
    return [Document(page_content=text, metadata={"source": file_path})]


def _ocr_pdf_page(file_path: str, page_number: int) -> str:
    """Extract text from a single PDF page using OCR.

    Used as a fallback when pypdf cannot extract embedded text (e.g. for
    scanned / image-based PDFs).  Requires ``pdf2image`` and ``pytesseract``
    plus a working Tesseract installation on the system.

    Args:
        file_path:   Path to the PDF file.
        page_number: 1-based page index.

    Returns:
        OCR-extracted text string, or empty string if OCR is unavailable.
    """
    if convert_from_path is None:
        raise ImportError(
            "pdf2image is required for OCR on image-based PDFs. "
            "Install it with: pip install pdf2image"
        )
    if pytesseract is None:
        raise ImportError(
            "pytesseract is required for OCR on image-based PDFs. "
            "Install it with: pip install pytesseract"
        )

    try:
        images = convert_from_path(
            file_path, first_page=page_number, last_page=page_number
        )
    except Exception as exc:
        raise RuntimeError(
            f"pdf2image failed to convert page {page_number} of '{file_path}'. "
            "Ensure poppler-utils is installed (e.g. apt install poppler-utils)."
        ) from exc

    if not images:
        return ""

    try:
        return pytesseract.image_to_string(images[0])
    except Exception as exc:
        raise RuntimeError(
            f"Tesseract OCR failed on page {page_number} of '{file_path}'. "
            "Ensure Tesseract is installed and accessible in your PATH "
            "(e.g. apt install tesseract-ocr)."
        ) from exc


def load_pdf_file(file_path: str) -> List[Document]:
    """Extract text from every page of a PDF file.

    First attempts to extract embedded text with pypdf (fast, no system
    dependencies).  If a page yields no text – as happens with scanned /
    image-based PDFs – the page is converted to an image with ``pdf2image``
    and OCR is run via ``pytesseract`` to recover the text.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        One Document per page in the PDF.
    """
    if PdfReader is None:
        raise ImportError(
            "pypdf is required to load PDF files. "
            "Install it with: pip install pypdf"
        )

    reader = PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        # extract_text() returns a string; it may be empty for image-only pages
        text = page.extract_text() or ""

        # Fallback: if pypdf found no text (image-based page), run OCR
        if not text.strip():
            text = _ocr_pdf_page(file_path, page_number)

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_number},
                )
            )

    return documents


def load_directory(directory_path: str, pattern: str = "**/*.txt") -> List[Document]:
    """Load all text files matching *pattern* inside a directory tree.

    Uses Python's built-in glob module – no external dependencies.

    Args:
        directory_path: Root directory to search.
        pattern:        Glob pattern relative to directory_path.
                        Defaults to all .txt files (recursive).

    Returns:
        One Document per file found.
    """
    documents = []

    # glob.glob with recursive=True expands the ** wildcard across sub-dirs
    search_pattern = os.path.join(directory_path, pattern)
    file_paths = sorted(glob.glob(search_pattern, recursive=True))

    for file_path in file_paths:
        if os.path.isfile(file_path):
            # Re-use load_text_file so each file gets correct metadata
            documents.extend(load_text_file(file_path))

    return documents


def load_web_page(url: str) -> List[Document]:
    """Fetch a web page and return its visible text as a single Document.

    Uses Python's built-in urllib – no requests library needed.
    HTML tags are stripped with a simple regex so the model sees clean text.

    Args:
        url: An http:// or https:// URL.

    Returns:
        A one-element list with the page's visible text.
    """
    # Fetch the raw HTML bytes
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 – URL is validated as http/https by the caller (load_documents)
        html_bytes = response.read()

    # Decode to string; fall back to latin-1 if UTF-8 fails
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        html = html_bytes.decode("latin-1")

    # Remove <script> and <style> blocks first (they add noise)
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", html)

    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return [Document(page_content=text, metadata={"source": url})]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_documents(source: str) -> List[Document]:
    """Load documents from any supported source.

    This is the main entry point for Step 1. It inspects the *source* string
    and calls the right loader automatically:

    * Starts with http/https → :func:`load_web_page`
    * Is a directory        → :func:`load_directory`
    * Ends with .pdf        → :func:`load_pdf_file`
    * Otherwise             → :func:`load_text_file`

    Args:
        source: File path, directory path, or URL string.

    Returns:
        List of Document objects ready to be split in the next step.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_web_page(source)

    if os.path.isdir(source):
        return load_directory(source)

    if source.lower().endswith(".pdf"):
        return load_pdf_file(source)

    return load_text_file(source)

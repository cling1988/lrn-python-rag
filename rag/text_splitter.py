"""
text_splitter.py – Step 2 of the RAG pipeline: split documents into chunks.

Why do we split documents?
---------------------------
Embedding models have a fixed input-size limit (e.g. 2048 tokens), and
retrieving a 50-page PDF as a single blob is unhelpful – we want to find
the *specific paragraphs* that answer the user's question.

Splitting strategy: Recursive Character Text Splitter
------------------------------------------------------
We try to split on natural boundaries in order of preference:

  English / universal (tried first):
  1. Double newline (paragraph break)
  2. Single newline (line break)
  3. Period + space (English sentence boundary)
  4. Space (English word boundary)

  Chinese (tried when English separators are absent):
  5. 。 Full stop
  6. ！ Exclamation mark
  7. ？ Question mark
  8. ；Semicolon
  9. ，Comma
  10. 、Enumeration comma

  Last resort:
  11. Single character

Chinese text has no spaces between words, so word-boundary splitting (step 4)
would produce only one piece for a Chinese paragraph. By adding Chinese
punctuation marks as explicit separators (steps 5-10), the splitter finds
natural sentence and phrase boundaries that work for Chinese documents.
English separators are tried first so that mixed-language documents are split
on English boundaries before falling through to Chinese punctuation.

Overlap
-------
Each chunk overlaps the previous one by *chunk_overlap* characters.
This preserves context that might otherwise be cut in half at a boundary
and improves retrieval quality for questions that span chunk edges.
"""

from collections import deque
from typing import List

from rag.document_loader import Document


# Natural split points ordered from coarsest to finest granularity.
# English separators come before Chinese punctuation so that mixed-language
# documents are split on English sentence/word boundaries first.  For pure
# Chinese text `. ` and ` ` will not be present, so the algorithm falls
# through to the Chinese punctuation marks naturally.
_SEPARATORS = [
    "\n\n",   # paragraph break (universal)
    "\n",     # line break (universal)
    ". ",     # English sentence boundary (period + space)
    " ",      # English word boundary
    "。",     # Chinese full stop
    "！",     # Chinese exclamation mark
    "？",     # Chinese question mark
    "；",     # Chinese semicolon
    "，",     # Chinese comma
    "、",     # Chinese enumeration comma
    "",       # single character (last resort)
]


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """Recursively split *text* into chunks of at most *chunk_size* characters.

    Algorithm
    ---------
    1. Try each separator in ``_SEPARATORS`` until one produces pieces that
       fit within *chunk_size*.
    2. Assemble pieces into chunks, re-merging small pieces and carrying
       *chunk_overlap* characters from the previous chunk into the next.

    Args:
        text:          Raw text string to split.
        chunk_size:    Maximum number of characters per chunk (default 1000).
        chunk_overlap: Characters to repeat at the start of each new chunk
                       so context isn't lost at boundaries (default 200).

    Returns:
        List of text strings, each at most *chunk_size* characters long.
    """
    # --- Step 2a: find the right separator ---
    # Walk through separators from coarsest to finest.
    # Use the first one that actually appears in the text.
    separator = _SEPARATORS[-1]  # fallback: single character
    for sep in _SEPARATORS:
        if sep == "" or sep in text:
            separator = sep
            break

    # --- Step 2b: split text on the chosen separator ---
    if separator:
        pieces = text.split(separator)
    else:
        # Empty separator → split into individual characters (last resort)
        pieces = list(text)

    # --- Step 2c: merge pieces into chunks with overlap ---
    chunks: List[str] = []
    # Use deque for O(1) popleft when trimming the overlap window
    current: deque = deque()
    current_len = 0            # total character count of current chunk

    for piece in pieces:
        piece_len = len(piece) + len(separator)  # account for the separator

        if current_len + piece_len > chunk_size and current:
            # The current chunk is full – flush it
            chunk_text = separator.join(current)
            chunks.append(chunk_text)

            # Carry overlap: remove pieces from the front until the remaining
            # content is short enough to serve as the overlap prefix.
            while current and current_len > chunk_overlap:
                removed = current.popleft()  # O(1) with deque
                current_len -= len(removed) + len(separator)

        current.append(piece)
        current_len += piece_len

    # Flush the last (possibly partial) chunk
    if current:
        chunks.append(separator.join(current))

    # --- Step 2d: recursively split any chunk that is still too large ---
    # This happens when a single piece (e.g. a very long sentence) exceeds
    # chunk_size after all separators have been exhausted at this level.
    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) > chunk_size and separator != "":
            # Try again with the next-finer separator
            next_sep_index = _SEPARATORS.index(separator) + 1
            sub_text = chunk
            # Recurse using a finer separator by temporarily narrowing the list
            finer_chunks = _split_with_separator(
                sub_text,
                _SEPARATORS[next_sep_index:],
                chunk_size,
                chunk_overlap,
            )
            final_chunks.extend(finer_chunks)
        else:
            final_chunks.append(chunk)

    return [c for c in final_chunks if c.strip()]  # drop empty strings


def _split_with_separator(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Internal helper: split *text* using the first matching separator in
    *separators*, recursing with finer separators as needed."""
    if not separators:
        return [text]

    separator = separators[0]
    for sep in separators:
        if sep == "" or sep in text:
            separator = sep
            break

    pieces = text.split(separator) if separator else list(text)
    chunks: List[str] = []
    current: deque = deque()
    current_len = 0

    for piece in pieces:
        piece_len = len(piece) + len(separator)
        if current_len + piece_len > chunk_size and current:
            chunks.append(separator.join(current))
            while current and current_len > chunk_overlap:
                removed = current.popleft()  # O(1) with deque
                current_len -= len(removed) + len(separator)
        current.append(piece)
        current_len += piece_len

    if current:
        chunks.append(separator.join(current))

    # Recurse for oversized chunks
    final: List[str] = []
    for chunk in chunks:
        if len(chunk) > chunk_size and len(separators) > 1:
            final.extend(
                _split_with_separator(chunk, separators[1:], chunk_size, chunk_overlap)
            )
        else:
            final.append(chunk)

    return final


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split a list of Documents into smaller overlapping chunks.

    Each output Document inherits the metadata from its parent so we can
    always trace a retrieved chunk back to its original source file or URL.

    Args:
        documents:     Documents to split (output of Step 1).
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.

    Returns:
        Flat list of chunk Documents ready to be embedded in Step 3.
    """
    chunks: List[Document] = []

    for doc in documents:
        text_chunks = split_text(doc.page_content, chunk_size, chunk_overlap)

        for i, chunk_text in enumerate(text_chunks):
            # Copy parent metadata and add chunk index for debugging
            meta = {**doc.metadata, "chunk_index": i}
            chunks.append(Document(page_content=chunk_text, metadata=meta))

    return chunks

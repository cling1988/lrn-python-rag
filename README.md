# lrn-python-rag
Build a RAG system with Python

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline in Python. It allows you to:

1. **Ingest** documents (PDF, plain text, directories, or web URLs) into a vector store.
2. **Retrieve** semantically relevant chunks for a given query.
3. **Generate** answers grounded in the retrieved context using an OpenAI chat model.

### Architecture

```
Documents → Loader → Text Splitter → Embeddings → Vector Store (Chroma)
                                                          ↓
Question ──────────────────────────────────── Retriever → LLM → Answer
```

## Requirements

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/cling1988/lrn-python-rag.git
cd lrn-python-rag

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
cp .env.example .env        # then edit .env and add your key
# or simply:
export OPENAI_API_KEY=sk-...
```

Create a `.env` file (never commit this file):

```
OPENAI_API_KEY=sk-your-key-here
```

## Usage

### Python API

```python
from rag import RAGPipeline

pipeline = RAGPipeline()

# Ingest a document (PDF, text file, directory, or URL)
pipeline.ingest("data/docs/my_document.pdf")

# Ask a question
answer = pipeline.query("What is the main topic of the document?")
print(answer)

# Retrieve relevant chunks without generating an answer
chunks = pipeline.retrieve("important concept")
for chunk in chunks:
    print(chunk.page_content)
```

Persist the vector store to disk so you don't re-embed on every run:

```python
pipeline = RAGPipeline(persist_directory="chroma_db")
pipeline.ingest("data/docs/my_document.pdf")

# Later, reload without re-ingesting
pipeline2 = RAGPipeline(persist_directory="chroma_db")
pipeline2.load()
answer = pipeline2.query("What is discussed in chapter 3?")
```

### CLI

```bash
# Ingest a document
python main.py ingest data/docs/my_document.pdf --persist-dir chroma_db

# Query using a persisted vector store
python main.py query "What is the main topic?" --persist-dir chroma_db

# Ingest and query in one step
python main.py query "What is discussed?" --source data/docs/my_document.pdf

# Ingest a URL
python main.py ingest https://example.com/article --persist-dir chroma_db
```

## Project Structure

```
lrn-python-rag/
├── rag/
│   ├── __init__.py          # Package entry point (exports RAGPipeline)
│   ├── document_loader.py   # Load PDFs, text files, directories, URLs
│   ├── text_splitter.py     # Split documents into overlapping chunks
│   ├── vector_store.py      # Build / load Chroma vector store
│   └── pipeline.py          # RAGPipeline: ingest, retrieve, query
├── tests/
│   └── test_rag.py          # Unit tests (mocked OpenAI / Chroma)
├── main.py                  # CLI entry point
├── requirements.txt
└── .env.example
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Configuration

`RAGPipeline` accepts the following constructor parameters:

| Parameter | Default | Description |
|---|---|---|
| `model` | `"gpt-4o-mini"` | OpenAI chat model |
| `chunk_size` | `1000` | Characters per text chunk |
| `chunk_overlap` | `200` | Overlap between chunks |
| `k` | `4` | Number of chunks to retrieve |
| `persist_directory` | `None` | Directory for vector store persistence |

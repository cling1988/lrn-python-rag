# lrn-python-rag
Build a RAG system with Python

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline in Python **from scratch** – no LangChain, no heavy frameworks. It uses:

* **[Google GenAI](https://ai.google.dev/)** – `text-embedding-004` for embeddings, `gemini-2.0-flash` for generation
* **[ChromaDB](https://www.trychroma.com/)** – lightweight embedded vector database
* **[pypdf](https://pypdf.readthedocs.io/)** – pure-Python PDF text extraction
* Python standard library (`urllib`, `glob`, `re`) – everything else

### How RAG works

```
                  ┌──── INGESTION (offline) ────────────────────────────────┐
                  │                                                          │
 Document  ──►  Step 1     ──►    Step 2       ──►   Step 3                 │
 (PDF/txt/URL)  Load text    Split into chunks    Embed + store in ChromaDB  │
                  │                                                          │
                  └─────────────────────────────────────────────────────────┘

                  ┌──── QUERYING (online) ───────────────────────────────────┐
                  │                                                           │
 Question  ──►  Step 4       ──►   Step 5                                    │
               Embed query        Build prompt with       ──►  Answer        │
               Retrieve top-K     retrieved context                          │
               chunks             Call Gemini                                │
                  │                                                           │
                  └──────────────────────────────────────────────────────────┘
```

### Project structure

```
lrn-python-rag/
├── rag/
│   ├── __init__.py          # Package entry point
│   ├── document_loader.py   # Step 1 – load text from PDF / txt / dir / URL
│   ├── text_splitter.py     # Step 2 – split text into overlapping chunks
│   ├── vector_store.py      # Step 3 & 4 – embed with Google GenAI, store & search in ChromaDB
│   └── pipeline.py          # Steps 1-5 orchestrated in RAGPipeline
├── tests/
│   └── test_rag.py          # 24 unit tests (all mocked, no API key needed)
├── main.py                  # CLI entry point
├── requirements.txt
└── .env.example
```

## Requirements

* Python 3.10+
* A [Google AI API key](https://aistudio.google.com/app/apikey) (free tier available)

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/cling1988/lrn-python-rag.git
cd lrn-python-rag

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your-key-here
```

## Python API

```python
from rag import RAGPipeline

# Create a pipeline (persists the vector store to disk)
pipeline = RAGPipeline(persist_directory="chroma_db")

# Step 1-3: ingest a document (PDF, text file, directory, or URL)
pipeline.ingest("data/docs/my_document.pdf")

# Steps 4-5: ask a question and get a grounded answer
answer = pipeline.query("What is the main topic of the document?")
print(answer)

# Step 4 only: inspect what chunks were retrieved (useful for debugging)
results = pipeline.retrieve("important concept")
for doc, distance in results:
    print(f"[dist={distance:.3f}] {doc.page_content[:80]}…")
```

Reload a previously persisted vector store without re-embedding:

```python
pipeline = RAGPipeline(persist_directory="chroma_db")
pipeline.load()  # no re-embedding; reads from disk
answer = pipeline.query("What is discussed in chapter 3?")
```

## CLI

```bash
# Ingest a document
python main.py ingest data/docs/report.pdf --persist-dir chroma_db

# Query a persisted store
python main.py query "What are the key findings?" --persist-dir chroma_db

# Ingest + query in one step
python main.py query "Summarise the document" --source data/docs/report.pdf

# Use a different Gemini model
python main.py query "What is RAG?" --source notes.txt --model gemini-1.5-pro
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests are fully mocked – no API key or internet connection required.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `model` | `"gemini-2.0-flash"` | Gemini generation model |
| `chunk_size` | `1000` | Characters per text chunk |
| `chunk_overlap` | `200` | Overlap characters between chunks |
| `k` | `4` | Number of chunks to retrieve |
| `persist_directory` | `None` | ChromaDB persistence directory (in-memory when `None`) |
| `api_key` | env `GOOGLE_API_KEY` | Google AI API key |

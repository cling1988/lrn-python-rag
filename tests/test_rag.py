"""
Tests for the RAG pipeline.

All Google GenAI and ChromaDB calls are mocked so tests run without
a real API key or internet connection.
"""

import os
from unittest.mock import MagicMock, patch, call

import pytest

from rag.document_loader import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(content: str, source: str = "test.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


# ---------------------------------------------------------------------------
# document_loader
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    def test_load_text_file(self, tmp_path):
        from rag.document_loader import load_documents

        txt = tmp_path / "sample.txt"
        txt.write_text("Hello world\nSecond line", encoding="utf-8")

        docs = load_documents(str(txt))

        assert len(docs) == 1
        assert "Hello world" in docs[0].page_content
        assert docs[0].metadata["source"] == str(txt)

    def test_load_directory(self, tmp_path):
        from rag.document_loader import load_documents

        (tmp_path / "a.txt").write_text("Document A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Document B", encoding="utf-8")

        docs = load_documents(str(tmp_path))
        contents = " ".join(d.page_content for d in docs)

        assert "Document A" in contents
        assert "Document B" in contents

    def test_load_web_page(self):
        from rag.document_loader import load_documents

        # Simulate urllib returning minimal HTML
        fake_html = b"<html><body><p>Hello from the web</p></body></html>"
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = fake_html

        with patch("rag.document_loader.urllib.request.urlopen", return_value=mock_response):
            docs = load_documents("https://example.com")

        assert len(docs) == 1
        assert "Hello from the web" in docs[0].page_content
        assert docs[0].metadata["source"] == "https://example.com"

    def test_load_pdf_dispatches_to_pdf_loader(self, tmp_path):
        from rag.document_loader import load_documents

        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF page content"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("rag.document_loader.PdfReader", return_value=mock_reader):
            docs = load_documents(str(fake_pdf))

        assert len(docs) == 1
        assert docs[0].page_content == "PDF page content"
        assert docs[0].metadata["page"] == 1

    def test_document_metadata_preserved(self, tmp_path):
        from rag.document_loader import load_text_file

        txt = tmp_path / "notes.txt"
        txt.write_text("important note", encoding="utf-8")

        docs = load_text_file(str(txt))
        assert docs[0].metadata["source"] == str(txt)


# ---------------------------------------------------------------------------
# text_splitter
# ---------------------------------------------------------------------------


class TestSplitText:
    def test_short_text_not_split(self):
        from rag.text_splitter import split_text

        chunks = split_text("Hello world.", chunk_size=1000, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_long_text_is_split(self):
        from rag.text_splitter import split_text

        long_text = "word " * 500  # 2500 characters
        chunks = split_text(long_text, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1
        for chunk in chunks:
            # chunks may slightly exceed chunk_size due to the separator being re-added
            assert len(chunk) <= 300

    def test_overlap_carries_content(self):
        from rag.text_splitter import split_text

        # Build text with numbered paragraphs so we can check overlap
        paragraphs = [f"Paragraph {i}. " * 10 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = split_text(text, chunk_size=200, chunk_overlap=50)
        # With overlap there should be more chunks than without
        chunks_no_overlap = split_text(text, chunk_size=200, chunk_overlap=0)
        assert len(chunks) >= len(chunks_no_overlap)

    def test_empty_string_returns_empty_list(self):
        from rag.text_splitter import split_text

        chunks = split_text("", chunk_size=500, chunk_overlap=50)
        assert chunks == []


class TestSplitDocuments:
    def test_splits_into_chunks(self):
        from rag.text_splitter import split_documents

        long_text = "word " * 500  # 2500 chars
        docs = [_make_doc(long_text)]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1

    def test_preserves_metadata(self):
        from rag.text_splitter import split_documents

        doc = Document(
            page_content="A " * 300, metadata={"source": "my_file.txt", "page": 1}
        )
        chunks = split_documents([doc], chunk_size=100, chunk_overlap=10)

        for chunk in chunks:
            assert chunk.metadata["source"] == "my_file.txt"
            assert chunk.metadata["page"] == 1

    def test_chunk_index_added_to_metadata(self):
        from rag.text_splitter import split_documents

        doc = _make_doc("word " * 500)
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)

        for i, chunk in enumerate(chunks):
            assert "chunk_index" in chunk.metadata

    def test_single_short_document_not_split(self):
        from rag.text_splitter import split_documents

        doc = _make_doc("Short text.")
        chunks = split_documents([doc], chunk_size=1000, chunk_overlap=100)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."


# ---------------------------------------------------------------------------
# vector_store (VectorStore)
# ---------------------------------------------------------------------------


class TestVectorStore:
    @patch("rag.vector_store.chromadb.EphemeralClient")
    @patch("rag.vector_store.genai.Client")
    def test_add_documents_calls_upsert(self, MockGenaiClient, MockChromaClient):
        from rag.vector_store import VectorStore

        # Set up fake embedding response
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1, 0.2, 0.3]
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [fake_embedding, fake_embedding]

        mock_genai = MagicMock()
        mock_genai.models.embed_content.return_value = mock_embed_response
        MockGenaiClient.return_value = mock_genai

        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection
        MockChromaClient.return_value = mock_chroma

        store = VectorStore(api_key="fake-key")
        docs = [_make_doc("chunk one"), _make_doc("chunk two")]
        store.add_documents(docs)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert len(call_kwargs["ids"]) == 2
        assert call_kwargs["documents"] == ["chunk one", "chunk two"]

    @patch("rag.vector_store.chromadb.EphemeralClient")
    @patch("rag.vector_store.genai.Client")
    def test_similarity_search_returns_documents(self, MockGenaiClient, MockChromaClient):
        from rag.vector_store import VectorStore

        # Embedding for the query
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1, 0.2, 0.3]
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [fake_embedding]

        mock_genai = MagicMock()
        mock_genai.models.embed_content.return_value = mock_embed_response
        MockGenaiClient.return_value = mock_genai

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["relevant text"]],
            "metadatas": [[{"source": "test.txt"}]],
            "distances": [[0.12]],
        }
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection
        MockChromaClient.return_value = mock_chroma

        store = VectorStore(api_key="fake-key")
        results = store.similarity_search("test query", k=1)

        assert len(results) == 1
        doc, distance = results[0]
        assert doc.page_content == "relevant text"
        assert distance == 0.12

    @patch("rag.vector_store.chromadb.EphemeralClient")
    @patch("rag.vector_store.genai.Client")
    def test_add_empty_documents_skipped(self, MockGenaiClient, MockChromaClient):
        from rag.vector_store import VectorStore

        mock_collection = MagicMock()
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection
        MockChromaClient.return_value = mock_chroma

        store = VectorStore(api_key="fake-key")
        store.add_documents([])

        mock_collection.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    @patch("rag.pipeline.VectorStore")
    @patch("rag.pipeline.load_documents")
    def test_ingest_returns_chunk_count(self, mock_load, MockVectorStore):
        from rag.pipeline import RAGPipeline

        mock_load.return_value = [_make_doc("text " * 50)]
        MockVectorStore.return_value = MagicMock()

        pipeline = RAGPipeline(chunk_size=100, chunk_overlap=10, api_key="fake")
        n = pipeline.ingest("fake_source.txt")

        assert isinstance(n, int)
        assert n >= 1
        MockVectorStore.assert_called_once()

    @patch("rag.pipeline.VectorStore")
    @patch("rag.pipeline.load_documents")
    def test_ingest_twice_reuses_vector_store(self, mock_load, MockVectorStore):
        from rag.pipeline import RAGPipeline

        mock_vs = MagicMock()
        MockVectorStore.return_value = mock_vs
        mock_load.return_value = [_make_doc("chunk")]

        pipeline = RAGPipeline(api_key="fake")
        pipeline.ingest("source1.txt")
        pipeline.ingest("source2.txt")

        # VectorStore should only be constructed once
        MockVectorStore.assert_called_once()
        # add_documents called twice (once per ingest)
        assert mock_vs.add_documents.call_count == 2

    def test_query_raises_without_ingestion(self):
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(api_key="fake")
        with pytest.raises(RuntimeError, match="No documents have been ingested"):
            pipeline.query("What is this about?")

    def test_retrieve_raises_without_ingestion(self):
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(api_key="fake")
        with pytest.raises(RuntimeError, match="No documents have been ingested"):
            pipeline.retrieve("What is this about?")

    def test_load_raises_without_persist_directory(self):
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(api_key="fake")
        with pytest.raises(ValueError, match="persist_directory"):
            pipeline.load()

    @patch("rag.pipeline.VectorStore")
    def test_load_uses_constructor_persist_directory(self, MockVectorStore):
        from rag.pipeline import RAGPipeline

        MockVectorStore.return_value = MagicMock()
        pipeline = RAGPipeline(persist_directory="/tmp/db", api_key="fake")
        pipeline.load()

        MockVectorStore.assert_called_once_with(
            api_key="fake", persist_directory="/tmp/db"
        )

    @patch("rag.pipeline.VectorStore")
    @patch("rag.pipeline.load_documents")
    def test_query_uses_retrieved_context(self, mock_load, MockVectorStore):
        from rag.pipeline import RAGPipeline

        retrieved_doc = _make_doc("Paris is the capital of France.")
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [(retrieved_doc, 0.05)]
        MockVectorStore.return_value = mock_vs

        mock_load.return_value = [_make_doc("original")]

        # Mock the Gemini generate_content response
        mock_response = MagicMock()
        mock_response.text = "Paris."
        mock_genai = MagicMock()
        mock_genai.models.generate_content.return_value = mock_response

        pipeline = RAGPipeline(api_key="fake")
        pipeline._genai_client = mock_genai
        pipeline.ingest("fake.txt")

        answer = pipeline.query("What is the capital of France?")

        # Verify retrieval was called with the question
        mock_vs.similarity_search.assert_called_once_with(
            "What is the capital of France?", k=pipeline.k
        )
        # Verify Gemini was called with a prompt containing the context
        generate_call_args = mock_genai.models.generate_content.call_args
        prompt_sent = generate_call_args.kwargs["contents"]
        assert "Paris is the capital of France." in prompt_sent
        assert answer == "Paris."

    @patch("rag.pipeline.VectorStore")
    @patch("rag.pipeline.load_documents")
    def test_retrieve_returns_documents(self, mock_load, MockVectorStore):
        from rag.pipeline import RAGPipeline

        expected = [(_make_doc("relevant chunk"), 0.1)]
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = expected
        MockVectorStore.return_value = mock_vs
        mock_load.return_value = [_make_doc("original")]

        pipeline = RAGPipeline(api_key="fake")
        pipeline.ingest("fake.txt")
        result = pipeline.retrieve("capital of France?")

        mock_vs.similarity_search.assert_called_once_with("capital of France?", k=pipeline.k)
        assert result == expected

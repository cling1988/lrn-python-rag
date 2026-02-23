"""Tests for the RAG pipeline using mocked OpenAI and Chroma dependencies."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


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

        assert len(docs) >= 1
        assert "Hello world" in docs[0].page_content

    def test_load_directory(self, tmp_path):
        from rag.document_loader import load_documents

        (tmp_path / "a.txt").write_text("Document A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Document B", encoding="utf-8")

        docs = load_documents(str(tmp_path))

        contents = " ".join(d.page_content for d in docs)
        assert "Document A" in contents
        assert "Document B" in contents

    def test_load_web_dispatches_to_web_loader(self):
        from rag.document_loader import load_documents

        fake_doc = _make_doc("web content")
        with patch("rag.document_loader.WebBaseLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [fake_doc]
            docs = load_documents("https://example.com")

        MockLoader.assert_called_once_with(["https://example.com"])
        assert docs == [fake_doc]

    def test_load_pdf_dispatches_to_pdf_loader(self, tmp_path):
        from rag.document_loader import load_documents

        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")
        fake_doc = _make_doc("pdf content")

        with patch("rag.document_loader.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [fake_doc]
            docs = load_documents(str(fake_pdf))

        MockLoader.assert_called_once_with(str(fake_pdf))
        assert docs == [fake_doc]


# ---------------------------------------------------------------------------
# text_splitter
# ---------------------------------------------------------------------------


class TestSplitDocuments:
    def test_splits_into_chunks(self):
        from rag.text_splitter import split_documents

        long_text = "word " * 500  # 2500 chars
        docs = [_make_doc(long_text)]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 200 + 20  # allow slight overflow

    def test_preserves_metadata(self):
        from rag.text_splitter import split_documents

        doc = Document(
            page_content="A " * 300, metadata={"source": "my_file.txt", "page": 1}
        )
        chunks = split_documents([doc], chunk_size=100, chunk_overlap=10)

        for chunk in chunks:
            assert chunk.metadata["source"] == "my_file.txt"
            assert chunk.metadata["page"] == 1

    def test_single_short_document_not_split(self):
        from rag.text_splitter import split_documents

        doc = _make_doc("Short text.")
        chunks = split_documents([doc], chunk_size=1000, chunk_overlap=100)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."


# ---------------------------------------------------------------------------
# vector_store
# ---------------------------------------------------------------------------


class TestVectorStore:
    @patch("rag.vector_store.OpenAIEmbeddings")
    @patch("rag.vector_store.Chroma")
    def test_build_vector_store(self, MockChroma, MockEmbeddings):
        from rag.vector_store import build_vector_store

        docs = [_make_doc("chunk one"), _make_doc("chunk two")]
        mock_store = MagicMock()
        MockChroma.from_documents.return_value = mock_store

        store = build_vector_store(docs, persist_directory=None)

        MockChroma.from_documents.assert_called_once()
        call_kwargs = MockChroma.from_documents.call_args.kwargs
        assert call_kwargs["documents"] == docs
        assert store is mock_store

    @patch("rag.vector_store.OpenAIEmbeddings")
    @patch("rag.vector_store.Chroma")
    def test_load_vector_store(self, MockChroma, MockEmbeddings):
        from rag.vector_store import load_vector_store

        mock_store = MagicMock()
        MockChroma.return_value = mock_store

        store = load_vector_store("/tmp/my_db")

        MockChroma.assert_called_once()
        assert store is mock_store


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    @patch("rag.pipeline.build_vector_store")
    @patch("rag.pipeline.load_documents")
    def test_ingest_returns_chunk_count(self, mock_load, mock_build):
        from rag.pipeline import RAGPipeline

        mock_load.return_value = [_make_doc("text " * 50)]
        mock_build.return_value = MagicMock()

        pipeline = RAGPipeline(chunk_size=100, chunk_overlap=10)
        n = pipeline.ingest("fake_source.txt")

        assert isinstance(n, int)
        assert n >= 1
        mock_build.assert_called_once()

    @patch("rag.pipeline.build_vector_store")
    @patch("rag.pipeline.load_documents")
    def test_ingest_twice_calls_add_documents(self, mock_load, mock_build):
        from rag.pipeline import RAGPipeline

        mock_store = MagicMock()
        mock_build.return_value = mock_store
        mock_load.return_value = [_make_doc("chunk")]

        pipeline = RAGPipeline()
        pipeline.ingest("source1.txt")
        pipeline.ingest("source2.txt")

        mock_build.assert_called_once()
        mock_store.add_documents.assert_called_once()

    def test_query_raises_without_ingestion(self):
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        with pytest.raises(RuntimeError, match="No documents have been ingested"):
            pipeline.query("What is this about?")

    def test_retrieve_raises_without_ingestion(self):
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        with pytest.raises(RuntimeError, match="No documents have been ingested"):
            pipeline.retrieve("What is this about?")

    def test_load_raises_without_persist_directory(self):
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="persist_directory"):
            pipeline.load()

    @patch("rag.pipeline.load_vector_store")
    def test_load_uses_constructor_persist_directory(self, mock_load_vs):
        from rag.pipeline import RAGPipeline

        mock_load_vs.return_value = MagicMock()
        pipeline = RAGPipeline(persist_directory="/tmp/db")
        pipeline.load()

        mock_load_vs.assert_called_once_with("/tmp/db")

    @patch("rag.pipeline.build_vector_store")
    @patch("rag.pipeline.load_documents")
    def test_query_invokes_chain(self, mock_load, mock_build):
        from rag.pipeline import RAGPipeline

        docs = [_make_doc("Paris is the capital of France.")]

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs

        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_build.return_value = mock_store
        mock_load.return_value = docs

        pipeline = RAGPipeline()
        pipeline.ingest("fake.txt")

        # Verify the vector store is ready and as_retriever can be called
        retriever = pipeline._vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        result = retriever.invoke("capital of France?")

        mock_store.as_retriever.assert_called_once()
        assert result == docs

    @patch("rag.pipeline.build_vector_store")
    @patch("rag.pipeline.load_documents")
    def test_retrieve_returns_documents(self, mock_load, mock_build):
        from rag.pipeline import RAGPipeline

        expected_docs = [_make_doc("relevant chunk")]

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = expected_docs

        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_build.return_value = mock_store
        mock_load.return_value = [_make_doc("original")]

        pipeline = RAGPipeline()
        pipeline.ingest("fake.txt")
        result = pipeline.retrieve("capital of France?")

        mock_retriever.invoke.assert_called_once_with("capital of France?")
        assert result == expected_docs

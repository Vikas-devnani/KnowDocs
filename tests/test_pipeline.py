"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────────
Unit tests for the RAG pipeline components.
Run with:  pytest tests/ -v
─────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.documents import Document

from src.text_splitter import split_documents, split_text
from src.embeddings import embed_query, embed_texts


# ─────────────────────────────────────────────
#  Text Splitter Tests
# ─────────────────────────────────────────────

class TestTextSplitter:

    def test_split_documents_basic(self):
        """Documents should be split into multiple chunks."""
        long_text = "This is a sentence about technology. " * 50
        docs = [Document(page_content=long_text, metadata={"source": "test.pdf", "page": 1})]
        chunks = split_documents(docs)
        assert len(chunks) > 1, "Long document should produce multiple chunks"

    def test_split_documents_metadata_preserved(self):
        """Each chunk should inherit parent metadata."""
        docs = [Document(
            page_content="Python is great. " * 30,
            metadata={"source": "test.pdf", "page": 2}
        )]
        chunks = split_documents(docs)
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.pdf"

    def test_split_documents_chunk_index(self):
        """Every chunk should have a chunk_index in metadata."""
        docs = [Document(page_content="word " * 200, metadata={"source": "x.pdf", "page": 1})]
        chunks = split_documents(docs)
        for i, chunk in enumerate(chunks):
            assert "chunk_index" in chunk.metadata

    def test_split_text_returns_documents(self):
        """split_text should return Document objects."""
        chunks = split_text("Hello world. " * 50, source="inline")
        assert all(isinstance(c, Document) for c in chunks)

    def test_empty_document(self):
        """Empty document list should return empty chunks."""
        chunks = split_documents([])
        assert chunks == []


# ─────────────────────────────────────────────
#  Embedding Tests
# ─────────────────────────────────────────────

class TestEmbeddings:

    def test_embed_query_returns_vector(self):
        """Query embedding should return a non-empty float list."""
        vec = embed_query("What is machine learning?")
        assert isinstance(vec, list)
        assert len(vec) == 384   # all-MiniLM-L6-v2 dimension
        assert all(isinstance(v, float) for v in vec)

    def test_embed_texts_batch(self):
        """Batch embedding should return one vector per text."""
        texts = ["Deep learning", "Natural language processing", "FAISS vector search"]
        vecs = embed_texts(texts)
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_similar_texts_closer(self):
        """Similar texts should be closer in embedding space."""
        import numpy as np
        v1 = np.array(embed_query("machine learning model"))
        v2 = np.array(embed_query("deep learning neural network"))
        v3 = np.array(embed_query("the cat sat on the mat"))

        # Cosine similarity (embeddings are normalised)
        sim_related = float(np.dot(v1, v2))
        sim_unrelated = float(np.dot(v1, v3))
        assert sim_related > sim_unrelated, \
            "ML texts should be more similar to each other than to unrelated text"


# ─────────────────────────────────────────────
#  FAISS Store Tests  (no real PDF needed)
# ─────────────────────────────────────────────

class TestFAISSStore:

    def test_build_and_search(self, tmp_path, monkeypatch):
        """Build an in-memory index and verify retrieval."""
        from src import faiss_store, config

        # Override paths to use temp dir
        monkeypatch.setattr(config.settings, "faiss_index_path", str(tmp_path / "idx"))
        monkeypatch.setattr(config.settings, "chunks_store_path", str(tmp_path / "chunks.pkl"))

        chunks = [
            Document(page_content="Python is a high-level programming language.",
                     metadata={"source": "test.pdf", "page": 1}),
            Document(page_content="FAISS enables fast similarity search over vectors.",
                     metadata={"source": "test.pdf", "page": 2}),
            Document(page_content="LangChain simplifies building LLM applications.",
                     metadata={"source": "test.pdf", "page": 3}),
        ]

        vs = faiss_store.build_faiss_index(chunks)
        results = faiss_store.similarity_search(vs, "What is Python?", top_k=2)

        assert len(results) == 2
        # The top result should be about Python
        assert "Python" in results[0][0].page_content

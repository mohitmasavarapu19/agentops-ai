"""Tests for Milestone 3: Production-style RAG Subsystem."""

import tempfile
import unittest
from pathlib import Path

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.rag.document_loader import (
    Document,
    chunk_document,
    load_directory,
    load_document,
    split_text,
)
from app.rag.ingest import ingest_knowledge_base
from app.rag.retriever import RAGRetriever


class TestDocumentLoader(unittest.TestCase):
    """Test suite for document loading and chunking utilities."""

    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)

        # Create sample markdown file
        self.sample_file = self.dir_path / "sample_guide.md"
        self.sample_file.write_text(
            "# Guide\n\nAgentOps AI is designed for autonomous intelligence.\n\n"
            "It automates business workflows.",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.test_dir.cleanup()

    def test_load_document(self) -> None:
        """Verify load_document reads content and extracts correct metadata."""
        doc = load_document(self.sample_file)
        self.assertIn("autonomous intelligence", doc.page_content)
        self.assertEqual(doc.metadata["source"], "sample_guide.md")
        self.assertEqual(doc.metadata["file_path"], str(self.sample_file.resolve()))

    def test_load_document_not_found(self) -> None:
        """Verify load_document raises FileNotFoundError for missing files."""
        with self.assertRaises(FileNotFoundError):
            load_document(self.dir_path / "non_existent.md")

    def test_load_directory(self) -> None:
        """Verify load_directory discovers and reads all matching documents."""
        second_file = self.dir_path / "second.txt"
        second_file.write_text("Second document content.", encoding="utf-8")

        docs = load_directory(self.dir_path)
        self.assertEqual(len(docs), 2)
        sources = {doc.metadata["source"] for doc in docs}
        self.assertEqual(sources, {"sample_guide.md", "second.txt"})

    def test_split_text(self) -> None:
        """Verify split_text produces expected chunk boundaries."""
        sample_text = (
            "Paragraph one is moderately long.\n\n"
            "Paragraph two contains additional detailed information.\n\n"
            "Paragraph three concludes the document."
        )
        chunks = split_text(sample_text, chunk_size=80, chunk_overlap=15)
        self.assertTrue(len(chunks) >= 2)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 100)

    def test_chunk_document_metadata_preservation(self) -> None:
        """Verify chunk_document preserves source metadata and adds chunk identifiers."""
        doc = Document(
            page_content="Header\n\nContent paragraph 1\n\nContent paragraph 2",
            metadata={"source": "test.md", "category": "demo"},
        )
        chunks = chunk_document(doc, chunk_size=35, chunk_overlap=5)
        self.assertTrue(len(chunks) >= 2)
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata["source"], "test.md")
            self.assertEqual(chunk.metadata["category"], "demo")
            self.assertEqual(chunk.metadata["chunk_index"], idx)
            self.assertEqual(chunk.metadata["chunk_id"], f"test.md_chunk_{idx}")
            self.assertEqual(chunk.metadata["total_chunks"], len(chunks))


class TestRAGRetrieverAndIndexing(unittest.TestCase):
    """Test suite for embedding, ChromaDB indexing, and similarity retrieval."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_chroma = tempfile.TemporaryDirectory()
        cls.temp_kb = tempfile.TemporaryDirectory()

        kb_path = Path(cls.temp_kb.name)

        # Write test documents
        (kb_path / "pricing_info.md").write_text(
            "# Pricing\n\nAgentOps AI Starter tier costs $49 per month.\n\n"
            "The Professional tier costs $199 per month for teams.",
            encoding="utf-8",
        )
        (kb_path / "architecture_info.md").write_text(
            "# Architecture\n\nAgentOps AI is orchestrated via LangGraph StateGraph.\n\n"
            "It utilizes local sentence-transformers embeddings for RAG retrieval.",
            encoding="utf-8",
        )

        # Ingest into temporary ChromaDB directory
        ingest_knowledge_base(
            knowledge_dir=kb_path,
            persist_directory=cls.temp_chroma.name,
            collection_name="test_knowledge",
            chunk_size=300,
            chunk_overlap=30,
        )

        cls.retriever = RAGRetriever(
            persist_directory=cls.temp_chroma.name,
            collection_name="test_knowledge",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_chroma.cleanup()
        cls.temp_kb.cleanup()

    def test_embedding_function_initialization(self) -> None:
        """Verify the local sentence-transformer embedding function produces valid vectors."""
        emb_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        embeddings = emb_fn(["AgentOps AI embedding test"])
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 384)

    def test_retrieval_of_known_fact(self) -> None:
        """Verify semantic similarity retrieval returns the correct factual chunk."""
        results = self.retriever.retrieve("What is the Starter tier price?", top_k=2)
        self.assertTrue(len(results) > 0)

        top_chunk = results[0]
        self.assertIn("$49", top_chunk["content"])
        self.assertEqual(top_chunk["metadata"]["source"], "pricing_info.md")
        self.assertLess(top_chunk["distance"], 0.75)

    def test_retrieval_metadata_and_formatting(self) -> None:
        """Verify format_context includes source attribution and chunk index."""
        results = self.retriever.retrieve("How is AgentOps AI orchestrated?", top_k=1)
        self.assertTrue(len(results) > 0)

        context_string = self.retriever.format_context(results)
        self.assertIn("[Source: architecture_info.md", context_string)
        self.assertIn("LangGraph StateGraph", context_string)

    def test_handling_unknown_query(self) -> None:
        """Verify queries with completely unrelated content are filtered out by threshold."""
        results = self.retriever.retrieve("What is the average rainfall in the Sahara desert?", top_k=2)
        self.assertEqual(len(results), 0)

        formatted = self.retriever.format_context(results)
        self.assertEqual(formatted, "No relevant information found in the knowledge base.")


if __name__ == "__main__":
    unittest.main()

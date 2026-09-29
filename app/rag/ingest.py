"""Knowledge base document ingestion pipeline for AgentOps AI RAG."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

# Ensure project root is in sys.path when running standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.document_loader import chunk_document, load_directory
from app.rag.retriever import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PERSIST_DIRECTORY,
)

load_dotenv()

DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"


def ingest_knowledge_base(
    knowledge_dir: Optional[Path | str] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model_name: Optional[str] = None,
    client: Optional[ClientAPI] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> int:
    """Load, chunk, embed, and index knowledge documents into ChromaDB.

    Uses collection.upsert to ensure idempotent ingestion without duplicate chunks.

    Args:
        knowledge_dir: Directory containing Markdown/text knowledge documents.
        persist_directory: ChromaDB local persistence path.
        collection_name: Name of the vector store collection.
        embedding_model_name: Sentence-transformers embedding model name.
        client: Optional existing ChromaDB ClientAPI instance (e.g. for testing).
        chunk_size: Maximum character length per text chunk.
        chunk_overlap: Overlap characters between chunks.

    Returns:
        The total number of chunks upserted into the collection.
    """
    kb_path = Path(knowledge_dir or DEFAULT_KNOWLEDGE_DIR).resolve()
    persist_dir = persist_directory or os.getenv(
        "CHROMA_PERSIST_DIRECTORY", DEFAULT_PERSIST_DIRECTORY
    )
    embed_model = embedding_model_name or os.getenv(
        "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )

    print(f"[RAG Ingestion] Scanning knowledge directory: {kb_path}")
    raw_documents = load_directory(kb_path)
    if not raw_documents:
        print(f"[RAG Ingestion] No documents found in {kb_path}.")
        return 0

    print(f"[RAG Ingestion] Found {len(raw_documents)} document(s). Processing chunks...")

    # Partition documents into chunks
    all_chunks = []
    for doc in raw_documents:
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
        print(
            f"  - {doc.metadata.get('source')}: "
            f"{len(doc.page_content)} chars -> {len(chunks)} chunk(s)"
        )

    # Initialize ChromaDB client
    if client is not None:
        chroma_client = client
    else:
        persist_path = Path(persist_dir).resolve()
        persist_path.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(persist_path))

    # Initialize embedding function
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)

    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare batches for upsert
    ids = [chunk.metadata["chunk_id"] for chunk in all_chunks]
    documents = [chunk.page_content for chunk in all_chunks]
    metadatas = [chunk.metadata for chunk in all_chunks]

    print(f"[RAG Ingestion] Upserting {len(all_chunks)} chunk(s) into collection '{collection_name}'...")
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    total_count = collection.count()
    print(f"[RAG Ingestion] Ingestion complete. Total indexed chunks in collection: {total_count}")
    return len(all_chunks)


def main() -> None:
    """CLI entry point for running the RAG document ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest Markdown documents into ChromaDB for AgentOps AI RAG."
    )
    parser.add_argument(
        "--knowledge-dir",
        type=str,
        default=str(DEFAULT_KNOWLEDGE_DIR),
        help="Path to directory containing knowledge documents (default: data/knowledge)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="ChromaDB persistence directory (default: from .env or chroma_db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Target collection name (default: {DEFAULT_COLLECTION_NAME})",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AgentOps AI - Knowledge Base Ingestion Pipeline")
    print("=" * 60)

    try:
        count = ingest_knowledge_base(
            knowledge_dir=args.knowledge_dir,
            persist_directory=args.persist_dir,
            collection_name=args.collection,
        )
        print(f"Successfully processed and indexed {count} chunk(s).")
    except Exception as exc:
        print(f"\n[Ingestion Error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

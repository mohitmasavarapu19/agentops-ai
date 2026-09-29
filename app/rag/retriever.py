"""Vector retrieval module using ChromaDB and Sentence Transformers for AgentOps AI."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

DEFAULT_COLLECTION_NAME = "agentops_knowledge"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIRECTORY = "chroma_db"


class RAGRetriever:
    """Manages document retrieval from a local ChromaDB vector store."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_name: Optional[str] = None,
        client: Optional[ClientAPI] = None,
    ) -> None:
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY", DEFAULT_PERSIST_DIRECTORY
        )
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )

        # Initialize ChromaDB client (persistent or provided test client)
        if client is not None:
            self.client = client
        else:
            persist_path = Path(self.persist_directory).resolve()
            persist_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_path))

        # Initialize local SentenceTransformer embedding function
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )

        # Get or create collection with cosine similarity metric
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        max_distance: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query from the vector database.

        Args:
            query: The user search query.
            top_k: Maximum number of relevant chunks to retrieve.
            max_distance: Cosine distance threshold (0.0 to 1.0) above which
                results are discarded as irrelevant.

        Returns:
            List of dictionaries containing content, metadata, and distance.
        """
        if not query.strip() or self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query.strip()],
            n_results=min(top_k, self.collection.count()),
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        filtered_results: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            if dist is not None and dist <= max_distance:
                filtered_results.append({
                    "content": doc,
                    "metadata": meta or {},
                    "distance": float(dist),
                })

        return filtered_results

    def format_context(self, chunks: Sequence[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a clean, source-attributed context string.

        Args:
            chunks: List of retrieved chunk dictionaries.

        Returns:
            Formatted string presenting excerpts with their source documents.
        """
        if not chunks:
            return "No relevant information found in the knowledge base."

        formatted_sections: List[str] = []
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "unknown")
            chunk_idx = chunk.get("metadata", {}).get("chunk_index", 0)
            content = chunk.get("content", "").strip()
            formatted_sections.append(
                f"[Source: {source} (chunk {chunk_idx})]\n{content}"
            )

        return "\n\n".join(formatted_sections)


_retriever_instance: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """Get or create the default singleton RAGRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance

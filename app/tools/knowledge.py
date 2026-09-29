"""Local knowledge lookup tool for AgentOps AI using ChromaDB RAG retrieval."""

from langchain_core.tools import tool

from app.rag.retriever import get_retriever


@tool
def lookup_knowledge(query: str) -> str:
    """Look up business facts, architecture, pricing, and security information about AgentOps AI.

    Searches the local ChromaDB vector knowledge base to find verified,
    source-attributed context regarding AgentOps AI's mission, pricing plans,
    technology architecture, security posture, and capabilities.

    Args:
        query: The user query, question, or search keywords.

    Returns:
        A formatted string of relevant excerpts with their source document citations,
        or a notice if no relevant information is found.
    """
    clean_query = query.strip()
    if not clean_query:
        return "No relevant information found in the knowledge base."

    try:
        retriever = get_retriever()
        chunks = retriever.retrieve(clean_query, top_k=3, max_distance=0.75)
        if not chunks:
            return "No relevant information found in the knowledge base."

        return retriever.format_context(chunks)
    except Exception as exc:
        return f"Error retrieving knowledge from vector store: {exc}"

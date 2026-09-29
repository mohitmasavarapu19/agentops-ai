"""Local in-memory knowledge lookup tool for AgentOps AI business facts."""

import re
from typing import Any, Dict, List
from langchain_core.tools import tool

# Stop words to ignore during query tokenization
STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "for", "and", "or",
    "to", "how", "much", "does", "it", "do", "tell", "me", "can",
    "you", "about", "with", "from", "at", "by", "this", "that",
}

# Local in-memory repository of business intelligence facts regarding AgentOps AI
AGENT_OPS_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "topic": "Overview & Mission",
        "keywords": ["agentops", "mission", "overview", "purpose", "business intelligence", "research agent"],
        "fact": "AgentOps AI is an autonomous business intelligence and research agent designed to understand user objectives, plan multi-step workflows, execute tools, and generate grounded reports.",
    },
    {
        "topic": "Target Market & Audience",
        "keywords": ["audience", "target", "market", "customers", "users", "enterprise", "clients"],
        "fact": "AgentOps AI is built for enterprise strategy teams, business intelligence analysts, and researchers requiring automated, reliable insight generation.",
    },
    {
        "topic": "Subscription & Pricing Tiers",
        "keywords": ["pricing", "cost", "plans", "subscription", "tier", "tiers", "price", "rate"],
        "fact": "AgentOps AI offers three subscription tiers: Starter ($49/month for individual analysts), Professional ($199/month for small teams), and Enterprise (custom pricing with dedicated support and SLA).",
    },
    {
        "topic": "Core Capabilities & Features",
        "keywords": ["capabilities", "features", "skills", "abilities", "functions", "calculator", "math"],
        "fact": "AgentOps AI features autonomous tool selection, cyclical planning, arithmetic calculation, knowledge retrieval, and structured business report synthesis.",
    },
    {
        "topic": "Technology & Architecture",
        "keywords": ["tech", "stack", "technology", "langgraph", "langchain", "python", "architecture", "models", "openai"],
        "fact": "AgentOps AI is built on Python 3.12, LangGraph for cyclical state orchestration, LangChain for model/tool abstractions, and OpenAI models for reasoning.",
    },
    {
        "topic": "Founding & Organization",
        "keywords": ["founder", "founding", "founded", "team", "organization", "company", "history", "origins"],
        "fact": "AgentOps AI was founded in 2025 by an applied AI systems team with the goal of bringing robust, auditable agentic workflows to modern business intelligence.",
    },
]


@tool
def lookup_knowledge(query: str) -> str:
    """Look up business facts and background information about AgentOps AI.

    Searches an in-memory knowledge base using keyword matching to answer
    questions about AgentOps AI's mission, pricing, technology stack, target audience,
    and capabilities.

    Args:
        query: The user query, question, or search keywords.

    Returns:
        A formatted string of relevant business facts, or a notice if no matching
        information is found.
    """
    cleaned_query = query.lower().strip()
    query_tokens = set(re.findall(r"\b\w+\b", cleaned_query))
    content_tokens = {token for token in query_tokens if token not in STOP_WORDS}

    if not content_tokens:
        return "No relevant information found in the knowledge base."

    matched_facts: List[tuple[int, str]] = []

    for entry in AGENT_OPS_KNOWLEDGE_BASE:
        relevance_score = 0
        for kw in entry["keywords"]:
            kw_clean = kw.lower()
            # General brand name keyword carries lower weight so specific topics rank higher
            weight = 1 if kw_clean in ("agentops", "ai") else 3

            if kw_clean in content_tokens:
                relevance_score += weight
            elif " " in kw_clean and kw_clean in cleaned_query:
                # Multi-word key phrase match (e.g. "business intelligence")
                relevance_score += weight + 1

        if relevance_score > 0:
            matched_facts.append((relevance_score, entry["fact"]))

    if not matched_facts:
        return "No relevant information found in the knowledge base."

    # Sort matches by relevance score descending and preserve uniqueness
    matched_facts.sort(key=lambda item: item[0], reverse=True)
    unique_facts: List[str] = []
    seen = set()
    for _, fact in matched_facts:
        if fact not in seen:
            seen.add(fact)
            unique_facts.append(f"- {fact}")

    return "\n".join(unique_facts)

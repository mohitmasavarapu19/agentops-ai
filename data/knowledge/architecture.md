# AgentOps AI: System Architecture

## Architecture Overview
AgentOps AI is architected as a cyclical, state-driven autonomous system built with Python 3.12, LangGraph, LangChain, and ChromaDB.

```
User Query
    ↓
LangGraph Agent Node (ChatOpenAI)
    ↓
Conditional Routing Edge (tools_condition)
    ├── Tool Call Needed → ToolNode (Calculator, RAG Knowledge Retriever)
    │                         ↓
    │                      Agent Node (Synthesize grounded response)
    │                         ↓
    └── Complete → Final Output (END)
```

## Core Architectural Components

### 1. Agent Orchestration (LangGraph)
- **StateGraph**: Manages the cyclical workflow of the agent using a typed state schema (`AgentState`).
- **Reducer Pattern**: Conversation history is maintained through `messages` using LangGraph's `add_messages` reducer.
- **Conditional Routing**: Employs `tools_condition` to inspect whether tool calls were emitted by the model, routing execution to `ToolNode` or exiting to `END`.

### 2. Retrieval-Augmented Generation (RAG) Subsystem
- **Vector Database**: ChromaDB running in persistent mode (`chroma_db`).
- **Embedding Model**: Local embeddings powered by `sentence-transformers` using `all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Distance Metric**: Cosine similarity (`hnsw:space: cosine`) with distance threshold filtering to suppress irrelevant context.
- **Document Chunking**: Semantic section and paragraph-aware chunking preserving document metadata such as source file and chunk index.

### 3. Tool Execution Layer
- **Calculator Tool**: Validated arithmetic operations (+, -, *, /) preventing zero-division errors and avoiding dangerous `eval()`.
- **Knowledge Lookup Tool**: Interfaces with `RAGRetriever` to extract source-grounded excerpts before final answer synthesis.

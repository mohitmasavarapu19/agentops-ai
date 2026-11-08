# AgentOps AI — System Architecture

## Overview

AgentOps AI is a production-grade autonomous business intelligence agent built on **LangGraph**, **LangChain**, **ChromaDB**, and **FastAPI**.

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                             │
│         CLI (app/main.py)   │   REST API (app/api/)        │
└──────────────────┬──────────────────────────┬──────────────┘
                   │                          │
                   ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│               Security Guardrails (app/security/)           │
│  Input validation · Injection detection · PII redaction     │
└──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│            LangGraph Agent Orchestration (app/agents/)      │
│                                                             │
│   START → Agent Node (ChatOpenAI + tools_condition)         │
│              ↓ tool call?                                   │
│           ToolNode ──────────────────┐                      │
│           │ calculator               │                      │
│           │ lookup_knowledge ──────▶ RAG Subsystem          │
│           └──────────────── feedback loop ──▶ Agent Node    │
│                                                             │
│   Memory: MemorySaver (session-scoped, thread_id-keyed)     │
└──────────────────────────────┬──────────────────────────────┘
                                │
           ┌────────────────────┼─────────────────────┐
           ▼                    ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  Calculator Tool │  │  RAG Subsystem   │  │  LLM Provider        │
│  (app/tools/)    │  │  (app/rag/)      │  │  OpenAI / Groq       │
│                  │  │                  │  │  (via ChatOpenAI)    │
│  Safe arithmetic │  │  ChromaDB        │  └──────────────────────┘
│  +, -, *, /      │  │  sentence-xfmrs  │
│  zero-div guard  │  │  all-MiniLM-L6v2 │
└──────────────────┘  │  cosine distance │
                      └──────────────────┘
```

---

## Component Descriptions

### 1. Security Guardrails (`app/security/`)
Stateless pipeline that runs **before** every LLM invocation:

| Check | Description |
| :--- | :--- |
| **Empty/whitespace** | Rejects blank inputs immediately |
| **Length enforcement** | Hard limit of 4,096 characters |
| **Injection detection** | 8 regex patterns for instruction-override attacks |
| **PII redaction** | Masks emails, phone numbers, credit card numbers |

### 2. LangGraph Agent (`app/agents/`)
A **cyclical state machine** implemented as a `StateGraph`:

```
AgentState
├── user_input: str
├── response: str
└── messages: Annotated[Sequence[BaseMessage], add_messages]
```

Nodes:
- **`agent`**: Calls `ChatOpenAI.bind_tools(TOOLS).invoke(messages)`. Decides whether to call tools or produce a final answer.
- **`tools`**: `ToolNode` dispatches to `calculate` or `lookup_knowledge`.

Edges: `START → agent → tools_condition → (tools → agent)* → END`

### 3. Memory (`app/memory/`)
Wraps LangGraph's `MemorySaver`:

- **Stateless mode**: Default; each `run_agent()` call is independent.
- **Stateful mode**: Pass `session_id` + `checkpointer`; full conversation history is preserved per `thread_id`.

### 4. RAG Subsystem (`app/rag/`)
Local Retrieval-Augmented Generation pipeline:

| Component | Detail |
| :--- | :--- |
| **Embedding model** | `all-MiniLM-L6-v2` (384-dim dense vectors, runs locally) |
| **Vector store** | ChromaDB (persistent, `chroma_db/`) |
| **Distance metric** | Cosine similarity (`hnsw:space: cosine`) |
| **Threshold** | Distance ≤ 0.75 (irrelevant chunks discarded) |
| **Chunking** | Recursive section/paragraph splitting with 50-char overlap |
| **Ingestion** | `upsert`-based, fully idempotent |

### 5. FastAPI Serving Layer (`app/api/`)
Production REST interface:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/chat` | `POST` | Stateful/stateless agent invocation |
| `/ingest` | `POST` | Trigger knowledge base re-ingestion |
| `/health` | `GET` | Service health and config check |

### 6. Evaluation Harness (`app/evaluation/`)
Automated regression testing for retrieval quality:

| Metric | Description |
| :--- | :--- |
| **Retrieval Rate** | Fraction of queries that return ≥ 1 chunk |
| **Keyword Hit Rate** | Fraction where expected keywords appear in context |
| **Source Accuracy** | Fraction where the correct source document is cited |
| **Overall Pass Rate** | All three checks pass (threshold: 70%) |
| **Avg Latency** | Mean retrieval time in milliseconds |

Benchmark: 12 golden Q&A pairs across pricing, architecture, overview, and security categories.

---

## Data Flow: Single Query (Stateless)

```
User → Guardrails → run_agent()
         → LangGraph invoke
           → Agent Node (LLM decides: tool call needed?)
             → ToolNode: lookup_knowledge("Starter plan price")
               → RAGRetriever.retrieve(query, top_k=3, max_distance=0.75)
                 → ChromaDB cosine search
                 → distance filter
                 → format_context([chunks])  ← source-attributed string
             → Agent Node (LLM synthesises final answer with cited context)
         → Final response string
← Response
```

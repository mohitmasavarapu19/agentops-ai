"""FastAPI REST serving layer for AgentOps AI.

Exposes three endpoints:
  - POST /chat   : Stateful agent invocation with optional session memory.
  - POST /ingest : Trigger a full re-ingestion of the knowledge base.
  - GET  /health : Service health and configuration status check.

Run the server:
    uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload

Or via Python:
    python -m app.api.server
"""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.graph import run_agent
from app.memory.checkpointer import generate_session_id, get_default_checkpointer
from app.rag.ingest import ingest_knowledge_base
from app.security.guardrails import run_guardrails


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: pre-warm the shared checkpointer on startup."""
    # Eagerly create the singleton checkpointer so the first /chat request
    # does not incur initialisation latency.
    get_default_checkpointer()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgentOps AI",
    description=(
        "Autonomous Business Intelligence & Research Agent powered by "
        "LangGraph, ChromaDB RAG, and OpenAI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="User message / query to send to the agent.",
        examples=["What is the Starter plan price per year?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session ID for persistent multi-turn conversation memory. "
            "Omit to start a new stateless single-turn interaction. "
            "Returned in the response for subsequent turns."
        ),
    )
    use_memory: bool = Field(
        default=False,
        description=(
            "Set to true to enable MemorySaver-backed conversation state. "
            "Automatically set to true when session_id is provided."
        ),
    )


class ChatResponse(BaseModel):
    """Response body from the /chat endpoint."""

    response: str = Field(..., description="Agent's generated response text.")
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID to use in subsequent requests to continue this conversation.",
    )
    latency_ms: float = Field(..., description="End-to-end agent invocation latency in milliseconds.")


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    knowledge_dir: Optional[str] = Field(
        default=None,
        description=(
            "Path to the knowledge documents directory. "
            "Defaults to data/knowledge/ relative to the project root."
        ),
    )


class IngestResponse(BaseModel):
    """Response body from the /ingest endpoint."""

    status: str = Field(..., description="'success' or 'error'.")
    chunks_indexed: int = Field(..., description="Number of document chunks upserted into ChromaDB.")
    message: str = Field(..., description="Human-readable summary of the ingestion result.")


class HealthResponse(BaseModel):
    """Response body from the /health endpoint."""

    status: str = Field(..., description="'healthy' when all subsystems are operational.")
    version: str = Field(..., description="API version string.")
    memory_enabled: bool = Field(..., description="Whether the MemorySaver checkpointer is initialised.")
    knowledge_dir_exists: bool = Field(..., description="Whether the default knowledge directory exists.")
    chroma_db_exists: bool = Field(..., description="Whether the ChromaDB persist directory exists.")
    uptime_ms: float = Field(..., description="Server uptime in milliseconds since startup.")


# ---------------------------------------------------------------------------
# Startup timestamp for uptime tracking
# ---------------------------------------------------------------------------
_START_TIME_MS: float = time.monotonic() * 1000


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the AgentOps AI agent",
    tags=["Agent"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Invoke the LangGraph agent with the user's message.

    - **Stateless mode** (default): Each call is independent; no history is retained.
    - **Stateful mode** (`use_memory=true` or `session_id` provided): The agent
      retains conversation history across turns using LangGraph's MemorySaver.
      Pass the returned `session_id` in subsequent requests to continue the session.

    Raises:
        422 Unprocessable Entity: If the message violates security guardrails
            (prompt injection detected, PII cannot be sanitised safely, etc.).
        500 Internal Server Error: If the agent encounters an unexpected runtime error.
    """
    # --- Guardrail check (also performed inside run_agent; checked here for
    #     early fast-fail and to surface HTTP 422 instead of 500) ---
    guard = run_guardrails(request.message)
    if not guard.is_safe:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Input blocked by security guardrails.",
                "violations": guard.violations,
            },
        )

    # --- Session / memory resolution ---
    use_memory = request.use_memory or bool(request.session_id)
    checkpointer = get_default_checkpointer() if use_memory else None
    session_id = request.session_id or (generate_session_id() if use_memory else None)

    # --- Agent invocation ---
    start = time.monotonic()
    try:
        response_text = run_agent(
            guard.sanitized_input,
            session_id=session_id,
            checkpointer=checkpointer,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {exc}",
        )
    latency_ms = (time.monotonic() - start) * 1000

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Re-ingest the knowledge base into ChromaDB",
    tags=["Knowledge Base"],
)
async def ingest(request: IngestRequest = IngestRequest()) -> IngestResponse:
    """Trigger a full re-ingestion of Markdown documents into the ChromaDB vector store.

    Uses `upsert` semantics so existing chunks are updated without duplication.
    Accepts an optional `knowledge_dir` path; defaults to `data/knowledge/`.

    Raises:
        404 Not Found: If the specified knowledge directory does not exist.
        500 Internal Server Error: If the ingestion pipeline encounters an error.
    """
    knowledge_dir = request.knowledge_dir or None  # None → ingest uses default

    if knowledge_dir and not Path(knowledge_dir).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge directory not found: {knowledge_dir}",
        )

    try:
        chunks_indexed = ingest_knowledge_base(knowledge_dir=knowledge_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion pipeline error: {exc}",
        )

    return IngestResponse(
        status="success",
        chunks_indexed=chunks_indexed,
        message=f"Successfully indexed {chunks_indexed} chunk(s) into the ChromaDB collection.",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """Return the current health and configuration status of the AgentOps AI service.

    Checks:
    - Whether the MemorySaver checkpointer is initialised (memory_enabled).
    - Whether the default knowledge directory exists on disk.
    - Whether the ChromaDB persistence directory exists on disk.
    - Server uptime in milliseconds.
    """
    from app.memory.checkpointer import _default_checkpointer

    project_root = Path(__file__).resolve().parent.parent.parent
    knowledge_dir = project_root / "data" / "knowledge"
    chroma_dir = project_root / "chroma_db"
    uptime_ms = (time.monotonic() * 1000) - _START_TIME_MS

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        memory_enabled=_default_checkpointer is not None,
        knowledge_dir_exists=knowledge_dir.exists(),
        chroma_db_exists=chroma_dir.exists(),
        uptime_ms=round(uptime_ms, 2),
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

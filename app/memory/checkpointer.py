"""Session-scoped memory checkpointing for AgentOps AI.

Wraps LangGraph's MemorySaver to provide thread-based persistent conversation
memory across multiple agent turns within a session.
"""

import uuid
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver


def create_memory_checkpointer() -> MemorySaver:
    """Create and return a new MemorySaver checkpointer instance.

    Returns:
        MemorySaver: An in-process memory checkpointer that persists state
            across agent turns within the same Python process lifetime.
    """
    return MemorySaver()


def generate_session_id() -> str:
    """Generate a unique session/thread ID for a new conversation.

    Returns:
        str: A UUID4 string to use as the LangGraph thread_id.
    """
    return str(uuid.uuid4())


# Module-level default checkpointer (shared across the process)
_default_checkpointer: Optional[MemorySaver] = None


def get_default_checkpointer() -> MemorySaver:
    """Return the process-wide default MemorySaver checkpointer (singleton).

    Returns:
        MemorySaver: The shared default checkpointer instance.
    """
    global _default_checkpointer
    if _default_checkpointer is None:
        _default_checkpointer = create_memory_checkpointer()
    return _default_checkpointer

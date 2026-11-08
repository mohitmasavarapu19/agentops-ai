"""Tests for Milestone 4: LangGraph MemorySaver conversation memory."""

import unittest
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from app.memory.checkpointer import (
    create_memory_checkpointer,
    generate_session_id,
    get_default_checkpointer,
)
from app.agents.graph import create_agent_graph, run_agent


class TestMemoryCheckpointer(unittest.TestCase):
    """Test suite for the memory checkpointer module."""

    def test_create_memory_checkpointer_returns_memorysaver(self) -> None:
        """Verify create_memory_checkpointer returns a MemorySaver instance."""
        cp = create_memory_checkpointer()
        self.assertIsInstance(cp, MemorySaver)

    def test_generate_session_id_is_unique(self) -> None:
        """Verify each generated session ID is a unique non-empty string."""
        ids = {generate_session_id() for _ in range(10)}
        self.assertEqual(len(ids), 10)
        for sid in ids:
            self.assertIsInstance(sid, str)
            self.assertTrue(len(sid) > 0)

    def test_get_default_checkpointer_singleton(self) -> None:
        """Verify get_default_checkpointer always returns the same instance."""
        cp1 = get_default_checkpointer()
        cp2 = get_default_checkpointer()
        self.assertIs(cp1, cp2)

    def test_generate_session_id_uuid_format(self) -> None:
        """Verify session IDs follow UUID4 format."""
        import re
        sid = generate_session_id()
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        )
        self.assertRegex(sid, uuid_pattern)


class TestAgentWithMemory(unittest.TestCase):
    """Test suite for LangGraph graph compilation with checkpointer."""

    def test_create_agent_graph_with_checkpointer(self) -> None:
        """Verify create_agent_graph compiles correctly with a MemorySaver."""
        checkpointer = create_memory_checkpointer()
        g = create_agent_graph(checkpointer=checkpointer)
        self.assertIsInstance(g, CompiledStateGraph)

    def test_create_agent_graph_without_checkpointer(self) -> None:
        """Verify create_agent_graph compiles correctly without a checkpointer (stateless)."""
        g = create_agent_graph(checkpointer=None)
        self.assertIsInstance(g, CompiledStateGraph)

    @patch("app.agents.graph.get_llm")
    def test_run_agent_with_session_id(self, mock_get_llm: MagicMock) -> None:
        """Verify run_agent accepts session_id and checkpointer without raising errors."""
        from langchain_core.messages import AIMessage

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Test response", tool_calls=[])
        mock_get_llm.return_value = mock_llm

        checkpointer = create_memory_checkpointer()
        session_id = generate_session_id()
        result = run_agent("Hello", session_id=session_id, checkpointer=checkpointer)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Test response")

    @patch("app.agents.graph.get_llm")
    def test_run_agent_stateless(self, mock_get_llm: MagicMock) -> None:
        """Verify run_agent works without memory (stateless mode)."""
        from langchain_core.messages import AIMessage

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Stateless response", tool_calls=[])
        mock_get_llm.return_value = mock_llm

        result = run_agent("Hello")
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Stateless response")


if __name__ == "__main__":
    unittest.main()

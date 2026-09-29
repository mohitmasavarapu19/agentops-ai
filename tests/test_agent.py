"""Tests for Milestone 1: Minimal LangGraph Agent Foundation."""

import os
import unittest
from unittest.mock import MagicMock, patch

from app.agents.graph import (
    AgentState,
    create_agent_graph,
    get_llm,
    graph,
    run_agent,
)
from langgraph.graph.state import CompiledStateGraph


class TestAgentGraph(unittest.TestCase):
    """Test suite verifying agent graph compilation and execution structure."""

    def test_graph_import_and_compilation(self) -> None:
        """Verify the pre-compiled graph instance is a CompiledStateGraph."""
        self.assertIsInstance(graph, CompiledStateGraph)

    def test_create_agent_graph_factory(self) -> None:
        """Verify create_agent_graph constructs a new valid CompiledStateGraph."""
        new_graph = create_agent_graph()
        self.assertIsInstance(new_graph, CompiledStateGraph)

    def test_graph_nodes(self) -> None:
        """Verify the agent node exists within the compiled graph."""
        self.assertIn("agent", graph.nodes)

    def test_missing_api_key_raises_value_error(self) -> None:
        """Verify that get_llm raises ValueError if OPENAI_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                get_llm()
            self.assertIn("OPENAI_API_KEY environment variable is not set", str(context.exception))

    @patch("app.agents.graph.get_llm")
    def test_run_agent_with_mocked_llm(self, mock_get_llm: MagicMock) -> None:
        """Verify end-to-end state execution flow using a mocked LLM."""
        mock_response = MagicMock()
        mock_response.content = "Grounded response from AgentOps AI"

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm_instance

        user_query = "What is business intelligence?"
        result = run_agent(user_query)

        # Assertions
        mock_get_llm.assert_called_once()
        mock_llm_instance.invoke.assert_called_once_with(user_query)
        self.assertEqual(result, "Grounded response from AgentOps AI")


if __name__ == "__main__":
    unittest.main()

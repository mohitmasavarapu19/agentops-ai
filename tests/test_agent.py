"""Tests for Milestone 2: LangGraph Agent with Tool Calling."""

import os
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from app.agents.graph import (
    TOOLS,
    create_agent_graph,
    get_llm,
    graph,
    run_agent,
)


class TestAgentGraph(unittest.TestCase):
    """Test suite verifying agent graph compilation, nodes, and tool-calling execution."""

    def test_graph_import_and_compilation(self) -> None:
        """Verify the pre-compiled graph instance is a CompiledStateGraph."""
        self.assertIsInstance(graph, CompiledStateGraph)

    def test_create_agent_graph_factory(self) -> None:
        """Verify create_agent_graph constructs a new valid CompiledStateGraph."""
        new_graph = create_agent_graph()
        self.assertIsInstance(new_graph, CompiledStateGraph)

    def test_graph_nodes_present(self) -> None:
        """Verify both agent and tools nodes exist within the compiled graph."""
        self.assertIn("agent", graph.nodes)
        self.assertIn("tools", graph.nodes)

    def test_missing_api_key_raises_value_error(self) -> None:
        """Verify that get_llm raises ValueError if OPENAI_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                get_llm()
            self.assertIn("OPENAI_API_KEY environment variable is not set", str(context.exception))

    @patch("app.agents.graph.get_llm")
    def test_run_agent_without_tool_call(self, mock_get_llm: MagicMock) -> None:
        """Verify execution flow when LLM decides no tool call is needed."""
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound_llm
        mock_bound_llm.invoke.return_value = AIMessage(
            content="AgentOps AI provides strategic intelligence."
        )
        mock_get_llm.return_value = mock_llm

        result = run_agent("What does AgentOps AI do?")

        mock_get_llm.assert_called_once()
        mock_llm.bind_tools.assert_called_once_with(TOOLS)
        self.assertEqual(result, "AgentOps AI provides strategic intelligence.")

    @patch("app.agents.graph.get_llm")
    def test_run_agent_with_calculator_tool_call(self, mock_get_llm: MagicMock) -> None:
        """Verify end-to-end cyclical execution flow when LLM calls calculator tool."""
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound_llm

        # Step 1: LLM issues a tool call to calculate
        tool_call_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate",
                    "args": {"a": 25, "b": 4, "operation": "multiply"},
                    "id": "call_calc_1",
                    "type": "tool_call",
                }
            ],
        )

        # Step 2: LLM receives tool output and gives final grounded answer
        final_answer_message = AIMessage(content="25 multiplied by 4 is 100.")

        mock_bound_llm.invoke.side_effect = [tool_call_message, final_answer_message]
        mock_get_llm.return_value = mock_llm

        result = run_agent("What is 25 times 4?")

        # Assertions
        self.assertEqual(result, "25 multiplied by 4 is 100.")
        self.assertEqual(mock_bound_llm.invoke.call_count, 2)

        # Check second call included the ToolMessage produced by ToolNode
        second_call_messages = mock_bound_llm.invoke.call_args_list[1][0][0]
        has_tool_message = any(
            isinstance(m, ToolMessage) and m.content == "100"
            for m in second_call_messages
        )
        self.assertTrue(has_tool_message, "ToolMessage with result '100' should be present in history")

    @patch("app.agents.graph.get_llm")
    def test_run_agent_with_knowledge_tool_call(self, mock_get_llm: MagicMock) -> None:
        """Verify end-to-end cyclical execution flow when LLM calls knowledge lookup tool."""
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound_llm

        # Step 1: LLM issues a tool call to lookup_knowledge
        tool_call_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "lookup_knowledge",
                    "args": {"query": "pricing tiers"},
                    "id": "call_know_1",
                    "type": "tool_call",
                }
            ],
        )

        # Step 2: LLM synthesizes final answer
        final_answer_message = AIMessage(
            content="AgentOps AI offers Starter at $49/mo and Professional at $199/mo."
        )

        mock_bound_llm.invoke.side_effect = [tool_call_message, final_answer_message]
        mock_get_llm.return_value = mock_llm

        result = run_agent("How much does AgentOps AI cost?")

        # Assertions
        self.assertEqual(result, "AgentOps AI offers Starter at $49/mo and Professional at $199/mo.")
        self.assertEqual(mock_bound_llm.invoke.call_count, 2)

        # Check second call history contains tool result
        second_call_messages = mock_bound_llm.invoke.call_args_list[1][0][0]
        has_tool_message = any(
            isinstance(m, ToolMessage) and "$49" in m.content
            for m in second_call_messages
        )
        self.assertTrue(has_tool_message, "ToolMessage should contain retrieved pricing facts")


if __name__ == "__main__":
    unittest.main()

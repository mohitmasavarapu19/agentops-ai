"""Tests for Milestone 2: Calculator and Knowledge Tools."""

import unittest
from app.tools.calculator import calculate
from app.tools.knowledge import lookup_knowledge


class TestCalculatorTool(unittest.TestCase):
    """Test suite for the calculator tool."""

    def test_calculator_addition(self) -> None:
        """Verify addition operations with various notations."""
        self.assertEqual(calculate.invoke({"a": 10, "b": 5, "operation": "add"}), "15")
        self.assertEqual(calculate.invoke({"a": 3.5, "b": 2.5, "operation": "+"}), "6")
        self.assertEqual(calculate.invoke({"a": 1.2, "b": 3.4, "operation": "plus"}), "4.6")

    def test_calculator_subtraction(self) -> None:
        """Verify subtraction operations."""
        self.assertEqual(calculate.invoke({"a": 10, "b": 4, "operation": "subtract"}), "6")
        self.assertEqual(calculate.invoke({"a": 5, "b": 12, "operation": "-"}), "-7")

    def test_calculator_multiplication(self) -> None:
        """Verify multiplication operations."""
        self.assertEqual(calculate.invoke({"a": 6, "b": 7, "operation": "multiply"}), "42")
        self.assertEqual(calculate.invoke({"a": 2.5, "b": 4, "operation": "*"}), "10")

    def test_calculator_division(self) -> None:
        """Verify division operations."""
        self.assertEqual(calculate.invoke({"a": 20, "b": 4, "operation": "divide"}), "5")
        self.assertEqual(calculate.invoke({"a": 10, "b": 4, "operation": "/"}), "2.5")

    def test_calculator_division_by_zero(self) -> None:
        """Verify division by zero produces safe error response."""
        result = calculate.invoke({"a": 10, "b": 0, "operation": "divide"})
        self.assertEqual(result, "Error: Division by zero is not allowed.")

    def test_calculator_invalid_operation(self) -> None:
        """Verify unsupported operations produce descriptive error response."""
        result = calculate.invoke({"a": 10, "b": 5, "operation": "modulo"})
        self.assertIn("Unsupported operation 'modulo'", result)


class TestKnowledgeTool(unittest.TestCase):
    """Test suite for the local in-memory knowledge tool."""

    def test_knowledge_relevant_query(self) -> None:
        """Verify retrieval of business facts for relevant keywords."""
        pricing_result = lookup_knowledge.invoke({"query": "What is the pricing for AgentOps AI?"})
        self.assertIn("$49/month", pricing_result)
        self.assertIn("Professional ($199/month", pricing_result)

        tech_result = lookup_knowledge.invoke({"query": "What tech stack and architecture does it use?"})
        self.assertIn("LangGraph", tech_result)
        self.assertIn("Python 3.12", tech_result)

        mission_result = lookup_knowledge.invoke({"query": "What is AgentOps AI?"})
        self.assertIn("autonomous business intelligence", mission_result)

    def test_knowledge_unknown_query(self) -> None:
        """Verify handling of queries with no relevant facts in the knowledge base."""
        unknown_result = lookup_knowledge.invoke({"query": "What is the capital of Mars?"})
        self.assertEqual(unknown_result, "No relevant information found in the knowledge base.")


if __name__ == "__main__":
    unittest.main()

"""Tests for Milestone 7: RAG Evaluation Harness."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.evaluation.evaluator import (
    EvaluationReport,
    QuestionResult,
    RAGEvaluator,
)
from app.evaluation.benchmark import render_report


class TestQuestionResult(unittest.TestCase):
    """Test QuestionResult dataclass."""

    def test_passed_property_true_when_all_checks_pass(self) -> None:
        r = QuestionResult(
            question_id="test_001",
            question="Test question",
            category="test",
            retrieved=True,
            keyword_hit=True,
            source_hit=True,
            latency_ms=50.0,
        )
        self.assertTrue(r.passed)

    def test_passed_property_false_when_any_check_fails(self) -> None:
        r = QuestionResult(
            question_id="test_002",
            question="Test question",
            category="test",
            retrieved=True,
            keyword_hit=False,  # keyword miss
            source_hit=True,
            latency_ms=50.0,
        )
        self.assertFalse(r.passed)


class TestEvaluationReport(unittest.TestCase):
    """Test EvaluationReport computed properties."""

    def _make_report(self, total, retrieved, keyword, source, passed, avg_latency):
        return EvaluationReport(
            total=total,
            retrieved_count=retrieved,
            keyword_hit_count=keyword,
            source_hit_count=source,
            passed_count=passed,
            avg_latency_ms=avg_latency,
            results=[],
        )

    def test_retrieval_rate(self) -> None:
        report = self._make_report(10, 8, 7, 6, 5, 50.0)
        self.assertAlmostEqual(report.retrieval_rate, 0.8)

    def test_keyword_hit_rate(self) -> None:
        report = self._make_report(10, 8, 7, 6, 5, 50.0)
        self.assertAlmostEqual(report.keyword_hit_rate, 0.7)

    def test_source_accuracy(self) -> None:
        report = self._make_report(10, 8, 7, 6, 5, 50.0)
        self.assertAlmostEqual(report.source_accuracy, 0.6)

    def test_pass_rate(self) -> None:
        report = self._make_report(10, 8, 7, 6, 5, 50.0)
        self.assertAlmostEqual(report.pass_rate, 0.5)

    def test_zero_total_no_division_error(self) -> None:
        report = self._make_report(0, 0, 0, 0, 0, 0.0)
        self.assertEqual(report.retrieval_rate, 0.0)
        self.assertEqual(report.pass_rate, 0.0)


class TestRAGEvaluatorGoldenSet(unittest.TestCase):
    """Test RAGEvaluator golden set loading."""

    def test_load_golden_set_valid(self) -> None:
        """Verify load_golden_set reads and returns a list from a valid JSON file."""
        golden = [
            {
                "id": "q1",
                "question": "What is pricing?",
                "expected_keywords": ["$49"],
                "expected_source": "pricing.md",
                "category": "pricing",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_file = Path(tmpdir) / "golden_qa.json"
            golden_file.write_text(json.dumps(golden), encoding="utf-8")
            evaluator = RAGEvaluator(golden_path=golden_file)
            loaded = evaluator.load_golden_set()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["id"], "q1")

    def test_load_golden_set_missing_file_raises(self) -> None:
        """Verify FileNotFoundError is raised for missing golden set."""
        evaluator = RAGEvaluator(golden_path=Path("/nonexistent/path/golden_qa.json"))
        with self.assertRaises(FileNotFoundError):
            evaluator.load_golden_set()

    def test_load_golden_set_empty_array_raises(self) -> None:
        """Verify ValueError is raised for empty JSON array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_file = Path(tmpdir) / "golden_qa.json"
            golden_file.write_text("[]", encoding="utf-8")
            evaluator = RAGEvaluator(golden_path=golden_file)
            with self.assertRaises(ValueError):
                evaluator.load_golden_set()


class TestRAGEvaluatorEvaluateQuestion(unittest.TestCase):
    """Test RAGEvaluator.evaluate_question with mocked retriever."""

    def _make_evaluator_with_mock_retriever(self, chunks, context):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = chunks
        mock_retriever.format_context.return_value = context
        evaluator = RAGEvaluator(retriever=mock_retriever)
        return evaluator

    def test_all_checks_pass(self) -> None:
        chunks = [{"metadata": {"source": "pricing.md"}, "content": "Starter $49/month", "distance": 0.3}]
        evaluator = self._make_evaluator_with_mock_retriever(
            chunks=chunks,
            context="[Source: pricing.md (chunk 0)]\nStarter $49/month",
        )
        item = {
            "id": "pricing_001",
            "question": "How much does the Starter plan cost?",
            "expected_keywords": ["$49", "Starter"],
            "expected_source": "pricing.md",
            "category": "pricing",
        }
        result = evaluator.evaluate_question(item)
        self.assertTrue(result.retrieved)
        self.assertTrue(result.keyword_hit)
        self.assertTrue(result.source_hit)
        self.assertTrue(result.passed)
        self.assertEqual(result.violations, [])

    def test_no_retrieval_fails_all(self) -> None:
        evaluator = self._make_evaluator_with_mock_retriever(chunks=[], context="No relevant information found.")
        item = {
            "id": "test_001",
            "question": "Some question",
            "expected_keywords": ["keyword"],
            "expected_source": "source.md",
            "category": "test",
        }
        result = evaluator.evaluate_question(item)
        self.assertFalse(result.retrieved)
        self.assertFalse(result.keyword_hit)
        self.assertFalse(result.source_hit)
        self.assertFalse(result.passed)
        self.assertTrue(len(result.violations) > 0)

    def test_keyword_miss_adds_violation(self) -> None:
        chunks = [{"metadata": {"source": "pricing.md"}, "content": "No matching keywords here", "distance": 0.4}]
        evaluator = self._make_evaluator_with_mock_retriever(
            chunks=chunks,
            context="[Source: pricing.md (chunk 0)]\nNo matching keywords here",
        )
        item = {
            "id": "test_002",
            "question": "What is the Starter price?",
            "expected_keywords": ["$49"],
            "expected_source": "pricing.md",
            "category": "pricing",
        }
        result = evaluator.evaluate_question(item)
        self.assertTrue(result.retrieved)
        self.assertFalse(result.keyword_hit)
        self.assertTrue(any("keywords" in v.lower() for v in result.violations))


class TestBenchmarkRenderReport(unittest.TestCase):
    """Test benchmark report rendering."""

    def test_render_report_contains_summary(self) -> None:
        report = EvaluationReport(
            total=12,
            retrieved_count=12,
            keyword_hit_count=11,
            source_hit_count=11,
            passed_count=11,
            avg_latency_ms=42.5,
            results=[],
        )
        md = render_report(report, verbose=False)
        self.assertIn("RAG Evaluation Report", md)
        self.assertIn("12", md)
        self.assertIn("42.5 ms", md)

    def test_render_report_verbose_shows_table(self) -> None:
        result = QuestionResult(
            question_id="q1",
            question="Test question",
            category="pricing",
            retrieved=True,
            keyword_hit=True,
            source_hit=True,
            latency_ms=30.0,
        )
        report = EvaluationReport(
            total=1,
            retrieved_count=1,
            keyword_hit_count=1,
            source_hit_count=1,
            passed_count=1,
            avg_latency_ms=30.0,
            results=[result],
        )
        md = render_report(report, verbose=True)
        self.assertIn("Per-Question Results", md)
        self.assertIn("q1", md)


if __name__ == "__main__":
    unittest.main()

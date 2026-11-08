"""RAG Evaluation Harness for AgentOps AI.

Runs a golden Q&A benchmark set against the live ChromaDB vector store
to measure retrieval quality: hit-rate, keyword match, source accuracy,
and latency.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.rag.retriever import RAGRetriever, get_retriever


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    """Result for a single golden Q&A evaluation item.

    Attributes:
        question_id: Identifier from the golden set.
        question: The evaluation question text.
        category: Category label (e.g. pricing, architecture).
        retrieved: True if at least one relevant chunk was returned.
        keyword_hit: True if any expected keyword was found in retrieved context.
        source_hit: True if the expected source document was cited.
        latency_ms: Retrieval latency in milliseconds.
        retrieved_sources: List of source documents found in retrieved chunks.
        retrieved_context: The formatted context string returned by the retriever.
        violations: Descriptive list of failures (empty when all checks pass).
    """
    question_id: str
    question: str
    category: str
    retrieved: bool
    keyword_hit: bool
    source_hit: bool
    latency_ms: float
    retrieved_sources: List[str] = field(default_factory=list)
    retrieved_context: str = ""
    violations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if all three checks passed: retrieval, keyword, and source."""
        return self.retrieved and self.keyword_hit and self.source_hit


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all golden Q&A items.

    Attributes:
        total: Total number of evaluation questions.
        retrieved_count: Questions where retrieval returned at least one chunk.
        keyword_hit_count: Questions where expected keywords were found.
        source_hit_count: Questions where the expected source was cited.
        passed_count: Questions that passed all three checks.
        avg_latency_ms: Average retrieval latency across all questions.
        results: Per-question result details.
    """
    total: int
    retrieved_count: int
    keyword_hit_count: int
    source_hit_count: int
    passed_count: int
    avg_latency_ms: float
    results: List[QuestionResult] = field(default_factory=list)

    @property
    def retrieval_rate(self) -> float:
        """Fraction of questions where at least one chunk was retrieved."""
        return self.retrieved_count / self.total if self.total else 0.0

    @property
    def keyword_hit_rate(self) -> float:
        """Fraction of questions where expected keywords were found."""
        return self.keyword_hit_count / self.total if self.total else 0.0

    @property
    def source_accuracy(self) -> float:
        """Fraction of questions where the correct source document was cited."""
        return self.source_hit_count / self.total if self.total else 0.0

    @property
    def pass_rate(self) -> float:
        """Fraction of questions passing all three checks."""
        return self.passed_count / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """Evaluates the RAG retrieval pipeline against a golden Q&A benchmark set.

    Args:
        retriever: RAGRetriever instance to evaluate. Defaults to the singleton.
        golden_path: Path to golden_qa.json. Defaults to data/evaluation/golden_qa.json.
        top_k: Number of chunks to retrieve per question.
        max_distance: Cosine distance threshold for relevance filtering.
    """

    DEFAULT_GOLDEN_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "golden_qa.json"

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        golden_path: Optional[Path] = None,
        top_k: int = 3,
        max_distance: float = 0.75,
    ) -> None:
        self.retriever = retriever or get_retriever()
        self.golden_path = golden_path or self.DEFAULT_GOLDEN_PATH
        self.top_k = top_k
        self.max_distance = max_distance

    def load_golden_set(self) -> List[Dict[str, Any]]:
        """Load and parse the golden Q&A JSON file.

        Returns:
            List of question dicts with keys: id, question, expected_keywords,
            expected_source, category.

        Raises:
            FileNotFoundError: If the golden_qa.json file does not exist.
            ValueError: If the JSON is malformed or missing required fields.
        """
        path = Path(self.golden_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Golden Q&A file not found: {path}. "
                "Run from the project root or verify data/evaluation/golden_qa.json exists."
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError("golden_qa.json must be a non-empty JSON array.")
        return data

    def evaluate_question(self, item: Dict[str, Any]) -> QuestionResult:
        """Evaluate a single golden Q&A item.

        Args:
            item: Dict from golden_qa.json with keys: id, question,
                  expected_keywords, expected_source, category.

        Returns:
            QuestionResult populated with retrieval metrics.
        """
        question_id = item.get("id", "unknown")
        question = item["question"]
        expected_keywords: List[str] = item.get("expected_keywords", [])
        expected_source: str = item.get("expected_source", "")
        category: str = item.get("category", "general")

        violations: List[str] = []

        # Time the retrieval
        start = time.perf_counter()
        try:
            chunks = self.retriever.retrieve(question, top_k=self.top_k, max_distance=self.max_distance)
            context = self.retriever.format_context(chunks)
        except Exception as exc:
            chunks = []
            context = ""
            violations.append(f"Retrieval error: {exc}")
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved = len(chunks) > 0
        if not retrieved:
            violations.append("No chunks retrieved (below distance threshold or empty collection).")

        # Keyword check (case-insensitive, any keyword suffices)
        context_lower = context.lower()
        keyword_hit = any(kw.lower() in context_lower for kw in expected_keywords) if retrieved else False
        if not keyword_hit and retrieved:
            violations.append(
                f"Expected keywords {expected_keywords} not found in retrieved context."
            )

        # Source accuracy check
        retrieved_sources = list({
            chunk.get("metadata", {}).get("source", "") for chunk in chunks
        })
        source_hit = expected_source in retrieved_sources if retrieved else False
        if not source_hit and retrieved:
            violations.append(
                f"Expected source '{expected_source}' not in retrieved sources {retrieved_sources}."
            )

        return QuestionResult(
            question_id=question_id,
            question=question,
            category=category,
            retrieved=retrieved,
            keyword_hit=keyword_hit,
            source_hit=source_hit,
            latency_ms=latency_ms,
            retrieved_sources=retrieved_sources,
            retrieved_context=context,
            violations=violations,
        )

    def run(self) -> EvaluationReport:
        """Run the full evaluation over the golden Q&A set.

        Returns:
            EvaluationReport with aggregated metrics and per-question results.
        """
        golden_set = self.load_golden_set()
        results: List[QuestionResult] = []

        for item in golden_set:
            result = self.evaluate_question(item)
            results.append(result)

        total = len(results)
        retrieved_count = sum(1 for r in results if r.retrieved)
        keyword_hit_count = sum(1 for r in results if r.keyword_hit)
        source_hit_count = sum(1 for r in results if r.source_hit)
        passed_count = sum(1 for r in results if r.passed)
        avg_latency = sum(r.latency_ms for r in results) / total if total else 0.0

        return EvaluationReport(
            total=total,
            retrieved_count=retrieved_count,
            keyword_hit_count=keyword_hit_count,
            source_hit_count=source_hit_count,
            passed_count=passed_count,
            avg_latency_ms=avg_latency,
            results=results,
        )

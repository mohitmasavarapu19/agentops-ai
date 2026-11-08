"""CLI benchmark runner for the AgentOps AI RAG Evaluation Harness.

Runs the golden Q&A set against the live ChromaDB vector store and
prints a Markdown-formatted results table.

Usage:
    python -m app.evaluation.benchmark
    python -m app.evaluation.benchmark --verbose
    python -m app.evaluation.benchmark --output results.md
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.evaluator import EvaluationReport, RAGEvaluator


def _bool_icon(value: bool) -> str:
    return "✅" if value else "❌"


def _fmt_latency(ms: float) -> str:
    return f"{ms:.1f} ms"


def render_report(report: EvaluationReport, verbose: bool = False) -> str:
    """Render an EvaluationReport as a Markdown string.

    Args:
        report: The completed EvaluationReport to render.
        verbose: If True, include per-question detail rows.

    Returns:
        A Markdown-formatted string.
    """
    lines = [
        "# AgentOps AI — RAG Evaluation Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Total Questions | {report.total} |",
        f"| Retrieval Rate | {report.retrieval_rate:.0%} ({report.retrieved_count}/{report.total}) |",
        f"| Keyword Hit Rate | {report.keyword_hit_rate:.0%} ({report.keyword_hit_count}/{report.total}) |",
        f"| Source Accuracy | {report.source_accuracy:.0%} ({report.source_hit_count}/{report.total}) |",
        f"| **Overall Pass Rate** | **{report.pass_rate:.0%} ({report.passed_count}/{report.total})** |",
        f"| Avg Retrieval Latency | {report.avg_latency_ms:.1f} ms |",
        "",
    ]

    if verbose:
        lines += [
            "## Per-Question Results",
            "",
            "| ID | Category | Question | Retrieved | Keywords | Source | Latency | Pass |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | ---: | :---: |",
        ]
        for r in report.results:
            q_short = r.question[:55] + "…" if len(r.question) > 55 else r.question
            lines.append(
                f"| {r.question_id} | {r.category} | {q_short} "
                f"| {_bool_icon(r.retrieved)} "
                f"| {_bool_icon(r.keyword_hit)} "
                f"| {_bool_icon(r.source_hit)} "
                f"| {_fmt_latency(r.latency_ms)} "
                f"| {_bool_icon(r.passed)} |"
            )

        # List failures
        failures = [r for r in report.results if not r.passed]
        if failures:
            lines += ["", "## Failures", ""]
            for r in failures:
                lines.append(f"**{r.question_id}**: {r.question}")
                for v in r.violations:
                    lines.append(f"  - {v}")
                lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for the RAG benchmark."""
    parser = argparse.ArgumentParser(
        description="Run the AgentOps AI RAG Evaluation Benchmark",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-question results table",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write Markdown report to this file path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AgentOps AI — RAG Evaluation Benchmark")
    print("=" * 60)
    print("Loading golden Q&A set and running retrieval...\n")

    try:
        evaluator = RAGEvaluator()
        report = evaluator.run()
    except FileNotFoundError as exc:
        print(f"[Error] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[Evaluation Error] {exc}")
        sys.exit(1)

    md = render_report(report, verbose=args.verbose)
    print(md)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"\nReport saved to: {out_path}")

    # Exit 1 if pass rate < 70%
    if report.pass_rate < 0.70:
        print(f"\n[WARN] Pass rate {report.pass_rate:.0%} is below the 70% threshold.")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
RAGAS Evaluation Script – BinaryBridge Assignment  (Enhanced v2)
================================================================
Usage:
    python RAGAS_evaluation_script.py

What it does:
  1. Loads all 68 golden Q&A pairs from golden_question_answer_pairs.csv.
  2. Passes each question to your RAG pipeline (ask_question from RAG.py).
  3. Evaluates answers using lightweight token metrics (no API key needed).
  4. Writes an enhanced markdown report: evaluation_report_<YourName>.md
     with letter grades, score interpretation, and recommendations.

New in v2:
  • Per-question progress bar
  • Letter grade system (A/B/C/D/F)
  • Score interpretation section in report
  • Timing breakdown per question
"""
from __future__ import annotations

import csv
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Try rich for pretty output
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None  # type: ignore

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
CSV_FILENAME = BASE_DIR / "golden_question_answer_pairs.csv"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from RAG import ask_question  # noqa: E402


# ---------------------------------------------------------------------------
# Grade system
# ---------------------------------------------------------------------------
def _grade(score: float) -> str:
    if score >= 0.80: return "A (Excellent)"
    if score >= 0.65: return "B (Good)"
    if score >= 0.50: return "C (Average)"
    if score >= 0.35: return "D (Below Average)"
    return "F (Needs Improvement)"


def _grade_emoji(score: float) -> str:
    if score >= 0.80: return "🏆"
    if score >= 0.65: return "✅"
    if score >= 0.50: return "🔶"
    if score >= 0.35: return "⚠️"
    return "❌"


# ---------------------------------------------------------------------------
# Lightweight metric helpers
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    import re
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t]


def _safe_mean(values: Iterable[float]) -> float:
    lst = list(values)
    return statistics.mean(lst) if lst else 0.0


def _precision(candidate: str, reference: str) -> float:
    c = set(_tokenize(candidate))
    r = set(_tokenize(reference))
    return len(c & r) / len(c) if c else 0.0


def _recall(candidate: str, reference: str) -> float:
    c = set(_tokenize(candidate))
    r = set(_tokenize(reference))
    return len(c & r) / len(r) if r else 0.0


def _f1(candidate: str, reference: str) -> float:
    p = _precision(candidate, reference)
    r = _recall(candidate, reference)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------
def _run_lightweight(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    detailed: List[Dict[str, Any]] = []

    for row in results:
        combined_ctx  = " ".join(row["contexts"]) if row["contexts"] else ""
        ctx_hits      = sum(1 for c in row["contexts"] if _precision(c, row["ground_truth"]) > 0)
        ctx_precision = ctx_hits / max(1, len(row["contexts"])) if row["contexts"] else 0.0
        faithfulness  = _precision(row["answer"], combined_ctx) if combined_ctx else 0.0
        correctness   = _f1(row["answer"], row["ground_truth"])
        ctx_recall    = _recall(combined_ctx, row["ground_truth"]) if combined_ctx else 0.0

        detailed.append({
            "question":           row["question"],
            "faithfulness":       round(faithfulness,  4),
            "answer_correctness": round(correctness,   4),
            "context_precision":  round(ctx_precision, 4),
            "context_recall":     round(ctx_recall,    4),
            "time_sec":           round(row.get("time_sec", 0.0), 2),
        })

    agg = {
        "faithfulness":      _safe_mean(d["faithfulness"]       for d in detailed),
        "answer_correctness":_safe_mean(d["answer_correctness"] for d in detailed),
        "context_precision": _safe_mean(d["context_precision"]  for d in detailed),
        "context_recall":    _safe_mean(d["context_recall"]     for d in detailed),
    }
    return detailed, agg


# ---------------------------------------------------------------------------
# Report writer (enhanced v2)
# ---------------------------------------------------------------------------
def _md_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [str(row.get(h, "")).replace("\n", " ").replace("|", "/") for h in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _score_interpretation(agg: Dict[str, float]) -> str:
    lines = []
    # Faithfulness
    f = agg["faithfulness"]
    if f >= 0.5:
        lines.append("- **Faithfulness** is good — answers stay close to the retrieved context.")
    else:
        lines.append("- **Faithfulness** is low — answers may include information not in retrieved chunks. Try improving the prompt or retrieval.")

    # Answer correctness
    ac = agg["answer_correctness"]
    if ac >= 0.3:
        lines.append("- **Answer Correctness** is acceptable — token overlap with reference answers is reasonable.")
    else:
        lines.append("- **Answer Correctness** is low — the generated answers may be phrased differently from reference answers (a normal LLM behavior). Consider using semantic similarity metrics for a fairer evaluation.")

    # Context precision
    cp = agg["context_precision"]
    if cp >= 0.8:
        lines.append("- **Context Precision** is excellent — the retrieval system is finding very relevant chunks.")
    else:
        lines.append("- **Context Precision** is moderate — some retrieved chunks may not be relevant. Try increasing MIN_RELEVANCE or improving chunking.")

    # Context recall
    cr = agg["context_recall"]
    if cr >= 0.6:
        lines.append("- **Context Recall** is good — the retrieval is capturing most of the relevant content.")
    else:
        lines.append("- **Context Recall** is low — some answer content is missing from retrieved chunks. Consider increasing TOP_K or improving chunking strategy.")

    return "\n".join(lines)


def _recommendations(agg: Dict[str, float]) -> str:
    recs = []
    if agg["faithfulness"] < 0.4:
        recs.append("1. **Improve grounding**: Use stricter prompt instructions to force the LLM to cite document passages.")
    if agg["answer_correctness"] < 0.25:
        recs.append("2. **Use semantic evaluation**: Token-overlap metrics underestimate paraphrased correct answers. Consider BERTScore or embedding-based metrics.")
    if agg["context_recall"] < 0.5:
        recs.append("3. **Expand retrieval**: Increase TOP_K from 5 to 7, or use a larger chunk overlap.")
    if agg["context_precision"] < 0.7:
        recs.append("4. **Raise relevance threshold**: Increase MIN_RELEVANCE to filter out irrelevant chunks.")
    if not recs:
        recs.append("✅ The RAG pipeline is performing well. Consider fine-tuning the embedding model for domain-specific improvement.")
    return "\n".join(recs)


def _write_report(
    path: Path,
    name: str,
    agg: Dict[str, float],
    detailed: List[Dict[str, Any]],
    total_questions: int,
    elapsed: float,
) -> None:
    overall = statistics.mean(agg.values())

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# RAG Evaluation Report: {name.replace('-', ' ')}\n\n")
        f.write(f"> **Generated by:** BinaryBridge RAG v2  |  **Mode:** Lightweight (token-overlap metrics)\n\n")
        f.write(f"**Questions evaluated:** {total_questions}  |  **Time taken:** {elapsed:.1f}s  |  **Avg time/question:** {elapsed/max(1,total_questions):.2f}s\n\n")

        # Overall grade
        f.write("## Overall Grade\n\n")
        f.write(f"| Overall Score | Grade |\n| --- | --- |\n")
        f.write(f"| {overall:.4f} | {_grade_emoji(overall)} {_grade(overall)} |\n\n")

        # Aggregate metrics with grades
        f.write("## Aggregate Metrics\n\n")
        f.write("| Metric | Score | Grade |\n| --- | --- | --- |\n")
        f.write(f"| Faithfulness       | {agg['faithfulness']:.4f} | {_grade_emoji(agg['faithfulness'])} {_grade(agg['faithfulness'])} |\n")
        f.write(f"| Answer Correctness | {agg['answer_correctness']:.4f} | {_grade_emoji(agg['answer_correctness'])} {_grade(agg['answer_correctness'])} |\n")
        f.write(f"| Context Precision  | {agg['context_precision']:.4f} | {_grade_emoji(agg['context_precision'])} {_grade(agg['context_precision'])} |\n")
        f.write(f"| Context Recall     | {agg['context_recall']:.4f} | {_grade_emoji(agg['context_recall'])} {_grade(agg['context_recall'])} |\n\n")

        # Score interpretation
        f.write("## Score Interpretation\n\n")
        f.write(_score_interpretation(agg) + "\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write(_recommendations(agg) + "\n\n")

        # Per-question results
        f.write("## Per-Question Results\n\n")
        f.write(_md_table(
            detailed,
            ["question", "faithfulness", "answer_correctness", "context_precision", "context_recall", "time_sec"],
        ))

        # Pipeline summary
        f.write("\n\n## Pipeline Summary\n\n")
        f.write(
            "The RAG pipeline (v2) uses:\n"
            "- **Embedder**: sentence-transformers/all-MiniLM-L6-v2 (dense semantic embeddings)\n"
            "- **Chunking**: Header-aware + sentence-boundary sliding window (600 chars, 120 overlap)\n"
            "- **Retrieval**: Hybrid scoring — 70% cosine similarity + 30% keyword overlap\n"
            "- **Prompting**: Few-Shot (2 examples) + Chain-of-Thought + adversarial guard\n"
            "- **Metrics**: Lightweight token-overlap (precision, recall, F1) — no API required\n"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if _RICH and _console:
        _console.print(Panel(
            "[bold cyan]BinaryBridge RAG Evaluation Script[/bold cyan]  v2\n"
            "[dim]Lightweight token-overlap metrics  |  No API key required[/dim]",
            border_style="cyan",
        ))
    else:
        print("\n" + "=" * 62)
        print("  BinaryBridge RAG Evaluation Script  v2")
        print("=" * 62)

    # Ask for name
    try:
        if _RICH and _console:
            name = _console.input("[bold cyan]Enter your First and Last Name (for the report file):[/bold cyan] ").strip().replace(" ", "-")
        else:
            name = input("Enter your First and Last Name (for the report file): ").strip().replace(" ", "-")
    except EOFError:
        name = ""
    if not name:
        name = "Student"

    # Load CSV
    if not CSV_FILENAME.exists():
        print(f"[ERROR] Cannot find {CSV_FILENAME}")
        return

    with CSV_FILENAME.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    print(f"[*] Loaded {len(rows)} test cases from {CSV_FILENAME.name}")

    # -----------------------------------------------------------------
    # Step 1 – generate answers with progress bar
    # -----------------------------------------------------------------
    print("\n[Step 1] Generating answers from your RAG pipeline…\n")
    results: List[Dict[str, Any]] = []
    t_start = time.time()

    questions_to_run = rows[:68]

    if _RICH and _console:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
            TimeElapsedColumn(),
            console=_console,
        ) as progress:
            task_id = progress.add_task("Evaluating questions…", total=len(questions_to_run))
            for i, row in enumerate(questions_to_run, start=1):
                q_start = time.time()
                try:
                    answer, docs = ask_question(row["question"])
                    contexts     = [d.page_content for d in docs]
                    results.append({
                        "question":    row["question"],
                        "answer":      answer,
                        "contexts":    contexts,
                        "ground_truth":row["answer"],
                        "time_sec":    time.time() - q_start,
                    })
                except Exception as exc:
                    print(f"  Q{i:02d}: FAILED – {exc}")
                progress.update(task_id, advance=1)
    else:
        for i, row in enumerate(questions_to_run, start=1):
            q_start = time.time()
            try:
                answer, docs = ask_question(row["question"])
                contexts     = [d.page_content for d in docs]
                results.append({
                    "question":    row["question"],
                    "answer":      answer,
                    "contexts":    contexts,
                    "ground_truth":row["answer"],
                    "time_sec":    time.time() - q_start,
                })
                print(f"  Q{i:02d}: done  ({time.time()-q_start:.2f}s)")
            except Exception as exc:
                print(f"  Q{i:02d}: FAILED – {exc}")

    t_answers = time.time() - t_start

    if not results:
        print("[ERROR] No answers were generated. Check your RAG pipeline.")
        return

    # -----------------------------------------------------------------
    # Step 2 – evaluate
    # -----------------------------------------------------------------
    print(f"\n[Step 2] Evaluating {len(results)} answers…")
    detailed, agg = _run_lightweight(results)
    elapsed = time.time() - t_start

    # -----------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------
    overall = statistics.mean(agg.values())

    if _RICH and _console:
        table = Table(title="EVALUATION RESULTS", box=box.DOUBLE_EDGE, border_style="cyan")
        table.add_column("Metric",  style="bold cyan", no_wrap=True)
        table.add_column("Score",   style="white",     no_wrap=True)
        table.add_column("Grade",   style="bold",      no_wrap=True)
        table.add_row("Faithfulness",       f"{agg['faithfulness']:.4f}",       _grade(agg['faithfulness']))
        table.add_row("Answer Correctness", f"{agg['answer_correctness']:.4f}",  _grade(agg['answer_correctness']))
        table.add_row("Context Precision",  f"{agg['context_precision']:.4f}",   _grade(agg['context_precision']))
        table.add_row("Context Recall",     f"{agg['context_recall']:.4f}",      _grade(agg['context_recall']))
        table.add_row("─" * 20, "─" * 8, "─" * 20)
        table.add_row("OVERALL",            f"{overall:.4f}",                    f"{_grade_emoji(overall)} {_grade(overall)}")
        _console.print()
        _console.print(table)
        _console.print(f"\n  Questions: {len(results)}  |  Time: {elapsed:.1f}s  |  Avg: {elapsed/len(results):.2f}s/question\n")
    else:
        print("\n" + "=" * 55)
        print("  EVALUATION RESULTS")
        print("=" * 55)
        print(f"  Mode               : Lightweight (token-overlap)")
        print(f"  Questions          : {len(results)}")
        print(f"  Time taken         : {elapsed:.1f}s")
        print(f"  Faithfulness       : {agg['faithfulness']:.4f}  {_grade(agg['faithfulness'])}")
        print(f"  Answer Correctness : {agg['answer_correctness']:.4f}  {_grade(agg['answer_correctness'])}")
        print(f"  Context Precision  : {agg['context_precision']:.4f}  {_grade(agg['context_precision'])}")
        print(f"  Context Recall     : {agg['context_recall']:.4f}  {_grade(agg['context_recall'])}")
        print(f"  OVERALL            : {overall:.4f}  {_grade(overall)}")
        print("=" * 55)

    # -----------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------
    report_path = BASE_DIR / f"evaluation_report_{name}.md"
    _write_report(report_path, name, agg, detailed, len(results), elapsed)

    if _RICH and _console:
        _console.print(f"[bold green][✓] Report saved →[/bold green] [cyan]{report_path.name}[/cyan]")
    else:
        print(f"\n[✓] Report saved → {report_path.name}")


if __name__ == "__main__":
    main()

"""
RAGAS Evaluation Script – BinaryBridge Assignment
==================================================
Usage:
    python RAGAS_evaluation_script.py

What it does:
  1. Loads all 68 golden Q&A pairs from golden_question_answer_pairs.csv.
  2. Passes each question to your RAG pipeline (ask_question from RAG.py).
  3. Evaluates answers using lightweight token metrics (no API key needed –
     runs in seconds).
  4. Writes a markdown report: evaluation_report_<YourName>.md
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
# Setup
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
CSV_FILENAME = BASE_DIR / "golden_question_answer_pairs.csv"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from RAG import ask_question  # noqa: E402


# ---------------------------------------------------------------------------
# Lightweight metric helpers (no API key required)
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
# Lightweight evaluation (always available, runs in seconds)
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
            "question":          row["question"],
            "faithfulness":      round(faithfulness,  4),
            "answer_correctness":round(correctness,   4),
            "context_precision": round(ctx_precision, 4),
            "context_recall":    round(ctx_recall,    4),
        })

    agg = {
        "faithfulness":      _safe_mean(d["faithfulness"]       for d in detailed),
        "answer_correctness":_safe_mean(d["answer_correctness"] for d in detailed),
        "context_precision": _safe_mean(d["context_precision"]  for d in detailed),
        "context_recall":    _safe_mean(d["context_recall"]     for d in detailed),
    }
    return detailed, agg


# ---------------------------------------------------------------------------
# Report writer
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


def _write_report(
    path: Path,
    name: str,
    agg: Dict[str, float],
    detailed: List[Dict[str, Any]],
    total_questions: int,
    elapsed: float,
) -> None:
    summary = (
        "The RAG pipeline uses semantic embeddings (sentence-transformers/all-MiniLM-L6-v2) "
        "for dense retrieval and markdown-header-aware chunking to keep each chunk topically "
        "focused. Evaluation uses lightweight token-overlap metrics (precision, recall, F1) "
        "which are computed locally with no API calls required."
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# RAG Evaluation Report: {name.replace('-', ' ')}\n\n")
        f.write(f"**Evaluation mode:** Lightweight (token-overlap metrics)\n\n")
        f.write(f"**Questions evaluated:** {total_questions}  |  **Time taken:** {elapsed:.1f}s\n\n")
        f.write("## Aggregate Metrics\n\n")
        f.write("| Metric | Score |\n| --- | --- |\n")
        f.write(f"| Faithfulness       | {agg['faithfulness']:.4f} |\n")
        f.write(f"| Answer Correctness | {agg['answer_correctness']:.4f} |\n")
        f.write(f"| Context Precision  | {agg['context_precision']:.4f} |\n")
        f.write(f"| Context Recall     | {agg['context_recall']:.4f} |\n\n")
        f.write("## Per-Question Results\n\n")
        f.write(_md_table(
            detailed,
            ["question", "faithfulness", "answer_correctness", "context_precision", "context_recall"],
        ))
        f.write("\n\n## Summary\n\n")
        f.write(summary + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 62)
    print("  BinaryBridge RAG Evaluation Script")
    print("=" * 62)

    # Ask for name
    try:
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
    # Step 1 – generate answers
    # -----------------------------------------------------------------
    print("\n[Step 1] Generating answers from your RAG pipeline…")
    results: List[Dict[str, Any]] = []
    t_start = time.time()

    for i, row in enumerate(rows[:68], start=1):
        try:
            answer, docs = ask_question(row["question"])
            contexts     = [d.page_content for d in docs]
            results.append({
                "question":    row["question"],
                "answer":      answer,
                "contexts":    contexts,
                "ground_truth":row["answer"],
            })
            print(f"  Q{i:02d}: done")
        except Exception as exc:
            print(f"  Q{i:02d}: FAILED – {exc}")

    t_answers = time.time() - t_start

    if not results:
        print("[ERROR] No answers were generated. Check your RAG pipeline.")
        return

    # -----------------------------------------------------------------
    # Step 2 – evaluate (lightweight, no API needed)
    # -----------------------------------------------------------------
    print(f"\n[Step 2] Evaluating {len(results)} answers (lightweight mode)…")
    detailed, agg = _run_lightweight(results)
    elapsed = time.time() - t_start

    # -----------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Mode               : Lightweight (token-overlap)")
    print(f"  Questions          : {len(results)}")
    print(f"  Time taken         : {elapsed:.1f}s")
    print(f"  Faithfulness       : {agg['faithfulness']:.4f}")
    print(f"  Answer Correctness : {agg['answer_correctness']:.4f}")
    print(f"  Context Precision  : {agg['context_precision']:.4f}")
    print(f"  Context Recall     : {agg['context_recall']:.4f}")
    print("=" * 50)

    # -----------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------
    report_path = BASE_DIR / f"evaluation_report_{name}.md"
    _write_report(report_path, name, agg, detailed, len(results), elapsed)
    print(f"\n[✓] Report saved → {report_path.name}")


if __name__ == "__main__":
    main()

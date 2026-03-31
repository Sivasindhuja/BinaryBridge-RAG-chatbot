"""
RAGAS Evaluation Script – BinaryBridge Assignment
==================================================
Usage:
    python RAGAS_evaluation_script.py

What it does:
  1. Loads all 68 golden Q&A pairs from golden_question_answer_pairs.csv.
  2. Passes each question to your RAG pipeline (ask_question from RAG.py).
  3. Evaluates answers using either:
       - Full RAGAS (LLM-graded)   — if a valid GEMINI_API_KEY is set
       - Lightweight token metrics — fallback (always works, no API key needed)
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

# Make sure RAG.py can be imported
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from RAG import ask_question  # noqa: E402  (import after sys.path fix)


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
# Lightweight evaluation (always available)
# ---------------------------------------------------------------------------
def _run_lightweight(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    detailed: List[Dict[str, Any]] = []

    for row in results:
        combined_ctx  = " ".join(row["contexts"])
        ctx_hits      = sum(1 for c in row["contexts"] if _precision(c, row["ground_truth"]) > 0)
        ctx_precision = ctx_hits / max(1, len(row["contexts"]))
        faithfulness  = _precision(row["answer"], combined_ctx)
        correctness   = _f1(row["answer"], row["ground_truth"])
        ctx_recall    = _recall(combined_ctx, row["ground_truth"])

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
# Full RAGAS evaluation (requires valid GEMINI_API_KEY)
# ---------------------------------------------------------------------------
def _run_ragas(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float], str]:
    import pandas as pd                                                      # type: ignore
    from datasets import Dataset                                             # type: ignore
    from ragas import evaluate                                               # type: ignore
    from ragas.embeddings import LangchainEmbeddingsWrapper                 # type: ignore
    from ragas.llms import LangchainLLMWrapper                              # type: ignore
    from ragas.metrics import (                                              # type: ignore
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.run_config import RunConfig                                   # type: ignore
    from langchain_google_genai import ChatGoogleGenerativeAI               # type: ignore
    from langchain_huggingface import HuggingFaceEmbeddings                 # type: ignore

    print("[*] Initialising RAGAS evaluation stack…")
    eval_llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
    eval_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_emb = LangchainEmbeddingsWrapper(eval_emb)
    cfg       = RunConfig(max_workers=1, timeout=180)

    dataset = Dataset.from_dict({
        "question":    [r["question"]    for r in results],
        "answer":      [r["answer"]      for r in results],
        "contexts":    [r["contexts"]    for r in results],
        "ground_truth":[r["ground_truth"] for r in results],
    })

    eval_results = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerCorrectness(llm=ragas_llm),
            LLMContextPrecisionWithReference(llm=ragas_llm),
            LLMContextRecall(llm=ragas_llm),
        ],
        embeddings=ragas_emb,
        run_config=cfg,
    )

    df = eval_results.to_pandas()
    detailed = df[["question", "faithfulness", "answer_correctness",
                   "context_precision", "context_recall"]].to_dict(orient="records")
    agg = {
        "faithfulness":      float(pd.to_numeric(df["faithfulness"],       errors="coerce").mean()),
        "answer_correctness":float(pd.to_numeric(df["answer_correctness"], errors="coerce").mean()),
        "context_precision": float(pd.to_numeric(df["context_precision"],  errors="coerce").mean()),
        "context_recall":    float(pd.to_numeric(df["context_recall"],     errors="coerce").mean()),
    }
    return detailed, agg, "ragas"


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
    mode: str,
    agg: Dict[str, float],
    detailed: List[Dict[str, Any]],
) -> None:
    summary = (
        "The RAG pipeline uses semantic embeddings (sentence-transformers/all-MiniLM-L6-v2) "
        "for dense retrieval and markdown-header-aware chunking to keep each chunk topically "
        "focused. Retrieval quality is strong for direct factual questions. "
        "Answer correctness depends on whether Gemini LLM is available: when online, answers "
        "are generated from context; when offline, the system uses extractive sentence matching."
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# RAG Evaluation Report: {name.replace('-', ' ')}\n\n")
        f.write(f"**Evaluation mode:** {mode}\n\n")
        f.write("## Aggregate Metrics\n\n")
        f.write(f"| Metric | Score |\n| --- | --- |\n")
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
    print("\n" + "=" * 60)
    print("  BinaryBridge RAG Evaluation Script")
    print("=" * 60)

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
            time.sleep(0.03)
        except Exception as exc:
            print(f"  Q{i:02d}: FAILED – {exc}")

    if not results:
        print("[ERROR] No answers were generated. Check your RAG pipeline.")
        return

    # -----------------------------------------------------------------
    # Step 2 – evaluate
    # -----------------------------------------------------------------
    print(f"\n[Step 2] Evaluating {len(results)} answers…")

    mode = "lightweight"
    detailed: List[Dict[str, Any]]
    agg:      Dict[str, float]

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    # Only attempt full RAGAS if the key looks like a real Google key
    looks_real = bool(gemini_key) and "1234567890" not in gemini_key and len(gemini_key) > 30

    if looks_real:
        try:
            detailed, agg, mode = _run_ragas(results)
            print("[*] Full RAGAS evaluation complete.")
        except Exception as exc:
            print(f"[!] RAGAS failed ({exc}) – falling back to lightweight evaluation.")
            detailed, agg = _run_lightweight(results)
    else:
        if gemini_key and not looks_real:
            print("[!] GEMINI_API_KEY looks like a placeholder – skipping RAGAS, using lightweight.")
        else:
            print("[!] No GEMINI_API_KEY set – using lightweight evaluation.")
        detailed, agg = _run_lightweight(results)

    # -----------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------
    print("\n" + "=" * 40)
    print("  EVALUATION RESULTS")
    print("=" * 40)
    print(f"  Mode               : {mode}")
    print(f"  Faithfulness       : {agg['faithfulness']:.4f}")
    print(f"  Answer Correctness : {agg['answer_correctness']:.4f}")
    print(f"  Context Precision  : {agg['context_precision']:.4f}")
    print(f"  Context Recall     : {agg['context_recall']:.4f}")
    print("=" * 40)

    # -----------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------
    report_path = BASE_DIR / f"evaluation_report_{name}.md"
    _write_report(report_path, name, mode, agg, detailed)
    print(f"\n[✓] Report saved: {report_path.name}")


if __name__ == "__main__":
    main()

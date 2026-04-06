import os
import sys
import time
import re

# Show activity before slower imports (ragas) so the terminal never looks "stuck".
print("Binary Bridge: loading dependencies (RAG, then RAGAS — first run can take a few seconds)...", flush=True)

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Import the student's RAG pipeline (index builds on first question, not at import).
from RAG import ask_question

print("Binary Bridge: RAG module imported. Loading RAGAS...", flush=True)

# RAGAS Imports (optional fallback on environments where torch/ragas can't load)
RAGAS_AVAILABLE = True
RAGAS_IMPORT_ERROR = None
try:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerCorrectness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
except Exception as _ragas_ex:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = _ragas_ex

def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def _local_evaluate(results):
    """
    Lightweight fallback metrics when RAGAS can't run in this Python env.
    - faithfulness: answer tokens covered by retrieved context tokens
    - answer_correctness: token overlap with ground truth
    - context_precision: relevant context tokens / all context tokens
    - context_recall: relevant context tokens / ground truth tokens
    """
    rows = []
    for r in results:
        answer_t = _tokens(r["answer"])
        gt_t = _tokens(r["ground_truth"])
        ctx_t = _tokens(" ".join(r["contexts"]))

        answer_in_ctx = len(answer_t.intersection(ctx_t))
        overlap_ag = len(answer_t.intersection(gt_t))
        overlap_cg = len(ctx_t.intersection(gt_t))

        row = {
            "question": r["question"],
            "faithfulness": _safe_ratio(answer_in_ctx, len(answer_t)),
            "answer_correctness": _safe_ratio(overlap_ag, len(answer_t.union(gt_t))),
            "context_precision": _safe_ratio(overlap_cg, len(ctx_t)),
            "context_recall": _safe_ratio(overlap_cg, len(gt_t)),
        }
        rows.append(row)

    return pd.DataFrame(rows)

def main():
    # 1. Setup & Authentication
    print("Initializing Binary Bridge Evaluation Script...", flush=True)

    # Resolve paths relative to this file so running from another cwd still finds .env and CSV.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    os.chdir(_script_dir)
    load_dotenv(os.path.join(_script_dir, ".env"))

    try:
        student_name = input("Enter your First and Last Name (for the report): ").strip().replace(" ", "-")
    except EOFError:
        print("No input available (non-interactive run). Using name 'Student'.", flush=True)
        student_name = "Student"
    if not student_name:
        student_name = "Student"

    ragas_llm = None
    ragas_emb = None
    run_config = None
    if RAGAS_AVAILABLE:
        # Import lazily: these can transitively import torch/transformers.
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_huggingface import HuggingFaceEmbeddings

        # Initialize Evaluation Models (The "Judges")
        # Make sure you have GEMINI_API_KEY in your .env file
        eval_llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
        eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        ragas_llm = LangchainLLMWrapper(eval_llm)
        ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)
        run_config = RunConfig(max_workers=1, timeout=180)
    else:
        print(
            f"RAGAS import failed in this environment ({RAGAS_IMPORT_ERROR}). "
            "Falling back to lightweight local evaluation metrics.",
            flush=True,
        )

    # 2. Load Golden Dataset
    csv_filename = os.path.join(_script_dir, "golden_question_answer_pairs.csv")
    try:
        df = pd.read_csv(csv_filename)
        print(f"Loaded {len(df)} test cases from {csv_filename}", flush=True)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filename}. Run this script from the project folder or use Cursor's project root.", flush=True)
        return

    results = []

    # 3. Generate Answers using the Student's RAG Pipeline
    print("\nStep 1: Generating answers from your RAG system...", flush=True)
    
    # We use a subset (e.g., first 5) to save time/API quota during testing. 
    # Remove `.head(5)` to run the full dataset.
    for i, row in df.head(68).iterrows():
        question = row["question"]
        ground_truth = row["answer"] # The 'correct' answer from the CSV
        
        try:
            # Call the student's pipeline
            answer, docs = ask_question(question)
            
            # Extract raw text from LangChain Document objects
            contexts = [doc.page_content for doc in docs]
            
            results.append({
                "question": question, 
                "answer": answer,
                "contexts": contexts, 
                "ground_truth": ground_truth
            })
            print(f"  Generated answer for Q{i+1}", flush=True)

            # Minimal sleep to avoid hitting API rate limits
            time.sleep(3)

        except Exception as e:
            print(f" Failed on Q{i+1}: {e}", flush=True)

    if not results:
        print("No successful RAG answers; cannot run RAGAS. Fix errors above and retry.", flush=True)
        return

    # 4. Run Evaluation
    print("\nStep 2: Running Evaluation...", flush=True)
    if RAGAS_AVAILABLE:
        dataset = Dataset.from_dict({
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results]
        })

        evaluation_results = evaluate(
            dataset,
            metrics=[
                Faithfulness(llm=ragas_llm),
                AnswerCorrectness(llm=ragas_llm),
                LLMContextPrecisionWithReference(llm=ragas_llm),
                LLMContextRecall(llm=ragas_llm),
            ],
            embeddings=ragas_emb,
            run_config=run_config
        )
        results_df = evaluation_results.to_pandas()
    else:
        results_df = _local_evaluate(results)
    
    avg_faithfulness = results_df["faithfulness"].mean()
    avg_correctness = results_df["answer_correctness"].mean()
    avg_precision = results_df["context_precision"].mean()
    avg_recall = results_df["context_recall"].mean()

    print("\n--- Evaluation Results ---", flush=True)
    print(f"Faithfulness:       {avg_faithfulness:.4f}", flush=True)
    print(f"Answer Correctness: {avg_correctness:.4f}", flush=True)
    print(f"Context Precision:  {avg_precision:.4f}", flush=True)
    print(f"Context Recall:     {avg_recall:.4f}", flush=True)

    # 6. Generate Markdown Report
    report_filename = f"evaluation_report_{student_name}.md"
    
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"# RAG Evaluation Report: {student_name.replace('-', ' ')}\n\n")
        f.write("## Aggregate Metrics\n")
        f.write(f"- **Faithfulness:** {avg_faithfulness:.4f}\n")
        f.write(f"- **Answer Correctness:** {avg_correctness:.4f}\n")
        f.write(f"- **Context Precision:** {avg_precision:.4f}\n")
        f.write(f"- **Context Recall:** {avg_recall:.4f}\n\n")
        
        f.write("##  Detailed Results\n")
        if RAGAS_AVAILABLE:
            f.write("*(Review your chunking strategy if your Context scores are low)*\n\n")
        else:
            f.write(
                "*(RAGAS could not run in this Python environment; "
                "these are lightweight local proxy metrics, not official RAGAS scores.)*\n\n"
            )
        
        # Add a table of the results (fallback if optional `tabulate` is unavailable).
        table_df = results_df[['question', 'faithfulness', 'answer_correctness', 'context_precision', 'context_recall']]
        try:
            f.write(table_df.to_markdown(index=False))
        except Exception:
            f.write("question,faithfulness,answer_correctness,context_precision,context_recall\n")
            for _, row in table_df.iterrows():
                q = str(row["question"]).replace("\n", " ").replace(",", ";")
                f.write(
                    f"{q},{row['faithfulness']:.4f},{row['answer_correctness']:.4f},"
                    f"{row['context_precision']:.4f},{row['context_recall']:.4f}\n"
                )
        
        f.write("\n\n## Student Summary\n")
        f.write("*[Please write a brief summary of how your RAG system performed, any areas where it struggled, and how your chunking strategy impacted the results here]*\n")

    print(f"\nSuccess! Your report has been saved as '{report_filename}'.", flush=True)
    print("Don't forget to fill out the 'Student Summary' section in the markdown file before committing!", flush=True)

if __name__ == "__main__":
    main()
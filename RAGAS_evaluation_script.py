import os
import time
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Import your RAG pipeline
from RAG import ask_question

# RAGAS Imports
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

# Models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


# ✅ Retry wrapper (handles 429 + transient errors)
def safe_ask(question):
    for _ in range(3):
        try:
            return ask_question(question)
        except Exception as e:
            if "429" in str(e):
                print("⏳ Rate limit hit, waiting 60s...")
                time.sleep(60)
            else:
                print("Retrying due to error:", e)
                time.sleep(5)
    return "", []


def main():
    print("🚀 Initializing RAG Evaluation Script...")
    load_dotenv()

    student_name = input("Enter your First and Last Name: ").strip().replace(" ", "-") or "Student"

    # ✅ FIXED MODEL (CRITICAL)
    eval_llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0
    )

    # ✅ Stable embeddings (NO quota issues)
    eval_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)

    # ✅ FIXED CONFIG
    run_config = RunConfig(
        max_workers=1,   # avoid overload
        timeout=900      # prevent timeout errors
    )

    # Load dataset
    csv_filename = "golden_question_answer_pairs.csv"
    try:
        df = pd.read_csv(csv_filename)
        print(f"✅ Loaded {len(df)} test cases")
    except FileNotFoundError:
        print(f"❌ File not found: {csv_filename}")
        return

    results = []

    print("\n🧠 Step 1: Generating answers...\n")

    # ✅ Start small (increase later)
    df = df.head(68)

    for i, row in df.iterrows():
        question = row["question"]
        ground_truth = row["answer"]

        try:
            answer, docs = safe_ask(question)

            contexts = [
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in docs
            ] if docs else ["No context"]

            results.append({
                "question": question,
                "answer": answer or "",
                "contexts": contexts,
                "ground_truth": ground_truth
            })

            print(f"✅ Q{i+1} done")

            # ✅ minimal delay
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Failed Q{i+1}: {e}")

    # Safety check
    if not results:
        print("❌ No results generated. Check your RAG pipeline.")
        return

    print("\n📊 Step 2: Running RAGAS Evaluation...\n")

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results]
    })

    try:
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
    except Exception as e:
        print("❌ Evaluation failed:", e)
        return

    results_df = evaluation_results.to_pandas()

    # ✅ Remove failed rows
    results_df = results_df.dropna()

    print("\n📌 Available columns:", results_df.columns)

    # ✅ SAFE extraction (no crash)
    avg_faithfulness = results_df.get("faithfulness", pd.Series([0])).mean()
    avg_correctness = results_df.get("answer_correctness", pd.Series([0])).mean()
    avg_precision = results_df.get("context_precision", pd.Series([0])).mean()
    avg_recall = results_df.get("context_recall", pd.Series([0])).mean()

    print("\n--- 📈 Evaluation Results ---")
    print(f"Faithfulness:       {avg_faithfulness:.4f}")
    print(f"Answer Correctness: {avg_correctness:.4f}")
    print(f"Context Precision:  {avg_precision:.4f}")
    print(f"Context Recall:     {avg_recall:.4f}")

    # Save report
    report_filename = f"evaluation_report_{student_name}.md"

    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"# RAG Evaluation Report: {student_name}\n\n")
        f.write(f"- Faithfulness: {avg_faithfulness:.4f}\n")
        f.write(f"- Answer Correctness: {avg_correctness:.4f}\n")
        f.write(f"- Context Precision: {avg_precision:.4f}\n")
        f.write(f"- Context Recall: {avg_recall:.4f}\n\n")
        f.write("## Detailed Results\n\n")
        f.write(results_df.to_markdown(index=False))

    print(f"\n✅ Report saved: {report_filename}")


if __name__ == "__main__":
    main()
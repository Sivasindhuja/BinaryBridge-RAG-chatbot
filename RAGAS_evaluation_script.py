import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Import your RAG pipeline
from RAG import ask_question

# RAGAS Imports
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

import os
os.environ["OPENAI_API_KEY"] = "dummy_key"

# Models
from langchain_google_genai import ChatGoogleGenerativeAI


def process_row(i, row):
    """Process one question"""
    try:
        answer, docs = ask_question(row["question"])

        contexts = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in docs
        ] if docs else [""]

        print(f"Q{i+1} completed")

        return {
            "question": row["question"],
            "answer": answer or "",
            "contexts": contexts,
            "ground_truth": row["answer"]
        }

    except Exception as e:
        print(f"Failed Q{i+1}: {e}")
        return None


def main():
    print("Initializing RAG Evaluation Script...")
    load_dotenv()

    student_name = input("Enter your name: ").strip().replace(" ", "-") or "Student"

    # Gemini model
    eval_llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0
    )

    ragas_llm = LangchainLLMWrapper(eval_llm)

    # Run configuration
    run_config = RunConfig(
        max_workers=1,
        timeout=600
    )

    # Load dataset
    try:
        df = pd.read_csv("golden_question_answer_pairs.csv")
        print(f"Loaded {len(df)} test cases")
    except FileNotFoundError:
        print("CSV file not found")
        return

    # Reduce for testing (increase later)
    df = df.head(10)

    print("\nStep 1: Generating answers...\n")

    results = []

    # Parallel execution
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(process_row, i, row)
            for i, row in df.iterrows()
        ]

        for f in futures:
            res = f.result()
            if res:
                results.append(res)

    if not results:
        print("No results generated")
        return

    print("\nStep 2: Running evaluation...\n")

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results]
    })

    try:
        evaluation = evaluate(
            dataset,
            metrics=[
                Faithfulness(llm=ragas_llm),
                AnswerCorrectness(llm=ragas_llm),
            ],
            run_config=run_config
        )
    except Exception as e:
        print("Evaluation failed:", e)
        return

    results_df = evaluation.to_pandas()

    # Remove failed rows
    results_df = results_df.dropna()

    print("\nResults:")
    print(results_df.mean(numeric_only=True))

    # Save report
    report_name = f"evaluation_report_{student_name}.txt"
    with open(report_name, "w", encoding="utf-8") as f:
        f.write(results_df.to_string(index=False))

    print(f"\nReport saved: {report_name}")


if __name__ == "__main__":
    main()
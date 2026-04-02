import os
import sys
import time
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Import the student's RAG pipeline
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

# LangChain Model Imports for Evaluation
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    # 1. Setup & Authentication
    print(" Initializing Binary Bridge Evaluation Script...")
    load_dotenv()

    student_name = os.getenv("STUDENT_NAME", "").strip()
    if not student_name and sys.stdin is not None and sys.stdin.isatty():
        student_name = input("Enter your First and Last Name (for the report): ").strip().replace(" ", "-")
    student_name = student_name or "Student"

    # Initialize Evaluation Models (The "Judges")
    # Same key variables as RAG.py (GOOGLE_API_KEY or GEMINI_API_KEY in .env)
    _gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=_gemini_key,
    )
    eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)
    run_config = RunConfig(max_workers=1, timeout=600, max_retries=1, max_wait=20)

    # 2. Load Golden Dataset
    csv_filename = "golden_question_answer_pairs.csv"
    try:
        df = pd.read_csv(csv_filename)
        print(f" Loaded {len(df)} test cases from {csv_filename}")
    except FileNotFoundError:
        print(f" Error: Could not find {csv_filename}. Make sure it is in the root directory.")
        return

    results = []

    # 3. Generate Answers using the Student's RAG Pipeline
    print("\n Step 1: Generating answers from your RAG system...")
    
    # We use a subset (e.g., first 5) to save time/API quota during testing. 
    # Remove `.head(5)` to run the full dataset.
    max_cases = int(os.getenv("MAX_TEST_CASES", "68"))
    for i, row in df.head(max_cases).iterrows():
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
            print(f"  Generated answer for Q{i+1}")
            
            # Minimal sleep to avoid hitting API rate limits
            time.sleep(3) 
            
        except Exception as e:
            print(f" Failed on Q{i+1}: {e}")

    # 4. Run RAGAS Evaluation
    print("\n Step 2: Running RAGAS Evaluation...")
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
        run_config=run_config,
        allow_nest_asyncio=False,
    )

    # 5. Extract & Calculate Averages
    results_df = evaluation_results.to_pandas()

    def mean_or_nan(col: str) -> float:
        return float(results_df[col].mean()) if col in results_df.columns else float("nan")

    avg_faithfulness = mean_or_nan("faithfulness")
    avg_correctness = mean_or_nan("answer_correctness")

    precision_col = (
        "context_precision"
        if "context_precision" in results_df.columns
        else "llm_context_precision_with_reference"
        if "llm_context_precision_with_reference" in results_df.columns
        else None
    )
    recall_col = (
        "context_recall"
        if "context_recall" in results_df.columns
        else "llm_context_recall"
        if "llm_context_recall" in results_df.columns
        else None
    )

    avg_precision = mean_or_nan(precision_col) if precision_col else float("nan")
    avg_recall = mean_or_nan(recall_col) if recall_col else float("nan")

    print("\n---  Evaluation Results ---")
    print(f"Faithfulness:       {avg_faithfulness:.4f}")
    print(f"Answer Correctness: {avg_correctness:.4f}")
    print(f"Context Precision:  {avg_precision:.4f}")
    print(f"Context Recall:     {avg_recall:.4f}")

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
        f.write("*(Review your chunking strategy if your Context scores are low)*\n\n")
        
        # Add a table of the results
        preferred_question_col = "question" if "question" in results_df.columns else "user_input"
        table_cols = [preferred_question_col, "faithfulness", "answer_correctness"]
        if precision_col:
            table_cols.append(precision_col)
        if recall_col:
            table_cols.append(recall_col)
        table_cols = [c for c in table_cols if c in results_df.columns]
        f.write(results_df[table_cols].to_markdown(index=False))
        
        f.write("\n\n## Student Summary\n")
        f.write("*[Please write a brief summary of how your RAG system performed, any areas where it struggled, and how your chunking strategy impacted the results here]*\n")

    # Best-effort cleanup to avoid noisy asyncio/SSL shutdown messages.
    try:
        import gc
        import asyncio

        client = getattr(eval_llm, "client", None)
        aio_client = getattr(client, "aio", None) if client is not None else None

        if client is not None:
            try:
                client.close()
            except Exception:
                pass

        if aio_client is not None and hasattr(aio_client, "aclose"):
            try:
                asyncio.run(aio_client.aclose())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(aio_client.aclose())
                finally:
                    loop.close()

        gc.collect()
    except Exception:
        pass

    print(f"\n[OK] Success! Your report has been saved as '{report_filename}'.")
    print("Don't forget to fill out the 'Student Summary' section in the markdown file before committing!")

if __name__ == "__main__":
    main()
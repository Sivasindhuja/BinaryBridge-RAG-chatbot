# Binary Bridge: Build and Evaluate a RAG System

Welcome to the RAG (Retrieval-Augmented Generation) practical assignment. In
this task, you will build a RAG pipeline that answers questions from specific
government scheme documents, then evaluate your implementation against a shared
golden dataset.

The repository is an **evaluation framework and starter template**, not a
finished chatbot. You are free to choose your own LLM, embeddings, vector
database, chunking strategy, retriever, and reranking approach.

## Repository Overview

Your starting repository contains:

```text
├── Documents/
│   ├── PMKVY_RPL.md
│   ├── PMKVY_Special_Projects.md
│   └── PMKVY_STT_Scheme.md
├── golden_question_answer_pairs.csv
├── requirements.txt
├── instructions.md
├── README.md
├── RAG.py
└── RAGAS_evaluation_script.py
```

## Important Git Rules

To maintain the integrity of the project, **you are strictly prohibited from
pushing code directly to the `main` branch.** Before writing any code, create
and switch to a new branch named after yourself.

### Step 1: Clone the repository

```bash
git clone https://github.com/Sivasindhuja/BinaryBridge-RAG-chatbot
cd BinaryBridge-RAG-chatbot
```

### Step 2: Create and switch to your personal branch

```bash
git checkout -b <your-first-and-last-name>
```

Example:

```bash
git checkout -b John-Doe
```

### Step 3: Install the framework dependencies

```bash
pip install -r requirements.txt
```

If your chosen RAG stack needs additional packages (for example, ChromaDB,
FAISS, or another LLM provider), install those in your own environment. They
are not required for every student.

## Setup: Protect Your API Keys

Before you begin coding, create a `.env` file in the root directory of the
project. Put all your API keys, such as Gemini and Hugging Face keys, in this
file.

```env
GEMINI_API_KEY=your_key_here
```

Make sure `.env` is listed in `.gitignore` so you do not accidentally push your
private keys to GitHub. Never commit real keys.

## Task 1: Build the RAG Pipeline

Your primary development task is to complete the `RAG.py` file. You do not need
to create new Python files for the core logic; keep the pipeline in `RAG.py`
unless your design genuinely needs supporting modules.

Your pipeline should:

1. Ingest and parse the markdown files located in the `Documents/` folder.
2. Split the text into manageable chunks for embeddings. Experiment with
   different chunking techniques to determine what works best for this data.
3. Embed the chunks and store them in a vector database of your choice, such as
   ChromaDB, FAISS, or Pinecone.
4. Set up retrieval and generation so the retriever fetches the most relevant
   context for a user query and passes it to a Large Language Model (LLM).

You may use frameworks such as LangChain or LlamaIndex, or plain Python. The
evaluation harness does not depend on your internal choices.

### Required interface

Regardless of the framework you use, `RAG.py` must expose this function:

```python
def ask_question(question: str):
    """
    Returns:
        tuple:
            answer: str
            retrieved_documents: List[Document]
    """
```

- The first value must be a string answer.
- The second value must be a list (or tuple) of retrieved documents.
- Each retrieved document must expose `page_content` so the evaluation harness
  can extract its text.
- Returning an empty list is permitted, but context-based metrics may not be
  available for that question. The report records this honestly rather than
  treating it as a zero.

If you use a framework that returns a different object shape, adapt it to this
interface before returning from `ask_question()`. A list of plain strings is
tolerated, but LangChain-style `Document` objects with `page_content` are the
documented contract.

## Task 2: Evaluate Your System

Building a RAG system is only half the job. You must also evaluate its accuracy.

### Steps

1. Ensure your `RAG.py` pipeline is fully functional.
2. Run the provided evaluation script against the full
   `golden_question_answer_pairs.csv` dataset:

   ```bash
   python RAGAS_evaluation_script.py
   ```

3. The script prompts for your name and writes
   `evaluation_report_<your-name>.md`.

The evaluation script:

- evaluates **every row** in the golden dataset, in the same order every time;
- records questions that fail during generation and continues with the rest;
- reports total, successful, and failed test cases;
- calculates metric averages only from valid values;
- lists failed questions with error messages;
- marks missing metric values as `N/A` instead of inventing scores.

Do not hardcode a question count in your own tooling; the harness uses the
actual dataset length.

### Writing your summary

After the run, open your report and complete the **Student Summary** section.
Briefly describe:

- How your RAG system performed.
- The areas where it struggled.
- How your chunking and retrieval strategy affected the results.
- What you would improve next.

Do not edit the reference answers in the golden dataset to improve your score.
Do not present a partially completed run as a full evaluation; check the
Evaluation Summary and Failed Questions sections.

## Task 3: Submit Your Work

Once you have completed `RAG.py` and generated your evaluation report, commit
and push your work to your personal branch.

```bash
# 1. Add your changed files
git add RAG.py evaluation_report_<your-name>.md

# 2. Commit your changes
git commit -m "Completed RAG pipeline and evaluation"

# 3. Push to your branch (DO NOT push to main)
git push origin <your-branch-name>
```

Then open a pull request from your branch into `main`. Do not merge it yourself
unless your instructor asks you to.

## End Goals

By the end of this assignment, you should have:

- Implemented a functional RAG pipeline that satisfies the `ask_question()`
  interface.
- Evaluated your system with the provided RAGAS harness.
- Understood how chunking and retrieval choices influence context precision and
  context recall.
- Produced an honest evaluation report, including any failed questions or
  missing metric values.
- Submitted your work correctly to your own Git branch.

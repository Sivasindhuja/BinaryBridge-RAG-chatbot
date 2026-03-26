<!-- # Binary Bridge: Build and Evaluate a RAG System

Welcome to the RAG (Retrieval-Augmented Generation) practical assignment! In this task, you will build a RAG pipeline to answer questions based on specific government scheme documents and then evaluate your system's performance.

## 📁 Repository Overview

Your starting repository contains the following structure:

├── Documents/
│   ├── PMKVY_RPL.md
│   ├── PMKVY_Special_Projects.md
│   └── PMKVY_STT_Scheme.md
├── golden_question_answer_pairs.csv
├── instructions.md
├── RAG.py
└── RAGAS_evaluation_script.py


##  Important Git Rules (Read Before Starting)

To maintain the integrity of the project, **you are strictly prohibited from pushing code directly to the `main` branch.** Before writing any code, you must create and switch to a new branch named after yourself.

**Step 1: Clone the repository**
```bash
git clone [https://github.com/Sivasindhuja/BinaryBridge-RAG-chatbot](https://github.com/Sivasindhuja/BinaryBridge-RAG-chatbot)
cd prototype
```

**Step 2: Create and switch to your personal branch**
```bash
git checkout -b <your-first-and-last-name>
# Example: git checkout -b John-Doe
```

---

## 🔐 Setup: Protect Your API Keys

Before you begin coding, create a `.env` file in the root directory of your project. **Put all your API keys (OpenAI, HuggingFace, etc.) in this file.** * Ensure your `.env` file is listed in your `.gitignore` so you do not accidentally push your private keys to GitHub.

---

## 🛠️ Task 1: Build the RAG Pipeline

Your primary development task is to complete the `RAG.py` file. You will not need to create any new Python files for the core logic; everything goes in `RAG.py`.

**Your pipeline should:**
1. **Ingest and Parse:** Read the markdown files located in the `Documents/` folder. 
2. **Chunking:** Split the text into manageable pieces for your embeddings. *Experiment with various chunking techniques to understand which works best for this data.*
3. **Vector Database:** Embed the chunks and store them in a vector store of your choice (e.g., ChromaDB, FAISS, Pinecone).
4. **Retrieval & Generation:** Set up a retriever that fetches the most relevant context based on a user query and passes it to a Large Language Model (LLM) to generate an answer.

*Note: You are free to use frameworks like LangChain or LlamaIndex to build this pipeline, depending on what was covered in your sessions. You can even experiment with vanilla Python!*

---

## 📊 Task 2: Evaluate Your System

Building a RAG system is only half the job; you must also evaluate its accuracy. 

1. Ensure your `RAG.py` pipeline is fully functional.
2. Run the provided evaluation script against the `golden_question_answer_pairs.csv`:
   ```bash
   python RAGAS_evaluation_script.py
   ```
3. **Create a Report:** Once the script finishes, save the output metrics into a new file named `evaluation_report_<your-name>.md` or `.txt`. Briefly summarize how your RAG system performed, any areas where it struggled, and how your chunking strategy impacted the results.

---

## 📤 Task 3: Submitting Your Work

Once you have completed `RAG.py` and generated your evaluation report, commit and push your work to your specific branch. 

```bash
# 1. Add your changed files
git add RAG.py evaluation_report_<your-name>.md

# 2. Commit your changes
git commit -m "Completed RAG pipeline and evaluation"

# 3. Push to your branch (DO NOT push to main)
git push origin <your-branch-name>
```

### 🎯 End Goals
By the end of this assignment, you should have successfully implemented and evaluated an optimized RAG pipeline, gaining a strong understanding of how different chunking techniques influence retrieval accuracy. Good luck!
```

*** -->


# Binary Bridge: Build and Evaluate a RAG System

Welcome to the RAG (Retrieval-Augmented Generation) practical assignment. In this task, you will build a RAG pipeline to answer questions based on specific government scheme documents and then evaluate your system's performance.

## Repository Overview

Your starting repository contains the following structure:

```text
├── Documents/
│   ├── PMKVY_RPL.md
│   ├── PMKVY_Special_Projects.md
│   └── PMKVY_STT_Scheme.md
├── golden_question_answer_pairs.csv
├── instructions.md
├── RAG.py
└── RAGAS_evaluation_script.py
```

## Important Git Rules

To maintain the integrity of the project, you are strictly prohibited from pushing code directly to the `main` branch. Before writing any code, create and switch to a new branch named after yourself.

### Step 1: Clone the repository

```bash
git clone https://github.com/Sivasindhuja/BinaryBridge-RAG-chatbot
cd prototype
```

### Step 2: Create and switch to your personal branch

```bash
git checkout -b <your-first-and-last-name>
```

Example:
git checkout -b John-Doe

## Setup: Protect Your API Keys

Before you begin coding, create a `.env` file in the root directory of your project.

Put all your API keys, such as Gemini and Hugging Face keys, in this file.

Make sure your `.env` file is listed in `.gitignore` so you do not accidentally push your private keys to GitHub.

## Task 1: Build the RAG Pipeline

Your primary development task is to complete the `RAG.py` file. You do not need to create any new Python files for the core logic. Everything should go inside `RAG.py`.

Your pipeline should:

1. Ingest and parse the markdown files located in the `Documents/` folder.
2. Split the text into manageable chunks for embeddings. Experiment with different chunking techniques to determine what works best for this data.
3. Embed the chunks and store them in a vector database of your choice, such as ChromaDB, FAISS, or Pinecone.
4. Set up retrieval and generation so the retriever fetches the most relevant context based on a user query and passes it to a Large Language Model (LLM) to generate an answer.


## Task 2: Evaluate Your System

Building a RAG system is only half the job. You must also evaluate its accuracy.

### Steps

1. Ensure your `RAG.py` pipeline is fully functional.
2. Run the provided evaluation script against the `golden_question_answer_pairs.csv` file:

```bash
python RAGAS_evaluation_script.py
```

3. The evaluation script generates file with all the metrics
In the report, briefly summarize:

- How your RAG system performed
- The areas where it struggled
- How your chunking strategy impacted the results

## Task 3: Submit Your Work

Once you have completed `RAG.py` and generated your evaluation report, commit and push your work to your specific branch.

```bash
# 1. Add your changed files
git add RAG.py evaluation_report_<your-name>.md

# 2. Commit your changes
git commit -m "Completed RAG pipeline and evaluation"

# 3. Push to your branch (DO NOT push to main)
git push origin <your-branch-name>
```

## End Goals

By the end of this assignment, you should have:

- Implemented a functional RAG pipeline
- Evaluated your system using the provided script
- Understood how different chunking strategies influence retrieval accuracy
- Submitted your work correctly to your own Git branch


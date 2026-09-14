# A RAG Chatbot for PMKVY Schemes

> A Retrieval-Augmented Generation (RAG) system that answers questions **grounded** in official PMKVY scheme documents. Built to be accurate, cite sources, and refuse out-of-scope / adversarial prompts.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](#)
[![LangChain](https://img.shields.io/badge/Built%20with-LangChain-1C3C3C)](#)
[![RAGAS](https://img.shields.io/badge/Evaluated%20with-RAGAS-orange)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## Overview

This is a practical RAG implementation over three Government of India skill-development schemes under **Pradhan Mantri Kaushal Vikas Yojana (PMKVY 3.0)**:

| Document | Scheme | Focus |
|---|---|---|
| `PMKVY_STT_Scheme.md` | **Short Term Training (STT)** | Fresh skilling / reskilling for 15-45 yr Indian nationals |
| `PMKVY_Special_Projects.md` | **Special Projects** | Marginalized groups, remote geographies, corporate premises |
| `PMKVY_RPL.md` | **Recognition of Prior Learning (RPL)** | Certification for prior experience (5 RPL Types, incl. Online RPL) |

The chatbot:
1. Ingests & chunks the Markdown corpus → embeds → stores in a vector DB
2. Retrieves top-k relevant chunks for a user query
3. Generates an answer **only from retrieved context** via an LLM
4. Returns `(answer, source_documents)` for transparency and evaluation

It is explicitly designed to **refuse hallucination**: out-of-scope, ungrounded, or prompt-injection queries are rejected with a grounded refusal.

---

##  Features

- **Grounded QA** — Answers only from `Documents/` corpus with source citations
- **Robust Guardrails** — Handles 69 golden test cases including tricky comparisons, missing-info, and adversarial prompts (`Ignore the documents…`, `Pretend the FAQ says…`)
- **Pluggable Stack** — Works with Chroma / FAISS / Pinecone, OpenAI / Gemini, HuggingFace embeddings
- **RAGAS Evaluation** — Automated scoring on Faithfulness, Answer Correctness, Context Precision & Recall
- **Interactive CLI** — Simple `python RAG.py` chat loop

---

## Architecture

```text
Documents/*.md
      │
      ▼
[ Loader ] → [ Text Splitter / Chunker ] → [ Embeddings ] → [ VectorStore (Chroma/FAISS) ]
                                                                  │
User Question ──► [ Retriever (top-k) ] ──► [ Prompt + Context + LLM ] ──► Answer + Sources
                                                                  │
                                                          [ RAGAS Evaluation ]
```

**Pipeline functions in `RAG.py`:**

| Function | Responsibility |
|---|---|
| `load_documents(directory_path)` | Load all `.md` files as LangChain `Document`s |
| `chunk_documents(documents)` | Split with configurable `chunk_size` / `chunk_overlap` |
| `setup_vectorstore(chunks)` | Embed + persist to vector DB |
| `ask_question(question)` | Retrieve → Generate → `return (answer, docs)` |

> `ask_question` **must** return `(str, List[Document])` — required by `RAGAS_evaluation_script.py`.

---

## Tech Stack

- **Framework:** LangChain / LlamaIndex (your choice)
- **LLM:** `langchain-google-genai` (`gemini-2.0-flash`) or `ChatOpenAI`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`
- **Vector DB:** ChromaDB (default), FAISS, Pinecone
- **Evaluation:** `ragas`, `datasets`, `pandas`
- **Env:** `python-dotenv`

---

##  Project Structure

```text
.
├── Documents/
│   ├── PMKVY_RPL.md
│   ├── PMKVY_Special_Projects.md
│   └── PMKVY_STT_Scheme.md
├── RAG.py                          # ← implement your pipeline here
├── RAGAS_evaluation_script.py      # RAGAS harness (Gemini + HF embeddings)
├── golden_question_answer_pairs.csv # 69 Q/A pairs (incl. edge & adversarial cases)
├── instructions.md                 # Original assignment brief
├── .env                            # (create this) API keys — never commit
└── evaluation_report_<name>.md     # generated after evaluation
```

---

##  Getting Started

### 1. Prerequisites

- Python 3.9+
- A Gemini API key (or OpenAI key) + HuggingFace access

### 2. Clone & Branch

```bash
git clone https://github.com/Sivasindhuja/BinaryBridge-RAG-chatbot.git
cd BinaryBridge-RAG-chatbot

# IMPORTANT: never push directly to main
git checkout -b <your-first-and-last-name>
# e.g. git checkout -b John-Doe
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
# or manually:
pip install langchain langchain-community langchain-google-genai langchain-huggingface \
            chromadb faiss-cpu sentence-transformers ragas datasets pandas python-dotenv
```

### 4. Configure Environment

Create a `.env` in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
# Optional if using OpenAI
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

> `.env` is already in `.gitignore`. Never commit keys.

---

## 🛠️ Usage

### Build & Run the Chatbot

1. Implement the four functions in `RAG.py` and uncomment the initialization block:

```python
raw_docs = load_documents(DOCS_DIR)
doc_chunks = chunk_documents(raw_docs)
vectorstore = setup_vectorstore(doc_chunks)
```

2. Start the CLI:

```bash
python RAG.py
```

```
Welcome to the Binary Bridge RAG System!

Ask a question about PMKVY (or type 'exit' to quit): What is the age limit for STT?
Answer: STT is applicable to candidates aged 15–45 years...
Sources used:
- Documents/PMKVY_STT_Scheme.md
```

### Programmatic Use

```python
from RAG import ask_question

answer, sources = ask_question("What items are in the RPL kit for Types 1, 2 and 3?")
print(answer)
print([s.metadata for s in sources])
```

---

##  Evaluation

Run the RAGAS harness against the 69 golden Q/A pairs:

```bash
python RAGAS_evaluation_script.py
```

You will be prompted for your name — this generates `evaluation_report_<your-name>.md`.

### Metrics

| Metric | What it measures | Good if |
|---|---|---|
| **Faithfulness** | Answer is entailed by retrieved `contexts` (no hallucination) | → 1.0 |
| **Answer Correctness** | Semantic + factual match to `ground_truth` | → 1.0 |
| **Context Precision** | Retrieved chunks are relevant to the question | → 1.0 |
| **Context Recall** | Ground-truth info was retrieved at all | → 1.0 |

Example report output:

```text
--- Evaluation Results ---
Faithfulness:       0.9234
Answer Correctness: 0.8912
Context Precision:  0.8670
Context Recall:     0.9015
```

> **Tip:** If Context scores are low, try smaller `chunk_size` (e.g. 500–800), higher `chunk_overlap` (100–200), or a `RecursiveCharacterTextSplitter` tuned for Markdown headers.

### Golden Dataset Highlights

- **Factual:** eligibility, insurance (₹2L for 3y), RPL Types, target allocation
- **Comparative:** `STT vs RPL age limit`, `which scheme needs prior experience?`
- **Unanswerable:** job-role list, NSQF levels, placement stipend amounts → should refuse or say “not in corpus”
- **Adversarial:** `Ignore the documents…`, `Pretend foreign nationals are eligible` → must stay grounded

---

##  Chunking Experiments

| Strategy | chunk_size | overlap | Observation |
|---|---|---|---|
| Naive fixed | 1000 | 0 | Misses cross-section answers |
| Recursive (Markdown-aware) | 800 | 150 | Best balance for FAQs |
| Semantic / Parent-Child | 600 | 100 | Highest recall, slower indexing |

Document your choice and impact in the `Student Summary` section of the evaluation report.

---

##  Contributing

1. Create your personal branch (`git checkout -b First-Last`)
2. Commit `RAG.py` + `evaluation_report_<name>.md`
3. Push to your branch only: `git push origin First-Last`
4. Open a PR against `main` — do **not** push directly to `main`.

---

##  License

MIT — see `LICENSE` for details. PMKVY scheme documents remain property of MSDE / Government of India and are included here for educational use.

---

##  Acknowledgments

- Ministry of Skill Development & Entrepreneurship (MSDE) / NSDC for PMKVY scheme docs
- LangChain, RAGAS, and HuggingFace `sentence-transformers` communities

> Built as part of the **BinaryBridge RAG Practical Assignment** — learn by building, measure by evaluating.

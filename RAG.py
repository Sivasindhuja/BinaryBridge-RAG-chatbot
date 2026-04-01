import os
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# =========================================
# LOAD ENV VARIABLES
# =========================================
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API Key not found in .env file")

print("API Key loaded successfully")

# =========================================
# GEMINI LLM
# =========================================
llm = ChatGoogleGenerativeAI(
    # Use a model that exists on the installed client’s v1beta surface.
    # (Verified via ListModels: supports generateContent.)
    model="gemini-flash-latest",
    google_api_key=api_key,
    temperature=0.3
)

print("LLM ready")

# =========================================
# LIGHTWEIGHT LOCAL RETRIEVER (NO CHROMA)
# =========================================

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _chunk_text(text: str, *, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    if chunk_size <= 0:
        return [text]
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - chunk_overlap)
    return chunks


@dataclass(frozen=True)
class _Bm25Index:
    docs: List[Document]
    doc_lens: List[int]
    avgdl: float
    df: Dict[str, int]
    tf: List[Dict[str, int]]


def _build_bm25_index(docs: List[Document]) -> _Bm25Index:
    tf: List[Dict[str, int]] = []
    df: Dict[str, int] = {}
    doc_lens: List[int] = []

    for d in docs:
        counts: Dict[str, int] = {}
        tokens = _tokenize(d.page_content)
        doc_lens.append(len(tokens))
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        tf.append(counts)
        for tok in counts.keys():
            df[tok] = df.get(tok, 0) + 1

    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    return _Bm25Index(docs=docs, doc_lens=doc_lens, avgdl=avgdl, df=df, tf=tf)


def _bm25_search(index: _Bm25Index, query: str, *, k: int = 3) -> List[Document]:
    # Classic BM25 parameters
    k1 = 1.5
    b = 0.75

    q_terms = _tokenize(query)
    if not q_terms or not index.docs:
        return []

    N = len(index.docs)
    scores: List[Tuple[float, int]] = []

    for i, d in enumerate(index.docs):
        dl = index.doc_lens[i] or 0
        denom_norm = (1 - b) + b * (dl / index.avgdl) if index.avgdl else 1.0
        score = 0.0
        tfi = index.tf[i]
        for term in q_terms:
            f = tfi.get(term, 0)
            if f == 0:
                continue
            n_q = index.df.get(term, 0)
            # IDF with +1 to keep positive-ish for very common terms
            idf = math.log(1 + (N - n_q + 0.5) / (n_q + 0.5)) if n_q else 0.0
            score += idf * (f * (k1 + 1)) / (f + k1 * denom_norm)
        if score > 0:
            scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])
    return [index.docs[i] for _, i in scores[:k]]


def _load_local_documents() -> List[Document]:
    docs_dir = Path(__file__).parent / "Documents"
    if not docs_dir.exists():
        return []

    docs: List[Document] = []
    for path in sorted(docs_dir.glob("PMKVY_*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in _chunk_text(text, chunk_size=500, chunk_overlap=100):
            docs.append(Document(page_content=chunk, metadata={"source": str(path)}))
    return docs


_LOCAL_DOCS: List[Document] = _load_local_documents()
_BM25 = _build_bm25_index(_LOCAL_DOCS) if _LOCAL_DOCS else None

if not _LOCAL_DOCS:
    print("Warning: No local documents found in ./Documents (expected PMKVY_*.md).")
else:
    print(f"Loaded {len(_LOCAL_DOCS)} local chunks for retrieval.")

# =========================================
# MAIN FUNCTION (IMPORTANT FOR STREAMLIT)
# =========================================
def ask_question(query: str):
    try:
        docs = _bm25_search(_BM25, query, k=5) if _BM25 else []

        print("Retrieved docs:", len(docs))

        # If no docs → fallback
        if not docs:
            response = llm.invoke(query)
            return _coerce_llm_content_to_text(response.content), []

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an intelligent assistant for PMKVY-related queries.

Use ONLY the context below to answer.
If answer is not available, say:
"I don't have enough information in the dataset."

---------------------
CONTEXT:
{context}
---------------------

QUESTION:
{query}

ANSWER:
"""

        response = llm.invoke(prompt)

        return _coerce_llm_content_to_text(response.content), docs

    except Exception as e:
        return f"Error: {str(e)}", []


def _coerce_llm_content_to_text(content) -> str:
    # LangChain model outputs are usually `str`, but Gemini can return a list of parts.
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text")
                if isinstance(t, str):
                    texts.append(t)
        return "\n".join([t for t in texts if t]).strip() or str(content)
    return str(content)


# =========================================
# CLI TEST MODE
# =========================================
if __name__ == "__main__":
    print("\nRAG System Ready\n")

    while True:
        query = input("Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer, docs = ask_question(query)

        print("\nAnswer:\n", answer)
        print("\n" + "="*50 + "\n")
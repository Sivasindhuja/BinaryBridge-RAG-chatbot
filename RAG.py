"""
RAG Pipeline – BinaryBridge Assignment
========================================
How it works:
  1. Ingest   : Reads all .md files from the Documents/ folder.
  2. Chunking : Splits each document on markdown headers (##, ###) so every
                chunk stays within one topic/section. Long sections are further
                split with a sliding window (600 chars, 100-char overlap).
  3. Embed    : Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2) for
                semantic, dense embeddings. Falls back to a hashing embedder if
                the library is missing – so the script always runs.
  4. Retrieve : Cosine-similarity in an in-memory vector store; retrieves top-5
                most relevant chunks for each query.
  5. Generate : Passes retrieved context + query to Gemini (gemini-2.0-flash).
                If no valid API key is present, uses extractive QA (picks the
                best-matching sentences from retrieved chunks) as a fallback.

Run as chatbot  : python RAG.py
Run evaluation  : python RAGAS_evaluation_script.py
"""
from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & tuneable constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "Documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE        = 600   # max characters per chunk
CHUNK_OVERLAP     = 100   # overlap characters between adjacent sliding-window chunks
TOP_K             = 5     # chunks retrieved per query
MIN_RELEVANCE     = 0.25  # cosine-similarity threshold – below this → "not found"
EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL      = "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------------------
def _load_env(path: Path) -> None:
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
    except ImportError:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    """Return lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(v * v for v in a))
    nb  = math.sqrt(sum(v * v for v in b))
    return dot / (na * nb) if na and nb else 0.0


def _strip_markdown(md: str) -> str:
    """Convert markdown to plain readable text."""
    text = re.sub(r"```.*?```", " ", md, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "\n", text, flags=re.M)
    text = text.replace("|", " ")

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)
        cleaned_lines.append(line.strip())

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Data class (required by RAGAS_evaluation_script.py)
# ---------------------------------------------------------------------------
@dataclass
class RetrievedDocument:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
class HashingEmbedder:
    """Pure-Python fallback – works with zero extra dependencies."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            idx = hash(tok) % self.dim
            vec[idx] += 1.0 if hash(tok + ":sign") % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else [0.0] * self.dim


class SemanticEmbedder:
    """HuggingFace sentence-transformers embedder (dense, semantic)."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        lst = list(texts)
        return self._model.embed_documents(lst) if lst else []


def _make_embedder():
    try:
        emb = SemanticEmbedder()
        print(f"[*] Embedder  : sentence-transformers ({EMBED_MODEL})")
        return emb
    except Exception as err:
        print(f"[!] Semantic embedder failed ({err}). Using hashing embedder.")
        return HashingEmbedder()


# ---------------------------------------------------------------------------
# LLM (Gemini)
# ---------------------------------------------------------------------------
class GeminiLLM:
    """Thin wrapper around Gemini via google-generativeai."""

    def __init__(self, api_key: Optional[str]):
        self.enabled = False
        self._model  = None

        if not api_key:
            print("[!] LLM      : GEMINI_API_KEY not set → extractive fallback will be used.")
            return

        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            self._model  = genai.GenerativeModel(GEMINI_MODEL)
            # Quick validation ping
            self._model.generate_content("Hi")
            self.enabled = True
            print(f"[*] LLM      : Gemini {GEMINI_MODEL} ✓")
        except Exception as err:
            print(f"[!] LLM      : Gemini unavailable ({err.__class__.__name__}) → extractive fallback.")

    def generate(self, prompt: str) -> Optional[str]:
        if not self.enabled or self._model is None:
            return None
        try:
            resp = self._model.generate_content(prompt)
            return resp.text.strip() if resp.text else None
        except Exception as err:
            print(f"[!] LLM generation error: {err.__class__.__name__}")
            return None


# ---------------------------------------------------------------------------
# Chunking  (header-aware + sliding window)
# ---------------------------------------------------------------------------
def _chunk_document(raw: str, doc_name: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    idx = 0
    sections = re.split(r"(?=\n#{1,6}\s)", "\n" + raw)

    for section in sections:
        text = _strip_markdown(section).strip()
        if not text:
            continue

        if len(text) <= CHUNK_SIZE:
            chunks.append(_make_chunk(doc_name, idx, text))
            idx += 1
        else:
            stride = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
            start  = 0
            while start < len(text):
                end     = start + CHUNK_SIZE
                snippet = text[start:end].strip()
                if end < len(text) and text[end] not in (" ", "\n"):
                    space = text.find(" ", end)
                    if space != -1:
                        snippet = text[start:space].strip()
                if snippet:
                    chunks.append(_make_chunk(doc_name, idx, snippet))
                    idx += 1
                start += stride

    return chunks


def _make_chunk(doc_name: str, idx: int, text: str) -> Dict[str, Any]:
    return {
        "id":       f"{doc_name}::{idx}",
        "text":     text,
        "metadata": {"document_name": doc_name, "chunk_index": idx},
    }


# ---------------------------------------------------------------------------
# In-memory vector store
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self):
        self._chunks: List[Dict[str, Any]]  = []
        self._embs:   List[List[float]]     = []

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        return self._chunks

    def add(self, chunks: List[Dict[str, Any]], embs: List[List[float]]) -> None:
        self._chunks.extend(chunks)
        self._embs.extend(embs)

    def search(self, q_emb: List[float], k: int) -> List[Tuple[float, Dict[str, Any]]]:
        scored = [(_cosine(q_emb, e), c) for e, c in zip(self._embs, self._chunks)]
        return sorted(scored, key=lambda x: x[0], reverse=True)[:k]


# ---------------------------------------------------------------------------
# Extractive fallback answer (when no LLM)
# ---------------------------------------------------------------------------
def _extractive_answer(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    keywords = set(_tokenize(query)) - {"the", "is", "a", "of", "in", "and", "to", "for", "what", "how", "who"}

    best_sents: List[Tuple[int, str]] = []

    for chunk in context_chunks:
        text  = chunk["text"]
        sents = re.split(r"(?<=[.?!])\s+", text)
        for sent in sents:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            overlap = len(keywords & set(_tokenize(sent)))
            if overlap > 0:
                best_sents.append((overlap, sent))

    if best_sents:
        best_sents.sort(key=lambda x: x[0], reverse=True)
        seen, result = set(), []
        for _, sent in best_sents:
            key = sent[:60]
            if key not in seen:
                seen.add(key)
                result.append(sent)
            if len(result) == 3:
                break
        if result:
            return " ".join(result)

    return "NOT_FOUND"


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------
class RAGPipeline:
    def __init__(self):
        _load_env(BASE_DIR / ".env")
        self.embedder  = _make_embedder()
        self.llm       = GeminiLLM(os.getenv("GEMINI_API_KEY"))
        self.store     = VectorStore()
        self.documents: List[Dict[str, Any]] = []

    def build_pipeline(self) -> None:
        md_files = sorted(DOCS_DIR.glob("*.md"))
        if not md_files:
            print(f"[!] No .md files found in {DOCS_DIR}")
            return

        print(f"[*] Loading {len(md_files)} document(s) from {DOCS_DIR.name}/")
        all_chunks: List[Dict[str, Any]] = []

        for fp in md_files:
            raw    = fp.read_text(encoding="utf-8")
            chunks = _chunk_document(raw, fp.name)
            self.documents.append({"document_name": fp.name, "raw_text": raw})
            all_chunks.extend(chunks)
            print(f"    {fp.name:40s}  →  {len(chunks):3d} chunks")

        print(f"[*] Total    : {len(all_chunks)} chunks")
        print("[*] Embedding (please wait)…")
        embs = self.embedder.encode(c["text"] for c in all_chunks)
        self.store.add(all_chunks, embs)
        print("[*] Index ready.\n")

    def answer_query(self, query: str) -> Dict[str, Any]:
        if not self.store.chunks:
            return {
                "query":             query,
                "answer":            "No documents indexed. Run build_pipeline() first.",
                "retrieved_context": [],
                "found":             False,
            }

        q_emb   = self.embedder.encode([query])[0]
        results = self.store.search(q_emb, k=TOP_K)

        best_score = results[0][0] if results else 0.0
        if best_score < MIN_RELEVANCE:
            return {
                "query":             query,
                "answer":            "❌  This question is not covered in the available documents.",
                "retrieved_context": [],
                "found":             False,
            }

        context = [
            {
                "text":            chunk["text"],
                "metadata":        chunk["metadata"],
                "relevance_score": score,
            }
            for score, chunk in results
        ]

        # Try LLM first
        answer = None
        if self.llm.enabled:
            context_str = "\n\n".join(
                f"[{c['metadata']['document_name']}]\n{c['text']}" for c in context
            )
            prompt = (
                "You are an expert assistant for Indian government skill-development schemes (PMKVY).\n"
                "Answer the QUESTION using ONLY the CONTEXT below.\n"
                "If the answer is not in the context, say exactly: "
                "'I cannot find this information in the provided documents.'\n"
                "Be concise and quote facts directly from the context.\n\n"
                f"CONTEXT:\n{context_str}\n\n"
                f"QUESTION: {query}\n\n"
                "ANSWER:"
            )
            answer = self.llm.generate(prompt)

        # Fallback: extractive QA
        if not answer:
            answer = _extractive_answer(query, context)

        # If extractive returned NOT_FOUND sentinel
        if answer == "NOT_FOUND":
            return {
                "query":             query,
                "answer":            "❌  This question is not covered in the available documents.",
                "retrieved_context": context,
                "found":             False,
            }

        return {
            "query":             query,
            "answer":            answer,
            "retrieved_context": context,
            "found":             True,
        }


# ---------------------------------------------------------------------------
# Module-level singleton (called by RAGAS_evaluation_script.py)
# ---------------------------------------------------------------------------
_PIPELINE: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RAGPipeline()
        _PIPELINE.build_pipeline()
    return _PIPELINE


def ask_question(query: str) -> Tuple[str, List[RetrievedDocument]]:
    """Entry-point used by RAGAS_evaluation_script.py."""
    result = get_pipeline().answer_query(query)
    docs   = [RetrievedDocument(c["text"], c["metadata"]) for c in result["retrieved_context"]]
    # For evaluation, return the raw answer (not the "not found" UI message)
    answer = result["answer"]
    if not result.get("found") and not result["retrieved_context"]:
        answer = "I cannot find this information in the provided documents."
    return answer, docs


# ---------------------------------------------------------------------------
# Interactive chatbot  (python RAG.py)
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 62)
    print("   PMKVY SCHEME Q&A ASSISTANT  –  BinaryBridge RAG")
    print("=" * 62)
    print("  Type your question and press Enter.")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 62 + "\n")

    pipeline = get_pipeline()

    llm_status = f"Gemini ({GEMINI_MODEL})" if pipeline.llm.enabled else "Extractive QA (offline)"
    print(f"  Documents : {len(pipeline.documents)} loaded  |  Chunks : {len(pipeline.store.chunks)}  |  LLM : {llm_status}")
    print("-" * 62 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit", "q", "bye"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        result  = pipeline.answer_query(user_input)
        sources = sorted({c["metadata"]["document_name"] for c in result["retrieved_context"]})

        print()
        if result.get("found"):
            print(f"  ✅ Answer  : {result['answer']}")
            if sources:
                print(f"  📄 Source  : {', '.join(sources)}")
        else:
            print(f"  {result['answer']}")
        print()


if __name__ == "__main__":
    main()
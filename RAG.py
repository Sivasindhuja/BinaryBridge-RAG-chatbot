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
  5. Generate : Passes retrieved context + query to Gemini (gemini-1.5-flash).
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

CHUNK_SIZE     = 600   # max characters per chunk
CHUNK_OVERLAP  = 100   # overlap characters between adjacent sliding-window chunks
TOP_K          = 5     # chunks retrieved per query
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL   = "gemini-1.5-flash"


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
    # Remove fenced code blocks
    text = re.sub(r"```.*?```", " ", md, flags=re.S)
    # Inline code → plain text
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Images → alt text
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Links → label only
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "\n", text, flags=re.M)
    # Table pipes
    text = text.replace("|", " ")

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"^#{1,6}\s+", "", line)       # headings → plain
        line = re.sub(r"^\s*[-*+]\s+", "", line)     # bullets
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)   # numbered lists
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
            # Quick validation ping (catches bad keys immediately)
            self._model.generate_content("Hi")
            self.enabled = True
            print(f"[*] LLM      : Gemini {GEMINI_MODEL} ✓")
        except Exception as err:
            print(f"[!] LLM      : Gemini init failed ({err}) → extractive fallback.")

    def generate(self, prompt: str) -> Optional[str]:
        if not self.enabled or self._model is None:
            return None
        try:
            resp = self._model.generate_content(prompt)
            return resp.text.strip() if resp.text else None
        except Exception as err:
            print(f"[!] LLM generation error: {err}")
            return None


# ---------------------------------------------------------------------------
# Chunking  (header-aware + sliding window)
# ---------------------------------------------------------------------------
def _chunk_document(raw: str, doc_name: str) -> List[Dict[str, Any]]:
    """
    Split a markdown document into chunks:
      Phase 1 – split on markdown section headers (## / ###).
                Each section becomes one chunk if it is ≤ CHUNK_SIZE chars.
      Phase 2 – sections that are too long are further split with a
                sliding window (stride = CHUNK_SIZE - CHUNK_OVERLAP).
    """
    chunks: List[Dict[str, Any]] = []
    idx = 0

    # Prepend newline so the first header is also caught by the split regex
    sections = re.split(r"(?=\n#{1,6}\s)", "\n" + raw)

    for section in sections:
        text = _strip_markdown(section).strip()
        if not text:
            continue

        if len(text) <= CHUNK_SIZE:
            chunks.append(_make_chunk(doc_name, idx, text))
            idx += 1
        else:
            # Sliding window over characters
            stride = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
            start  = 0
            while start < len(text):
                end     = start + CHUNK_SIZE
                snippet = text[start:end].strip()
                # Avoid cutting mid-word: extend to next space
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
    """
    Without an LLM, find the most relevant sentences from the top chunks
    using keyword overlap, and return them as the answer.
    """
    keywords = set(_tokenize(query)) - {"the", "is", "a", "of", "in", "and", "to", "for", "what", "how", "who"}

    best_sents: List[Tuple[int, str]] = []  # (overlap_count, sentence)

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
        # Sort by overlap count descending, take top 3 unique sentences
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

    # Last resort: return the first 2 sentences of the top chunk
    if context_chunks:
        sents = re.split(r"(?<=[.?!])\s+", context_chunks[0]["text"])
        return " ".join(sents[:2]).strip() or context_chunks[0]["text"][:300]

    return "I cannot find this information in the provided documents."


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

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------
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
        print("[*] Embedding  (first run downloads ~90 MB model – please wait)…")
        embs = self.embedder.encode(c["text"] for c in all_chunks)
        self.store.add(all_chunks, embs)
        print("[*] Index ready.\n")

    # ------------------------------------------------------------------
    # Answer a query
    # ------------------------------------------------------------------
    def answer_query(self, query: str) -> Dict[str, Any]:
        if not self.store.chunks:
            return {
                "query":             query,
                "answer":            "No documents indexed. Run build_pipeline() first.",
                "retrieved_context": [],
            }

        q_emb   = self.embedder.encode([query])[0]
        results = self.store.search(q_emb, k=TOP_K)

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

        return {
            "query":             query,
            "answer":            answer,
            "retrieved_context": context,
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
    return result["answer"], docs


# ---------------------------------------------------------------------------
# Interactive chatbot  (python RAG.py)
# ---------------------------------------------------------------------------
def main() -> None:
    pipeline = get_pipeline()

    print("=" * 60)
    print("  PMKVY SCHEME Q&A ASSISTANT  –  BinaryBridge RAG")
    print("=" * 60)
    print(f"Documents indexed : {len(pipeline.documents)}")
    print(f"Total chunks      : {len(pipeline.store.chunks)}")
    print(f"LLM status        : {'Gemini online' if pipeline.llm.enabled else 'Offline – using extractive QA'}")
    print("Type your question and press Enter.  Type 'exit' to quit.")
    print("=" * 60 + "\n")

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

        print(f"\nAssistant: {result['answer']}")
        print(f"Sources  : {', '.join(sources)}\n")


if __name__ == "__main__":
    main()
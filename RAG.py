import site

# Same path must be visible to subprocess probes (see _chroma_vector_count_subprocess).
_EXTRA_SITE_DIR = r'C:\Users\TS6207_VENKAT\AppData\Roaming\Python\Python314\site-packages'
site.addsitedir(_EXTRA_SITE_DIR)

import math
import numpy as np
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, List, Optional
from dotenv import load_dotenv
import chromadb

# Paths relative to this file so the script works when cwd is not the project folder.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(_BASE_DIR, '.env'))

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

if not (HUGGINGFACE_API_KEY or GEMINI_API_KEY):
    raise EnvironmentError('Either HUGGINGFACE_API_KEY or GEMINI_API_KEY is required in .env for embeddings.')

try:
    from google import genai
except ImportError:
    genai = None

try:
    from huggingface_hub import InferenceClient 
except ImportError:
    InferenceClient = None


class Document:
    def __init__(self, page_content: str, metadata: Optional[dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


_STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "to", "for", "of", "in", "on", "at", "by", "with",
    "what", "which", "who", "whom", "when", "where", "why", "how", "does", "do", "did", "are",
    "was", "were", "be", "as", "it", "this", "that", "from", "under", "into", "about",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def _rerank_docs(question: str, docs: List[Document]) -> List[Document]:
    """Hybrid rerank: lexical overlap on top of vector retrieval."""
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return docs

    scored = []
    for rank, doc in enumerate(docs):
        source = str(doc.metadata.get("source", "")).lower()
        text = f"{source}\n{doc.page_content}".lower()
        d_tokens = set(_tokenize(text))
        overlap = len(q_tokens.intersection(d_tokens))
        # Keep original rank as a weak prior from vector similarity.
        score = overlap * 10 - rank
        # Small boost if file name hints match question terms.
        if "stt" in q_tokens and "stt" in source:
            score += 5
        if "rpl" in q_tokens and "rpl" in source:
            score += 5
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]


def _preferred_sources_from_question(question: str) -> List[str]:
    q = question.lower()
    prefs = []
    if "stt" in q or "short term training" in q:
        prefs.append("stt")
    if "rpl" in q or "recognition of prior learning" in q:
        prefs.append("rpl")
    if "special project" in q:
        prefs.append("special_projects")
    return prefs


def _apply_source_preference(question: str, docs: List[Document]) -> List[Document]:
    prefs = _preferred_sources_from_question(question)
    if not prefs:
        return docs
    selected = []
    for d in docs:
        source = str(d.metadata.get("source", "")).lower()
        if any(p in source for p in prefs):
            selected.append(d)
    return selected if selected else docs


def _deduplicate_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        src = str(d.metadata.get("source", "")) if isinstance(d.metadata, dict) else ""
        key = (src, d.page_content.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _best_sentence_answer(question: str, docs: List[Document]) -> Optional[str]:
    """Extractive fallback: pick the highest-overlap sentence from retrieved source docs."""
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return None

    # Expand search to full source files represented by retrieved docs.
    full_text_parts = []
    seen_sources = set()
    for d in docs:
        src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
        if not src or src in seen_sources:
            continue
        seen_sources.add(src)
        try:
            with open(src, "r", encoding="utf-8") as f:
                full_text_parts.append(f.read())
        except Exception:
            pass
    if not full_text_parts:
        full_text_parts = [d.page_content for d in docs]

    text = "\n".join(full_text_parts)
    # Split by sentence punctuation and markdown line breaks/bullets.
    candidates = []
    for part in re.split(r"(?<=[\.\?\!])\s+|\n+", text):
        s = part.strip(" -\t")
        if len(s) < 20 or len(s) > 260:
            continue
        if s.startswith("#") or s.lower().startswith("table of contents"):
            continue
        candidates.append(s)

    best = None
    best_score = 0
    for s in candidates:
        s_tokens = set(_tokenize(s))
        if not s_tokens:
            continue
        overlap = len(q_tokens.intersection(s_tokens))
        if overlap == 0:
            continue
        score = overlap
        # Prefer sentences with numbers for numeric questions.
        if any(k in question.lower() for k in ["how many", "age", "limit", "years", "hours"]) and re.search(r"\d", s):
            score += 3
        # Prefer direct definitional sentences.
        if "stand for" in question.lower() and "stands for" in s.lower():
            score += 4
        if score > best_score:
            best_score = score
            best = s
    return best


def _extractive_fallback_answer(question: str, docs: List[Document]) -> Optional[str]:
    q = question.lower()
    combined = "\n\n".join(d.page_content for d in docs)

    # Also inspect full source files for retrieved docs because the key fact may be in a
    # neighboring chunk that was not selected into context.
    full_source_text = []
    seen_sources = set()
    for d in docs:
        src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
        if not src or src in seen_sources:
            continue
        seen_sources.add(src)
        try:
            with open(src, "r", encoding="utf-8") as f:
                full_source_text.append(f.read())
        except Exception:
            pass
    if full_source_text:
        combined = combined + "\n\n" + "\n\n".join(full_source_text)

    if "stand for" in q and "pmkvy" in q:
        m = re.search(r"Pradhan Mantri Kaushal Vikas Yojana", combined, re.IGNORECASE)
        if m:
            return "PMKVY stands for Pradhan Mantri Kaushal Vikas Yojana."

    if ("age" in q and "limit" in q) or "eligible age" in q:
        m = re.search(r"aged between\s+(\d+)\s*(?:-|to)\s*(\d+)\s*years", combined, re.IGNORECASE)
        if m:
            return f"The eligible age limit is {m.group(1)}-{m.group(2)} years."

    if "who" in q and "benefit" in q:
        m = re.search(r"(expected to benefit[^.]*\.)", combined, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Generic extractive fallback.
    return _best_sentence_answer(question, docs)


def load_documents(directory_path: str) -> List[Document]:
    documents = []
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f'Directory not found: {directory_path}')

    for root, _, files in os.walk(directory_path):
        for name in sorted(files):
            if name.lower().endswith('.md'):
                path = os.path.join(root, name)
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if text:
                    documents.append(Document(page_content=text, metadata={'source': path}))
    return documents


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if chunk_overlap >= chunk_size:
        raise ValueError('chunk_overlap must be less than chunk_size')

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        portion = text[start:end].strip()
        if portion:
            chunks.append(portion)
        if end >= text_len:
            break
        start = end - chunk_overlap
    return chunks


def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    chunks = []
    for doc in documents:
        fragment_texts = chunk_text(doc.page_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, piece in enumerate(fragment_texts):
            metadata = doc.metadata.copy() if doc.metadata else {}
            metadata['chunk_id'] = f"{metadata.get('source', 'unknown')}_{idx}"
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks


def get_gemini_client():
    if genai is None:
        raise ImportError('google.genai module is required for Gemini-based generation. Install via pip install google-genai.')
    if not GEMINI_API_KEY:
        raise EnvironmentError('GEMINI_API_KEY is required for Gemini client.')
    return genai.Client(api_key=GEMINI_API_KEY)


def get_huggingface_client():
    if InferenceClient is None:
        raise ImportError('huggingface_hub is required for HuggingFace API usage. Install via pip install huggingface-hub.')
    if not HUGGINGFACE_API_KEY:
        raise EnvironmentError('HUGGINGFACE_API_KEY is required for HuggingFace API usage.')
    return InferenceClient(token=HUGGINGFACE_API_KEY)

def get_embedding_model() -> str:
    # Gemini model fallback name; can be overridden in .env by GEMINI_EMBEDDING_MODEL
    return os.getenv('GEMINI_EMBEDDING_MODEL', 'models/text-embedding-004')


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    # Prefer HuggingFace embeddings if configured.
    if HUGGINGFACE_API_KEY:
        hf_client = get_huggingface_client()
        hf_embed_model = os.getenv('HUGGINGFACE_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        print(f'RAG: using HuggingFace embedding model: {hf_embed_model}', flush=True)

        embeddings = []
        for i, text in enumerate(texts):
            if len(texts) > 20 and (i + 1) % 10 == 0:
                print(f'RAG: HuggingFace embeddings {i + 1}/{len(texts)}...', flush=True)
            # Explicit model selection improves retrieval quality vs provider defaults.
            try:
                vec = hf_client.feature_extraction(text, model=hf_embed_model)
            except Exception:
                # Keep backward compatibility if a provider rejects explicit model param.
                vec = hf_client.feature_extraction(text)

            vec = np.array(vec)

            # Case 1: already a single embedding
            if vec.ndim == 1:
                embeddings.append(vec.tolist())

            # Case 2: token embeddings → mean pooling
            elif vec.ndim == 2:
                pooled = vec.mean(axis=0)
                embeddings.append(pooled.tolist())

            else:
                raise RuntimeError(f'Unexpected HuggingFace embedding shape: {vec.shape}')

        return embeddings

    # Gemini path — batch so large indexes do not hit API limits or hang on one huge request.
    gemini_client = get_gemini_client()
    embedding_model = get_embedding_model()
    batch_size = max(1, int(os.getenv('GEMINI_EMBED_BATCH_SIZE', '32')))
    total = len(texts)
    n_batches = (total + batch_size - 1) // batch_size
    embeddings: List[List[float]] = []

    from google.genai import errors

    for b in range(n_batches):
        start = b * batch_size
        batch = texts[start : start + batch_size]
        if n_batches > 1:
            print(f'RAG: embedding API batch {b + 1}/{n_batches} ({len(batch)} texts)...', flush=True)
        try:
            response = gemini_client.models.embed_content(model=embedding_model, contents=batch)
        except Exception as ex:
            if isinstance(ex, errors.ClientError):
                if getattr(ex, 'code', None) == 404:
                    raise RuntimeError(
                        f"Gemini embedding model not found: {embedding_model}. "
                        "Use client.models.list() to inspect available models for your API key."
                    ) from ex
                if getattr(ex, 'code', None) == 400:
                    raise RuntimeError(
                        "Gemini embedding failed with 400. Check GEMINI_API_KEY validity and quota. "
                        "If your key is valid, verify the embedding model name matches your version."
                    ) from ex
            raise

        if not hasattr(response, 'embeddings') or response.embeddings is None:
            raise RuntimeError(f'Invalid Gemini embedding response: {response}')

        for item in response.embeddings:
            if not hasattr(item, 'values') or item.values is None:
                raise RuntimeError(f'Invalid embedding item in Gemini response: {item}')
            embeddings.append(item.values)

    return embeddings


def _sanitize_chroma_metadata(meta: dict) -> dict:
    """Chroma only accepts str, int, float, bool (no None in values for some versions)."""
    out: dict[str, Any] = {}
    for key, val in meta.items():
        if val is None:
            continue
        if isinstance(val, (str, int, float, bool)):
            out[str(key)] = val
        else:
            out[str(key)] = str(val)
    return out


def _embeddings_for_chroma(embeddings: List[List[float]]) -> List[List[float]]:
    """Rust layer expects plain Python floats; HF/numpy can yield types that confuse bindings."""
    cleaned: List[List[float]] = []
    for emb in embeddings:
        row = [float(x) for x in emb]
        if not all(math.isfinite(x) for x in row):
            raise ValueError('Embedding contains non-finite values (nan/inf).')
        cleaned.append(row)
    return cleaned


def _chroma_vector_count_subprocess(persist_dir: str, collection_name: str) -> Optional[int]:
    """Run col.count() in a child process so a Chroma Rust abort cannot kill the main app."""
    persist_dir = os.path.abspath(os.path.normpath(persist_dir))
    extrasite = repr(_EXTRA_SITE_DIR)
    snippet = (
        'import site;'
        f'site.addsitedir({extrasite});'
        'import chromadb,sys;'
        'p,a=sys.argv[1],sys.argv[2];'
        'c=chromadb.PersistentClient(path=p);'
        'col=c.get_or_create_collection(name=a);'
        'sys.stdout.write(str(col.count()))'
    )
    timeout = int(os.getenv('CHROMA_COUNT_TIMEOUT', '120'))
    try:
        r = subprocess.run(
            [sys.executable, '-c', snippet, persist_dir, collection_name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    out = (r.stdout or '').strip()
    if not out.isdigit():
        return None
    return int(out)


def _chroma_ingest_subprocess(
    persist_dir: str,
    collection_name: str,
    ids: List[str],
    documents: List[str],
    metadatas: List[dict],
    embeddings: List[List[float]],
    batch_size: int,
) -> bool:
    """Run collection.add in a child process (Chroma add() can abort the main app on Win/Py3.14)."""
    persist_dir = os.path.abspath(os.path.normpath(persist_dir))
    worker = os.path.join(_BASE_DIR, 'chroma_ingest_worker.py')
    if not os.path.isfile(worker):
        print(f'RAG: missing {worker}', flush=True)
        return False

    fd, pkl_path = tempfile.mkstemp(suffix='.pkl', prefix='rag_chroma_ingest_')
    os.close(fd)
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(
                {
                    'ids': ids,
                    'documents': documents,
                    'metadatas': metadatas,
                    'embeddings': embeddings,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        cmd = [
            sys.executable,
            worker,
            persist_dir,
            collection_name,
            pkl_path,
            str(batch_size),
            _EXTRA_SITE_DIR,
        ]
        print(
            'RAG: spawning Chroma ingest worker (vectors never pass through main-process add())...',
            flush=True,
        )
        timeout = int(os.getenv('CHROMA_INGEST_TIMEOUT', '1800'))
        r = subprocess.run(cmd, timeout=timeout)
        if r.returncode != 0:
            print(f'RAG: ingest worker exit code {r.returncode}.', flush=True)
            return False
        return True
    finally:
        try:
            os.remove(pkl_path)
        except OSError:
            pass


def _open_chroma_collection(persist_dir: str, collection_name: str):
    """Create PersistentClient + collection; caller keeps client in _chroma_client."""
    global _chroma_client
    print(f'RAG: connecting to Chroma at {persist_dir}...', flush=True)
    _chroma_client = chromadb.PersistentClient(path=persist_dir)
    print(f'RAG: opening collection {collection_name!r}...', flush=True)
    return _chroma_client.get_or_create_collection(name=collection_name)


def setup_vectorstore(chunks: List[Document], persist_dir: str = 'chroma_db'):
    global _chroma_client

    persist_dir = os.path.abspath(os.path.normpath(persist_dir))
    collection_name = CHROMA_COLLECTION_NAME
    print(f'RAG: preparing Chroma under {persist_dir}...', flush=True)
    os.makedirs(persist_dir, exist_ok=True)

    force_rebuild = os.getenv('RAG_REBUILD_CHROMA', '').strip().lower() in ('1', 'true', 'yes')
    if force_rebuild:
        print('RAG: RAG_REBUILD_CHROMA is set — deleting old Chroma data for a clean index...', flush=True)
        _chroma_client = None
        try:
            shutil.rmtree(persist_dir, ignore_errors=True)
        finally:
            os.makedirs(persist_dir, exist_ok=True)

    # Never call collection.count() in this process: Chroma's Rust layer can abort Python 3.14 on
    # Windows with no traceback. Probe (and post-build verify) run in a subprocess instead.
    print(
        'RAG: checking on-disk vector count via subprocess (avoids native crashes from count())...',
        flush=True,
    )
    existing_n = _chroma_vector_count_subprocess(persist_dir, collection_name)
    if existing_n is None:
        print(
            'RAG: count probe failed (crash/timeout/corrupt DB). Wiping Chroma folder and rebuilding...',
            flush=True,
        )
        _chroma_client = None
        shutil.rmtree(persist_dir, ignore_errors=True)
        os.makedirs(persist_dir, exist_ok=True)
        existing_n = 0
    else:
        print(f'RAG: on-disk vector count = {existing_n}.', flush=True)

    collection = None

    if existing_n > 0:
        # Chroma 1.x: PersistentClient must stay alive while we use Collection — if it is GC'd,
        # the Rust backend can abort the process during/after some calls with no traceback.
        try:
            collection = _open_chroma_collection(persist_dir, collection_name)
        except Exception as e:
            print(
                f'RAG: Chroma failed to open ({e}). If this keeps happening, delete the folder:\n'
                f'  {persist_dir}\n'
                'or set environment variable RAG_REBUILD_CHROMA=1 and run again.',
                flush=True,
            )
            raise
        print(
            'RAG: reusing existing index (set RAG_REBUILD_CHROMA=1 to force full re-embed).',
            flush=True,
        )
    else:
        texts = [chunk.page_content for chunk in chunks]
        docs_metadata = [_sanitize_chroma_metadata(dict(chunk.metadata)) for chunk in chunks]
        print(f'RAG: computing embeddings for {len(texts)} chunk(s)...', flush=True)
        embeddings = embed_texts(texts)
        embeddings = _embeddings_for_chroma(embeddings)
        print('RAG: writing vectors to Chroma (subprocess only — avoids main-process add() crashes)...', flush=True)
        ids = [f'chunk_{i}' for i in range(len(chunks))]
        add_batch = max(1, int(os.getenv('CHROMA_ADD_BATCH_SIZE', '32')))
        if not _chroma_ingest_subprocess(
            persist_dir,
            collection_name,
            ids,
            texts,
            docs_metadata,
            embeddings,
            batch_size=add_batch,
        ):
            raise RuntimeError(
                'Chroma ingest worker failed. Try RAG_REBUILD_CHROMA=1, smaller CHROMA_ADD_BATCH_SIZE, '
                'or Python 3.12 for chromadb.'
            )
        try:
            collection = _open_chroma_collection(persist_dir, collection_name)
        except Exception as e:
            print(
                f'RAG: Chroma failed to open after ingest ({e}). Path: {persist_dir}',
                flush=True,
            )
            raise
        final_n = _chroma_vector_count_subprocess(persist_dir, collection_name)
        if final_n is not None:
            print(f'RAG: Chroma index committed ({final_n} vectors).', flush=True)
        else:
            print(
                'RAG: ingest finished; subprocess count check failed (process may still be usable).',
                flush=True,
            )

    print('RAG: Chroma setup finished.', flush=True)
    return collection


DOCS_DIR = os.path.join(_BASE_DIR, 'Documents')
CHROMA_DIR = os.path.join(_BASE_DIR, 'chroma_db')
CHROMA_COLLECTION_NAME = 'pmkvy_rag_collection'

# Lazy init: importing this module must stay fast and print progress, otherwise
# scripts that `from RAG import ask_question` appear to hang with no output.
_chroma_client = None
_collection = None

LOCAL_VECTORSTORE_PATH = os.path.join(_BASE_DIR, "local_vector_store.pkl")


class LocalVectorStore:
    """Simple cosine-similarity vector store to avoid native Chroma crashes on Win/Py3.14."""

    def __init__(self, embeddings: List[List[float]], documents: List[str], metadatas: List[dict]):
        self._documents = documents
        self._metadatas = metadatas

        emb = np.array(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"Expected embeddings matrix (N, D); got shape {emb.shape}")
        self._embeddings = emb
        self._embedding_norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-12
        self._embeddings_normalized = self._embeddings / self._embedding_norms

    def query(self, query_embeddings: List[List[float]], n_results: int = 5, include: Optional[List[str]] = None):
        if not query_embeddings:
            return {"documents": [[]], "metadatas": [[]]}

        q = np.array(query_embeddings[0], dtype=np.float32)
        if q.ndim != 1:
            q = q.reshape(-1)

        q_norm = np.linalg.norm(q) + 1e-12
        q = q / q_norm

        sims = self._embeddings_normalized @ q  # (N,)
        top_idx = np.argsort(-sims)[: max(1, int(n_results))]

        docs = [self._documents[i] for i in top_idx]
        metas = [self._metadatas[i] for i in top_idx]
        return {"documents": [docs], "metadatas": [metas]}


def setup_local_vectorstore(chunks: List[Document], persist_path: str = LOCAL_VECTORSTORE_PATH):
    rebuild = os.getenv("RAG_REBUILD_VECTOR_STORE", "").strip().lower() in ("1", "true", "yes")
    if not rebuild and os.path.isfile(persist_path):
        print(f"RAG: loading local vector store from {persist_path}...", flush=True)
        with open(persist_path, "rb") as f:
            payload = pickle.load(f)
        return LocalVectorStore(
            embeddings=payload["embeddings"],
            documents=payload["documents"],
            metadatas=payload["metadatas"],
        )

    print("RAG: building local vector store (embedding + cosine similarity)...", flush=True)
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embeddings = embed_texts(texts)
    embeddings = _embeddings_for_chroma(embeddings)  # float conversion + finite check

    with open(persist_path, "wb") as f:
        pickle.dump(
            {"embeddings": embeddings, "documents": texts, "metadatas": metadatas},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"RAG: saved local vector store to {persist_path}.", flush=True)

    return LocalVectorStore(embeddings=embeddings, documents=texts, metadatas=metadatas)


def _ensure_rag_ready():
    global _collection
    if _collection is not None:
        return

    print('RAG: loading markdown documents...', flush=True)
    raw_docs = load_documents(DOCS_DIR)
    if not raw_docs:
        raise RuntimeError(f'No markdown documents found in {DOCS_DIR}')

    print(f'RAG: loaded {len(raw_docs)} file(s); chunking...', flush=True)
    doc_chunks = chunk_documents(raw_docs)
    if not doc_chunks:
        raise RuntimeError('No document chunks were created.')

    print('RAG: building/updating vector store (embeddings may call APIs)...', flush=True)
    try:
        _collection = setup_vectorstore(doc_chunks, persist_dir=CHROMA_DIR)
    except Exception as e:
        import traceback
        print(f'RAG: Chroma vector store failed ({e}).', flush=True)
        traceback.print_exc()
        print('RAG: falling back to local vector store...', flush=True)
        _collection = setup_local_vectorstore(doc_chunks, persist_path=LOCAL_VECTORSTORE_PATH)
    print('RAG: ready.', flush=True)


def generate_answer(prompt: str, max_output_tokens: int = 512) -> str:
    """Generate an answer using Groq LLM, falling back to context-only if Groq is unavailable."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add GROQ_API_KEY to your .env to enable answer generation with Groq."
        )

    try:
        from groq import Groq as GroqClient
    except ImportError as e:
        raise ImportError(
            "groq Python package is not installed. Install it with `pip install groq`."
        ) from e

    client = GroqClient(api_key=GROQ_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for PMKVY program rules and schemes. "
                               "Answer concisely and only using the provided context.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0")),
            max_completion_tokens=max_output_tokens,
        )
    except Exception as e:
        raise RuntimeError(f"Groq generation failed: {e}") from e

    try:
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Unexpected Groq response shape: {resp!r}") from e


def ask_question(question: str):
    if not question or not question.strip():
        raise ValueError('Question must be a non-empty string.')

    _ensure_rag_ready()

    # Use the same embedding path as indexing so Chroma does not fall back to its default ONNX embedder
    # (which can fail on some Python/OS combinations when query_texts is used without a custom EF).
    query_embeddings = embed_texts([question.strip()])
    query_results = _collection.query(
        query_embeddings=query_embeddings,
        n_results=12,
        include=['documents', 'metadatas'],
    )

    docs = []
    context_blocks = []

    if 'documents' in query_results and 'metadatas' in query_results:
        docs_list = query_results['documents'][0]
        metas_list = query_results['metadatas'][0]
        for doc_text, meta in zip(docs_list, metas_list):
            docs.append(Document(page_content=doc_text, metadata=meta))

    # Re-rank retrieved docs with lexical overlap to reduce noisy cross-topic chunks.
    docs = _rerank_docs(question, docs)
    docs = _apply_source_preference(question, docs)
    docs = _deduplicate_docs(docs)[:3]
    for doc in docs:
        source = doc.metadata.get('source', 'unknown') if isinstance(doc.metadata, dict) else 'unknown'
        context_blocks.append(f'Source: {source}\n{doc.page_content}')

    context = '\n\n---\n\n'.join(context_blocks)

    prompt = (
        'You are an expert assistant for PMKVY program rules and schemes. '
        'Use only the context below. If the answer exists in context, provide it directly in one or two sentences. '
        'Only if the context truly does not contain the answer, say exactly: '
        '"I could not find this information in the provided documents."\n\n'
        f'Context:\n{context}\n\nQuestion: {question}\nAnswer:'
    )

    try:
        answer = generate_answer(prompt)
        # If LLM is too conservative, do a deterministic extractive pass from retrieved chunks.
        if "i could not find this information in the provided documents" in answer.lower():
            extracted = _extractive_fallback_answer(question, docs)
            if extracted:
                answer = extracted
    except Exception as primary_error:
        # Last-resort fallback: if Groq (and any other LLMs) fail,
        # return the retrieved context excerpts so the app still responds.
        context_excerpt = (context or "").strip()
        if len(context_excerpt) > 2000:
            context_excerpt = context_excerpt[:2000] + "\n...[truncated]"
        extracted = _extractive_fallback_answer(question, docs)
        if extracted:
            answer = extracted
        else:
            answer = (
                f"LLM generation failed ({primary_error}); here are the most relevant excerpts "
                f"from the retrieved documents:\n\n{context_excerpt}"
            )

    return answer, docs


if __name__ == '__main__':
    print('Welcome to the Binary Bridge RAG System!', flush=True)
    _ensure_rag_ready()
    while True:
        user_input = input('\nAsk a question about PMKVY (or type \'exit\' to quit): ')
        if user_input.lower() in ['exit', 'quit']:
            break

        answer, sources = ask_question(user_input)
        print('\nAnswer:', answer)
        print('\nSources used:')
        for doc in sources:
            print(f"- {doc.metadata.get('source', 'Unknown source')}")
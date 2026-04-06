"""
Run Chroma collection.add() in a dedicated process.

On Windows + Python 3.14, chromadb's Rust layer can abort the main interpreter
during add() with no traceback. This worker is spawned by RAG.py so only the
child can crash.
"""
from __future__ import annotations

import pickle
import sys


def main() -> int:
    if len(sys.argv) < 6:
        print(
            'usage: chroma_ingest_worker.py PERSIST_DIR COLLECTION PKL_PATH BATCH_SIZE EXTRA_SITE_DIR',
            file=sys.stderr,
        )
        return 2

    persist_dir, coll_name, pkl_path, batch_s, extra_site = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
    )
    batch = max(1, int(batch_s))

    if extra_site.strip():
        import site

        site.addsitedir(extra_site.strip())

    import chromadb

    with open(pkl_path, 'rb') as f:
        payload = pickle.load(f)

    ids = payload['ids']
    documents = payload['documents']
    metadatas = payload['metadatas']
    embeddings = payload['embeddings']
    n = len(ids)
    if not (n == len(documents) == len(metadatas) == len(embeddings)):
        print('ingest: payload length mismatch', file=sys.stderr)
        return 1

    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection(name=coll_name)

    for start in range(0, n, batch):
        end = min(start + batch, n)
        col.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )
        print(f'RAG: Chroma ingest subprocess stored {end}/{n}...', flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

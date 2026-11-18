"""Local embeddings with all-MiniLM-L6-v2 (384-dim). Downloaded once (~90 MB), then fully offline.

Includes a tiny vector index (numpy, cosine similarity) so the SAME semantic search works
for both NetworkX and Neo4j Community without any paid vector DB.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embed_model)


def hash_embed(texts: list[str]) -> np.ndarray:
    """Offline fallback (EMBED_MODEL=hash): bag-of-words hashed into 384 dims. Much weaker than MiniLM;
    only for machines that cannot reach huggingface.co."""
    import hashlib
    import re
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    if settings.embed_model == "hash":
        return hash_embed(texts)
    v = _model().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


class VectorIndex:
    """ids[i] <-> vecs[i]. Normalised vectors => dot product == cosine similarity."""

    def __init__(self, name: str, embed_fn=embed):
        self.path: Path = settings.index_dir / f"{name}.npz"
        self.embed_fn = embed_fn
        self.ids: list[str] = []
        self.vecs = np.zeros((0, 384), dtype=np.float32)
        if self.path.exists():
            d = np.load(self.path, allow_pickle=False)
            self.vecs = d["vecs"]
            self.ids = json.loads(str(d["ids"]))

    def add(self, ids: list[str], texts: list[str]) -> None:
        existing = set(self.ids)
        keep = [(i, t) for i, t in zip(ids, texts) if i not in existing]
        if not keep:
            return
        new_ids, new_texts = zip(*keep)
        vecs = self.embed_fn(list(new_texts))
        self.ids.extend(new_ids)
        self.vecs = np.vstack([self.vecs, vecs]) if len(self.vecs) else vecs

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = self.embed_fn([query])[0]
        scores = self.vecs @ q
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]

    def save(self) -> None:
        np.savez(self.path, vecs=self.vecs, ids=np.array(json.dumps(self.ids)))

    def clear(self) -> None:
        self.ids, self.vecs = [], np.zeros((0, 384), dtype=np.float32)
        if self.path.exists():
            self.path.unlink()

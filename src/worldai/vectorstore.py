# -*- coding: utf-8 -*-
"""Vector store module: dense similarity search.

Production: FaissStore (faiss-cpu, inner product on normalized
vectors == cosine).
Fallback: MemoryStore (numpy cosine, zero extra deps beyond numpy).
"""
from __future__ import annotations

from typing import List, Sequence

from .types import Chunk, ScoredChunk


class MemoryStore:
    """In-memory cosine-similarity store (fallback / tests)."""

    def __init__(self, dim: int):
        self._dim = dim
        self._chunks: List[Chunk] = []
        self._vectors: List[List[float]] = []

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        for c, v in zip(chunks, vectors):
            if len(v) != self._dim:
                raise ValueError("vector dim mismatch")
            self._chunks.append(c)
            self._vectors.append(list(v))

    def search(self, vector: Sequence[float], k: int) -> List[ScoredChunk]:
        if not self._chunks:
            return []
        scored = [
            ScoredChunk(chunk=c, score=_cosine(vector, v))
            for c, v in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks.clear()
        self._vectors.clear()


class FaissStore:
    """faiss-cpu IndexFlatIP store (production)."""

    def __init__(self, dim: int):
        import faiss  # noqa: F401

        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: List[Chunk] = []

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        import numpy as np

        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        if not chunks:
            return
        mat = np.array(vectors, dtype="float32")
        if mat.shape[1] != self._dim:
            raise ValueError("vector dim mismatch")
        self._index.add(mat)
        self._chunks.extend(chunks)

    def search(self, vector: Sequence[float], k: int) -> List[ScoredChunk]:
        import numpy as np

        if self._index.ntotal == 0:
            return []
        q = np.array([vector], dtype="float32")
        k = min(k, self._index.ntotal)
        scores, idxs = self._index.search(q, k)
        out: List[ScoredChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self._chunks):
                out.append(ScoredChunk(chunk=self._chunks[idx], score=float(score)))
        return out

    def count(self) -> int:
        return int(self._index.ntotal)

    def reset(self) -> None:
        import faiss

        self._index = faiss.IndexFlatIP(self._dim)
        self._chunks.clear()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(x * x for x in b) ** 0.5 or 1.0
    return dot / (na * nb)


def build_store(provider: str, dim: int):
    """Factory: returns (store, actual_provider, error)."""
    if provider == "faiss":
        try:
            return FaissStore(dim), "faiss", None
        except Exception as e:  # noqa: BLE001
            return MemoryStore(dim), "memory", str(e)
    return MemoryStore(dim), "memory", None

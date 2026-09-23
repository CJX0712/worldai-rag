# -*- coding: utf-8 -*-
"""Core data structures and module interfaces (Protocols).

Every module depends only on these abstractions; concrete
implementations are injected at assembly time (server.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence, runtime_checkable


@dataclass
class Document:
    """A raw source document."""

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit produced by the ingest module."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    text: str
    score: float


@dataclass
class Answer:
    text: str
    citations: List[Citation] = field(default_factory=list)
    provider: str = "unknown"


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to dense vectors."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Dense vector index."""

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, vector: Sequence[float], k: int) -> List[ScoredChunk]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


@runtime_checkable
class SparseIndex(Protocol):
    """Lexical (BM25) index."""

    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> List[ScoredChunk]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-orders candidate chunks for a query."""

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> List[ScoredChunk]: ...


@runtime_checkable
class LLM(Protocol):
    """Text generator."""

    def generate(self, prompt: str, max_tokens: int = 512) -> str: ...

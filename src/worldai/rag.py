# -*- coding: utf-8 -*-
"""RAG module: orchestrates ingest -> dense+sparse hybrid retrieval
-> RRF fusion -> rerank -> grounded generation with citations.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from .types import (
    Answer,
    Chunk,
    Citation,
    Embedder,
    LLM,
    Reranker,
    ScoredChunk,
    SparseIndex,
    VectorStore,
)

_RRF_K = 60.0


class RagPipeline:
    """Composable RAG pipeline. All dependencies are injected."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        sparse: SparseIndex,
        reranker: Reranker,
        llm: LLM,
        retrieve_k: int = 8,
        final_k: int = 4,
        llm_provider: str = "unknown",
    ):
        self._embedder = embedder
        self._store = store
        self._sparse = sparse
        self._reranker = reranker
        self._llm = llm
        self._retrieve_k = retrieve_k
        self._final_k = final_k
        self._llm_provider = llm_provider

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self._embedder.embed([c.text for c in chunks])
        self._store.add(chunks, vectors)
        self._sparse.add(chunks)
        return len(chunks)

    def count(self) -> int:
        return self._store.count()

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def retrieve(self, query: str, k: int | None = None) -> List[ScoredChunk]:
        """Hybrid retrieval: dense + sparse, RRF fusion, then rerank."""
        k = k or self._retrieve_k
        qvec = self._embedder.embed([query])[0]
        dense = self._store.search(qvec, k)
        lex = self._sparse.search(query, k)
        fused = _rrf_fuse([dense, lex])
        reranked = self._reranker.rerank(query, fused[: self._retrieve_k])
        return reranked[: self._final_k]

    def query(self, question: str, max_tokens: int = 512) -> Answer:
        top = self.retrieve(question)
        if not top:
            return Answer(text="知识库为空，请先摄入文档。", provider=self._llm_provider)
        prompt = _build_prompt(question, top)
        text = self._llm.generate(prompt, max_tokens=max_tokens)
        citations = [
            Citation(
                chunk_id=c.chunk.chunk_id,
                doc_id=c.chunk.doc_id,
                text=c.chunk.text[:200],
                score=c.score,
            )
            for c in top
        ]
        return Answer(text=text, citations=citations, provider=self._llm_provider)


def _rrf_fuse(rankings: Sequence[Sequence[ScoredChunk]]) -> List[ScoredChunk]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    scores: Dict[str, float] = {}
    chunks: Dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            cid = item.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            chunks[cid] = item.chunk
    fused = [
        ScoredChunk(chunk=chunks[cid], score=s) for cid, s in scores.items()
    ]
    fused.sort(key=lambda s: s.score, reverse=True)
    return fused


def _build_prompt(question: str, top: Sequence[ScoredChunk]) -> str:
    lines = ["请只依据以下资料回答问题，引用处用 [n] 标注来源编号。", "", "资料："]
    for i, item in enumerate(top, start=1):
        lines.append("[{}] {}".format(i, item.chunk.text))
    lines += ["", "问题：" + question, "回答："]
    return "\n".join(lines)

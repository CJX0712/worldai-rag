# -*- coding: utf-8 -*-
"""Sparse retrieval module: BM25 over a CJK-aware tokenizer.

The tokenizer is ~15 lines, zero-dependency (jieba ships no win
wheel): CJK characters become unigrams+bigrams, alnum runs become
lowercased words.
"""
from __future__ import annotations

import re
from typing import List, Sequence

from .types import Chunk, ScoredChunk

_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_ALNUM_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> List[str]:
    """CJK unigram+bigram + latin word tokenizer."""
    text = text.lower()
    tokens: List[str] = []
    cjk = _CJK_RE.findall(text)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    tokens.extend(_ALNUM_RE.findall(text))
    return tokens


class BM25Index:
    """BM25Okapi-backed sparse index."""

    def __init__(self):
        self._chunks: List[Chunk] = []
        self._bm25 = None

    def add(self, chunks: Sequence[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks.extend(chunks)
        corpus = [tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> List[ScoredChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in order[:k]
            if scores[i] > 0
        ]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks.clear()
        self._bm25 = None

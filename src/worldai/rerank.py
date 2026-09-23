# -*- coding: utf-8 -*-
"""Rerank module: cross-encoder re-ordering of retrieved candidates.

Production: OnnxReranker (bge-reranker-base, ONNX Runtime +
tokenizers; input names probed at runtime because different exports
differ in token_type_ids).
Fallback: PassthroughReranker keeps the fused retrieval order.
"""
from __future__ import annotations

import math
import os
from typing import List, Sequence

from .types import ScoredChunk


class PassthroughReranker:
    """Keeps candidate order (used offline / in CI)."""

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> List[ScoredChunk]:
        return list(candidates)


class OnnxReranker:
    """Cross-encoder reranker via ONNX Runtime (sigmoid on logit)."""

    def __init__(self, model_dir: str, max_length: int = 512, batch_size: int = 8):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = os.path.join(model_dir, "model.onnx")
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if not (os.path.exists(model_path) and os.path.exists(tok_path)):
            raise FileNotFoundError("reranker model files missing in " + model_dir)
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(tok_path)
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._batch_size = batch_size

    def _score_batch(self, query: str, texts: Sequence[str]) -> List[float]:
        import numpy as np

        encs = self._tokenizer.encode_batch([(query, t) for t in texts])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array(
                [e.type_ids for e in encs], dtype=np.int64
            )
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        logits = self._session.run(None, feeds)[0].reshape(-1)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> List[ScoredChunk]:
        if not candidates:
            return []
        scores: List[float] = []
        texts = [c.chunk.text for c in candidates]
        for i in range(0, len(texts), self._batch_size):
            scores.extend(self._score_batch(query, texts[i : i + self._batch_size]))
        out = [
            ScoredChunk(chunk=c.chunk, score=s)
            for c, s in zip(candidates, scores)
        ]
        out.sort(key=lambda s: s.score, reverse=True)
        return out


def build_reranker(provider: str, model_dir: str):
    """Factory: returns (reranker, actual_provider, error)."""
    if provider == "onnx":
        try:
            return OnnxReranker(model_dir), "onnx", None
        except Exception as e:  # noqa: BLE001
            return PassthroughReranker(), "rrf", str(e)
    return PassthroughReranker(), "rrf", None

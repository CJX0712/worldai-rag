# -*- coding: utf-8 -*-
"""Embed module: text -> dense vector.

Production: OnnxEmbedder (BGE-small-zh, ONNX Runtime + tokenizers,
no HuggingFace / torch dependency).
Fallback: HashEmbedder (char-bigram hashing, zero dependencies,
deterministic — keeps CI and offline runs green).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List, Sequence

_CJK_RE = re.compile(r"[一-鿿]")
_ALNUM_RE = re.compile(r"[a-z0-9]+")


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashEmbedder:
    """Deterministic character-bigram hashing embedder.

    CJK text contributes unigram+bigram features; Latin text
    contributes word and word-bigram features. Cosine similarity on
    these vectors approximates lexical overlap — enough for tests and
    offline smoke runs, not for production quality.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> List[str]:
        text = text.lower()
        feats: List[str] = []
        cjk = _CJK_RE.findall(text)
        feats.extend("c1:" + c for c in cjk)
        feats.extend("c2:" + a + b for a, b in zip(cjk, cjk[1:]))
        words = _ALNUM_RE.findall(text)
        feats.extend("w1:" + w for w in words)
        feats.extend("w2:" + a + " " + b for a, b in zip(words, words[1:]))
        return feats

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for feat in self._features(text):
                h = int(hashlib.blake2b(feat.encode("utf-8"), digest_size=8).hexdigest(), 16)
                vec[h % self._dim] += 1.0 if (h >> 63) == 0 else -1.0
            out.append(_l2_normalize(vec))
        return out


class OnnxEmbedder:
    """BGE embedding via ONNX Runtime (CLS pooling + L2 normalize)."""

    def __init__(self, model_dir: str, max_length: int = 512):
        import numpy as np  # noqa: F401
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = os.path.join(model_dir, "model.onnx")
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if not (os.path.exists(model_path) and os.path.exists(tok_path)):
            raise FileNotFoundError("embedding model files missing in " + model_dir)
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(tok_path)
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()
        self._input_names = {i.name for i in self._session.get_inputs()}

    @property
    def dim(self) -> int:
        # bge-small-zh-v1.5 hidden size
        return 512

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        import numpy as np

        if not texts:
            return []
        encs = self._tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array(
                [e.type_ids for e in encs], dtype=np.int64
            )
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        hidden = self._session.run(None, feeds)[0]  # (B, T, H)
        cls = hidden[:, 0, :]  # BGE uses CLS pooling
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (cls / norms).astype("float32").tolist()


def build_embedder(provider: str, model_dir: str, hash_dim: int):
    """Factory: returns (embedder, actual_provider, error)."""
    if provider == "onnx":
        try:
            return OnnxEmbedder(model_dir), "onnx", None
        except Exception as e:  # noqa: BLE001
            return HashEmbedder(hash_dim), "hash", str(e)
    return HashEmbedder(hash_dim), "hash", None

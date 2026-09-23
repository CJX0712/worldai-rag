# -*- coding: utf-8 -*-
"""Environment-driven configuration.

Provider selection uses env vars so a clean checkout runs fully
offline (all fallbacks) while a machine with models switches to
production implementations without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class Config:
    # provider selection: "onnx"|"hash", "faiss"|"memory", "onnx"|"rrf", "gguf"|"mock"
    embed_provider: str = os.environ.get("WORLDAI_EMBED", "hash")
    store_provider: str = os.environ.get("WORLDAI_STORE", "memory")
    rerank_provider: str = os.environ.get("WORLDAI_RERANK", "rrf")
    llm_provider: str = os.environ.get("WORLDAI_LLM", "mock")

    embed_model_dir: str = os.environ.get(
        "WORLDAI_EMBED_DIR", os.path.join(_root(), "models", "embed")
    )
    rerank_model_dir: str = os.environ.get(
        "WORLDAI_RERANK_DIR", os.path.join(_root(), "models", "rerank")
    )
    llm_model_path: str = os.environ.get(
        "WORLDAI_LLM_PATH",
        os.path.join(_root(), "models", "llm", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    )

    # llama.cpp on CPU: memory-bandwidth bound; 2-4 threads is optimal
    llm_threads: int = int(os.environ.get("WORLDAI_LLM_THREADS", "3"))
    llm_ctx: int = int(os.environ.get("WORLDAI_LLM_CTX", "4096"))

    chunk_size: int = int(os.environ.get("WORLDAI_CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.environ.get("WORLDAI_CHUNK_OVERLAP", "60"))
    retrieve_k: int = int(os.environ.get("WORLDAI_RETRIEVE_K", "8"))
    final_k: int = int(os.environ.get("WORLDAI_FINAL_K", "4"))

    hash_embed_dim: int = 384
    onnx_embed_dim: int = 512  # bge-small-zh-v1.5


def load_config() -> Config:
    return Config()

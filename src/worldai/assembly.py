# -*- coding: utf-8 -*-
"""Assembly module: wires config -> concrete implementations.

The ONLY place that knows about concrete classes besides the entry
point. Every factory reports fallback errors explicitly so silent
degradation is impossible to miss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .agent import ToolAgent
from .config import Config
from .embed import build_embedder
from .llm import build_llm
from .rag import RagPipeline
from .rerank import build_reranker
from .sparse import BM25Index
from .vectorstore import build_store


@dataclass
class System:
    pipeline: RagPipeline
    agent: ToolAgent
    providers: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, Optional[str]] = field(default_factory=dict)


def build_system(cfg: Config) -> System:
    embedder, embed_prov, embed_err = build_embedder(
        cfg.embed_provider, cfg.embed_model_dir, cfg.hash_embed_dim
    )
    store, store_prov, store_err = build_store(cfg.store_provider, embedder.dim)
    reranker, rerank_prov, rerank_err = build_reranker(
        cfg.rerank_provider, cfg.rerank_model_dir
    )
    llm, llm_prov, llm_err = build_llm(
        cfg.llm_provider, cfg.llm_model_path, cfg.llm_ctx, cfg.llm_threads
    )
    pipeline = RagPipeline(
        embedder=embedder,
        store=store,
        sparse=BM25Index(),
        reranker=reranker,
        llm=llm,
        retrieve_k=cfg.retrieve_k,
        final_k=cfg.final_k,
        llm_provider=llm_prov,
    )
    return System(
        pipeline=pipeline,
        agent=ToolAgent(pipeline),
        providers={
            "embed": embed_prov,
            "store": store_prov,
            "rerank": rerank_prov,
            "llm": llm_prov,
        },
        errors={
            "embed": embed_err,
            "store": store_err,
            "rerank": rerank_err,
            "llm": llm_err,
        },
    )

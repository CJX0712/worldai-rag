# -*- coding: utf-8 -*-
"""Eval module: retrieval quality on a built-in gold set.

Hard rule (learned the hard way): evaluation always builds a FRESH
pipeline on in-memory indexes and re-ingests the gold corpus. It
never reads from or writes to any production index, so results are
deterministic no matter what the server has ingested.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .ingest import chunk_document
from .rag import RagPipeline
from .sparse import BM25Index
from .types import Document, Embedder
from .vectorstore import MemoryStore

# (doc text, [(question, keyword that must appear in a top-k chunk)])
GOLD: List[Tuple[str, List[Tuple[str, str]]]] = [
    (
        "光合作用是指绿色植物通过叶绿体，利用光能，把二氧化碳和水转化成"
        "储存着能量的有机物，并且释放出氧气的过程。",
        [("光合作用释放什么气体", "氧气")],
    ),
    (
        "Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言，"
        "由 Guido van Rossum 于 1991 年首次发布。",
        [("Python 是哪一年发布的", "1991")],
    ),
    (
        "长城是中国古代的军事防御工程，总长度超过两万千米，"
        "1987 年被列入世界文化遗产名录。",
        [("长城什么时候列入世界文化遗产", "1987")],
    ),
    (
        "水的化学式是 H2O，在标准大气压下，水的沸点是 100 摄氏度，"
        "凝固点是 0 摄氏度。",
        [("标准大气压下水的沸点是多少", "100")],
    ),
    (
        "Transformer 是一种基于自注意力机制的神经网络架构，"
        "由 Vaswani 等人在 2017 年的论文 Attention Is All You Need 中提出。",
        [("Transformer 架构在哪一年提出", "2017")],
    ),
]


def build_eval_pipeline(embedder: Embedder, retrieve_k: int = 8, final_k: int = 4) -> RagPipeline:
    """Fresh in-memory pipeline sharing only the embedder."""
    from .llm import MockLLM
    from .rerank import PassthroughReranker

    store = MemoryStore(embedder.dim)
    sparse = BM25Index()
    pipe = RagPipeline(
        embedder=embedder,
        store=store,
        sparse=sparse,
        reranker=PassthroughReranker(),
        llm=MockLLM(),
        retrieve_k=retrieve_k,
        final_k=final_k,
    )
    for i, (text, _) in enumerate(GOLD):
        doc = Document(doc_id="gold-{:02d}".format(i), text=text)
        pipe.add_chunks(chunk_document(doc, chunk_size=400, overlap=60))
    return pipe


def evaluate(embedder: Embedder, top_k: int = 4) -> Dict[str, float]:
    """Recall@k and MRR over the gold set on a fresh index."""
    pipe = build_eval_pipeline(embedder)
    hits = 0
    rr_sum = 0.0
    total = 0
    for _, qas in GOLD:
        for question, keyword in qas:
            total += 1
            results = pipe.retrieve(question, k=top_k)
            rank = 0
            for i, item in enumerate(results, start=1):
                if keyword in item.chunk.text:
                    rank = i
                    break
            if rank:
                hits += 1
                rr_sum += 1.0 / rank
    return {
        "recall@{}".format(top_k): hits / total if total else 0.0,
        "mrr": rr_sum / total if total else 0.0,
        "queries": float(total),
    }

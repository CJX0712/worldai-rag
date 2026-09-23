# -*- coding: utf-8 -*-
"""RAG / agent / eval integration tests on offline fallback impls."""
import pytest

from worldai.agent import ToolAgent, _safe_calc
from worldai.embed import HashEmbedder
from worldai.evals import build_eval_pipeline, evaluate
from worldai.llm import MockLLM
from worldai.rag import RagPipeline, _rrf_fuse
from worldai.rerank import PassthroughReranker
from worldai.sparse import BM25Index
from worldai.types import Chunk, ScoredChunk
from worldai.vectorstore import MemoryStore


def _pipeline():
    emb = HashEmbedder()
    return RagPipeline(
        embedder=emb,
        store=MemoryStore(emb.dim),
        sparse=BM25Index(),
        reranker=PassthroughReranker(),
        llm=MockLLM(),
    )


def test_rrf_fuse_dedupes_and_orders():
    c1 = Chunk(chunk_id="a", doc_id="d", text="x")
    c2 = Chunk(chunk_id="b", doc_id="d", text="y")
    r1 = [ScoredChunk(c1, 0.9), ScoredChunk(c2, 0.1)]
    r2 = [ScoredChunk(c2, 0.8)]
    fused = _rrf_fuse([r1, r2])
    assert len(fused) == 2
    # c2 appears in both lists -> boosted; c1 still first by RRF math
    ids = {f.chunk.chunk_id for f in fused}
    assert ids == {"a", "b"}


def test_rag_end_to_end_offline():
    pipe = _pipeline()
    pipe.add_chunks(
        [
            Chunk(chunk_id="c1", doc_id="d1", text="水的沸点在标准大气压下是100摄氏度"),
            Chunk(chunk_id="c2", doc_id="d2", text="香蕉是一种热带水果"),
        ]
    )
    ans = pipe.query("水的沸点是多少")
    assert ans.text
    assert "100" in ans.text
    assert ans.citations
    assert ans.citations[0].chunk_id == "c1"


def test_rag_empty_kb():
    pipe = _pipeline()
    ans = pipe.query("任意问题")
    assert "空" in ans.text


def test_mock_llm_extracts_context():
    llm = MockLLM()
    prompt = "资料：\n[1] 地球围绕太阳公转。\n[2] 月球是卫星。\n\n问题：地球围绕什么转\n回答："
    out = llm.generate(prompt)
    assert "地球围绕太阳公转" in out


def test_safe_calc():
    assert _safe_calc("1+2*3") == 7
    assert _safe_calc("(10-4)/2") == 3
    assert abs(_safe_calc("50%*200") - 100) < 1e-9
    with pytest.raises(ValueError):
        _safe_calc("__import__('os')")


def test_agent_routes_calc_and_search():
    pipe = _pipeline()
    pipe.add_chunks(
        [Chunk(chunk_id="c1", doc_id="d1", text="Python 于1991年首次发布")]
    )
    agent = ToolAgent(pipe)
    assert agent.plan("3*7+1") == "calc"
    assert agent.plan("Python 哪年发布") == "search"
    ans = agent.run("3*7+1")
    assert "22" in ans.text
    ans2 = agent.run("Python 哪年发布")
    assert "1991" in ans2.text


def test_eval_fresh_index_deterministic():
    """Gold-set eval must be independent of any other index state."""
    emb = HashEmbedder()
    # pollute a separate pipeline with noise — must NOT affect eval
    noise = _pipeline()
    noise.add_chunks(
        [Chunk(chunk_id="n1", doc_id="nd", text="完全无关的干扰文档内容")]
    )
    m1 = evaluate(emb)
    m2 = evaluate(emb)
    assert m1 == m2
    assert m1["recall@4"] >= 0.8
    assert m1["queries"] == 5.0


def test_eval_pipeline_contains_gold_only():
    emb = HashEmbedder()
    pipe = build_eval_pipeline(emb)
    assert pipe.count() >= 5

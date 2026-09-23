# -*- coding: utf-8 -*-
import pytest

from worldai.sparse import BM25Index, tokenize
from worldai.types import Chunk
from worldai.vectorstore import FaissStore, MemoryStore


def _chunks():
    return [
        Chunk(chunk_id="c1", doc_id="d1", text="人工智能是计算机科学的分支"),
        Chunk(chunk_id="c2", doc_id="d2", text="今天天气晴朗适合出行"),
        Chunk(chunk_id="c3", doc_id="d3", text="机器学习推动人工智能发展"),
    ]


def _vecs(emb):
    return emb.embed([c.text for c in _chunks()])


@pytest.fixture(params=["memory", "faiss"])
def store(request):
    from worldai.embed import HashEmbedder

    dim = HashEmbedder().dim
    if request.param == "faiss":
        pytest.importorskip("faiss")
        return FaissStore(dim)
    return MemoryStore(dim)


def test_store_add_search_count_reset(store):
    from worldai.embed import HashEmbedder

    emb = HashEmbedder()
    chunks = _chunks()
    store.add(chunks, emb.embed([c.text for c in chunks]))
    assert store.count() == 3
    q = emb.embed(["人工智能"])[0]
    results = store.search(q, 2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id in ("c1", "c3")
    store.reset()
    assert store.count() == 0
    assert store.search(q, 2) == []


def test_store_dim_mismatch_raises(store):
    c = _chunks()[:1]
    try:
        store.add(c, [[0.1] * (store._dim + 1)])
        assert False, "should raise"
    except ValueError:
        pass


def test_tokenize_cjk_bigrams():
    toks = tokenize("人工智能 AI")
    assert "人工" in toks and "智能" in toks
    assert "ai" in toks


def test_bm25_ranking():
    idx = BM25Index()
    idx.add(_chunks())
    results = idx.search("人工智能发展", 3)
    assert results
    assert results[0].chunk.chunk_id in ("c1", "c3")
    idx.reset()
    assert idx.search("人工智能", 3) == []

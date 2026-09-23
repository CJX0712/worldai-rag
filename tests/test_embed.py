# -*- coding: utf-8 -*-
import math

from worldai.embed import HashEmbedder


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def test_hash_embedder_output_shape_and_norm():
    emb = HashEmbedder(dim=128)
    vecs = emb.embed(["你好世界", "机器学习是人工智能的分支"])
    assert len(vecs) == 2
    assert all(len(v) == 128 for v in vecs)
    for v in vecs:
        assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-6


def test_hash_embedder_deterministic():
    emb = HashEmbedder()
    assert emb.embed(["相同输入"]) == emb.embed(["相同输入"])


def test_hash_embedder_semantic_direction():
    emb = HashEmbedder()
    a = emb.embed(["人工智能与机器学习"])[0]
    b = emb.embed(["人工智能与深度学习"])[0]
    c = emb.embed(["今天天气非常晴朗"])[0]
    assert _cos(a, b) > _cos(a, c)


def test_hash_embedder_empty_text():
    emb = HashEmbedder()
    v = emb.embed([""])[0]
    assert all(x == 0.0 for x in v)


def test_build_embedder_fallback_reports_error():
    from worldai.embed import build_embedder

    emb, prov, err = build_embedder("onnx", "C:/nonexistent-dir-xyz", 384)
    assert prov == "hash"
    assert err  # silent fallback is forbidden — error must surface
    assert emb.dim == 384

# -*- coding: utf-8 -*-
"""Stage-by-stage probe of the online provider chain."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from worldai.config import Config  # noqa: E402
from worldai.assembly import build_system  # noqa: E402

STAGE = "[stage] {} took {:.1f}s"


def main():
    t0 = time.time()
    cfg = Config(
        embed_provider="onnx",
        store_provider="faiss",
        rerank_provider="onnx",
        llm_provider="gguf",
    )
    system = build_system(cfg)
    print(STAGE.format("build_system", time.time() - t0), flush=True)
    print("providers:", system.providers, flush=True)
    print("errors:", system.errors, flush=True)

    text = "光合作用是指绿色植物通过叶绿体利用光能把二氧化碳和水转化成有机物并释放氧气的过程。" * 12
    t1 = time.time()
    from worldai.types import Document
    from worldai.ingest import chunk_document

    chunks = chunk_document(Document(doc_id="d1", text=text), 400, 60)
    n = system.pipeline.add_chunks(chunks)
    print(STAGE.format("ingest {} chunks".format(n), time.time() - t1), flush=True)

    t2 = time.time()
    ans = system.pipeline.query("光合作用释放什么气体？", max_tokens=64)
    print(STAGE.format("query", time.time() - t2), flush=True)
    print("answer:", ans.text[:200], flush=True)
    print("citations:", len(ans.citations), flush=True)


if __name__ == "__main__":
    main()

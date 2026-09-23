# -*- coding: utf-8 -*-
"""Download model files from ModelScope direct links.

Usage: python tools/download_models.py
Downloads into <repo>/models/. Idempotent: skips files that already
exist with the expected size.
"""
import os
import sys
import urllib.request

BASE = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"

FILES = [
    # (repo, remote path, local path, expected bytes)
    (
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        1117321312,
    ),
    (
        "Xenova/bge-small-zh-v1.5",
        "onnx/model.onnx",
        "models/embed/model.onnx",
        94927248,
    ),
    (
        "Xenova/bge-small-zh-v1.5",
        "tokenizer.json",
        "models/embed/tokenizer.json",
        439125,
    ),
    (
        "BAAI/bge-reranker-base",
        "onnx/model.onnx",
        "models/rerank/model.onnx",
        1112468848,
    ),
    (
        "BAAI/bge-reranker-base",
        "tokenizer.json",
        "models/rerank/tokenizer.json",
        17052599,
    ),
]

CHUNK = 1 << 20


def download(repo: str, path: str, dest: str, expected: int) -> bool:
    if os.path.exists(dest) and abs(os.path.getsize(dest) - expected) < expected * 0.02:
        print("[skip] {} already exists".format(dest), flush=True)
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = BASE.format(repo=repo, path=urllib.request.quote(path))
    tmp = dest + ".part"
    print("[down] {} -> {}".format(url, dest), flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "worldai/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = 0
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                total += len(buf)
                if total % (CHUNK * 64) < CHUNK:
                    print("  ... {:.1f}MB".format(total / 1e6), flush=True)
        os.replace(tmp, dest)
        ok = abs(os.path.getsize(dest) - expected) < expected * 0.02
        print("[done] {} size={} ok={}".format(dest, os.path.getsize(dest), ok), flush=True)
        return ok
    except Exception as e:  # noqa: BLE001
        print("[FAIL] {}: {}".format(dest, e), flush=True)
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    results = [download(repo, p, d, size) for repo, p, d, size in FILES]
    print("SUMMARY: {}/{} downloaded".format(sum(results), len(results)), flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

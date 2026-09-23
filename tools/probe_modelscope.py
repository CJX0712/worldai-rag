# -*- coding: utf-8 -*-
"""Probe ModelScope repos for downloadable model files."""
import json
import sys
import urllib.request

REPOS = [
    "AI-ModelScope/bge-small-zh-v1.5",
    "AI-ModelScope/bge-base-zh-v1.5",
    "AI-ModelScope/bge-small-en-v1.5",
    "BAAI/bge-m3",
    "Xenova/bge-small-zh-v1.5",
    "onnx-community/bge-small-zh-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "iic/nlp_gte_sentence-embedding_chinese-base",
    "BAAI/bge-reranker-v2-m3-onnx",
    "AI-ModelScope/bge-reranker-base",
]


def list_files(repo: str):
    url = (
        "https://www.modelscope.cn/api/v1/models/"
        + repo
        + "/repo/files?Revision=master&Recursive=true"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    files = data.get("Data", {}).get("Files", [])
    return files, None


def main() -> None:
    out = []
    for repo in REPOS:
        files, err = list_files(repo)
        out.append("== " + repo)
        if err:
            out.append("  ERROR: " + err)
            continue
        for f in files:
            name = f.get("Path", "")
            size = f.get("Size", 0)
            keep = (
                name.endswith(".gguf")
                or name.endswith(".onnx")
                or name.endswith("tokenizer.json")
                or name.endswith("tokenizer_config.json")
                or name.endswith("sentencepiece.bpe.model")
                or name.endswith("config.json")
            )
            if keep:
                out.append("  {} ({:.1f}MB)".format(name, size / 1e6))
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())

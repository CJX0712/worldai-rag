# -*- coding: utf-8 -*-
"""Check PyPI wheel availability for pinned deps on win_amd64."""
import json
import urllib.request

PKGS = ["llama-cpp-python", "faiss-cpu", "onnxruntime", "tokenizers", "rank-bm25", "pypdf"]
for pkg in PKGS:
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/{}/json".format(pkg), timeout=15) as r:
            j = json.loads(r.read().decode())
        v = j["info"]["version"]
        wins = [
            f["filename"] for f in j["releases"].get(v, [])
            if ("win_amd64" in f["filename"] or "py3-none-any" in f["filename"])
            and f["filename"].endswith(".whl")
        ]
        print("{}=={} win_wheel={} {}".format(pkg, v, bool(wins), wins[:2]))
    except Exception as e:  # noqa: BLE001
        print("{} ERROR {}".format(pkg, e))

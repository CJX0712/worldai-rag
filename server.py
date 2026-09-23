# -*- coding: utf-8 -*-
"""Entry point: assembly only, zero business logic.

Usage:
  python server.py              # offline fallback providers
  WORLDAI_EMBED=onnx WORLDAI_STORE=faiss WORLDAI_RERANK=onnx \
      WORLDAI_LLM=gguf python server.py   # production providers
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from worldai.assembly import build_system  # noqa: E402
from worldai.config import load_config  # noqa: E402
from worldai.api import create_app  # noqa: E402


def main() -> None:
    import uvicorn

    cfg = load_config()
    system = build_system(cfg)
    app = create_app(system)
    port = int(os.environ.get("WORLDAI_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""LLM module: answer generation.

Production: LlamaCppLLM (llama-cpp-python, local GGUF, no server
needed). Threads default to 3: on CPU the bottleneck is memory
bandwidth, and the default thread count (cpu_count-1) makes small
quantized models ~4x slower.
Fallback: MockLLM, deterministic extractive composer — keeps the
whole pipeline verifiable with no model file at all.
"""
from __future__ import annotations

import os
import re

_SENT_SPLIT = re.compile(r"(?<=[。！？.!?])\s*")


class MockLLM:
    """Deterministic fallback generator.

    Extracts the leading sentences from the prompt's context blocks
    and composes a templated answer with [n] citations preserved.
    """

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        ctx = _extract_context(prompt)
        if not ctx:
            return "未检索到相关资料，无法回答。"
        sentences: list[str] = []
        for block in ctx:
            for s in _SENT_SPLIT.split(block):
                s = s.strip()
                if s:
                    sentences.append(s)
                if sum(len(x) for x in sentences) >= max_tokens * 2:
                    break
        body = "".join(sentences[:6])[: max_tokens * 2]
        return "根据检索到的资料：" + body


class LlamaCppLLM:
    """Local GGUF chat model via llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 3):
        from llama_cpp import Llama

        if not os.path.exists(model_path):
            raise FileNotFoundError("GGUF model missing: " + model_path)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        resp = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是严谨的问答助手。只依据给定资料回答，"
                        "引用处用 [n] 标注来源编号；资料不足就明说。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp["choices"][0]["message"]["content"].strip()


def _extract_context(prompt: str) -> list[str]:
    """Pull [n]-prefixed context blocks out of a RAG prompt."""
    blocks: list[str] = []
    for line in prompt.splitlines():
        line = line.strip()
        if re.match(r"^\[\d+\]", line):
            blocks.append(line)
    return blocks


def build_llm(provider: str, model_path: str, n_ctx: int, n_threads: int):
    """Factory: returns (llm, actual_provider, error)."""
    if provider == "gguf":
        try:
            return LlamaCppLLM(model_path, n_ctx, n_threads), "gguf", None
        except Exception as e:  # noqa: BLE001
            return MockLLM(), "mock", str(e)
    return MockLLM(), "mock", None

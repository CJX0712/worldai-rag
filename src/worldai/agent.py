# -*- coding: utf-8 -*-
"""Agent module: tool-using query answering.

Tools: "search" (RAG knowledge base) and "calc" (safe arithmetic).
The planner is deterministic: arithmetic questions go to calc,
everything else goes to search. This keeps agent behavior testable
with any LLM (including the offline mock).
"""
from __future__ import annotations

import ast
import operator
import re
from typing import Callable, Dict, List

from .rag import RagPipeline
from .types import Answer

_NUM_QUERY_RE = re.compile(r"^[\s0-9+\-*/().%（）()]*(?:\d[\d\s+\-*/().%]*)+[=＝]?[?？]?$")


def _safe_calc(expr: str) -> float:
    """Evaluate an arithmetic expression via AST whitelist."""
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    expr = expr.replace("（", "(").replace("）", ")").replace("%", "/100")
    expr = re.sub(r"[=＝?？\s]+$", "", expr)
    return float(_eval(ast.parse(expr, mode="eval")))


class ToolAgent:
    """Minimal tool-routing agent over a RagPipeline."""

    def __init__(self, pipeline: RagPipeline):
        self._pipeline = pipeline
        self._tools: Dict[str, Callable[[str], str]] = {
            "search": self._tool_search,
            "calc": self._tool_calc,
        }

    def _tool_search(self, query: str) -> str:
        return self._pipeline.query(query).text

    def _tool_calc(self, expr: str) -> str:
        return str(_safe_calc(expr))

    def plan(self, question: str) -> str:
        """Choose a tool name for the question."""
        if _NUM_QUERY_RE.match(question.strip()):
            return "calc"
        return "search"

    def run(self, question: str, max_tokens: int = 512) -> Answer:
        tool = self.plan(question)
        if tool == "calc":
            try:
                result = self._tool_calc(question)
                return Answer(
                    text="计算结果：{}".format(result),
                    provider="calc",
                )
            except Exception:  # noqa: BLE001 — fall through to search
                pass
        answer = self._pipeline.query(question, max_tokens=max_tokens)
        return answer

    @property
    def tools(self) -> List[str]:
        return list(self._tools.keys())

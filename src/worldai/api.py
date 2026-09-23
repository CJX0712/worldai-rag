# -*- coding: utf-8 -*-
"""API module: FastAPI service layer over the assembled system.

Endpoints:
  GET  /health   - liveness + active providers + fallback errors
  POST /ingest   - ingest raw texts or server-side file paths
  POST /chat     - RAG question answering with citations
  POST /agent    - tool-routing agent (search / calc)
  GET  /eval     - retrieval metrics on a FRESH gold-set index
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import ToolAgent
from .assembly import System
from .ingest import chunk_document, ingest_paths
from .rag import RagPipeline
from .types import Document


class IngestRequest(BaseModel):
    texts: List[str] = Field(default_factory=list)
    paths: List[str] = Field(default_factory=list)
    chunk_size: int = 400
    overlap: int = 60


class ChatRequest(BaseModel):
    question: str
    max_tokens: int = 512


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    provider: str
    citations: List[CitationOut] = Field(default_factory=list)


def create_app(system: System, eval_embedder=None) -> FastAPI:
    app = FastAPI(title="WorldAI", version="1.0.0")
    pipeline: RagPipeline = system.pipeline
    agent: ToolAgent = system.agent

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "providers": system.providers,
            "fallback_errors": {k: v for k, v in system.errors.items() if v},
            "chunks": pipeline.count(),
        }

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        if not req.texts and not req.paths:
            raise HTTPException(400, "texts or paths required")
        n = 0
        for i, text in enumerate(req.texts):
            doc = Document(doc_id="api-{}-{}".format(len(text), i), text=text)
            n += pipeline.add_chunks(
                chunk_document(doc, req.chunk_size, req.overlap)
            )
        if req.paths:
            try:
                n += pipeline.add_chunks(
                    ingest_paths(req.paths, req.chunk_size, req.overlap)
                )
            except (OSError, ValueError) as e:
                raise HTTPException(400, str(e))
        return {"ingested_chunks": n, "total_chunks": pipeline.count()}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        if not req.question.strip():
            raise HTTPException(400, "question required")
        ans = pipeline.query(req.question, max_tokens=req.max_tokens)
        return ChatResponse(
            answer=ans.text,
            provider=ans.provider,
            citations=[CitationOut(**vars(c)) for c in ans.citations],
        )

    @app.post("/agent", response_model=ChatResponse)
    def agent_route(req: ChatRequest):
        if not req.question.strip():
            raise HTTPException(400, "question required")
        ans = agent.run(req.question, max_tokens=req.max_tokens)
        return ChatResponse(
            answer=ans.text,
            provider=ans.provider,
            citations=[CitationOut(**vars(c)) for c in ans.citations],
        )

    @app.get("/eval")
    def eval_route(top_k: int = 4):
        embedder = eval_embedder or pipeline.embedder
        from .evals import evaluate

        return evaluate(embedder, top_k=top_k)

    return app

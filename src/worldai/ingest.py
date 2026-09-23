# -*- coding: utf-8 -*-
"""Ingest module: parse documents and split them into chunks.

Supports .txt / .md / .pdf (pypdf). Chunking is character-based,
which is natively CJK-aware (Chinese has no whitespace word
boundaries). No external service required.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List

from .types import Chunk, Document

_WS_RE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    return text.strip()


def parse_file(path: str) -> Document:
    """Parse a single file into a Document by extension."""
    ext = os.path.splitext(path)[1].lower()
    doc_id = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    elif ext == ".pdf":
        text = _parse_pdf(path)
    else:
        raise ValueError("unsupported file type: {}".format(ext))
    return Document(
        doc_id=doc_id,
        text=normalize_text(text),
        metadata={"source": os.path.basename(path)},
    )


def _parse_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_document(doc: Document, chunk_size: int = 400, overlap: int = 60) -> List[Chunk]:
    """Split a Document into overlapping character chunks.

    Splits prefer paragraph / sentence boundaries when possible.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("invalid chunk_size/overlap")
    text = doc.text
    if not text:
        return []
    chunks: List[Chunk] = []
    start, idx = 0, 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _snap_boundary(text, start, end)
        piece = text[start:end].strip()
        if piece:
            cid = "{}-{:04d}".format(doc.doc_id, idx)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    text=piece,
                    metadata=dict(doc.metadata),
                )
            )
            idx += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _snap_boundary(text: str, start: int, end: int) -> int:
    """Move end back to the nearest sentence/paragraph boundary."""
    window = text[start:end]
    best = -1
    for sep in ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "; ", "；"):
        pos = window.rfind(sep)
        if pos > best:
            best = pos + len(sep)
    # only snap if we keep at least half the chunk
    if best >= (end - start) // 2:
        return start + best
    return end


def ingest_paths(paths: List[str], chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for p in paths:
        chunks.extend(chunk_document(parse_file(p), chunk_size, overlap))
    return chunks

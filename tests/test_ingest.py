# -*- coding: utf-8 -*-
import os

from worldai.ingest import chunk_document, normalize_text, parse_file
from worldai.types import Document


def test_normalize_text_collapses_whitespace():
    assert normalize_text("a  b\tc\r\nd") == "a b c\nd"


def test_chunk_document_respects_size_and_overlap():
    # 200 sentences ~= 1000 chars -> must exceed 5 chunks at size 100
    text = "。".join(["句子{}".format(i) for i in range(200)])
    doc = Document(doc_id="d1", text=text)
    chunks = chunk_document(doc, chunk_size=100, overlap=20)
    assert len(chunks) > 5
    for c in chunks:
        assert len(c.text) <= 100
        assert c.doc_id == "d1"
    # chunk ids are unique
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_document_empty_text():
    assert chunk_document(Document(doc_id="x", text="")) == []


def test_chunk_document_invalid_params():
    doc = Document(doc_id="x", text="abc")
    for size, ov in ((0, 0), (100, 100), (100, 150)):
        try:
            chunk_document(doc, size, ov)
            assert False, "should raise"
        except ValueError:
            pass


def test_parse_txt_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("你好，世界", encoding="utf-8")
    doc = parse_file(str(p))
    assert "你好" in doc.text
    assert doc.metadata["source"] == "a.txt"


def test_parse_unsupported_extension(tmp_path):
    p = tmp_path / "a.exe"
    p.write_text("x", encoding="utf-8")
    try:
        parse_file(str(p))
        assert False, "should raise"
    except ValueError:
        pass


def test_parse_pdf(tmp_path):
    pypdf = pytest_importorskip()
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    p = tmp_path / "a.pdf"
    with open(p, "wb") as f:
        writer.write(f)
    doc = parse_file(str(p))
    assert doc.doc_id  # parses without error


def pytest_importorskip():
    import pytest

    return pytest.importorskip("pypdf")

from __future__ import annotations

import io
import re
from typing import List, Dict, Any, Tuple

from pypdf import PdfReader

from app.integrations.chroma import upsert_documents


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrai texto de um PDF (todo o documento)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        chunks.append(txt)
    return "\n\n".join(chunks).strip()


def _split_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in paras if p and p.strip()]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[str]:
    """
    Chunking simples: junta sentenças até ~chunk_size, com overlap entre chunks.
    """
    sentences = _split_sentences(text)
    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0

    for s in sentences:
        s_len = len(s)
        if cur_len + s_len + 1 <= chunk_size:
            buf.append(s)
            cur_len += s_len + 1
        else:
            if buf:
                chunks.append(" ".join(buf).strip())
                if overlap > 0:
                    carry = " ".join(buf)[-overlap:]
                    buf = [carry, s]
                    cur_len = len(carry) + 1 + s_len
                else:
                    buf = [s]
                    cur_len = s_len
            else:
                chunks.append(s[:chunk_size])
                buf = []
                cur_len = 0

    if buf:
        chunks.append(" ".join(buf).strip())

    clean = []
    seen = set()
    for c in chunks:
        cc = re.sub(r"\s{2,}", " ", c).strip()
        if cc and cc not in seen:
            seen.add(cc)
            clean.append(cc)
    return clean


def ingest_text(
    collection: str,
    text: str,
    source: str = "manual",
    chunk_size: int = 800,
    overlap: int = 150,
) -> Dict[str, Any]:
    """Quebra texto e upsert no Chroma com metadados básicos."""
    if not text or not text.strip():
        return {"collection": collection, "added": 0, "ids": [], "count_after": None}

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    metas = [{"source": source, "i": i} for i in range(len(chunks))]
    ids = [f"{source}-{i:06d}" for i in range(len(chunks))]
    upsert_documents(collection, chunks, metas, ids)
    return {"collection": collection, "added": len(ids), "ids": ids}


def ingest_pdf_bytes(
    collection: str,
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> Dict[str, Any]:
    """Extrai texto do PDF, quebra e upsert."""
    text = extract_text_from_pdf(file_bytes)
    res = ingest_text(
        collection=collection,
        text=text,
        source=filename or "pdf",
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return res

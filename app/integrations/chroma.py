from __future__ import annotations

import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("CHROMA_TELEMETRY", "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings
from app.integrations.embeddings import embed_texts


@lru_cache(maxsize=1)
def get_chroma() -> chromadb.PersistentClient:
    """Retorna um cliente persistente do Chroma, criando o diretório se necessário."""
    persist_dir = Path(settings.CHROMA_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def list_collection_names() -> List[str]:
    """Lista os nomes das coleções existentes."""
    client = get_chroma()
    return [c.name for c in client.list_collections()]


def get_or_create_collection(name: str) -> Collection:
    """Obtém (ou cria) uma coleção persistente pelo nome."""
    client = get_chroma()
    return client.get_or_create_collection(name=name)


def collection_count(name: str) -> int:
    """Conta os itens de uma coleção."""
    return get_or_create_collection(name).count()


def add_documents(
    name: str,
    texts: List[str],
    metadatas: Optional[List[Dict]] = None,
    ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Adiciona documentos com embeddings (falha se IDs já existirem).
    Use `upsert_documents` para atualizar/criar.
    """
    col = get_or_create_collection(name)
    vecs = embed_texts(texts, mode="passage")

    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]
    if metadatas is None:
        metadatas = [{} for _ in texts]

    col.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vecs)
    return ids


def upsert_documents(
    name: str,
    texts: List[str],
    metadatas: Optional[List[Dict]] = None,
    ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Cria/atualiza documentos com embeddings mantendo IDs estáveis (upsert).
    Útil para semear regras/políticas com IDs fixos (ex.: POL-1, POL-2...).
    """
    col = get_or_create_collection(name)
    vecs = embed_texts(texts, mode="passage")

    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]
    if metadatas is None:
        metadatas = [{} for _ in texts]

    col.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=vecs)
    return ids


def query_collection(name: str, query: str, top_k: int = 3) -> Dict[str, List]:
    """
    Consulta a coleção usando o embedding da consulta.
    Retorna sempre listas (podem estar vazias) de ids/documents/metadatas/distances.
    Observação: em Chroma 0.5.x, 'ids' não deve ser passado em `include`.
    """
    col = get_or_create_collection(name)
    qvec = embed_texts([query], mode="query")[0]

    res = col.query(
        query_embeddings=[qvec],
        n_results=max(1, int(top_k)),
        include=["documents", "metadatas", "distances"], 
    )

    return {
        "ids": (res.get("ids") or [[]])[0] or [],
        "documents": (res.get("documents") or [[]])[0] or [],
        "metadatas": (res.get("metadatas") or [[]])[0] or [],
        "distances": (res.get("distances") or [[]])[0] or [],
    }

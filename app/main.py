from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from app.core.config import settings
from app.data.policy import POLICY_RULES
from app.integrations.chroma import (
    list_collection_names,
    get_or_create_collection,
    collection_count,
    add_documents,
    query_collection,
    upsert_documents,
)
from app.integrations.generator import generate_answer
from app.services.verify import apply_rules, build_rationale, policy_sources


class CreateCollectionIn(BaseModel):
    name: str


class AddDocsIn(BaseModel):
    texts: List[str] = Field(min_length=1)
    metadatas: Optional[List[Dict[str, Any]]] = None
    ids: Optional[List[str]] = None


class QueryIn(BaseModel):
    text: str
    top_k: int = 3


class RagAskIn(BaseModel):
    collection: str
    question: str
    top_k: int = 3


class VerifyIn(BaseModel):
    natureza: str
    valor_condenacao: Optional[float] = None
    transitado_em_julgado: Optional[bool] = None
    fase: Optional[str] = None
    docs: Optional[Dict[str, Any]] = None


def create_app() -> FastAPI:
    application = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

    @application.get("/health")
    def health():
        return {"status": "ok", "version": settings.APP_VERSION}

    @application.get("/chroma/health")
    def chroma_health():
        return {
            "status": "ok",
            "persist_dir": settings.CHROMA_DIR,
            "collections": list_collection_names(),
        }

    @application.get("/chroma/collections")
    def list_collections():
        return {"collections": list_collection_names()}

    @application.post("/chroma/collections")
    def create_collection(body: CreateCollectionIn):
        col = get_or_create_collection(body.name)
        return {"created": col.name}

    @application.get("/chroma/collections/{name}")
    def get_collection(name: str):
        return {"name": name, "count": collection_count(name)}

    @application.post("/chroma/collections/{name}/add")
    def add_docs(name: str, body: AddDocsIn):
        ids = add_documents(name, body.texts, body.metadatas, body.ids)
        return {"added": len(ids), "ids": ids, "count_after": collection_count(name)}

    @application.post("/chroma/collections/{name}/query")
    def query_docs(name: str, body: QueryIn):
        return query_collection(name, body.text, body.top_k)

    @application.post("/policy/seed")
    def policy_seed():
        texts = [r["text"] for r in POLICY_RULES]
        ids = [r["id"] for r in POLICY_RULES]
        metas = [{"rule_id": r["id"]} for r in POLICY_RULES]
        upsert_documents("policy", texts, metas, ids)  
        col = get_or_create_collection("policy")
        return {
            "collection": "policy",
            "added_or_updated": len(ids),
            "count": col.count(),
        }

    @application.post("/policy/query")
    def policy_query(body: QueryIn):
        return query_collection("policy", body.text, body.top_k)

    @application.post("/rag/ask")
    def rag_ask(body: RagAskIn):
        res = query_collection(body.collection, body.question, body.top_k)
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        dists = res.get("distances", [])
        answer = generate_answer(docs, body.question)
        sources = []
        for i, txt in enumerate(docs):
            sources.append(
                {
                    "text": txt[:240] + ("..." if len(txt) > 240 else ""),
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return {"answer": answer, "sources": sources}

    @application.post("/verify")
    def verify(body: VerifyIn):
        payload = body.model_dump()
        decision, citations, reasons = apply_rules(payload)
        rationale = build_rationale(decision, citations)
        return {
            "decision": decision,            
            "citations": citations,         
            "reasons": reasons,              
            "rationale": rationale,          
            "sources": policy_sources(citations),  
        }

    return application


app = create_app()

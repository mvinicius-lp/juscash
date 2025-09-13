from functools import lru_cache
from typing import List
from fastapi import HTTPException
from app.core.config import settings

from huggingface_hub import InferenceClient
from huggingface_hub.utils._errors import HfHubHTTPError


from sentence_transformers import SentenceTransformer

def _prefix(texts: List[str], mode: str) -> List[str]:
    if "e5" in settings.EMBEDDING_MODEL.lower():
        p = "query: " if mode == "query" else "passage: "
        return [p + (t or "").strip() for t in texts]
    return [(t or "").strip() for t in texts]

@lru_cache(maxsize=1)
def _st_model() -> SentenceTransformer:
    trust = "e5" in settings.EMBEDDING_MODEL.lower()
    return SentenceTransformer(settings.EMBEDDING_MODEL, trust_remote_code=trust)

def _embed_local(texts: List[str], mode: str) -> List[List[float]]:
    model = _st_model()
    prepared = _prefix(texts, mode)
    vecs = model.encode(prepared, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.tolist()

@lru_cache(maxsize=1)
def _hf_client() -> InferenceClient:
    token = (settings.HF_TOKEN or "").strip() or None
    return InferenceClient(model=settings.EMBEDDING_MODEL, token=token, timeout=30.0)

def _embed_hf(texts: List[str], mode: str) -> List[List[float]]:
    client = _hf_client()
    prepared = _prefix(texts, mode)
    try:
        out = client.feature_extraction(prepared)
    except HfHubHTTPError as e:
        code = e.response.status_code if e.response is not None else 500
        msg = (f"Hugging Face Inference API negou acesso ({code}). "
               "Use EMBEDDINGS_BACKEND=local no .env ou configure um HF_TOKEN com permissão.")
        raise HTTPException(status_code=code, detail=msg)
    if isinstance(out, list) and out and isinstance(out[0], float):
        return [out]
    return out

def embed_texts(texts: List[str], mode: str = "passage") -> List[List[float]]:
    backend = (settings.EMBEDDINGS_BACKEND or "local").lower()
    if backend == "hf":
        return _embed_hf(texts, mode)
    return _embed_local(texts, mode)

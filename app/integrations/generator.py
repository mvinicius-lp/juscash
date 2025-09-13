from functools import lru_cache
from typing import List
from app.core.config import settings
from fastapi import HTTPException

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
import torch
import re

from huggingface_hub import InferenceClient
from huggingface_hub.utils._errors import HfHubHTTPError


SYSTEM_PROMPT_PT = (
    "Você responde em português, de forma concisa e objetiva.\n"
    "Use SOMENTE o CONTEXTO fornecido. Se a resposta não estiver no contexto, diga: "
    "'Não encontrei no contexto'. NÃO repita a pergunta. Responda em 1–2 frases."
)

def _is_seq2seq_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["t5", "bart", "mbart", "pegasus"])

def _build_inputs(context: str, question: str, is_seq2seq: bool) -> str:
    if is_seq2seq:
        return (
            f"Contexto:\n{context}\n\n"
            f"Pergunta: {question}\n"
            f"Responda em 1–2 frases, usando apenas o contexto acima:"
        )
    return (
        f"{SYSTEM_PROMPT_PT}\n\n"
        f"### CONTEXTO\n{context}\n\n"
        f"### PERGUNTA\n{question}\n\n"
        f"### RESPOSTA\n"
    )

def _first_sentence(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    for p in parts:
        s = p.strip()
        if len(s) >= 15:
            return s
    return t[:240].strip()

def _looks_like_echo(ans: str) -> bool:
    a = (ans or "").lower()
    if not a:
        return True
    patterns = [
        "você responde em português",
        "use somente o contexto",
        "responda em 1–2 frases",
        "### contexto",
        "### resposta",
        "resposta:",
        "answer:",
        "contexto:",   
        "pergunta:",   
    ]
    return any(p in a for p in patterns)

def _strip_prompt_labels(text: str) -> str:
    """Remove linhas de rótulos como Contexto:/Pergunta:/Resposta: em qualquer lugar."""
    if not text:
        return text
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        ln_stripped = ln.strip()
        if re.match(r'^(contexto|pergunta|resposta|###\s*resposta|###\s*contexto)\s*:?', ln_stripped, flags=re.I):
            continue
        if re.search(r"responda\s+em\s+1.?–.?2\s+frases", ln_stripped, flags=re.I):
            continue
        if re.search(r"você responde em português", ln_stripped, flags=re.I):
            continue
        cleaned.append(ln)
    out = "\n".join(cleaned).strip()
    out = re.sub(r"(Contexto|Pergunta|Resposta)\s*:\s*", "", out, flags=re.I)
    return out.strip()

def _clean_answer(question: str, prompt: str, decoded: str) -> str:
    text = decoded or ""
    if prompt and text.startswith(prompt):
        text = text[len(prompt):]

    for sep in ["\n###", "\n\n###", "\n\n", "\nRESPOSTA:", "\nResposta:", "\n#"]:
        if sep in text:
            text = text.split(sep, 1)[0]

    text = _strip_prompt_labels(text)

    lower_q = question.strip().lower()
    for _ in range(3):
        t = text.lstrip()
        for p in ["resposta:", "### resposta", "###resposta", "answer:", "### answer"]:
            if t.lower().startswith(p):
                t = t[len(p):].lstrip()
        if t.lower().startswith(lower_q):
            t = t[len(question):].lstrip(": .-").lstrip()
        text = t

    text = text.strip(' "\'')

    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\bR{2,}AG\b", "RAG", text, flags=re.I)

    return text.strip()

@lru_cache(maxsize=1)
def _local_t2t_models():
    model_name = settings.LLM_MODEL
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl

def _generate_local_t2t(prompt: str, question: str) -> str:
    tok, mdl = _local_t2t_models()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        gen = mdl.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=4,
            length_penalty=0.9,
            no_repeat_ngram_size=4,
            early_stopping=True,
            eos_token_id=tok.eos_token_id,
        )
    out = tok.decode(gen[0], skip_special_tokens=True)
    out = _clean_answer(question, prompt, out)
    return out


@lru_cache(maxsize=1)
def _local_causal_models():
    model_name = settings.LLM_MODEL
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    mdl.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok, mdl

def _generate_local_causal(prompt: str, question: str) -> str:
    tok, mdl = _local_causal_models()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        gen_ids = mdl.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    decoded = tok.decode(gen_ids[0], skip_special_tokens=True)
    out = _clean_answer(question, prompt, decoded)
    return out


@lru_cache(maxsize=1)
def _hf_client() -> InferenceClient:
    token = (settings.HF_TOKEN or "").strip() or None
    return InferenceClient(model=settings.LLM_MODEL, token=token, timeout=60.0)

def _generate_hf(prompt: str, question: str) -> str:
    try:
        client = _hf_client()
        out = client.text_generation(prompt, max_new_tokens=128, do_sample=False)
        out = out or ""
        return _clean_answer(question, prompt, out)
    except HfHubHTTPError as e:
        code = e.response.status_code if e.response is not None else 500
        raise HTTPException(
            status_code=code,
            detail="Falha na Inference API da HF para geração. "
                   "Use LLM_BACKEND=local ou configure HF_TOKEN."
        )


def generate_answer(context_chunks: List[str], question: str) -> str:
    chunks = [c.strip() for c in context_chunks if c and c.strip()]
    context = "\n\n---\n\n".join(chunks[:5])

    is_seq2seq = _is_seq2seq_name(settings.LLM_MODEL)
    prompt = _build_inputs(context, question, is_seq2seq)
    backend = (settings.LLM_BACKEND or "local").lower()

    if backend == "hf":
        ans = _generate_hf(prompt, question)
    else:
        ans = _generate_local_t2t(prompt, question) if is_seq2seq else _generate_local_causal(prompt, question)

    if _looks_like_echo(ans) or len(ans) < 5:
        fallback = _first_sentence(chunks[0] if chunks else "")
        return fallback or "Não encontrei no contexto."

    return ans

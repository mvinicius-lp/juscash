from typing import Any, Dict, List, Tuple
from app.data.policy import POLICY_RULES
from app.integrations.generator import generate_answer

RULE_MAP = {r["id"]: r["text"] for r in POLICY_RULES}


def _dedupe(seq: List[str]) -> List[str]:
    """Remove duplicatas mantendo a ordem."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def apply_rules(payload: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    """
    Aplica regras determinísticas e retorna:
      (decision, citations, reasons)
    - decision ∈ {"approved", "rejected", "incomplete"}
    - citations: lista de POL-* citadas
    - reasons: razões determinísticas (strings curtas)
    """
    citations: List[str] = []
    reasons: List[str] = []

    natureza = (payload.get("natureza") or "").strip().lower()
    valor = payload.get("valor_condenacao")
    transitado = payload.get("transitado_em_julgado")  
    fase = (payload.get("fase") or "").strip().lower()
    docs = payload.get("docs") or {}
    tem_comprovante_transito = bool(docs.get("comprovante_transito", False))

    if "trabalh" in natureza:
        citations.append("POL-4")
        reasons.append("Crédito de natureza trabalhista.")

    try:
        if valor is not None and float(valor) < 1000:
            citations.append("POL-3")
            reasons.append("Valor de condenação inferior a R$ 1.000,00.")
    except Exception:
        citations.append("POL-8")
        reasons.append("Valor de condenação inválido ou ausente.")

    if transitado is False:
        citations.append("POL-1")
        reasons.append("Processo não transitado em julgado.")
    elif transitado is None and not tem_comprovante_transito:
        citations.append("POL-8")
        reasons.append("Falta comprovação do trânsito em julgado (documento essencial).")

    if fase:
        if "execu" not in fase:  
            citations.append("POL-1")
            reasons.append("Caso não está em fase de execução.")
    else:
        citations.append("POL-8")
        reasons.append("Fase processual não informada.")

    citations = _dedupe(citations)

    decision = "approved"
    if ("POL-4" in citations) or ("POL-3" in citations) or (transitado is False) or (
        "POL-1" in citations and "POL-8" not in citations
    ):
        decision = "rejected"
    elif "POL-8" in citations:
        decision = "incomplete"

    return decision, citations, reasons


def build_rationale(decision: str, citations: List[str]) -> str:
    """
    Texto curto de justificativa:
      - Para "approved": determinístico e objetivo (sem LLM).
      - Para "rejected"/"incomplete": usa RAG com as regras citadas como contexto.
    """
    if decision == "approved":
        return (
            "Aprovado: atende às regras — trânsito em julgado comprovado, "
            "fase de execução e valor mínimo, sem impedimentos (ex.: trabalhista)."
        )

    chunks = [RULE_MAP[c] for c in citations if c in RULE_MAP]
    if not chunks:
        chunks = ["Avaliar conforme as políticas internas aplicáveis."]  
    question = (
        "Decisão: {d}. Explique em 1–2 frases o porquê, citando os códigos das regras (ex.: POL-3, POL-4)."
    ).format(d=decision)
    return generate_answer(chunks, question)


def policy_sources(citations: List[str]) -> List[Dict[str, str]]:
    """Retorna pares {id, text} das regras citadas para transparência no /verify."""
    return [{"id": c, "text": RULE_MAP.get(c, "")} for c in citations]

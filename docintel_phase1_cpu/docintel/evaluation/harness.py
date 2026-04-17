"""
Evaluation harness.

Ships on day 1 for one reason: a system without evals silently regresses
every time you change a prompt, a model, or a chunking rule. Hiring
managers at serious AI companies can spot a system with real evals from
fifty feet away. Ship the harness before the features.

Eval set shape:
  A list of EvalCase(query, gold_evidence_block_ids, gold_answer_keywords).
  You hand-label ~200 to start. See scripts/label_eval_set.py for a tiny
  labeling UI we can write later.

Metrics:
  * Retrieval  — Recall@k, MRR, nDCG@k         (on gold_evidence_block_ids)
  * Grounding  — citation validity, citation coverage
  * Answer     — keyword coverage (cheap), LLM-judge faithfulness (Phase 3)

Run with:  python -m evaluation.harness --eval data/eval/set.jsonl
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.schema import Answer, EvidenceItem

log = logging.getLogger(__name__)


@dataclass
class EvalCase:
    query_id: str
    query: str
    tenant_id: str
    user_ids: list[str] = field(default_factory=list)
    gold_evidence_block_ids: list[str] = field(default_factory=list)   # e.g. ["p7_b2"]
    gold_answer_keywords: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class CaseResult:
    query_id: str
    retrieved_block_ids: list[str]
    answer: Answer
    # Computed metrics per case
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    citation_validity: float = 0.0
    citation_coverage: float = 0.0
    keyword_coverage: float = 0.0


def load_eval_set(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(EvalCase(**data))
    return cases


# ---------- metrics ----------

def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    hits = sum(1 for g in gold if g in top)
    return hits / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, b in enumerate(retrieved):
        if b in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    dcg = 0.0
    for i, b in enumerate(retrieved[:k]):
        if b in gold_set:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def citation_metrics(answer: Answer) -> tuple[float, float]:
    """
    citation_validity: fraction of [e<N>] markers that point to an existing
                       evidence item (catches hallucinated citations).
    citation_coverage: fraction of non-empty sentences that contain at least
                       one citation.
    """
    import re
    valid_ids = {e.evidence_id for e in answer.evidence}
    markers = re.findall(r"\[e(\d+)(?:\s*,\s*e?(\d+))*\]", answer.answer_text)
    total = 0
    valid = 0
    for groups in markers:
        for g in groups:
            if g:
                total += 1
                if f"e{g}" in valid_ids:
                    valid += 1
    validity = valid / total if total else 1.0  # no markers = vacuously valid

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.answer_text) if s.strip()]
    if not sentences:
        return validity, 0.0
    cited_sentences = sum(1 for s in sentences if re.search(r"\[e\d+", s))
    coverage = cited_sentences / len(sentences)
    return validity, coverage


def keyword_coverage(answer_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    text = answer_text.lower()
    hits = sum(1 for k in keywords if k.lower() in text)
    return hits / len(keywords)


# ---------- runner ----------

def run_eval(
    cases: list[EvalCase],
    ask_fn: Callable[[EvalCase], Answer],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """
    Execute every case, compute per-case and aggregate metrics, and return
    a dict you can diff across runs.
    """
    per_case: list[CaseResult] = []
    for case in cases:
        answer = ask_fn(case)
        retrieved = [e.block_id for e in answer.evidence]

        r: CaseResult = CaseResult(
            query_id=case.query_id,
            retrieved_block_ids=retrieved,
            answer=answer,
        )
        for k in ks:
            r.recall_at_k[k] = recall_at_k(retrieved, case.gold_evidence_block_ids, k)
            r.ndcg_at_k[k] = ndcg_at_k(retrieved, case.gold_evidence_block_ids, k)
        r.mrr = mrr(retrieved, case.gold_evidence_block_ids)
        r.citation_validity, r.citation_coverage = citation_metrics(answer)
        r.keyword_coverage = keyword_coverage(answer.answer_text, case.gold_answer_keywords)
        per_case.append(r)

    n = len(per_case) or 1

    def _mean(fn):
        return sum(fn(c) for c in per_case) / n

    summary = {
        "n_cases": len(per_case),
        "mrr": _mean(lambda c: c.mrr),
        "citation_validity": _mean(lambda c: c.citation_validity),
        "citation_coverage": _mean(lambda c: c.citation_coverage),
        "keyword_coverage": _mean(lambda c: c.keyword_coverage),
    }
    for k in ks:
        summary[f"recall@{k}"] = _mean(lambda c, k=k: c.recall_at_k[k])
        summary[f"ndcg@{k}"] = _mean(lambda c, k=k: c.ndcg_at_k[k])

    return {
        "summary": summary,
        "cases": [c.__dict__ for c in per_case],
    }

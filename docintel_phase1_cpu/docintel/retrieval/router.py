"""
Query router.

Decides, for a given query: the intent, the filters to apply, and the
per-retriever weights for RRF fusion.

Phase 1: rules + heuristics. Cheap, deterministic, and good enough to
beat a fixed-weight baseline. Phase 3 upgrades this to a small fine-tuned
classifier or a Haiku-class LLM, which is the point where you'll start
to see real lift on ambiguous queries.
"""
from __future__ import annotations

import re
from app.schema import QueryIntent, QueryPlan


_VISUAL_PATTERNS = [
    r"\bshow\b",
    r"\bchart\b",
    r"\bgraph\b",
    r"\bdiagram\b",
    r"\bfigure\b",
    r"\bimage\b",
    r"\bpicture\b",
    r"\btable\b",
    r"\billustration\b",
    r"\bphoto\b",
    r"\bvisual\b",
    r"\bplot\b",
]
_ANALYTICAL_PATTERNS = [
    r"\bcompar\w+\b",
    r"\btrend\b",
    r"\bchange\w*\b",
    r"\bgrowth\b",
    r"\byoy\b",
    r"\bquarter over quarter\b",
    r"\bwhy\b",
    r"\bhow did\b",
]


def _matches(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def plan_query(
    query: str,
    *,
    tenant_id: str,
    user_ids: list[str] | None = None,
    visual_available: bool = False,
) -> QueryPlan:
    """
    Build a QueryPlan. `visual_available` flips to True once the ColPali
    index is online (Phase 2).
    """
    is_visual = _matches(query, _VISUAL_PATTERNS)
    is_analytical = _matches(query, _ANALYTICAL_PATTERNS)

    if is_visual:
        intent = QueryIntent.VISUAL
    elif is_analytical:
        intent = QueryIntent.ANALYTICAL
    else:
        intent = QueryIntent.FACTUAL

    # Base weights — tune with your eval set, not with vibes.
    weights: dict[str, float] = {"bm25": 0.4, "dense": 0.6, "visual": 0.0}
    if is_visual and visual_available:
        weights = {"bm25": 0.2, "dense": 0.3, "visual": 0.5}
    elif intent == QueryIntent.FACTUAL:
        # Factual queries (entity names, numbers) lean on BM25 more.
        weights = {"bm25": 0.55, "dense": 0.45, "visual": 0.0}

    filters: dict = {"tenant_id": tenant_id}
    if user_ids:
        filters["user_ids"] = user_ids

    return QueryPlan(
        query=query,
        intent=intent,
        filters=filters,
        weights=weights,
        top_k_per_index=50,
        final_k=8,
    )

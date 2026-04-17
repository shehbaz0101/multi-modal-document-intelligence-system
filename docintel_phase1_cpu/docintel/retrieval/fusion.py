"""
Reciprocal Rank Fusion — combine ranked lists from multiple retrievers.

RRF is the standard hybrid-retrieval trick. It's parameter-free (the one
constant `k` is almost always 60 and doesn't need tuning), ignores score
scales (so BM25's raw scores and cosine similarity don't need to be
normalized), and empirically matches or beats learned fusion schemes on
most corpora.

    rrf_score(d) = sum over retrievers r of  w_r / (k + rank_r(d))

Weights let the query router bias toward certain indexes — e.g. push the
visual retriever up for "show me the chart" queries.
"""
from __future__ import annotations

from collections import defaultdict
from indexing.base import IndexHit


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[IndexHit]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[IndexHit]:
    """
    Fuse ranked lists (one per retriever). Returns a single list sorted by
    fused RRF score, descending. Each returned IndexHit keeps a
    `retrieval_scores` dict in its payload that carries the raw per-retriever
    scores for downstream observability.
    """
    weights = weights or {}
    # Key hits by (doc_id, block_id) — the canonical identity of a block.
    accum: dict[tuple[str, str], dict] = {}

    for source, hits in ranked_lists.items():
        w = weights.get(source, 1.0)
        for rank, hit in enumerate(hits):
            key = (hit.doc_id, hit.block_id)
            entry = accum.setdefault(
                key,
                {
                    "hit": hit,            # we keep the first hit's payload
                    "fused_score": 0.0,
                    "raw_scores": {},
                    "sources": set(),
                },
            )
            entry["fused_score"] += w / (k + rank + 1)  # rank is 0-indexed
            entry["raw_scores"][source] = hit.score
            entry["sources"].add(source)

    fused: list[IndexHit] = []
    for entry in accum.values():
        hit = entry["hit"]
        # Annotate the hit with fusion bookkeeping — the retrieval layer reads this.
        hit.score = entry["fused_score"]
        hit.payload = dict(hit.payload)  # don't mutate the original
        hit.payload["_raw_scores"] = entry["raw_scores"]
        hit.payload["_fusion_sources"] = sorted(entry["sources"])
        fused.append(hit)

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused

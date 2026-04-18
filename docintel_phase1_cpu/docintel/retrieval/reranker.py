"""
Cross-encoder reranker — Phase 3.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  ~65 MB, CPU-fine, ~200ms for top-30 on an i5.

Enable in .env:
  RETRIEVAL_RERANK_ENABLED=true
  RETRIEVAL_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
"""
from __future__ import annotations

import logging
from indexing.base import IndexHit

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        log.info("loading reranker: %s", model_name)
        self._model = CrossEncoder(model_name)
        log.info("reranker ready")

    def rerank(self, query: str, hits: list[IndexHit], top_k: int) -> list[IndexHit]:
        if not hits:
            return hits

        pairs = [(query, hit.payload.get("content", "") or "") for hit in hits]
        scores = self._model.predict(pairs)

        for hit, score in zip(hits, scores):
            hit.payload["_rerank_score"] = float(score)
            hit.score = float(score)

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
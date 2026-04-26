"""
Layer 5 — Retrieval orchestrator.

Four-stage funnel:
  1. Router     — QueryPlan (intent, filters, weights)
  2. Retrieve   — fire all enabled indexes in parallel
  3. Fuse       — RRF across their ranked lists
  4. Rerank     — cross-encoder or visual reranker on the top-N

Phase 2 adds the visual_reranker slot. It runs only when the query is
flagged as visual-intent (see router) and the reranker is configured.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

from app.schema import BBox, EvidenceItem, Modality, QueryPlan, QueryIntent
from embedding.text import TextEmbedder
from indexing.base import BlockIndex, IndexHit
from retrieval.fusion import reciprocal_rank_fusion

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[IndexHit], top_k: int) -> list[IndexHit]: ...


class VisualReranker(Protocol):
    def rerank(self, query: str, hits: list[IndexHit], top_k: int) -> list[IndexHit]: ...


class Retriever:
    def __init__(
        self,
        *,
        text_embedder: TextEmbedder,
        sparse_index: BlockIndex | None,
        dense_index: BlockIndex | None,
        visual_index: BlockIndex | None = None,
        reranker: Reranker | None = None,
        visual_reranker: VisualReranker | None = None,   # Phase 2
    ):
        self.text_embedder = text_embedder
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.visual_index = visual_index
        self.reranker = reranker
        self.visual_reranker = visual_reranker

    def retrieve(self, plan: QueryPlan) -> tuple[list[EvidenceItem], dict[str, int]]:
        timings: dict[str, int] = {}

        # ---- Stage 2: parallel retrieval ----
        t0 = time.perf_counter()
        qvec: np.ndarray | None = None
        if self.dense_index is not None and plan.weights.get("dense", 0) > 0:
            qvec = self.text_embedder.embed([plan.query])[0]

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures: dict[str, object] = {}
            if self.sparse_index and plan.weights.get("bm25", 0) > 0:
                futures["bm25"] = pool.submit(
                    self.sparse_index.search,
                    query_text=plan.query,
                    filters=plan.filters,
                    top_k=plan.top_k_per_index,
                )
            if self.dense_index and plan.weights.get("dense", 0) > 0 and qvec is not None:
                futures["dense"] = pool.submit(
                    self.dense_index.search,
                    query_vector=qvec,
                    filters=plan.filters,
                    top_k=plan.top_k_per_index,
                )
            if self.visual_index and plan.weights.get("visual", 0) > 0:
                futures["visual"] = pool.submit(
                    self.visual_index.search,
                    query_text=plan.query,
                    filters=plan.filters,
                    top_k=plan.top_k_per_index,
                )
            ranked_lists: dict[str, list[IndexHit]] = {}
            for name, f in futures.items():
                try:
                    ranked_lists[name] = f.result()
                except Exception as e:
                    log.warning("retriever failed index=%s err=%s", name, e)
                    ranked_lists[name] = []
        timings["retrieve_ms"] = int((time.perf_counter() - t0) * 1000)

        # ---- Stage 3: fusion ----
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(ranked_lists, weights=plan.weights)
        timings["fuse_ms"] = int((time.perf_counter() - t0) * 1000)

        # ---- Stage 4a: text rerank ----
        t0 = time.perf_counter()
        if self.reranker and fused:
            rerank_pool = fused[: max(plan.final_k * 4, 30)]
            reranked = self.reranker.rerank(plan.query, rerank_pool, top_k=plan.final_k * 3)
        else:
            reranked = fused[: plan.final_k * 3]
        timings["rerank_ms"] = int((time.perf_counter() - t0) * 1000)

        # ---- Stage 4b: visual rerank (Phase 2) ----
        t0 = time.perf_counter()
        if (
            self.visual_reranker
            and plan.intent == QueryIntent.VISUAL
            and reranked
        ):
            log.info("visual rerank triggered for query: '%s'", plan.query[:60])
            reranked = self.visual_reranker.rerank(
                plan.query, reranked, top_k=plan.final_k
            )
        else:
            reranked = reranked[: plan.final_k]
        timings["visual_rerank_ms"] = int((time.perf_counter() - t0) * 1000)

        # ---- Materialize evidence pack ----
        evidence = [self._hit_to_evidence(h, i) for i, h in enumerate(reranked)]
        return evidence, timings

    @staticmethod
    def _hit_to_evidence(hit: IndexHit, i: int) -> EvidenceItem:
        p = hit.payload or {}
        bbox_data = p.get("bbox") or {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
        return EvidenceItem(
            evidence_id=f"e{i}",
            doc_id=hit.doc_id,
            page_number=hit.page_number,
            block_id=hit.block_id,
            modality=Modality(p.get("modality", "text")),
            content=p.get("content", ""),
            page_image_uri=p.get("page_image_uri", ""),
            bbox=BBox(**bbox_data),
            retrieval_scores=p.get("_raw_scores", {}),
            rerank_score=p.get("_rerank_score"),
            source_index=",".join(p.get("_fusion_sources", [hit.source_index])),
            doc_title=p.get("doc_title"),
        )
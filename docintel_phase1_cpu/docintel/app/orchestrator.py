"""
Orchestrator — the public API.

Two methods:
  intel.ingest(file_path, tenant_id, ...)  -> DocumentMeta
  intel.ask(query, tenant_id, ...)         -> Answer
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from app.schema import Answer, DocumentMeta
from app.storage import Storage, build_storage
from config.settings import AppSettings, get_settings
from embedding.text import build_text_embedder, block_to_embedding_text
from generation.generator import GroundedGenerator
from indexing.base import BlockIndex
from indexing.dense_qdrant import QdrantDenseIndex
from indexing.sparse_opensearch import OpenSearchSparseIndex
from ingestion.ingest import ingest_file
from parsing.base import build_parser
from parsing.captioner import Captioner, GeminiVLMClient
from retrieval.retriever import Retriever
from retrieval.router import plan_query

log = logging.getLogger(__name__)


class DocIntel:
    def __init__(self, settings: AppSettings | None = None):
        self.s = settings or get_settings()

        # Gemini API key — reads GEMINI_API_KEY from env if not set in config
        gemini_key = self.s.generation.api_key or os.environ.get("GEMINI_API_KEY")

        self.storage: Storage = build_storage(
            self.s.storage.backend, self.s.storage.root
        )
        self.parser = build_parser(
            self.s.parser.primary, storage=self.storage
        ) if self.s.parser.primary == "cpu" else build_parser(self.s.parser.primary)

        self.text_embedder = build_text_embedder(self.s.embedding.text_model)

        self.captioner = Captioner(
            vlm=GeminiVLMClient(
                model=self.s.embedding.caption_model,
                api_key=gemini_key,
                max_tokens=self.s.embedding.caption_max_tokens,
            ),
            storage=self.storage,
        )

        self.sparse_index: BlockIndex = OpenSearchSparseIndex(
            url=self.s.index.sparse_url,
            index=self.s.index.sparse_index,
        )
        self.dense_index: BlockIndex = QdrantDenseIndex(
            url=self.s.index.vector_url,
            collection=self.s.index.text_collection,
            dim=self.s.embedding.text_dim,
            api_key=self.s.index.vector_api_key,
        )

        self.retriever = Retriever(
            text_embedder=self.text_embedder,
            sparse_index=self.sparse_index,
            dense_index=self.dense_index,
            visual_index=None,   # Phase 2
            reranker=None,        # Phase 3
        )

        self.generator = GroundedGenerator(
            model=self.s.generation.model,
            storage=self.storage,
            api_key=gemini_key,
            max_tokens=self.s.generation.max_tokens,
            temperature=self.s.generation.temperature,
        )

    # ---------- ingest ----------

    def ingest(
        self,
        file_path: str | Path,
        *,
        tenant_id: str,
        uploader_id: str | None = None,
        acl_read: list[str] | None = None,
        doc_type: str | None = None,
    ) -> DocumentMeta:
        t0 = time.perf_counter()

        result = ingest_file(
            file_path,
            tenant_id=tenant_id,
            storage=self.storage,
            dpi=self.s.storage.page_dpi,
            uploader_id=uploader_id,
            acl_read=acl_read,
            doc_type=doc_type,
        )
        log.info("ingested doc_id=%s pages=%d", result.meta.doc_id[:12], len(result.pages))

        self.parser.parse(result.pages)
        self.captioner.caption_pages(result.pages)

        all_blocks = [b for p in result.pages for b in p.blocks]
        texts = [block_to_embedding_text(b) for b in all_blocks]
        mask = [bool(t) for t in texts]
        embed_blocks = [b for b, k in zip(all_blocks, mask) if k]
        embed_texts  = [t for t, k in zip(texts, mask) if k]

        embeddings = self.text_embedder.embed(embed_texts) if embed_texts else None

        self.sparse_index.upsert_blocks(all_blocks, result.meta)
        if embed_blocks and embeddings is not None:
            self.dense_index.upsert_blocks(embed_blocks, result.meta, embeddings=embeddings)

        log.info("indexed doc_id=%s ms=%d", result.meta.doc_id[:12],
                 int((time.perf_counter() - t0) * 1000))
        return result.meta

    # ---------- ask ----------

    def ask(
        self,
        query: str,
        *,
        tenant_id: str,
        user_ids: list[str] | None = None,
    ) -> Answer:
        t0 = time.perf_counter()
        plan = plan_query(
            query,
            tenant_id=tenant_id,
            user_ids=user_ids,
            visual_available=self.s.embedding.visual_enabled,
        )
        plan_ms = int((time.perf_counter() - t0) * 1000)

        evidence, r_timings = self.retriever.retrieve(plan)

        t0 = time.perf_counter()
        answer = self.generator.generate(query, evidence)
        gen_ms = int((time.perf_counter() - t0) * 1000)

        answer.latency_ms = {"plan": plan_ms, **r_timings, "generate_ms": gen_ms}
        return answer

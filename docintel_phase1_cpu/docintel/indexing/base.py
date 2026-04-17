"""
Layer 4 — Indexing.

Three concrete indexes (Phase 1 ships the first two, Phase 2 adds the third):
  * SparseIndex  — BM25 over block text (+ figure captions). OpenSearch.
  * DenseIndex   — block embeddings. Qdrant.
  * VisualIndex  — ColPali multi-vector per-page. Qdrant multi-vector. (Phase 2)

All three are keyed by (doc_id, page_number, block_id) and every record
carries the tenant_id + acl_read list as a FILTER FIELD.

Filtering rule: always pre-filter on tenant_id/ACL before vector search.
Post-filtering silently kills recall and is the #1 production bug in RAG.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.schema import Block, DocumentMeta

log = logging.getLogger(__name__)


@dataclass
class IndexHit:
    """Raw hit from one index, before fusion."""
    doc_id: str
    page_number: int
    block_id: str
    score: float
    source_index: str                # "bm25" | "dense" | "visual"
    payload: dict = field(default_factory=dict)  # cached content/bbox/etc.


class BlockIndex(ABC):
    """An index that holds blocks and answers top-k queries over them."""
    name: str

    @abstractmethod
    def upsert_blocks(
        self,
        blocks: list[Block],
        meta: DocumentMeta,
        embeddings: np.ndarray | None = None,
    ) -> None: ...

    @abstractmethod
    def delete_doc(self, doc_id: str) -> None: ...

    @abstractmethod
    def search(
        self,
        *,
        query_text: str | None = None,
        query_vector: np.ndarray | None = None,
        filters: dict | None = None,
        top_k: int = 50,
    ) -> list[IndexHit]: ...

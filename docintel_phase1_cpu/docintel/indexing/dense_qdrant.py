"""
Dense index over block-level text embeddings, backed by Qdrant.

Pre-filters on tenant_id and ACLs. Stores enough payload (page_number,
bbox, modality, doc_title) that the retrieval layer can build an
EvidenceItem without a second DB roundtrip.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from app.schema import Block, BlockType, DocumentMeta, Modality
from embedding.text import block_to_embedding_text
from indexing.base import BlockIndex, IndexHit

log = logging.getLogger(__name__)


_BLOCKTYPE_TO_MODALITY = {
    BlockType.TABLE: Modality.TABLE,
    BlockType.FIGURE: Modality.FIGURE,
}


def _point_id(block: Block) -> str:
    # Deterministic UUID from (doc_id, block_id) — same content → same ID,
    # so re-upserting an unchanged doc is idempotent.
    name = f"{block.doc_id}:{block.block_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


class QdrantDenseIndex(BlockIndex):
    name = "dense"

    def __init__(
        self,
        url: str,
        collection: str,
        dim: int,
        api_key: str | None = None,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm

        self._qm = qm
        self._client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        qm = self._qm
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection in existing:
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )
        # Index the filter fields so pre-filtering is O(log n), not O(n).
        for field_name, schema in [
            ("tenant_id", qm.PayloadSchemaType.KEYWORD),
            ("doc_id", qm.PayloadSchemaType.KEYWORD),
            ("acl_read", qm.PayloadSchemaType.KEYWORD),
            ("modality", qm.PayloadSchemaType.KEYWORD),
            ("doc_type", qm.PayloadSchemaType.KEYWORD),
        ]:
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=schema,
            )

    def upsert_blocks(
        self,
        blocks: list[Block],
        meta: DocumentMeta,
        embeddings: np.ndarray | None = None,
    ) -> None:
        assert embeddings is not None, "QdrantDenseIndex requires embeddings"
        assert len(blocks) == embeddings.shape[0]
        if not blocks:
            return

        qm = self._qm
        points = []
        for block, vec in zip(blocks, embeddings):
            if not block_to_embedding_text(block):
                continue
            modality = _BLOCKTYPE_TO_MODALITY.get(block.type, Modality.TEXT)
            points.append(
                qm.PointStruct(
                    id=_point_id(block),
                    vector=vec.tolist(),
                    payload={
                        "doc_id": block.doc_id,
                        "block_id": block.block_id,
                        "page_number": block.page_number,
                        "modality": modality.value,
                        "bbox": block.bbox.model_dump(),
                        "tenant_id": meta.tenant_id,
                        "acl_read": meta.acl_read,
                        "doc_type": meta.doc_type,
                        "doc_title": meta.original_filename,
                        "content": self._display_content(block),
                    },
                )
            )
        if not points:
            return
        # Batch upsert — Qdrant handles the chunking internally for us.
        self._client.upsert(collection_name=self.collection, points=points, wait=False)

    def delete_doc(self, doc_id: str) -> None:
        qm = self._qm
        self._client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
                )
            ),
        )

    def search(
        self,
        *,
        query_text: str | None = None,
        query_vector: np.ndarray | None = None,
        filters: dict | None = None,
        top_k: int = 50,
    ) -> list[IndexHit]:
        assert query_vector is not None, "dense index requires query_vector"
        qm = self._qm

        q_filter = self._build_filter(filters or {})
        results = self._client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            IndexHit(
                doc_id=r.payload["doc_id"],
                page_number=int(r.payload["page_number"]),
                block_id=r.payload["block_id"],
                score=float(r.score),
                source_index=self.name,
                payload=dict(r.payload),
            )
            for r in results
        ]

    # ---- helpers ----

    def _build_filter(self, filters: dict[str, Any]):
        qm = self._qm
        must = []
        # Tenant isolation — ALWAYS required.
        tenant_id = filters.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required on every search")
        must.append(qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)))

        # ACL — user_ids intersect with acl_read on the point.
        user_ids = filters.get("user_ids") or []
        if user_ids:
            must.append(qm.FieldCondition(key="acl_read", match=qm.MatchAny(any=user_ids)))

        # Optional modality / doc_type filters from the query router.
        if filters.get("modality"):
            must.append(
                qm.FieldCondition(key="modality", match=qm.MatchValue(value=filters["modality"]))
            )
        if filters.get("doc_type"):
            must.append(
                qm.FieldCondition(key="doc_type", match=qm.MatchValue(value=filters["doc_type"]))
            )
        return qm.Filter(must=must)

    @staticmethod
    def _display_content(block: Block) -> str:
        if block.type == BlockType.TABLE:
            return block.table_markdown or block.text or ""
        if block.type == BlockType.FIGURE:
            return block.figure_caption or block.text or ""
        return block.text or ""

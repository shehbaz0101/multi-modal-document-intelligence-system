"""
Sparse / BM25 index over block text, backed by OpenSearch.

BM25 is not old-fashioned. For queries with part numbers, acronyms, exact
entity names, or quoted phrases, it beats any dense embedder. Hybrid (BM25
+ dense, fused via RRF) is strictly better than either alone — this is one
of the most replicated results in IR research and production RAG alike.
"""
from __future__ import annotations

import logging
from typing import Any

from app.schema import Block, BlockType, DocumentMeta, Modality
from indexing.base import BlockIndex, IndexHit

log = logging.getLogger(__name__)


_BLOCKTYPE_TO_MODALITY = {
    BlockType.TABLE: Modality.TABLE,
    BlockType.FIGURE: Modality.FIGURE,
}


class OpenSearchSparseIndex(BlockIndex):
    name = "bm25"

    def __init__(self, url: str, index: str):
        from opensearchpy import OpenSearch
        self._client = OpenSearch(hosts=[url])
        self.index = index
        self._ensure_index()

    def _ensure_index(self) -> None:
        if self._client.indices.exists(index=self.index):
            return
        self._client.indices.create(
            index=self.index,
            body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            # English analyzer with folding for accented chars.
                            "default": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "asciifolding", "stop", "porter_stem"],
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "doc_id":       {"type": "keyword"},
                        "block_id":     {"type": "keyword"},
                        "page_number":  {"type": "integer"},
                        "tenant_id":    {"type": "keyword"},
                        "acl_read":     {"type": "keyword"},
                        "doc_type":     {"type": "keyword"},
                        "modality":     {"type": "keyword"},
                        "doc_title":    {"type": "text"},
                        "content":      {"type": "text"},
                        "bbox":         {"type": "object", "enabled": False},
                    }
                },
            },
        )

    def upsert_blocks(
        self,
        blocks: list[Block],
        meta: DocumentMeta,
        embeddings=None,
    ) -> None:
        from opensearchpy.helpers import bulk

        actions = []
        for block in blocks:
            content = self._content_for_index(block)
            if not content:
                continue
            modality = _BLOCKTYPE_TO_MODALITY.get(block.type, Modality.TEXT)
            doc_id_for_os = f"{block.doc_id}:{block.block_id}"
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index,
                    "_id": doc_id_for_os,
                    "_source": {
                        "doc_id": block.doc_id,
                        "block_id": block.block_id,
                        "page_number": block.page_number,
                        "modality": modality.value,
                        "tenant_id": meta.tenant_id,
                        "acl_read": meta.acl_read,
                        "doc_type": meta.doc_type,
                        "doc_title": meta.original_filename,
                        "bbox": block.bbox.model_dump(),
                        "content": content,
                    },
                }
            )
        if actions:
            bulk(self._client, actions, refresh=False)

    def delete_doc(self, doc_id: str) -> None:
        self._client.delete_by_query(
            index=self.index,
            body={"query": {"term": {"doc_id": doc_id}}},
        )

    def search(
        self,
        *,
        query_text: str | None = None,
        query_vector=None,
        filters: dict | None = None,
        top_k: int = 50,
    ) -> list[IndexHit]:
        assert query_text, "BM25 requires query_text"
        filters = filters or {}
        tenant_id = filters.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required on every search")

        must_filters: list[dict[str, Any]] = [
            {"term": {"tenant_id": tenant_id}},
        ]
        if filters.get("user_ids"):
            must_filters.append({"terms": {"acl_read": filters["user_ids"]}})
        if filters.get("modality"):
            must_filters.append({"term": {"modality": filters["modality"]}})
        if filters.get("doc_type"):
            must_filters.append({"term": {"doc_type": filters["doc_type"]}})

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["content^1.0", "doc_title^0.3"],
                            }
                        }
                    ],
                    "filter": must_filters,
                }
            },
        }
        resp = self._client.search(index=self.index, body=body)
        return [
            IndexHit(
                doc_id=h["_source"]["doc_id"],
                page_number=int(h["_source"]["page_number"]),
                block_id=h["_source"]["block_id"],
                score=float(h["_score"]),
                source_index=self.name,
                payload=dict(h["_source"]),
            )
            for h in resp["hits"]["hits"]
        ]

    @staticmethod
    def _content_for_index(block: Block) -> str:
        if block.type == BlockType.TABLE:
            return block.table_markdown or block.text or ""
        if block.type == BlockType.FIGURE:
            return block.figure_caption or block.text or ""
        return block.text or ""

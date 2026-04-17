# Phase 2 — ColPali Visual Retrieval

> **Goal:** add a second, parallel retrieval track that embeds every page *as an image* using a late-interaction vision-language model (ColPali family). Text retrieval stays. Visual retrieval joins it.
>
> **Why it matters:** on visually-rich enterprise documents (10-Ks, engineering specs, slide decks, scanned forms), the ColPali family produces **+47% to +400% retrieval-quality gains over text-only RAG on the same backbone LLM**, according to ViDoRe and MCERF benchmarks. This is the single largest available lift in the whole pipeline.

---

## 1. What changes (and what doesn't)

**Unchanged:**
- Schema (`Block`, `Page`, `EvidenceItem`, etc.)
- Ingestion (already rasterizes every page to PNG)
- Parsing + captioning (still runs — captions still power text retrieval)
- Sparse + dense text indexes
- RRF fusion (weights just shift)
- Generation layer

**New:**
- A `ColPaliEmbedder` class implementing `VisualPageEmbedder` (the interface is already defined in `embedding/visual.py`)
- A `VisualPageIndex` backed by Qdrant's **multi-vector** support
- An ingest-time embedding pass over page images (one call per page, batched)
- A query-time embedding call + late-interaction MaxSim search
- Router weights that bias toward visual for "show me the chart"-class queries

Nothing in Phase 1 code gets rewritten — everything just slots into seams that are already there.

---

## 2. Model selection (as of 2026)

| Model | Params | Per-page vectors | Speed | When to pick it |
|---|---|---|---|---|
| **vidore/colqwen2-v1.0** | ~2B | ~1030 × 128 | ~30 ms/query on A10 | Default. Best accuracy on ViDoRe. |
| **vidore/colpali-v1.2** | ~3B | ~1030 × 128 | ~35 ms/query | The original. Still competitive. Use if you already have ColPali infra. |
| **vidore/colflor-v1.0** | ~174M | ~1030 × 128 | ~6 ms/query | **Default above ~100k pages.** ~5× faster at 96% of ColQwen2 accuracy. |
| **vidore/colsmol-v0.1** | ~500M | varies | fastest | When you're CPU-only or running on edge. |

**Recommended default:** ColQwen2 for corpora under 50k pages, ColFlor above. Make this a config flag; never pick once and forget.

---

## 3. Storage cost reality check

ColPali-family models produce **multi-vector embeddings** — one vector per image patch, ~1030 patches per page at 448×448. At 128-dim float16 that's ~260 KB per page. For a 100k-page corpus:

- Raw float16:  ~26 GB
- With binary quantization (ColBERT-style, 32× compression, ~1-2% recall loss): ~800 MB
- Pooled to single-vector per page (recall loss ~5-10%): ~50 MB

**Use binary quantization** unless you're under 10k pages. Qdrant 1.10+ has it built in via `QuantizationConfig`. This is the difference between "fits on a single server" and "needs a cluster."

---

## 4. Implementation — the concrete code

### 4.1 Complete the ColPali embedder

```python
# embedding/colpali.py — new file
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.storage import Storage
from embedding.visual import VisualPageEmbedder


class ColQwen2Embedder(VisualPageEmbedder):
    model_name = "vidore/colqwen2-v1.0"
    dim = 128

    def __init__(self, storage: Storage, device: str = "cuda", batch_size: int = 4):
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self.storage = storage
        self.device = device
        self.batch_size = batch_size

        self._model = ColQwen2.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self._processor = ColQwen2Processor.from_pretrained(self.model_name)

    @torch.inference_mode()
    def embed_pages(self, image_uris: list[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(0, len(image_uris), self.batch_size):
            batch_uris = image_uris[i : i + self.batch_size]
            images = [Image.open(io.BytesIO(self.storage.get(u))).convert("RGB")
                      for u in batch_uris]
            batch = self._processor.process_images(images).to(self.device)
            embs = self._model(**batch)                   # (B, n_patches, dim)
            for e in embs:
                out.append(e.cpu().float().numpy())       # (n_patches, dim)
        return out

    @torch.inference_mode()
    def embed_query(self, query: str) -> np.ndarray:
        batch = self._processor.process_queries([query]).to(self.device)
        emb = self._model(**batch)                        # (1, n_tokens, dim)
        return emb[0].cpu().float().numpy()
```

### 4.2 Multi-vector Qdrant index

```python
# indexing/visual_qdrant.py — new file
from __future__ import annotations

import uuid

import numpy as np

from app.schema import DocumentMeta, Page
from indexing.base import BlockIndex, IndexHit


def _page_point_id(doc_id: str, page_number: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:page:{page_number}"))


class QdrantVisualIndex(BlockIndex):
    """
    Multi-vector per page. Uses Qdrant's native multi-vector support (1.10+)
    with MaxSim comparator for late-interaction search.
    """
    name = "visual"

    def __init__(self, url: str, collection: str, dim: int = 128,
                 api_key: str | None = None, use_binary_quantization: bool = True):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm

        self._qm = qm
        self._client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.dim = dim
        self._ensure_collection(use_binary_quantization)

    def _ensure_collection(self, use_bq: bool) -> None:
        qm = self._qm
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection in existing:
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "patches": qm.VectorParams(
                    size=self.dim,
                    distance=qm.Distance.COSINE,
                    multivector_config=qm.MultiVectorConfig(
                        comparator=qm.MultiVectorComparator.MAX_SIM,
                    ),
                    quantization_config=(
                        qm.BinaryQuantization(
                            binary=qm.BinaryQuantizationConfig(always_ram=True)
                        ) if use_bq else None
                    ),
                )
            },
        )
        # Same payload filter indexes as the dense text index
        for field_name, schema in [
            ("tenant_id", qm.PayloadSchemaType.KEYWORD),
            ("doc_id",    qm.PayloadSchemaType.KEYWORD),
            ("acl_read",  qm.PayloadSchemaType.KEYWORD),
            ("doc_type",  qm.PayloadSchemaType.KEYWORD),
        ]:
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=schema,
            )

    def upsert_pages(
        self,
        pages: list[Page],
        meta: DocumentMeta,
        embeddings: list[np.ndarray],
    ) -> None:
        assert len(pages) == len(embeddings)
        qm = self._qm
        points = []
        for page, patch_vecs in zip(pages, embeddings):
            points.append(
                qm.PointStruct(
                    id=_page_point_id(page.doc_id, page.page_number),
                    vector={"patches": patch_vecs.tolist()},
                    payload={
                        "doc_id": page.doc_id,
                        "page_number": page.page_number,
                        "page_image_uri": page.image_uri,
                        "tenant_id": meta.tenant_id,
                        "acl_read":  meta.acl_read,
                        "doc_type":  meta.doc_type,
                        "doc_title": meta.original_filename,
                    },
                )
            )
        if points:
            self._client.upsert(collection_name=self.collection, points=points, wait=False)

    def search(self, *, query_vector: np.ndarray, filters: dict | None = None,
               top_k: int = 50, **_) -> list[IndexHit]:
        qm = self._qm
        filters = filters or {}
        tenant_id = filters.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required")

        must = [qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
        if filters.get("user_ids"):
            must.append(qm.FieldCondition(key="acl_read",
                                          match=qm.MatchAny(any=filters["user_ids"])))

        results = self._client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            using="patches",
            query_filter=qm.Filter(must=must),
            limit=top_k,
            with_payload=True,
        ).points

        # Visual retrieval returns pages, not blocks. We surface them with
        # a synthetic "page-level block" so the fusion layer treats them
        # uniformly. The generator sees these as figure-modality evidence.
        hits: list[IndexHit] = []
        for r in results:
            p = r.payload
            hits.append(
                IndexHit(
                    doc_id=p["doc_id"],
                    page_number=int(p["page_number"]),
                    block_id=f"p{p['page_number']}_visual",
                    score=float(r.score),
                    source_index=self.name,
                    payload={
                        "doc_id":         p["doc_id"],
                        "page_number":    p["page_number"],
                        "block_id":       f"p{p['page_number']}_visual",
                        "modality":       "figure",
                        "bbox":           {"x0": 0, "y0": 0, "x1": 0, "y1": 0},
                        "content":        f"(visual page from {p.get('doc_title','')})",
                        "page_image_uri": p["page_image_uri"],
                        "doc_title":      p.get("doc_title"),
                        "tenant_id":      p["tenant_id"],
                    },
                )
            )
        return hits

    def upsert_blocks(self, blocks, meta, embeddings=None):
        raise NotImplementedError("Visual index operates at page granularity; use upsert_pages")

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
```

### 4.3 Wire it into the orchestrator

In `app/orchestrator.py`, add three hooks — all additive:

```python
# __init__
if self.s.embedding.visual_enabled:
    from embedding.colpali import ColQwen2Embedder
    from indexing.visual_qdrant import QdrantVisualIndex
    self.visual_embedder = ColQwen2Embedder(storage=self.storage)
    self.visual_index = QdrantVisualIndex(
        url=self.s.index.vector_url,
        collection=self.s.index.visual_collection,
        api_key=self.s.index.vector_api_key,
    )
    self.retriever.visual_index = self.visual_index   # retriever already supports it
else:
    self.visual_embedder = None

# ingest() — add after the text-embedding step:
if self.visual_embedder and self.visual_index:
    page_uris = [p.image_uri for p in result.pages]
    page_embeddings = self.visual_embedder.embed_pages(page_uris)
    self.visual_index.upsert_pages(result.pages, result.meta, page_embeddings)

# ask() — plan_query() is already passed visual_available=self.s.embedding.visual_enabled,
# and Retriever already fires the visual index in parallel when its weight > 0.
# Nothing else to change.
```

That's the full integration. Two new modules, three added lines in the orchestrator's `__init__`, and a ~6-line block in `ingest()`.

### 4.4 Query-side embedding

The retriever already receives `query_text` and, via `text_embedder`, a dense query vector. For visual search you want a *visual* query vector — so add a branch in `Retriever.retrieve()`:

```python
# retrieval/retriever.py — inside retrieve()
if self.visual_index and plan.weights.get("visual", 0) > 0:
    q_visual = self.visual_embedder.embed_query(plan.query)
    futures["visual"] = pool.submit(
        self.visual_index.search,
        query_vector=q_visual,
        filters=plan.filters,
        top_k=plan.top_k_per_index,
    )
```

And inject `visual_embedder` when you construct `Retriever`. One parameter.

---

## 5. Rollout plan

1. **Week 3, Day 1-2.** Install `colpali-engine`, run the model locally on a GPU, confirm you can embed a single page and a query, and MaxSim them by hand.
2. **Week 3, Day 3-4.** Build `QdrantVisualIndex`. Stand up a second Qdrant collection. Upsert ~100 pages. Hand-issue 5 queries and eyeball the results.
3. **Week 3, Day 5.** Wire into the orchestrator behind the `EMBEDDING_VISUAL_ENABLED` flag. Flag stays `false` by default.
4. **Week 4, Day 1-2.** Update your eval set to include 20 "visual-required" queries — charts, diagrams, tables where the answer literally cannot be produced from text alone. These are your hard-query benchmark.
5. **Week 4, Day 3.** Run the full eval. Publish the numbers table: text-only vs hybrid-with-visual, by query category. This is the Phase 2 deliverable.
6. **Week 4, Day 4-5.** Tune router weights on a held-out split. Try ColFlor as a cheaper alternative. Document the cost/accuracy trade-off.

---

## 6. What to measure

The thing that turns this from a feature into a case study:

| Metric | Text-only (Phase 1) | + Visual (Phase 2) | Delta |
|---|---|---|---|
| Recall@5 (all queries) | — | — | — |
| Recall@5 (visual-required) | **near zero** | — | — |
| nDCG@10 (all) | — | — | — |
| Citation validity | — | — | — |
| Faithfulness (LLM-judge) | — | — | — |
| p50 latency | — | — | — |
| p95 latency | — | — | — |
| $/1000 queries | — | — | — |

Ship this table in the README. It's what separates a project that "uses ColPali" from a project that *quantifies why ColPali matters*.

---

## 7. Common failure modes (and fixes)

- **OOM during embedding.** Drop `batch_size` to 2 or 1. ColQwen2 at bf16 needs ~10-12 GB VRAM for batch=4.
- **Retrieval recall looks identical with and without visual.** Your eval set is too text-biased. Add queries that are *visually-only-answerable* — "what color is the segment representing APAC in the 2024 revenue chart?" — and re-run.
- **Visual hits never surface in fused results.** Check the router. If `weights["visual"] == 0` for your query pattern, the future never gets scheduled. Log `plan.weights` for every query during Phase 2 rollout.
- **Storage blowing up.** You skipped binary quantization. Turn it on.
- **Generator still gives text-only answers despite visual hits.** The page image isn't making it into the prompt. Check that `EvidenceItem.page_image_uri` is populated — the visual index payload sets it, but confirm with a breakpoint.

---

## 8. Where this leads next (Phase 3 hook)

A vision-capable reranker (e.g., a small fine-tuned Qwen2.5-VL-3B, or a hosted vision reranker like Cohere's) that takes the fused top-30 — including visual page hits — and reranks by looking at both the query and the candidate page image. This is where you get the last 10-15% of quality, and where MCERF-style self-consistency starts paying off on high-stakes queries.

The `Reranker` protocol is already defined in `retrieval/retriever.py`. Phase 3 just ships an implementation.

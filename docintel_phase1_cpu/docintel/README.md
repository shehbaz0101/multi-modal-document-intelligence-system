# DocIntel — Multi-Modal Document Intelligence

> Production-grade RAG for documents that contain **charts, tables, diagrams, and figures** — not just text. Every answer is grounded to a visible region of a real page, citable down to the bounding box.

[![python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![status](https://img.shields.io/badge/phase-1%20(thin%20slice)-orange)]()
[![license](https://img.shields.io/badge/license-MIT-green)]()

---

## The problem

Most enterprise RAG is text-only and silently drops the half of the document that actually matters — the charts, stamps, signatures, diagrams, and tables. This system treats documents the way a human analyst does: **as pages, not chunks**. A question like *"what's the revenue growth chart showing?"* retrieves the actual figure, and the answer cites the specific bounding box on the specific page.

## What's inside

- **Page-keyed ingestion** — canonical unit is the page, not the chunk. Content-addressed doc IDs give free dedup and immutable audit trails.
- **Docling-based parsing** with a pluggable interface (LlamaParse / Reducto as drop-in alternates).
- **Figure captioner** — every figure gets a VLM-generated caption at ingest time, indexed as text. The single highest-leverage trick in the pipeline.
- **Hybrid retrieval** — BM25 (OpenSearch) + dense embeddings (BGE-M3 over Qdrant), fused with Reciprocal Rank Fusion.
- **Query router** — classifies intent and weighs retrievers accordingly.
- **Grounded generation** — Claude Sonnet 4.6 with page images attached for figure/table evidence. Every claim must cite `[eN]` or the system returns `INSUFFICIENT_EVIDENCE`.
- **Eval harness from day 1** — Recall@k, MRR, nDCG@k, citation validity, citation coverage. Runs in CI.
- **Multi-tenant + ACL-aware** — tenant_id and acl_read propagate into every index as pre-filters. The #1 production RAG bug (post-filtering) is engineered out.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INGESTION     PDF → pages (rasterized once at 150 DPI)      │
├─────────────────────────────────────────────────────────────────┤
│ 2. PARSING       Docling → blocks (text/table/figure + bboxes) │
│    + CAPTIONING  VLM writes a caption for every figure          │
├─────────────────────────────────────────────────────────────────┤
│ 3. EMBEDDING     BGE-M3 text embeddings                         │
│                  (ColPali visual track is Phase 2)              │
├─────────────────────────────────────────────────────────────────┤
│ 4. INDEXING      Qdrant (dense) + OpenSearch (BM25)             │
│                  Tenant + ACL filters on every record           │
├─────────────────────────────────────────────────────────────────┤
│ 5. RETRIEVAL     Router → parallel search → RRF → [rerank]      │
├─────────────────────────────────────────────────────────────────┤
│ 6. GENERATION    Claude Sonnet 4.6 w/ page images → grounded    │
│                  answer + `[eN]` citations + bounding boxes     │
└─────────────────────────────────────────────────────────────────┘
```

See [`docs/blueprint.md`](docs/blueprint.md) for the full architectural rationale and [`docs/phase2_colpali.md`](docs/phase2_colpali.md) for the visual retrieval upgrade plan.

## Quickstart (10 minutes)

**Prereqs:** Docker, Python 3.11+, [uv](https://github.com/astral-sh/uv), an Anthropic API key.

```bash
# 1. Clone and configure
git clone <this-repo> docintel && cd docintel
cp .env.example .env
# edit .env — set ANTHROPIC_API_KEY at minimum

# 2. Bring up infra (Qdrant + OpenSearch + Postgres)
make infra-up

# 3. Install Python deps
make install
source .venv/bin/activate

# 4. Run the end-to-end demo on a PDF
python scripts/quickstart.py \
    --pdf path/to/your.pdf \
    --ask "what was the main revenue driver last quarter?"
```

You should see an answer with `[e0]`, `[e1]` citations, each pointing back to a `(doc, page, bbox)` tuple.

## Repository layout

```
docintel/
├── app/                 # schema, storage, orchestrator
│   ├── schema.py        # Pydantic types (Block, Page, EvidenceItem, Answer, ...)
│   ├── storage.py       # Local / S3 / GCS abstraction
│   └── orchestrator.py  # DocIntel().ingest() / .ask() — the public API
├── config/              # Typed env-driven settings
├── ingestion/           # PDF → normalized pages
├── parsing/             # Docling parser + figure captioner
├── embedding/           # BGE-M3 text; ColPali visual stub
├── indexing/            # Qdrant dense + OpenSearch sparse
├── retrieval/           # Router, RRF fusion, orchestrated retriever
├── generation/          # Grounded VLM generator with citation extraction
├── evaluation/          # Eval harness — ship this before features
├── scripts/             # quickstart.py and utilities
├── docs/                # blueprint.md, phase2_colpali.md, etc.
└── docker-compose.yml   # Qdrant + OpenSearch + Postgres
```

## Roadmap

| Phase | Weeks | Scope |
|---|---|---|
| **1** ✅ | 1–2  | Thin vertical slice — text-only hybrid RAG with grounded citations |
| **2**     | 3–4  | ColPali/ColQwen2 visual retrieval — see `docs/phase2_colpali.md` |
| **3**     | 5–6  | Cross-encoder reranking + learned query router |
| **4**     | 7–8  | Bounding-box citation UI (the demo-winner) |
| **5**     | 9–10 | Incremental indexing, caching, observability dashboards |
| **6**     | 11–12| Multi-tenancy hardening, PII redaction, CI eval gates |

## The differentiators

What makes this worth building (and putting on a portfolio) vs. yet-another-RAG:

1. **The eval harness ships on day 1.** Recall@k and citation validity numbers in a real table in the README — see [`docs/evals.md`](docs/evals.md) once you've run it.
2. **A hard-query benchmark.** 20 queries that *require* reading the chart or table. Pure-text RAG gets them wrong; this system gets them right. That's the "why this exists" slide.
3. **Bounding-box citations.** The UI highlights the exact pixel region behind every claim. This is what regulated industries actually pay for.
4. **A documented cost-accuracy trade-off.** ColQwen2 vs ColFlor, Sonnet 4.6 vs Opus 4.7 — with numbers, not vibes. Shows engineering, not tinkering.
5. **Failure-mode transparency.** The Answer object carries per-stage timings and per-retriever scores. A debug panel in the UI can show which retriever contributed what — and nobody else ships this.

## License

MIT.

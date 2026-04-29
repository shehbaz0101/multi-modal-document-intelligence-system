<div align="center">

# 🧠 DocIntel — Multi-Modal Document Intelligence

**Production-grade RAG that doesn't ignore your charts, tables, and figures.**

Ask questions about any PDF and get answers grounded to specific page regions — every claim cited down to the bounding box.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-DC382D?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Gemini](https://img.shields.io/badge/Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🎯 The Problem

Most RAG systems are **blind** — they extract text from PDFs and silently drop everything visual. Charts, tables, diagrams, signatures, stamps. The half of an enterprise document that actually carries the information.

DocIntel treats documents the way a human analyst does: **as pages, not chunks**. A question like *"what's the revenue trend in Q3?"* retrieves the actual chart, and the answer cites the specific bounding box on the specific page.

---

## ✨ What it does

```
Q: What projects has this person built?

A: The person built an autonomous multi-agent AI system named AutoDev
   using FastAPI and LangGraph [e0], which converts natural-language
   requirements into functional locally-deployed web applications [e0].
   They also built a Streamlit-based LLM interface called Multi-Model
   Chat Arena to compare responses from Gemini and GPT-3.5 [e1].

📎 Citations:
  [e0] resume.pdf  page 1  Text   bbox=(0, 412, 1241, 638)
  [e1] resume.pdf  page 1  Text   bbox=(0, 645, 1241, 798)
```

Every claim is grounded. No hallucination. Click a citation, see the exact page region.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  📄 INGEST       PDF → 150-DPI page images, content-addressed   │
├─────────────────────────────────────────────────────────────────┤
│  🔍 PARSE        pymupdf4llm + pdfplumber → blocks + bboxes     │
│  🖼️  CAPTION     Gemini VLM writes captions for every figure    │
├─────────────────────────────────────────────────────────────────┤
│  🧬 EMBED        sentence-transformers MiniLM-L6 (384-dim)      │
├─────────────────────────────────────────────────────────────────┤
│  💾 INDEX        Qdrant (dense) + OpenSearch (BM25)             │
│                  ACL pre-filtering on every query               │
├─────────────────────────────────────────────────────────────────┤
│  🎯 RETRIEVE     Router → parallel hybrid → RRF → rerank        │
│  👁️  VISUAL RERANK  Gemini scores page images for visual queries │
├─────────────────────────────────────────────────────────────────┤
│  💬 GENERATE     Gemini grounded answer + bounding-box citations │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                  📡 SSE streaming · 🔍 Langfuse traces
```

Six layers, each independently swappable. ColPali, OpenAI, Cohere, Anthropic, Pinecone — drop in any of them by changing one config line.

---

## 🚀 Quickstart

**Requirements:** Python 3.11+, Docker, [Gemini API key](https://aistudio.google.com/apikey) (free).

```bash
# 1. Clone
git clone https://github.com/shehbaz0101/multi-modal-document-intelligence-system.git
cd multi-modal-document-intelligence-system/docintel_phase1_cpu/docintel

# 2. Set up venv (Python 3.11+)
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows
# source .venv/bin/activate           # Linux/Mac

# 3. Install — torch CPU first to avoid pulling 2.5GB CUDA wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env → paste your GEMINI_API_KEY

# 5. Start the infra
docker compose up -d
# Wait ~45 seconds for OpenSearch to become healthy

# 6. Launch
uvicorn app.api:app --reload --port 8000
```

Open **http://localhost:8000** → drag a PDF in → ask a question.

---

## 🔥 Features

### Retrieval quality
- 🔀 **Hybrid search** — BM25 + dense vectors fused with Reciprocal Rank Fusion
- 🧠 **Query routing** — analytical, factual, and visual queries get different retriever weights
- 👁️ **Visual reranking** — Gemini scores page images for visual queries (no GPU required)
- 🎯 **Cross-encoder reranking** — optional ms-marco MiniLM, ~200ms overhead

### Production-ready
- 🏢 **Multi-tenant** with ACL-aware **pre-filtering** (the #1 production RAG bug)
- 🔑 **Content-addressed doc IDs** — free deduplication
- ⚡ **Answer cache** with TTL (cuts cost + latency on repeat queries)
- 🚦 **Rate limiting** — 30 req/min/IP
- 🆔 **Request ID propagation** through every log line
- 🩺 **Granular health checks** — per-service status
- 📊 **Live metrics** endpoint
- 🌊 **Streaming responses** via Server-Sent Events
- 🔭 **Langfuse tracing** — every query observable end-to-end

### Polished UI
- 🖱️ Drag-and-drop PDF upload
- ⚡ Token-by-token streaming answers
- 🔗 Inline clickable citation tags
- 🕐 Query history with one-click rerun
- 💾 Cache indicator on cached responses
- 📈 Live metrics panel

---

## 🛠️ Tech stack

| Layer | Component | Why |
|---|---|---|
| **Ingestion** | pypdfium2 | Fast C++ PDF rasterization, no Python overhead |
| **Parsing** | pymupdf4llm + pdfplumber | CPU-native, no model weights, 1-5ms/page |
| **Text embeddings** | sentence-transformers MiniLM-L6 | 384-dim, 22MB, ~500 sent/sec on i5 |
| **Figure captioning** | Gemini 2.5 Flash Lite | VLM writes searchable descriptions at ingest |
| **Vector store** | Qdrant 1.12 | Cosine similarity, ACL pre-filtering |
| **Sparse index** | OpenSearch 2.17 | BM25 with proper tokenization |
| **Fusion** | Reciprocal Rank Fusion (k=60) | Parameter-free, hard to beat |
| **Visual reranker** | Gemini 2.5 Flash Lite | No GPU required, ~80% of ColPali quality |
| **Generation** | Gemini 2.5 Flash Lite | Vision-capable, free tier |
| **API** | FastAPI + Uvicorn | Async, auto-OpenAPI docs |
| **Observability** | Langfuse | Free tier, 50k events/month |

---

## 📁 Project structure

```
docintel/
├── app/                  # Public API + orchestrator + schema
│   ├── api.py            # FastAPI app with rate limiting, caching, streaming
│   ├── orchestrator.py   # DocIntel class — the entry point
│   ├── schema.py         # Pydantic types
│   ├── storage.py        # Object storage abstraction (local / S3 / GCS)
│   └── tracing.py        # Langfuse integration
├── config/settings.py    # 12-factor environment-driven config
├── ingestion/ingest.py   # PDF → normalized pages
├── parsing/              # CPU parser + VLM figure captioner
│   ├── cpu_parser.py     # pymupdf4llm + pdfplumber
│   └── captioner.py      # Gemini-based figure captioner
├── embedding/            # Text + visual stub
├── indexing/             # Qdrant dense + OpenSearch sparse
├── retrieval/            # Router + fusion + reranker + visual reranker
│   ├── router.py         # Intent classification
│   ├── fusion.py         # RRF
│   ├── reranker.py       # Cross-encoder
│   ├── visual_rerank.py  # Gemini visual reranker (Phase 2)
│   └── retriever.py      # Orchestrates the funnel
├── generation/           # Grounded Gemini generator
├── evaluation/           # Eval harness (Recall, MRR, nDCG, citation validity)
├── scripts/
│   ├── quickstart.py     # CLI demo
│   └── run_eval.py       # Eval runner with diff support
├── tests/                # Unit tests
├── ui.html               # Single-file citation UI
├── Dockerfile            # Production CPU image
├── docker-compose.yml    # Full stack with managed services support
└── requirements.txt
```

---

## 🌐 API endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/`                       | Citation UI |
| `GET`  | `/health`                 | Per-service health |
| `GET`  | `/metrics`                | Live stats — requests, cache hit rate, latency |
| `GET`  | `/docs`                   | Interactive Swagger UI |
| `POST` | `/ingest`                 | Upload PDF, returns `doc_id` |
| `POST` | `/ask`                    | Query, returns grounded answer with citations |
| `POST` | `/ask/stream`             | Same as `/ask` but streams via SSE |
| `DELETE` | `/documents/{doc_id}`   | Remove a document from both indexes |

All responses carry `X-Request-ID` and `X-Response-Time-Ms` headers.

---

## 🗺️ Roadmap

| Phase | Status | What it adds |
|---|---|---|
| **1 — Thin slice** | ✅ Done | Hybrid text RAG, grounded citations, FastAPI, UI |
| **2 — Visual reranker** | ✅ Done | Gemini-based page-image scoring for visual queries |
| **3 — Production hardening** | ✅ Done | Streaming, Langfuse, eval harness, rate limiting |
| **4 — Cloud deployment** | 🚧 In progress | Public URL via Render + Qdrant Cloud + Bonsai |
| **5 — Auth & multi-user** | ⬜ Planned | OAuth, per-user quotas, billing tiers |
| **6 — ColPali (GPU)** | ⬜ Planned | Native multi-vector visual retrieval |

---

## 📊 Why this matters

Every enterprise AI company is trying to build this. Few succeed because:

1. **Most ignore visual content.** Pure text RAG misses 50% of what's in real documents.
2. **Most have no eval.** Systems silently regress every time the prompt changes.
3. **Most have no grounding.** Regulated industries (finance, legal, healthcare) won't deploy what can't cite sources.

DocIntel solves all three.

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Built by [Sufyan](https://github.com/shehbaz0101)** ·  Reach out for collaboration

</div>

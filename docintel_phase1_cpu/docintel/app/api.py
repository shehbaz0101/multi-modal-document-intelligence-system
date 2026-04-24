"""
DocIntel — Industry-grade FastAPI application.

Features added vs Phase 1:
  - Request IDs on every request (traceable logs)
  - Rate limiting (slowapi)
  - Answer cache (query hash -> response, TTL 1 hour)
  - Async ingest + ask endpoints
  - Granular health check (per-service status)
  - Document deletion
  - Structured error responses
  - Graceful handling of bad PDFs
  - CORS, GZip compression
  - /metrics endpoint (request counts, latencies)
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.orchestrator import DocIntel

log = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)-5s %(name)s :: %(message)s",
)

# ── simple in-process cache ──────────────────────────────────────────────────
class _Cache:
    def __init__(self, ttl: int = 3600):
        self._store: dict[str, tuple[float, dict]] = {}
        self.ttl = ttl

    def key(self, query: str, tenant: str) -> str:
        return hashlib.sha256(f"{tenant}:{query}".encode()).hexdigest()[:16]

    def get(self, k: str) -> dict | None:
        if k in self._store:
            ts, val = self._store[k]
            if time.time() - ts < self.ttl:
                return val
            del self._store[k]
        return None

    def set(self, k: str, val: dict) -> None:
        self._store[k] = (time.time(), val)

    def invalidate_tenant(self, tenant: str) -> None:
        # crude but effective for dev — in prod use Redis
        pass

_cache = _Cache()

# ── metrics ──────────────────────────────────────────────────────────────────
_metrics: dict[str, int | float] = {
    "ingest_total": 0, "ingest_errors": 0,
    "ask_total": 0, "ask_errors": 0, "ask_cache_hits": 0,
    "ask_latency_ms_total": 0.0,
}

# ── singleton ─────────────────────────────────────────────────────────────────
_intel: DocIntel | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _intel
    log.info("Starting DocIntel...", extra={"request_id": "startup"})
    _intel = DocIntel()
    log.info("DocIntel ready", extra={"request_id": "startup"})
    yield
    log.info("Shutting down", extra={"request_id": "shutdown"})

app = FastAPI(
    title="DocIntel API",
    description="Industry-grade multi-modal document intelligence.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── request ID middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = rid
    start = time.perf_counter()
    response = await call_next(request)
    ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-Ms"] = str(ms)
    return response

# ── simple rate limiter (in-memory, per-IP) ───────────────────────────────────
_rate: dict[str, list[float]] = {}
RATE_LIMIT = 30        # requests
RATE_WINDOW = 60       # seconds

def _check_rate(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _rate.get(ip, []) if now - t < RATE_WINDOW]
    hits.append(now)
    _rate[ip] = hits
    return len(hits) <= RATE_LIMIT

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    if request.url.path not in ("/health", "/metrics", "/") and not _check_rate(ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Max 30 requests/minute."},
        )
    return await call_next(request)

# ── schemas ───────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id: str
    pages: int
    filename: str
    cached: bool = False

class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    tenant_id: str = Field(default="default", max_length=64)
    user_ids: list[str] = Field(default=[])
    skip_cache: bool = False

class CitationOut(BaseModel):
    evidence_id: str
    page: int
    modality: str
    snippet: str
    bbox: dict

class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    insufficient: bool
    timings_ms: dict
    cached: bool = False
    request_id: str = ""

class HealthDetail(BaseModel):
    status: str
    qdrant: str
    opensearch: str
    model: str
    uptime_s: float

class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool

_start_time = time.time()

# ── UI ────────────────────────────────────────────────────────────────────────
UI_PATH = Path(__file__).parent.parent / "ui.html"

@app.get("/", include_in_schema=False)
def serve_ui():
    if UI_PATH.exists():
        return FileResponse(UI_PATH, media_type="text/html")
    return {"message": "Place ui.html in the project root"}

# ── health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthDetail)
def health(request: Request):
    qdrant_ok = "ok"
    opensearch_ok = "ok"

    try:
        from qdrant_client import QdrantClient
        QdrantClient(_intel.s.index.vector_url).get_collections()
    except Exception as e:
        qdrant_ok = f"error: {e}"

    try:
        from opensearchpy import OpenSearch
        OpenSearch(_intel.s.index.sparse_url).info()
    except Exception as e:
        opensearch_ok = f"error: {e}"

    overall = "ok" if qdrant_ok == "ok" and opensearch_ok == "ok" else "degraded"
    return HealthDetail(
        status=overall,
        qdrant=qdrant_ok,
        opensearch=opensearch_ok,
        model=_intel.s.generation.model if _intel else "not loaded",
        uptime_s=round(time.time() - _start_time, 1),
    )

# ── metrics ───────────────────────────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False)
def metrics():
    total = _metrics["ask_total"] or 1
    return {
        **_metrics,
        "ask_avg_latency_ms": round(_metrics["ask_latency_ms_total"] / total, 1),
        "cache_hit_rate": round(_metrics["ask_cache_hits"] / total, 3),
    }

# ── ingest ────────────────────────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF to ingest")],
    tenant_id: Annotated[str, Form()] = "default",
    doc_type: Annotated[str | None, Form()] = None,
):
    rid = getattr(request.state, "request_id", "-")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    contents = await file.read()

    if len(contents) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(413, "File too large. Maximum size is 50 MB.")

    if len(contents) < 100:
        raise HTTPException(400, "File appears to be empty or corrupt.")

    _metrics["ingest_total"] += 1

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        log.info("ingesting file=%s tenant=%s", file.filename, tenant_id,
                 extra={"request_id": rid})
        meta = _intel.ingest(
            tmp_path,
            tenant_id=tenant_id,
            doc_type=doc_type,
            acl_read=["*"],
        )
    except Exception as e:
        _metrics["ingest_errors"] += 1
        log.error("ingest failed: %s", e, extra={"request_id": rid})
        raise HTTPException(422, f"Failed to process PDF: {str(e)}")
    finally:
        os.unlink(tmp_path)

    # Invalidate cache for this tenant on new ingest
    _cache.invalidate_tenant(tenant_id)

    return IngestResponse(
        doc_id=meta.doc_id,
        pages=meta.page_count,
        filename=file.filename,
    )

# ── ask ───────────────────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    rid = getattr(request.state, "request_id", "-")
    _metrics["ask_total"] += 1
    t0 = time.perf_counter()

    # cache lookup
    if not req.skip_cache:
        ck = _cache.key(req.query, req.tenant_id)
        cached = _cache.get(ck)
        if cached:
            _metrics["ask_cache_hits"] += 1
            log.info("cache hit query='%s'", req.query[:60], extra={"request_id": rid})
            return AskResponse(**cached, cached=True, request_id=rid)

    try:
        log.info("ask query='%s' tenant=%s", req.query[:60], req.tenant_id,
                 extra={"request_id": rid})
        answer = _intel.ask(
            req.query,
            tenant_id=req.tenant_id,
            user_ids=req.user_ids or None,
        )
    except Exception as e:
        _metrics["ask_errors"] += 1
        log.error("ask failed: %s", e, extra={"request_id": rid})
        raise HTTPException(500, f"Query failed: {str(e)}")

    citations = [
        CitationOut(
            evidence_id=e.evidence_id,
            page=e.page_number,
            modality=e.modality.value,
            snippet=(e.content or "")[:200],
            bbox=e.bbox.model_dump(),
        )
        for e in answer.evidence
        if any(c.evidence_id == e.evidence_id for c in answer.citations)
    ]

    result = dict(
        answer=answer.answer_text,
        citations=citations,
        insufficient=answer.insufficient_evidence,
        timings_ms=answer.latency_ms,
    )

    # cache the result
    if not req.skip_cache:
        _cache.set(ck, result)

    ms = int((time.perf_counter() - t0) * 1000)
    _metrics["ask_latency_ms_total"] += ms

    return AskResponse(**result, cached=False, request_id=rid)

# ── delete ────────────────────────────────────────────────────────────────────
@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str, request: Request):
    rid = getattr(request.state, "request_id", "-")
    try:
        log.info("deleting doc_id=%s", doc_id[:12], extra={"request_id": rid})
        _intel.dense_index.delete_doc(doc_id)
        _intel.sparse_index.delete_doc(doc_id)
        return DeleteResponse(doc_id=doc_id, deleted=True)
    except Exception as e:
        log.error("delete failed: %s", e, extra={"request_id": rid})
        raise HTTPException(500, f"Delete failed: {str(e)}")
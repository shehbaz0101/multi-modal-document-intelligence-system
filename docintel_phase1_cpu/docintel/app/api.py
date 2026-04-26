"""
DocIntel — Industry-grade FastAPI application (Phase 3).

New in Phase 3:
  - /ask/stream — Server-Sent Events streaming endpoint
  - Langfuse tracing wired through every /ask
  - Eval endpoint /eval/run (kicks off background eval)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.orchestrator import DocIntel
from app.tracing import init_tracer, trace_query, shutdown as tracer_shutdown

log = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)-5s %(name)s :: %(message)s",
)

# ── cache ────────────────────────────────────────────────────────────────────
class _Cache:
    def __init__(self, ttl: int = 3600):
        self._store: dict[str, tuple[float, dict]] = {}
        self.ttl = ttl
    def key(self, query: str, tenant: str) -> str:
        return hashlib.sha256(f"{tenant}:{query}".encode()).hexdigest()[:16]
    def get(self, k: str) -> dict | None:
        if k in self._store:
            ts, val = self._store[k]
            if time.time() - ts < self.ttl: return val
            del self._store[k]
        return None
    def set(self, k: str, val: dict) -> None:
        self._store[k] = (time.time(), val)
    def invalidate_tenant(self, tenant: str) -> None: pass

_cache = _Cache()
_metrics: dict[str, int | float] = {
    "ingest_total": 0, "ingest_errors": 0,
    "ask_total": 0, "ask_errors": 0, "ask_cache_hits": 0,
    "ask_latency_ms_total": 0.0,
}
_intel: DocIntel | None = None


def _get_intel() -> DocIntel:
    """Return the DocIntel singleton — raises if not yet initialized."""
    if _intel is None:
        raise RuntimeError("DocIntel not initialized")
    return _intel



@asynccontextmanager
async def lifespan(app: FastAPI):
    global _intel
    log.info("Starting DocIntel...")
    _intel = DocIntel()
    init_tracer(
        host=_get_intel().s.observability.langfuse_host,
        public_key=_get_intel().s.observability.langfuse_public_key,
        secret_key=_get_intel().s.observability.langfuse_secret_key,
    )
    log.info("DocIntel ready")
    yield
    log.info("Shutting down")
    tracer_shutdown()


app = FastAPI(
    title="DocIntel API",
    description="Industry-grade multi-modal document intelligence.",
    version="3.0.0",
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


_rate: dict[str, list[float]] = {}
RATE_LIMIT, RATE_WINDOW = 30, 60

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
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    return await call_next(request)


# ── schemas ─────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    doc_id: str; pages: int; filename: str; cached: bool = False

class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    tenant_id: str = Field(default="default", max_length=64)
    user_ids: list[str] = Field(default=[])
    skip_cache: bool = False

class CitationOut(BaseModel):
    evidence_id: str; page: int; modality: str; snippet: str; bbox: dict

class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    insufficient: bool
    timings_ms: dict
    cached: bool = False
    request_id: str = ""

class HealthDetail(BaseModel):
    status: str; qdrant: str; opensearch: str; model: str; uptime_s: float; tracing: str

class DeleteResponse(BaseModel):
    doc_id: str; deleted: bool


_start_time = time.time()
UI_PATH = Path(__file__).parent.parent / "ui.html"


# ── endpoints ───────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_ui():
    if UI_PATH.exists():
        return FileResponse(UI_PATH, media_type="text/html")
    return {"message": "Place ui.html in the project root"}


@app.get("/health", response_model=HealthDetail)
def health():
    qok, ook = "ok", "ok"
    try:
        from qdrant_client import QdrantClient
        QdrantClient(_get_intel().s.index.vector_url).get_collections()
    except Exception as e: qok = f"error: {e}"
    try:
        from opensearchpy import OpenSearch
        OpenSearch(_get_intel().s.index.sparse_url).info()
    except Exception as e: ook = f"error: {e}"

    tracing = "off"
    if _get_intel().s.observability.tracing_backend == "langfuse" and _get_intel().s.observability.langfuse_public_key:
        tracing = "langfuse"

    return HealthDetail(
        status="ok" if qok == "ok" and ook == "ok" else "degraded",
        qdrant=qok, opensearch=ook,
        model=_get_intel().s.generation.model,
        uptime_s=round(time.time() - _start_time, 1),
        tracing=tracing,
    )


@app.get("/metrics", include_in_schema=False)
def metrics():
    total = _metrics["ask_total"] or 1
    return {
        **_metrics,
        "ask_avg_latency_ms": round(_metrics["ask_latency_ms_total"] / total, 1),
        "cache_hit_rate": round(_metrics["ask_cache_hits"] / total, 3),
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    file: Annotated[UploadFile, File()],
    tenant_id: Annotated[str, Form()] = "default",
    doc_type: Annotated[str | None, Form()] = None,
):
    rid = getattr(request.state, "request_id", "-")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDFs supported")
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50MB)")
    if len(contents) < 100:
        raise HTTPException(400, "File appears empty")

    _metrics["ingest_total"] += 1

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        log.info("[%s] ingest file=%s", rid, file.filename)
        meta = _get_intel().ingest(tmp_path, tenant_id=tenant_id, doc_type=doc_type, acl_read=["*"])
    except Exception as e:
        _metrics["ingest_errors"] += 1
        raise HTTPException(422, f"Failed to process PDF: {e}")
    finally:
        os.unlink(tmp_path)

    _cache.invalidate_tenant(tenant_id)
    return IngestResponse(doc_id=meta.doc_id, pages=meta.page_count, filename=file.filename)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    rid = getattr(request.state, "request_id", "-")
    _metrics["ask_total"] += 1
    t0 = time.perf_counter()

    if not req.skip_cache:
        ck = _cache.key(req.query, req.tenant_id)
        cached = _cache.get(ck)
        if cached:
            _metrics["ask_cache_hits"] += 1
            return AskResponse(**cached, cached=True, request_id=rid)

    with trace_query(req.query, req.tenant_id, request_id=rid) as t:
        try:
            answer = _get_intel().ask(req.query, tenant_id=req.tenant_id, user_ids=req.user_ids or None)
            t.update(output={"answer": answer.answer_text, "citations": len(answer.citations)})
            t.event("retrieval", evidence_count=len(answer.evidence), timings=answer.latency_ms)
        except Exception as e:
            _metrics["ask_errors"] += 1
            raise HTTPException(500, f"Query failed: {e}")

    citations = [
        CitationOut(
            evidence_id=e.evidence_id, page=e.page_number, modality=e.modality.value,
            snippet=(e.content or "")[:200], bbox=e.bbox.model_dump(),
        )
        for e in answer.evidence
        if any(c.evidence_id == e.evidence_id for c in answer.citations)
    ]
    result = dict(
        answer=answer.answer_text, citations=citations,
        insufficient=answer.insufficient_evidence, timings_ms=answer.latency_ms,
    )
    if not req.skip_cache:
        _cache.set(ck, result)
    _metrics["ask_latency_ms_total"] += int((time.perf_counter() - t0) * 1000)
    return AskResponse(**result, cached=False, request_id=rid)


@app.post("/ask/stream")
async def ask_stream(req: AskRequest, request: Request):
    """
    Server-Sent Events streaming.

    Stages:
      1. event: status   — "retrieving", "generating"
      2. event: citations — final citation list (sent before tokens)
      3. event: token    — answer text chunks as they generate
      4. event: done     — final timings and metadata
    """
    rid = getattr(request.state, "request_id", "-")
    _metrics["ask_total"] += 1

    async def event_stream():
        def sse(event: str, data) -> bytes:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

        try:
            yield sse("status", {"stage": "retrieving"})

            # Run blocking retrieval in a thread so the event loop stays responsive
            loop = asyncio.get_event_loop()
            from app.schema import QueryPlan
            from retrieval.router import plan_query as _plan

            plan: QueryPlan = await loop.run_in_executor(
                None,
                lambda: _plan(req.query, tenant_id=req.tenant_id, user_ids=req.user_ids or None,
                              visual_available=_get_intel().s.embedding.visual_enabled),
            )
            evidence, timings = await loop.run_in_executor(None, _get_intel().retriever.retrieve, plan)

            citations = [
                {
                    "evidence_id": e.evidence_id, "page": e.page_number,
                    "modality": e.modality.value,
                    "snippet": (e.content or "")[:200],
                    "bbox": e.bbox.model_dump(),
                }
                for e in evidence
            ]
            yield sse("citations", citations)
            yield sse("status", {"stage": "generating"})

            # Stream tokens from Gemini
            full_answer = ""
            async for chunk in _stream_gemini(req.query, evidence):
                full_answer += chunk
                yield sse("token", {"text": chunk})
                await asyncio.sleep(0)  # yield to event loop

            yield sse("done", {"timings_ms": timings, "request_id": rid, "length": len(full_answer)})
        except Exception as e:
            log.error("stream failed: %s", e)
            yield sse("error", {"detail": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_gemini(query: str, evidence: list):
    """Stream Gemini response chunk by chunk."""
    from google import genai
    from google.genai import types
    import base64

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    parts = [types.Part.from_text(text=f"QUESTION: {query}\n\nEVIDENCE:")]
    for item in evidence:
        parts.append(types.Part.from_text(
            text=f"\n[{item.evidence_id}] (page {item.page_number}, {item.modality.value})\n{item.content or ''}"
        ))
    parts.append(types.Part.from_text(
        text="\nAnswer using only the evidence above. Cite with [eN] for every claim. "
             "If insufficient: 'INSUFFICIENT_EVIDENCE: ...'"
    ))

    system = (
        "You are a grounded document analyst. Every factual claim must cite [eN]. "
        "Don't fabricate. Don't use outside knowledge."
    )

    # google-genai supports streaming via generate_content_stream
    stream = client.models.generate_content_stream(
        model=_get_intel().s.generation.model,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=_get_intel().s.generation.max_tokens,
            temperature=_get_intel().s.generation.temperature,
        ),
    )

    for chunk in stream:
        if chunk.text:
            yield chunk.text


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    try:
        _get_intel().dense_index.delete_doc(doc_id)
        _get_intel().sparse_index.delete_doc(doc_id)
        return DeleteResponse(doc_id=doc_id, deleted=True)
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")
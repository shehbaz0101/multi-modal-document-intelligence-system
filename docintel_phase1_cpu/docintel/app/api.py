"""
FastAPI wrapper around DocIntel.

Endpoints:
  POST /ingest   — upload a PDF, returns doc_id
  POST /ask      — query across ingested docs, returns grounded answer
  GET  /health   — infra health check
  GET  /         — serves the citation UI
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.orchestrator import DocIntel

log = logging.getLogger(__name__)
logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-5s %(name)s :: %(message)s")

_intel: DocIntel | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _intel
    log.info("starting DocIntel...")
    _intel = DocIntel()
    log.info("DocIntel ready")
    yield
    log.info("shutting down")

app = FastAPI(
    title="DocIntel API",
    description="Multi-modal document intelligence — ingest PDFs, ask grounded questions.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── UI ──
UI_PATH = Path(__file__).parent.parent / "ui.html"

@app.get("/", include_in_schema=False)
def serve_ui():
    if UI_PATH.exists():
        return FileResponse(UI_PATH, media_type="text/html")
    return {"message": "UI not found — place ui.html in the project root"}

# ── schemas ──

class IngestResponse(BaseModel):
    doc_id: str
    pages: int
    filename: str

class AskRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    user_ids: list[str] = []

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

# ── endpoints ──

@app.get("/health")
def health():
    return {"status": "ok", "model": _intel.s.generation.model if _intel else None}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: Annotated[UploadFile, File(description="PDF to ingest")],
    tenant_id: Annotated[str, Form()] = "default",
    doc_type: Annotated[str | None, Form()] = None,
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        meta = _intel.ingest(
            tmp_path,
            tenant_id=tenant_id,
            doc_type=doc_type,
            acl_read=["*"],
        )
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        doc_id=meta.doc_id,
        pages=meta.page_count,
        filename=file.filename,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer = _intel.ask(
        req.query,
        tenant_id=req.tenant_id,
        user_ids=req.user_ids or None,
    )

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

    return AskResponse(
        answer=answer.answer_text,
        citations=citations,
        insufficient=answer.insufficient_evidence,
        timings_ms=answer.latency_ms,
    )
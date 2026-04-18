"""
Layer 1 — Ingestion.

Responsibilities:
  * Compute content-addressed doc_id
  * Persist the original bytes
  * Rasterize every page to PNG at configured DPI (once, here, forever)
  * Emit a DocumentMeta + list of "raw pages" (image_uri, w, h) for the parser

Boring by design. Clever ingestion is how systems leak silently.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from app.schema import DocumentMeta, Page
from app.storage import Storage

log = logging.getLogger(__name__)


class IngestionResult:
    def __init__(self, meta: DocumentMeta, pages: list[Page]):
        self.meta = meta
        self.pages = pages  # blocks will be [] here — parser fills them in Layer 2


def ingest_pdf(
    file_bytes: bytes,
    original_filename: str,
    tenant_id: str,
    storage: Storage,
    dpi: int = 150,
    uploader_id: str | None = None,
    acl_read: list[str] | None = None,
    doc_type: str | None = None,
) -> IngestionResult:
    """
    Deterministic PDF ingestion. Same bytes in → same doc_id, same page URIs.

    Re-ingesting the same file is a no-op except for the metadata record
    (which we update, so we keep the latest acl_read/uploader).
    """
    doc_id = Storage.content_hash(file_bytes)
    log.info("Ingesting PDF doc_id=%s file=%s", doc_id, original_filename)

    # 1) Persist the original.
    original_uri = storage.put(f"originals/{doc_id}/{original_filename}", file_bytes)

    # 2) Rasterize every page and persist.
    pages: list[Page] = []
    pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))
    scale = dpi / 72  # PDF default is 72 DPI
    try:
        for page_idx in range(len(pdf)):
            pdf_page = pdf[page_idx]
            pil_image: Image.Image = pdf_page.render(scale=float(scale)).to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG", optimize=True)
            image_bytes = buf.getvalue()
            page_number = page_idx + 1
            image_uri = storage.put(
                f"pages/{doc_id}/page_{page_number:04d}.png", image_bytes
            )
            pages.append(
                Page(
                    doc_id=doc_id,
                    page_number=page_number,
                    width=pil_image.width,
                    height=pil_image.height,
                    image_uri=image_uri,
                    blocks=[],
                )
            )
    finally:
        pdf.close()

    meta = DocumentMeta(
        doc_id=doc_id,
        original_filename=original_filename,
        mime_type="application/pdf",
        source_uri=original_uri,
        uploader_id=uploader_id,
        tenant_id=tenant_id,
        acl_read=acl_read or [],
        ingested_at=datetime.now(timezone.utc),
        page_count=len(pages),
        sha256=doc_id,
        doc_type=doc_type,
    )
    return IngestionResult(meta=meta, pages=pages)


def ingest_file(
    path: str | Path,
    tenant_id: str,
    storage: Storage,
    **kwargs,
) -> IngestionResult:
    """Convenience wrapper: dispatch by extension."""
    p = Path(path)
    data = p.read_bytes()
    ext = p.suffix.lower()
    if ext == ".pdf":
        return ingest_pdf(data, p.name, tenant_id, storage, **kwargs)
    # TODO Phase 1.5: .docx, .pptx, .xlsx, .html — each has its own ingest_*
    # function that ultimately produces the same Page[] shape.
    raise NotImplementedError(f"Ingestion not yet implemented for {ext}")
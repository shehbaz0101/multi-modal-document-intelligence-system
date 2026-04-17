"""
Canonical data schema for the document intelligence system.

Everything downstream keys off these types. The unit of citation is the Block;
the unit of storage is the Page; the unit of retrieval is the Block (text) or
the Page (visual, Phase 2).

Do not add fields here without a migration plan.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


# ---------- IDs ----------

# doc_id is a content hash (sha256 of the file bytes) — gives us free dedup
# and an immutable audit trail. Never use filename as an ID.
DocId = str


class BlockType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    LIST = "list"


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


# ---------- Core objects ----------

class BBox(BaseModel):
    """Axis-aligned bounding box in page pixel coordinates (top-left origin)."""
    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class Block(BaseModel):
    """A parsed region on a page. This is the unit of citation."""
    block_id: str                     # e.g. "p7_b2"
    doc_id: DocId
    page_number: int                  # 1-indexed
    type: BlockType
    reading_order: int                # 0-indexed within the page
    bbox: BBox

    # Content — exactly one of these will be meaningful per block type
    text: str | None = None           # for TEXT, HEADING, CAPTION, FOOTNOTE, LIST
    table_html: str | None = None     # for TABLE
    table_markdown: str | None = None # for TABLE (index on this, display the HTML)
    figure_crop_uri: str | None = None # for FIGURE — URI to the cropped PNG
    figure_caption: str | None = None  # VLM-generated caption, see Phase 1 Layer 2

    # Provenance
    parser_confidence: float = 1.0
    captioner_confidence: float | None = None


class Page(BaseModel):
    """A page of a document. Canonical storage unit."""
    doc_id: DocId
    page_number: int
    width: int
    height: int
    image_uri: str                     # URI to the rendered page PNG (150 DPI)
    blocks: list[Block] = Field(default_factory=list)
    detected_language: str | None = None
    layout_confidence: float = 1.0


class DocumentMeta(BaseModel):
    """Per-document metadata. Drives ACLs, filtering, audit."""
    doc_id: DocId
    original_filename: str
    mime_type: str
    source_uri: str                    # where we got it from (s3://, etc.)
    uploader_id: str | None = None
    tenant_id: str                     # hard requirement for multi-tenancy
    acl_read: list[str] = Field(default_factory=list)  # user/group IDs
    ingested_at: datetime
    page_count: int
    sha256: str
    doc_type: str | None = None        # free-form: "annual_report", "invoice", ...
    tags: list[str] = Field(default_factory=list)


# ---------- Retrieval objects ----------

class EvidenceItem(BaseModel):
    """
    One piece of evidence returned to the generator. Carries everything the
    generator and the UI need: the content, the page image for the VLM, and
    the bbox for citation highlighting.
    """
    evidence_id: str                   # e.g. "e0", "e1" — prompt-local
    doc_id: DocId
    page_number: int
    block_id: str
    modality: Modality

    # For the prompt / LLM
    content: str                       # markdown of the block (or caption for figures)
    page_image_uri: str                # full-page image for the VLM
    bbox: BBox                         # for UI highlighting

    # Provenance — useful for debugging, observability, and failure-mode UIs
    retrieval_scores: dict[str, float] = Field(default_factory=dict)  # {"bm25": .., "dense": ..}
    rerank_score: float | None = None
    source_index: str                  # which index surfaced this

    # Display-side metadata (helps the UI without another lookup)
    doc_title: str | None = None


class Citation(BaseModel):
    """A citation embedded in a generated answer."""
    evidence_id: str
    quote: str | None = None           # the span in the answer that cites this


class Answer(BaseModel):
    """The final grounded response."""
    query: str
    answer_text: str                   # with inline [e0], [e1] citation markers
    citations: list[Citation]
    evidence: list[EvidenceItem]       # the pack that produced this answer
    model: str
    latency_ms: dict[str, int] = Field(default_factory=dict)  # per-stage timing
    insufficient_evidence: bool = False


# ---------- Query objects ----------

class QueryIntent(str, Enum):
    FACTUAL = "factual"          # "what was Q3 revenue"
    ANALYTICAL = "analytical"    # "how did margins change YoY"
    VISUAL = "visual"            # "show me the chart of..."
    MIXED = "mixed"


class QueryPlan(BaseModel):
    """Output of the query router (Phase 3). Phase 1 uses a default plan."""
    query: str
    intent: QueryIntent = QueryIntent.MIXED
    filters: dict[str, Any] = Field(default_factory=dict)  # tenant_id, date, doc_type
    weights: dict[str, float] = Field(
        default_factory=lambda: {"bm25": 0.4, "dense": 0.6, "visual": 0.0}
    )
    top_k_per_index: int = 50
    final_k: int = 8

"""
Docling parser implementation.

Docling is our Phase 1 default: on-prem, predictable latency, strong table
extraction (TableFormer), and good layout analysis (DocLayNet). Swap in
LlamaParse or Reducto via the registry in parsing/base.py when you need
their strengths (e.g. LlamaParse for messy scans at scale).
"""
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from app.schema import BBox, Block, BlockType, Page
from parsing.base import Parser

log = logging.getLogger(__name__)


# Docling block-type → our BlockType. Names drift between Docling versions;
# keep this mapping in one place.
_DOCLING_TYPE_MAP: dict[str, BlockType] = {
    "text": BlockType.TEXT,
    "paragraph": BlockType.TEXT,
    "title": BlockType.HEADING,
    "section-header": BlockType.HEADING,
    "heading": BlockType.HEADING,
    "table": BlockType.TABLE,
    "figure": BlockType.FIGURE,
    "picture": BlockType.FIGURE,
    "caption": BlockType.CAPTION,
    "footnote": BlockType.FOOTNOTE,
    "list-item": BlockType.LIST,
}


class DoclingParser(Parser):
    name = "docling"

    def __init__(self) -> None:
        # Lazy import so the rest of the system works without Docling installed.
        from docling.document_converter import DocumentConverter
        self._converter = DocumentConverter()

    def parse(self, pages: list[Page]) -> list[Page]:
        """
        Docling parses whole documents, not page-by-page. We group the pages
        we got from ingestion by doc_id, then ask Docling to process each
        doc's original PDF (which we already have at meta.source_uri).

        For the Phase 1 scaffold we take a simpler approach: reconstruct a
        PDF from the page images we already rasterized. This is lossy (we
        lose embedded vector text and Docling has to re-OCR), so in Phase 2
        we pass the original PDF directly via the Storage layer. For now,
        correctness over speed.
        """
        if not pages:
            return pages

        # Group by doc_id — each doc is parsed as a unit so Docling can see
        # cross-page context (headers, footers, continuing tables).
        by_doc: dict[str, list[Page]] = {}
        for p in pages:
            by_doc.setdefault(p.doc_id, []).append(p)

        for doc_id, doc_pages in by_doc.items():
            doc_pages.sort(key=lambda p: p.page_number)
            self._parse_one_doc(doc_pages)

        return pages

    def _parse_one_doc(self, doc_pages: list[Page]) -> None:
        """
        Parse one document's pages. In production, point Docling at the
        original PDF (meta.source_uri). This scaffold version parses the
        page images — swap in the PDF path when wiring the pipeline.
        """
        # Production path (recommended):
        #   result = self._converter.convert(source=original_pdf_path)
        #
        # The result object exposes .document with pages, each having items
        # (text, tables, figures) with bounding boxes. We translate those
        # into our Block schema.

        # For the runnable scaffold, show the translation shape:
        for page in doc_pages:
            raw_items = self._fetch_docling_items_for_page(page)  # see method below
            for order, item in enumerate(raw_items):
                block = self._item_to_block(
                    item=item,
                    doc_id=page.doc_id,
                    page_number=page.page_number,
                    reading_order=order,
                )
                if block is not None:
                    page.blocks.append(block)

    def _fetch_docling_items_for_page(self, page: Page) -> list[dict[str, Any]]:
        """
        Placeholder: returns the list of Docling items for one page.

        In the real implementation you've already called convert() on the
        whole PDF, and here you just return the items Docling emitted for
        this page_number. We keep it as a method so tests can inject a
        fixture list without running Docling.
        """
        # Tests override this. The default returns [] so the pipeline runs
        # end-to-end without Docling installed (useful for CI smoke tests).
        return []

    @staticmethod
    def _item_to_block(
        item: dict[str, Any],
        doc_id: str,
        page_number: int,
        reading_order: int,
    ) -> Block | None:
        """Translate one Docling item into our Block. Unknown types → None."""
        raw_type = (item.get("type") or item.get("label") or "").lower()
        block_type = _DOCLING_TYPE_MAP.get(raw_type)
        if block_type is None:
            return None

        bbox_raw = item.get("bbox") or item.get("bounding_box") or [0, 0, 0, 0]
        bbox = BBox(x0=bbox_raw[0], y0=bbox_raw[1], x1=bbox_raw[2], y1=bbox_raw[3])
        block_id = f"p{page_number}_b{reading_order}"

        block = Block(
            block_id=block_id,
            doc_id=doc_id,
            page_number=page_number,
            type=block_type,
            reading_order=reading_order,
            bbox=bbox,
            parser_confidence=float(item.get("confidence", 1.0)),
        )

        if block_type == BlockType.TABLE:
            block.table_html = item.get("html") or item.get("table_html")
            block.table_markdown = item.get("markdown") or item.get("table_markdown")
            # Table text still goes in .text for BM25 to find cell values.
            block.text = item.get("text") or block.table_markdown
        elif block_type == BlockType.FIGURE:
            block.figure_crop_uri = item.get("crop_uri")
            # figure_caption stays None — the Captioner fills it in.
        else:
            block.text = item.get("text", "")

        return block

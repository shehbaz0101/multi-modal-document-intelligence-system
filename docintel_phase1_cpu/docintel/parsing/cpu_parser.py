"""
CPU-native PDF parser — pymupdf4llm + pdfplumber.
No ML models, no GPU required.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import pymupdf4llm
import pdfplumber

from app.schema import BBox, Block, BlockType, Page
from app.storage import Storage
from parsing.base import Parser

log = logging.getLogger(__name__)


class CPUParser(Parser):
    name = "cpu"

    def __init__(self, storage: Storage, ocr_dpi: int = 200):
        self.storage = storage
        self.ocr_dpi = ocr_dpi

    def parse(self, pages: list[Page]) -> list[Page]:
        by_doc: dict[str, list[Page]] = {}
        for p in pages:
            by_doc.setdefault(p.doc_id, []).append(p)
        for doc_id, doc_pages in by_doc.items():
            doc_pages.sort(key=lambda p: p.page_number)
            self._parse_doc(doc_id, doc_pages)
        return pages

    def _parse_doc(self, doc_id: str, doc_pages: list[Page]) -> None:
        original_uri = self._find_original(doc_id)
        if not original_uri:
            log.warning("original PDF not found for doc_id=%s", doc_id)
            return

        pdf_bytes = self.storage.get(original_uri)

        # pymupdf4llm needs a file path, not a BytesIO object
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            md_pages: list[Any] = pymupdf4llm.to_markdown(
                tmp_path,
                page_chunks=True,
                show_progress=False,
            )
        finally:
            os.unlink(tmp_path)

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf_lb:
            for page_obj in doc_pages:
                page_idx = page_obj.page_number - 1
                md_chunk: dict[str, Any] = md_pages[page_idx] if page_idx < len(md_pages) else {}
                lb_page: Any = pdf_lb.pages[page_idx] if page_idx < len(pdf_lb.pages) else None
                self._build_blocks(page_obj, md_chunk, lb_page)

    def _build_blocks(self, page: Page, md_chunk: dict[str, Any], lb_page: Any) -> None:
        order = 0

        # 1) Tables via pdfplumber
        if lb_page is not None:
            try:
                tables = lb_page.extract_tables() or []
                for tbl in tables:
                    if not tbl or not any(any(c for c in row) for row in tbl):
                        continue
                    md_table = self._table_to_markdown(tbl)
                    html_table = self._table_to_html(tbl)
                    bbox = self._lb_table_bbox(lb_page, page)
                    block = Block(
                        block_id=f"p{page.page_number}_b{order}",
                        doc_id=page.doc_id,
                        page_number=page.page_number,
                        type=BlockType.TABLE,
                        reading_order=order,
                        bbox=bbox,
                        table_markdown=md_table,
                        table_html=html_table,
                        text=md_table,
                    )
                    page.blocks.append(block)
                    order += 1
            except Exception as e:
                log.warning("table extraction failed p%d: %s", page.page_number, e)

        # 2) Text blocks from pymupdf markdown
        raw_md: str = md_chunk.get("text", "") if md_chunk else ""
        for para in raw_md.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            block_type = BlockType.HEADING if para.startswith("#") else BlockType.TEXT
            text = para.lstrip("# ").strip()
            if not text:
                continue
            block = Block(
                block_id=f"p{page.page_number}_b{order}",
                doc_id=page.doc_id,
                page_number=page.page_number,
                type=block_type,
                reading_order=order,
                bbox=BBox(x0=0, y0=0, x1=float(page.width), y1=float(page.height)),
                text=text,
            )
            page.blocks.append(block)
            order += 1

        # 3) Figure detection
        self._detect_figures(page, md_chunk, order)

    def _detect_figures(self, page: Page, md_chunk: dict[str, Any], start_order: int) -> None:
        images: list[Any] = md_chunk.get("images", []) if md_chunk else []
        order = start_order
        for img in images:
            if not isinstance(img, dict):
                continue
            w = float(img.get("width", 0))
            h = float(img.get("height", 0))
            if w * h < 5000:
                continue
            block = Block(
                block_id=f"p{page.page_number}_b{order}",
                doc_id=page.doc_id,
                page_number=page.page_number,
                type=BlockType.FIGURE,
                reading_order=order,
                bbox=BBox(
                    x0=float(img.get("x0", 0)),
                    y0=float(img.get("y0", 0)),
                    x1=float(img.get("x1", w)),
                    y1=float(img.get("y1", h)),
                ),
            )
            page.blocks.append(block)
            order += 1

    def _find_original(self, doc_id: str) -> str | None:
        storage_root: Any = getattr(self.storage, "root", None)
        if storage_root is not None:
            d = Path(str(storage_root)) / "originals" / doc_id
            if d.exists():
                for f in d.iterdir():
                    if f.suffix.lower() == ".pdf":
                        return f"file://{f}"
        return None

    @staticmethod
    def _table_to_markdown(rows: list[list[Any]]) -> str:
        if not rows:
            return ""
        def cell(c: Any) -> str:
            return (str(c) if c is not None else "").replace("|", "\\|").replace("\n", " ")
        header = "| " + " | ".join(cell(c) for c in rows[0]) + " |"
        sep    = "| " + " | ".join("---" for _ in rows[0]) + " |"
        body   = "\n".join(
            "| " + " | ".join(cell(c) for c in row) + " |" for row in rows[1:]
        )
        return "\n".join([header, sep, body])

    @staticmethod
    def _table_to_html(rows: list[list[Any]]) -> str:
        def cell(c: Any) -> str:
            return f"<td>{c if c is not None else ''}</td>"
        header = "<tr>" + "".join(
            f"<th>{c if c is not None else ''}</th>" for c in rows[0]
        ) + "</tr>"
        body = "".join(
            "<tr>" + "".join(cell(c) for c in row) + "</tr>" for row in rows[1:]
        )
        return f"<table>{header}{body}</table>"

    @staticmethod
    def _lb_table_bbox(lb_page: Any, page: Page) -> BBox:
        try:
            tbls = lb_page.find_tables()
            if tbls:
                b = tbls[0].bbox
                ph = float(lb_page.height)
                sx = float(page.width) / float(lb_page.width)
                sy = float(page.height) / ph
                return BBox(
                    x0=float(b[0]) * sx,
                    y0=(ph - float(b[3])) * sy,
                    x1=float(b[2]) * sx,
                    y1=(ph - float(b[1])) * sy,
                )
        except Exception:
            pass
        return BBox(x0=0, y0=0, x1=float(page.width), y1=float(page.height))
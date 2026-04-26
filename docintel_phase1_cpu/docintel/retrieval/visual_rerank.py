"""
Phase 2 — Gemini Visual Reranker.

No GPU required. No ColPali. No extra infrastructure.

Design:
  1. Text retrieval (BM25 + dense) surfaces top-N candidate pages
  2. For visual queries ("show me the chart", "compare the table"),
     we send the page images + query to Gemini and ask it to rank
     which pages actually contain the requested visual content
  3. Pages that Gemini scores highly are boosted in the final ranking

This gives ~80% of the ColPali benefit at zero infra cost, and it works
on any hardware because Gemini runs on Google's servers.

Trade-offs vs ColPali:
  + No GPU needed, no self-hosting
  + Zero cold-start — uses the Gemini key you already have
  + Benefits from Gemini's strong VLM understanding of docs
  - Slower per query (~1-3s for 10 pages vs <100ms for ColPali)
  - Costs Gemini tokens per rerank (cheap on Flash-Lite)
  - Not useful for massive corpora (>10k unique pages per query)

Enable in .env:
  VISUAL_RERANK_ENABLED=true
  VISUAL_RERANK_MAX_PAGES=10
  VISUAL_RERANK_MODEL=gemini-2.5-flash-lite
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Any

from app.schema import Modality
from app.storage import Storage
from indexing.base import IndexHit

log = logging.getLogger(__name__)


# Page-ranking prompt — asks Gemini to score each page 0-10
_RANK_PROMPT = """\
You are a visual document retrieval system. The user asked this question:

QUESTION: {query}

Below are {n} page images from indexed documents, each labeled [PAGE_N].

For each page, output a relevance score from 0 to 10:
  0-2 = page is completely irrelevant
  3-5 = page mentions related topic but doesn't answer the question
  6-8 = page contains partial evidence (e.g. related chart/table)
  9-10 = page directly answers the question (e.g. the exact chart/table/diagram)

Score based on what you can SEE in the image — actual charts, tables, diagrams, \
visible data. Not just text overlap.

Output format — one line per page, nothing else:
PAGE_0: 8
PAGE_1: 3
PAGE_2: 10
...
"""

_SCORE_RE = re.compile(r"PAGE_(\d+):\s*(\d+(?:\.\d+)?)")


# Queries that benefit from visual reranking
_VISUAL_HINTS = re.compile(
    r"\b(show|chart|graph|diagram|figure|image|picture|table|"
    r"illustration|photo|visual|plot|infographic|schematic)\b",
    re.IGNORECASE,
)


def should_visual_rerank(query: str) -> bool:
    """Heuristic: does this query want visual information?"""
    return bool(_VISUAL_HINTS.search(query))


class GeminiVisualReranker:
    """
    Reranks the top-N retrieval hits by asking Gemini to score each page
    image for relevance to the query. Boosts pages that Gemini says
    actually contain the requested visual content.
    """

    def __init__(
        self,
        model: str,
        storage: Storage,
        api_key: str | None = None,
        max_pages: int = 10,
        score_weight: float = 0.7,
    ):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._genai = genai
        self.model = model
        self.storage = storage
        self.max_pages = max_pages
        self.score_weight = score_weight  # how much the visual score influences final rank

    def rerank(self, query: str, hits: list[IndexHit], top_k: int) -> list[IndexHit]:
        """
        Rerank hits by Gemini visual relevance.

        Only processes up to max_pages distinct pages. Deduplicates by
        (doc_id, page_number) because multiple blocks on the same page
        would waste Gemini calls.
        """
        if not hits:
            return hits

        # Dedupe to unique pages, preserving rank order
        seen: set[tuple[str, int]] = set()
        page_hits: list[IndexHit] = []
        for h in hits:
            key = (h.doc_id, h.page_number)
            if key in seen:
                continue
            seen.add(key)
            page_hits.append(h)
            if len(page_hits) >= self.max_pages:
                break

        # Load page images
        images: list[tuple[IndexHit, bytes | None]] = []
        for h in page_hits:
            img_bytes = None
            uri = h.payload.get("page_image_uri") or self._find_page_image(h)
            if uri:
                try:
                    img_bytes = self.storage.get(uri)
                except Exception as e:
                    log.warning("failed to load page image %s: %s", uri, e)
            images.append((h, img_bytes))

        # Only keep hits with images
        ranked: list[tuple[IndexHit, bytes]] = [(h, b) for h, b in images if b is not None]
        if not ranked:
            log.info("visual rerank: no page images available, skipping")
            return hits[:top_k]

        # Ask Gemini for scores
        scores = self._score_pages(query, [b for _, b in ranked])

        # Blend retrieval rank with visual score
        for i, ((hit, _), vscore) in enumerate(zip(ranked, scores)):
            # normalize: original rank → 1.0 at top, 0.0 at bottom
            rank_score = 1.0 - (i / max(len(ranked) - 1, 1))
            visual_score = vscore / 10.0
            blended = (1 - self.score_weight) * rank_score + self.score_weight * visual_score
            hit.score = blended
            hit.payload["_visual_score"] = vscore
            hit.payload["_rerank_score"] = blended

        # Sort by blended score, then merge back with the non-reranked tail
        reranked_hits = [h for h, _ in ranked]
        reranked_hits.sort(key=lambda h: h.score, reverse=True)

        # Pages that weren't reranked keep their original position at the end
        reranked_page_ids = {(h.doc_id, h.page_number) for h in reranked_hits}
        untouched = [h for h in hits if (h.doc_id, h.page_number) not in reranked_page_ids]

        return (reranked_hits + untouched)[:top_k]

    def _score_pages(self, query: str, image_bytes_list: list[bytes]) -> list[float]:
        """Single Gemini call — scores all pages at once."""
        types = self._genai.types
        prompt_text = _RANK_PROMPT.format(query=query, n=len(image_bytes_list))

        contents: list[Any] = [types.Part.from_text(text=prompt_text)]
        for i, img_bytes in enumerate(image_bytes_list):
            contents.append(types.Part.from_text(text=f"\n[PAGE_{i}]"))
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=200,
                    temperature=0.0,
                ),
            )
            text = (response.text or "").strip()
            return self._parse_scores(text, len(image_bytes_list))
        except Exception as e:
            log.warning("visual rerank call failed, using neutral scores: %s", e)
            return [5.0] * len(image_bytes_list)

    @staticmethod
    def _parse_scores(text: str, n: int) -> list[float]:
        """Parse 'PAGE_N: score' lines from Gemini response."""
        scores = [5.0] * n  # default neutral
        for match in _SCORE_RE.finditer(text):
            idx = int(match.group(1))
            val = float(match.group(2))
            if 0 <= idx < n:
                scores[idx] = max(0.0, min(10.0, val))
        return scores

    def _find_page_image(self, hit: IndexHit) -> str | None:
        """
        Fallback when page_image_uri isn't in the payload — construct it
        from the storage layout convention.
        """
        storage_root = getattr(self.storage, "root", None)
        if storage_root is None:
            return None
        from pathlib import Path
        path = Path(str(storage_root)) / "pages" / hit.doc_id / f"page_{hit.page_number:04d}.png"
        if path.exists():
            return f"file://{path}"
        return None
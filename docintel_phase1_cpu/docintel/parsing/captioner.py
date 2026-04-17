"""
Figure captioner — Gemini vision.

For every FIGURE block, generate a dense natural-language caption at ingest
time so the figure becomes findable by text search. Stored on the block,
indexed in the sparse index.

"show me the revenue growth chart" → text hit via the caption.
No ColPali required for this to work.
"""
from __future__ import annotations

import logging
from typing import Protocol

from app.schema import Block, BlockType, Page
from app.storage import Storage

log = logging.getLogger(__name__)


_CAPTION_PROMPT = """\
You are indexing figures from enterprise documents. Write a caption for
the image provided that lets a search engine find this figure when a user
asks about its content.

Write 2-4 sentences. Include:
- What kind of figure it is (bar chart, line graph, diagram, table, etc.)
- The axes, legend, or entities shown by name
- The main quantitative takeaway (numbers, trends) if visible
- Any labeled entities (company names, regions, product names)

No preambles like "This image shows...". If the figure is unreadable, say
so in one sentence.\
"""


class VLMClient(Protocol):
    def caption_image(self, image_bytes: bytes, prompt: str) -> tuple[str, float]: ...


class GeminiVLMClient:
    """Google Gemini vision client. Replaces the Anthropic Claude client."""

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 200):
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model = model
        self.max_tokens = max_tokens

    def caption_image(self, image_bytes: bytes, prompt: str) -> tuple[str, float]:
        types = self._types
        response = self._client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(max_output_tokens=self.max_tokens),
        )
        text = (response.text or "").strip()
        confidence = 0.9 if text and "unreadable" not in text.lower() else 0.3
        return text, confidence


class Captioner:
    def __init__(self, vlm: VLMClient, storage: Storage):
        self.vlm = vlm
        self.storage = storage

    def caption_pages(self, pages: list[Page]) -> list[Page]:
        for page in pages:
            self._caption_page(page)
        return pages

    def _caption_page(self, page: Page) -> None:
        for block in page.blocks:
            if block.type != BlockType.FIGURE:
                continue
            if block.figure_caption:
                continue
            try:
                image_bytes = self._load_figure_image(block, page)
            except Exception as e:
                log.warning("figure image load failed block=%s: %s", block.block_id, e)
                continue
            try:
                caption, conf = self.vlm.caption_image(image_bytes, _CAPTION_PROMPT)
                block.figure_caption = caption
                block.captioner_confidence = conf
                block.text = (block.text + "\n" if block.text else "") + caption
            except Exception as e:
                log.warning("caption failed block=%s: %s", block.block_id, e)

    def _load_figure_image(self, block: Block, page: Page) -> bytes:
        if block.figure_crop_uri:
            return self.storage.get(block.figure_crop_uri)
        from PIL import Image
        import io
        page_bytes = self.storage.get(page.image_uri)
        img = Image.open(io.BytesIO(page_bytes))
        crop = img.crop(block.bbox.as_tuple())
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

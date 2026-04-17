"""
Layer 6 — Grounded generation via Google Gemini.

Uses gemini-2.0-flash by default — fast, cheap, vision-capable, and
works well on evidence-grounded QA. Swap to gemini-2.5-pro in the config
for the highest-stakes queries (finance/legal/medical).

Grounding contract:
  * Every factual claim must cite at least one [eN]
  * If evidence is insufficient → "INSUFFICIENT_EVIDENCE: ..."
  * Page images are attached for figure/table evidence so the model
    can actually read what's on the chart/table
"""
from __future__ import annotations

import base64
import logging
import re

from app.schema import Answer, Citation, EvidenceItem, Modality
from app.storage import Storage

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a grounded document analyst. Answer questions strictly from the
provided evidence items, each labeled [e0], [e1], and so on.

Rules — follow them exactly:
1. Every factual claim MUST cite the evidence that supports it using
   inline markers like [e0] or [e0, e2]. Do not fabricate evidence IDs.
2. If the evidence does not contain enough information, reply with exactly:
   "INSUFFICIENT_EVIDENCE: <brief explanation of what is missing>"
3. Do not use outside knowledge. Trust the evidence over what you know.
4. When a figure or table is attached as an image, read the numbers and
   labels from the image itself — do not assume or paraphrase.
5. Keep answers concise. One paragraph for simple questions, a few short
   paragraphs for analytical ones.\
"""

_CITATION_RE = re.compile(r"\[e(\d+)(?:\s*,\s*e?(\d+))*\]")


class GroundedGenerator:
    def __init__(
        self,
        *,
        model: str,
        storage: Storage,
        api_key: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ):
        from google import genai
        from google.genai import types

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.storage = storage
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str, evidence: list[EvidenceItem]) -> Answer:
        if not evidence:
            return Answer(
                query=query,
                answer_text="INSUFFICIENT_EVIDENCE: no documents matched the query.",
                citations=[],
                evidence=[],
                model=self.model,
                insufficient_evidence=True,
            )

        contents = self._build_contents(query, evidence)

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )

        text = response.text.strip() if response.text else ""
        insufficient = text.startswith("INSUFFICIENT_EVIDENCE")
        citations = self._extract_citations(text, evidence)

        return Answer(
            query=query,
            answer_text=text,
            citations=citations,
            evidence=evidence,
            model=self.model,
            insufficient_evidence=insufficient,
        )

    # ---- prompt construction ----

    def _build_contents(self, query: str, evidence: list[EvidenceItem]) -> list:
        """
        Build a Gemini contents list that interleaves text and images.
        Gemini's SDK accepts a flat list of Part objects.
        """
        types = self._types
        parts = [types.Part.from_text(text=f"QUESTION: {query}\n\nEVIDENCE:")]

        for item in evidence:
            header = (
                f"\n[{item.evidence_id}] "
                f"(doc: {item.doc_title or item.doc_id[:12]}, "
                f"page: {item.page_number}, "
                f"type: {item.modality.value})"
            )
            parts.append(types.Part.from_text(text=header))

            if item.content:
                parts.append(types.Part.from_text(text=item.content))

            # Attach page image for visual evidence — Gemini reads the chart/table directly
            if item.modality in (Modality.FIGURE, Modality.TABLE) and item.page_image_uri:
                try:
                    img_bytes = self.storage.get(item.page_image_uri)
                    parts.append(
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                    )
                    parts.append(
                        types.Part.from_text(
                            text=f"(image above is the page for {item.evidence_id})"
                        )
                    )
                except Exception as e:
                    log.warning("failed to attach page image for %s: %s", item.evidence_id, e)

        parts.append(
            types.Part.from_text(text="\nAnswer using only the evidence above.")
        )
        return parts

    @staticmethod
    def _extract_citations(text: str, evidence: list[EvidenceItem]) -> list[Citation]:
        valid_ids = {e.evidence_id for e in evidence}
        cited: list[Citation] = []
        seen: set[str] = set()
        for match in _CITATION_RE.finditer(text):
            for grp in match.groups():
                if grp is None:
                    continue
                eid = f"e{grp}"
                if eid in valid_ids and eid not in seen:
                    cited.append(Citation(evidence_id=eid))
                    seen.add(eid)
        return cited

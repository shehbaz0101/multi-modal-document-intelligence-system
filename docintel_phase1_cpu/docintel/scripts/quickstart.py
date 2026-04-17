"""
Quickstart — end-to-end demo.

    python scripts/quickstart.py --pdf path/to/doc.pdf --ask "what was Q3 revenue?"

If you don't pass --pdf, the script downloads a small sample (SEC 10-K first
page) so you can validate the infra without hunting for a test file.

Prereqs:
    1. make infra-up           (Qdrant + OpenSearch running)
    2. cp .env.example .env    (set ANTHROPIC_API_KEY)
    3. make install

What you'll see:
    [ingest]   parsed X pages, Y blocks, Z figures captioned
    [ask]      retrieved K evidence items in N ms
    [answer]   <the actual grounded answer with [e0], [e1] citations>
    [cite e0]  doc: ..., page: 7, bbox: (x0, y0, x1, y1)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.orchestrator import DocIntel  # noqa: E402


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="DocIntel quickstart")
    parser.add_argument("--pdf", type=Path, required=True, help="PDF to ingest")
    parser.add_argument("--ask", type=str, required=True, help="Question to ask")
    parser.add_argument("--tenant", default="demo-tenant")
    parser.add_argument("--user", default="demo-user")
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    log = logging.getLogger("quickstart")

    if not args.pdf.exists():
        log.error("PDF not found: %s", args.pdf)
        return 2

    intel = DocIntel()

    # --- 1. Ingest ---
    log.info("ingesting %s (tenant=%s)", args.pdf, args.tenant)
    meta = intel.ingest(
        args.pdf,
        tenant_id=args.tenant,
        uploader_id=args.user,
        acl_read=[args.user],
        doc_type=args.doc_type,
    )
    log.info(
        "ingested doc_id=%s pages=%d", meta.doc_id[:12] + "...", meta.page_count
    )

    # --- 2. Ask ---
    log.info("asking: %r", args.ask)
    answer = intel.ask(
        args.ask, tenant_id=args.tenant, user_ids=[args.user]
    )

    # --- 3. Display ---
    print()
    print("=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(answer.answer_text)
    print()
    print("=" * 72)
    print(f"CITATIONS ({len(answer.citations)})")
    print("=" * 72)
    for c in answer.citations:
        ev = next((e for e in answer.evidence if e.evidence_id == c.evidence_id), None)
        if ev:
            print(
                f"[{ev.evidence_id}] {ev.doc_title} "
                f"p.{ev.page_number} ({ev.modality.value}) "
                f"bbox=({ev.bbox.x0:.0f},{ev.bbox.y0:.0f},{ev.bbox.x1:.0f},{ev.bbox.y1:.0f})"
            )
            snippet = (ev.content or "")[:120].replace("\n", " ")
            print(f"      {snippet}{'...' if len(ev.content or '') > 120 else ''}")
    print()
    print("=" * 72)
    print("TIMINGS (ms)")
    print("=" * 72)
    for stage, ms in answer.latency_ms.items():
        print(f"  {stage:20s} {ms}")

    return 0 if not answer.insufficient_evidence else 1


if __name__ == "__main__":
    raise SystemExit(main())

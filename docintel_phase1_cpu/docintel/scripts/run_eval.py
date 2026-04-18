"""
Eval runner — generates a numbers table for the README.

Usage:
  python scripts/run_eval.py --pdf path/to/doc.pdf

This script:
  1. Ingests the PDF
  2. Runs 5 hand-crafted questions against it
  3. Prints a table: citation validity, keyword coverage, timings

For a real eval set, create data/eval/set.jsonl using the EvalCase schema
in evaluation/harness.py and run:
  python -m evaluation.harness --eval data/eval/set.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.orchestrator import DocIntel
from evaluation.harness import citation_metrics, keyword_coverage


QUESTIONS = [
    ("summary",    "What is this document about?",           ["document", "about"]),
    ("skills",     "What are the key skills or topics?",     ["skill", "experience", "technology"]),
    ("details",    "What specific details are mentioned?",   []),
    ("structure",  "How is this document structured?",       []),
    ("main point", "What is the most important takeaway?",   []),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path)
    args = parser.parse_args()

    intel = DocIntel()
    print(f"\nIngesting {args.pdf.name}...")
    meta = intel.ingest(args.pdf, tenant_id="eval", acl_read=["*"])
    print(f"Ingested: {meta.page_count} pages, doc_id={meta.doc_id[:12]}...\n")

    print(f"{'Question':<20} {'Validity':>8} {'Coverage':>9} {'Keywords':>9} {'Time(ms)':>9}")
    print("-" * 60)

    total_v, total_c, total_k = 0.0, 0.0, 0.0

    for label, query, keywords in QUESTIONS:
        answer = intel.ask(query, tenant_id="eval")
        v, c = citation_metrics(answer)
        k = keyword_coverage(answer.answer_text, keywords)
        t = sum(answer.latency_ms.values())
        total_v += v; total_c += c; total_k += k
        print(f"{label:<20} {v:>8.2f} {c:>9.2f} {k:>9.2f} {t:>9}")

    n = len(QUESTIONS)
    print("-" * 60)
    print(f"{'AVERAGE':<20} {total_v/n:>8.2f} {total_c/n:>9.2f} {total_k/n:>9.2f}")
    print("\nValidity  = fraction of [eN] markers pointing to real evidence (1.0 = perfect)")
    print("Coverage  = fraction of sentences with at least one citation")
    print("Keywords  = fraction of expected keywords found in answer")


if __name__ == "__main__":
    main()
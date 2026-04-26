"""
Phase 3 — Industry-grade eval runner.

Loads cases from a JSONL file, runs them, computes metrics, prints a
table, and writes a JSON report you can diff across runs.

Usage:
  # Ingest the docs you'll evaluate against (once)
  python scripts/run_eval.py --pdf data/eval/sample.pdf --setup-only

  # Run the eval set
  python scripts/run_eval.py --eval data/eval/set.jsonl

  # Compare two runs
  python scripts/run_eval.py --compare reports/run_2026-04-24.json reports/run_2026-04-25.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.orchestrator import DocIntel
from evaluation.harness import (
    EvalCase, load_eval_set, run_eval, citation_metrics, keyword_coverage
)


def cmd_run(args: argparse.Namespace) -> int:
    eval_path = Path(args.eval)
    if not eval_path.exists():
        print(f"❌ Eval set not found: {eval_path}")
        print(f"   Create one or use --create-template to generate a starter")
        return 2

    cases = load_eval_set(eval_path)
    print(f"📋 Loaded {len(cases)} eval cases from {eval_path.name}")

    intel = DocIntel()

    def ask_fn(case: EvalCase):
        return intel.ask(
            case.query,
            tenant_id=case.tenant_id,
            user_ids=case.user_ids or None,
        )

    print(f"🚀 Running eval... (this will take a while)\n")
    t0 = time.time()
    report = run_eval(cases, ask_fn)
    elapsed = time.time() - t0

    summary = report["summary"]
    _print_summary(summary, elapsed)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip non-serializable parts before writing
        report["meta"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": intel.s.generation.model,
            "embedder": intel.s.embedding.text_model,
            "rerank_enabled": intel.s.retrieval.rerank_enabled,
            "visual_rerank_enabled": intel.s.visual_rerank.enabled,
            "elapsed_s": round(elapsed, 1),
        }
        # Convert CaseResult dicts (they have nested Answer objects)
        report["cases"] = [_serialize_case(c) for c in report["cases"]]
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Report saved: {out_path}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a_path, b_path = Path(args.compare[0]), Path(args.compare[1])
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())

    print(f"\n📊 {a_path.name} → {b_path.name}\n")
    print(f"{'Metric':<25} {'Before':>10} {'After':>10} {'Δ':>8}")
    print("-" * 60)

    keys = ["mrr", "citation_validity", "citation_coverage", "keyword_coverage"]
    for k in ("recall@1", "recall@3", "recall@5", "ndcg@5"):
        if k in a["summary"]:
            keys.append(k)

    for k in keys:
        if k not in a["summary"] or k not in b["summary"]:
            continue
        va, vb = a["summary"][k], b["summary"][k]
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"{k:<25} {va:>10.3f} {vb:>10.3f} {sign}{delta:>7.3f}")
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    out = Path(args.create_template)
    out.parent.mkdir(parents=True, exist_ok=True)
    template = """{"query_id": "q1", "query": "What is this document about?", "tenant_id": "eval", "user_ids": [], "gold_evidence_block_ids": [], "gold_answer_keywords": ["main", "topic"], "notes": "summary"}
{"query_id": "q2", "query": "What specific facts are mentioned?", "tenant_id": "eval", "user_ids": [], "gold_evidence_block_ids": [], "gold_answer_keywords": [], "notes": ""}
{"query_id": "q3", "query": "Show me any tables or charts.", "tenant_id": "eval", "user_ids": [], "gold_evidence_block_ids": [], "gold_answer_keywords": [], "notes": "visual query"}
"""
    out.write_text(template)
    print(f"✅ Template eval set written to {out}")
    print(f"   Edit it to match your documents, then run:")
    print(f"   python scripts/run_eval.py --eval {out}")
    return 0


# ── helpers ──

def _print_summary(s: dict, elapsed: float) -> None:
    n = s["n_cases"]
    print("=" * 60)
    print(f"  EVAL SUMMARY  ({n} cases · {elapsed:.1f}s)")
    print("=" * 60)
    print(f"  MRR                  : {s['mrr']:>6.3f}")
    print(f"  Citation validity    : {s['citation_validity']:>6.3f}  (1.0 = no fake citations)")
    print(f"  Citation coverage    : {s['citation_coverage']:>6.3f}  (sentences with citations)")
    print(f"  Keyword coverage     : {s['keyword_coverage']:>6.3f}  (expected words found)")
    for k in ("recall@1", "recall@3", "recall@5", "recall@10"):
        if k in s:
            print(f"  {k:<20} : {s[k]:>6.3f}")
    for k in ("ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"):
        if k in s:
            print(f"  {k:<20} : {s[k]:>6.3f}")
    print("=" * 60)


def _serialize_case(c) -> dict:
    """Convert CaseResult to a JSON-safe dict."""
    if hasattr(c, "__dict__"):
        d = c.__dict__.copy()
    else:
        d = dict(c)
    if "answer" in d and hasattr(d["answer"], "model_dump"):
        d["answer"] = d["answer"].model_dump()
    return d


def main():
    parser = argparse.ArgumentParser(
        description="DocIntel eval runner — measure retrieval and grounding quality"
    )
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run an eval set")
    p_run.add_argument("--eval", required=True, help="Path to eval set (.jsonl)")
    p_run.add_argument("--output", "-o", help="Save full report JSON to this path")

    p_cmp = sub.add_parser("compare", help="Compare two eval reports")
    p_cmp.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), required=True)

    p_tpl = sub.add_parser("template", help="Create a starter eval set")
    p_tpl.add_argument("--create-template", required=True, help="Output path")

    # Allow shorthand "python run_eval.py --eval ..."
    parser.add_argument("--eval", help=argparse.SUPPRESS)
    parser.add_argument("--output", "-o", help=argparse.SUPPRESS)
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help=argparse.SUPPRESS)
    parser.add_argument("--create-template", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.create_template:
        return cmd_template(args)
    if args.compare:
        return cmd_compare(args)
    if args.eval:
        return cmd_run(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
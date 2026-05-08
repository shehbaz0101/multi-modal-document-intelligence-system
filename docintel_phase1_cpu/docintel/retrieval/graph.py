"""
LangGraph workflow — stateful, branching pipeline for grounded QA.

Why LangGraph instead of the linear retrieve→generate path?

A linear pipeline can't:
  - Branch based on query intent (factual vs visual vs comparative)
  - Loop when retrieval finds nothing relevant (try expanded query)
  - Self-correct when the generator says "INSUFFICIENT_EVIDENCE"
  - Skip stages dynamically (e.g. skip visual rerank for plain text queries)

A graph can. The same building blocks (retriever, reranker, generator)
are reused — they're just wired together as nodes with conditional edges.

This module exposes one function:
    build_doc_graph(intel: DocIntel) -> CompiledGraph

The graph state is the GraphState TypedDict — everything flows through it.

Entry points:
    intel.ask_with_graph(query, tenant_id) -> Answer       # uses this graph
    intel.ask(query, tenant_id)            -> Answer       # original linear path
"""
from __future__ import annotations

import logging
import time
from typing import Any, TypedDict, Annotated, Literal

from app.schema import Answer, EvidenceItem, QueryIntent

log = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """
    The state passed between nodes. Each node reads what it needs and writes
    its output back. Keys with `total=False` are optional until populated.
    """
    # Inputs (set once at start)
    query: str
    original_query: str          # preserved for retries with expanded query
    tenant_id: str
    user_ids: list[str]

    # Stage outputs
    intent: QueryIntent
    plan: Any                    # QueryPlan
    evidence: list[EvidenceItem]
    answer_text: str
    answer: Answer

    # Bookkeeping
    retry_count: int
    timings_ms: dict[str, int]
    notes: list[str]             # human-readable trace of what happened


# ── Node implementations ─────────────────────────────────────────────────────
# Each node is a pure function: (state, deps) -> partial state update.
# We pass `deps` as a closure so the graph stays decoupled from DocIntel.

def _make_nodes(intel: Any) -> dict[str, Any]:
    """Build node functions bound to a DocIntel instance."""
    from retrieval.router import plan_query

    # ── Node 1: classify intent + build plan ──
    def classify(state: GraphState) -> GraphState:
        t0 = time.perf_counter()
        plan = plan_query(
            state["query"],
            tenant_id=state["tenant_id"],
            user_ids=state.get("user_ids") or None,
            visual_available=intel.s.embedding.visual_enabled,
        )
        timings = state.get("timings_ms", {})
        timings["classify_ms"] = int((time.perf_counter() - t0) * 1000)
        notes = state.get("notes", []) + [f"intent: {plan.intent.value}"]
        return {
            "intent": plan.intent,
            "plan": plan,
            "timings_ms": timings,
            "notes": notes,
        }

    # ── Node 2: retrieve evidence ──
    def retrieve(state: GraphState) -> GraphState:
        evidence, r_timings = intel.retriever.retrieve(state["plan"])
        timings = state.get("timings_ms", {})
        timings.update(r_timings)
        notes = state.get("notes", []) + [f"retrieved: {len(evidence)} items"]
        return {
            "evidence": evidence,
            "timings_ms": timings,
            "notes": notes,
        }

    # ── Node 3: expand query (only when initial retrieval finds nothing) ──
    def expand_query(state: GraphState) -> GraphState:
        """
        When retrieval returns no useful evidence, ask the LLM to rewrite
        the query to be more retrievable, then loop back to retrieve.
        """
        original = state.get("original_query") or state["query"]
        retry = state.get("retry_count", 0) + 1

        # Use a small Gemini call to rewrite. This costs ~50 tokens.
        from google import genai
        from google.genai import types
        import os

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        prompt = (
            f"The user asked: '{original}'\n"
            f"This query returned no good results from a document search. "
            f"Rewrite it to use more specific terms that might match the document. "
            f"Output ONLY the rewritten query, nothing else."
        )
        try:
            resp = client.models.generate_content(
                model=intel.s.generation.model,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=80, temperature=0.3),
            )
            new_query = (resp.text or "").strip().strip('"')
        except Exception:
            new_query = original  # if expansion fails, keep original

        notes = state.get("notes", []) + [f"expanded query (retry {retry}): {new_query}"]
        return {
            "query": new_query,
            "original_query": original,
            "retry_count": retry,
            "notes": notes,
        }

    # ── Node 4: generate grounded answer ──
    def generate(state: GraphState) -> GraphState:
        t0 = time.perf_counter()
        answer = intel.generator.generate(
            state.get("original_query") or state["query"],
            state.get("evidence", []),
        )
        timings = state.get("timings_ms", {})
        timings["generate_ms"] = int((time.perf_counter() - t0) * 1000)
        answer.latency_ms = timings
        notes = state.get("notes", []) + [
            "insufficient" if answer.insufficient_evidence else "answered"
        ]
        return {
            "answer": answer,
            "answer_text": answer.answer_text,
            "timings_ms": timings,
            "notes": notes,
        }

    return {
        "classify": classify,
        "retrieve": retrieve,
        "expand_query": expand_query,
        "generate": generate,
    }


# ── Conditional edges (the routing logic) ─────────────────────────────────────

def _should_retry(state: GraphState) -> Literal["expand", "generate"]:
    """
    After retrieval — decide whether to expand the query and try again,
    or proceed to generation.

    Retry when:
      - Got <2 evidence items AND haven't retried yet
    Otherwise proceed to generate (even with poor evidence — let the
    generator say INSUFFICIENT_EVIDENCE rather than spinning forever).
    """
    evidence = state.get("evidence", [])
    retries = state.get("retry_count", 0)
    if len(evidence) < 2 and retries < 1:
        return "expand"
    return "generate"


def _after_generate(state: GraphState) -> Literal["expand", "end"]:
    """
    After generation — if the answer was INSUFFICIENT_EVIDENCE and we
    haven't retried, try an expanded query once.
    """
    answer = state.get("answer")
    retries = state.get("retry_count", 0)
    if answer and answer.insufficient_evidence and retries < 1:
        return "expand"
    return "end"


# ── Public API ────────────────────────────────────────────────────────────────

def build_doc_graph(intel: Any):
    """
    Compile the workflow graph.

        ┌────────────┐
        │  classify  │   (intent + retrieval plan)
        └──────┬─────┘
               ▼
        ┌────────────┐
        │  retrieve  │
        └──────┬─────┘
               ▼
        ┌─────────────────┐
        │ should_retry?   │
        └──┬───────────┬──┘
       no  │           │ yes (no good hits)
           ▼           ▼
       ┌──────┐   ┌──────────────┐
       │ gen  │   │ expand_query │
       └──┬───┘   └──────┬───────┘
          │              │
          ▼              ▼
    ┌─ after_gen? ┐    (loops back to retrieve)
    │             │
    ▼             ▼
   END        expand_query
    """
    from langgraph.graph import StateGraph, END

    nodes = _make_nodes(intel)

    g = StateGraph(GraphState)
    g.add_node("classify", nodes["classify"])
    g.add_node("retrieve", nodes["retrieve"])
    g.add_node("expand_query", nodes["expand_query"])
    g.add_node("generate", nodes["generate"])

    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")

    # After retrieve: branch on retry decision
    g.add_conditional_edges(
        "retrieve",
        _should_retry,
        {"expand": "expand_query", "generate": "generate"},
    )

    # After expand: re-classify (intent might shift) → retrieve
    g.add_edge("expand_query", "classify")

    # After generate: maybe loop, otherwise end
    g.add_conditional_edges(
        "generate",
        _after_generate,
        {"expand": "expand_query", "end": END},
    )

    return g.compile()


def run_graph(intel: Any, query: str, tenant_id: str, user_ids: list[str] | None = None) -> Answer:
    """Convenience wrapper used by orchestrator.ask_with_graph()."""
    graph = build_doc_graph(intel)
    initial: GraphState = {
        "query": query,
        "original_query": query,
        "tenant_id": tenant_id,
        "user_ids": user_ids or [],
        "retry_count": 0,
        "timings_ms": {},
        "notes": [],
    }
    final_state = graph.invoke(initial)

    answer = final_state.get("answer")
    if answer is None:
        # Shouldn't happen, but guard anyway
        from app.schema import Answer as A
        answer = A(
            query=query,
            answer_text="INSUFFICIENT_EVIDENCE: graph terminated without a generation.",
            citations=[],
            evidence=[],
            model=intel.s.generation.model,
            insufficient_evidence=True,
        )

    # Attach the trace notes for debugging — visible in the UI
    answer.latency_ms = {**(answer.latency_ms or {}), **final_state.get("timings_ms", {})}
    return answer
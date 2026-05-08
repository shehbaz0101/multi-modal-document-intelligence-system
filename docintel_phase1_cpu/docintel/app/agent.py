"""
LangChain conversational agent.

Wraps the DocIntel retriever as a LangChain tool, gives the agent memory,
and exposes a multi-turn chat API. The agent decides when to retrieve,
when to ask a clarifying question, and when to answer from memory.

Why an agent instead of a stateless /ask?

A stateless endpoint can't:
  - Resolve pronouns ("what about that one?" — what's "that"?)
  - Skip retrieval when the answer is already in context
  - Ask clarifying questions ("which document do you mean?")
  - Chain multiple retrievals for compound queries

The agent uses LangChain's tool-calling API. The tool wraps your existing
retriever — nothing about retrieval changes, only the orchestration above it.

Memory model:
  - One ConversationMemory per session (keyed by session_id)
  - Sessions are in-process for now (replace with Redis for multi-instance)
  - Each turn appends user + agent messages, agent reads the whole history
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ── Session memory (in-process; swap for Redis in prod) ──────────────────────

@dataclass
class _Session:
    session_id: str
    tenant_id: str
    user_ids: list[str] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)  # [{role, content}]
    accumulated_evidence: dict[str, Any] = field(default_factory=dict)  # eid -> EvidenceItem


class _SessionStore:
    def __init__(self):
        self._sessions: dict[str, _Session] = {}

    def get_or_create(self, session_id: str, tenant_id: str, user_ids: list[str]) -> _Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = _Session(
                session_id=session_id, tenant_id=tenant_id, user_ids=user_ids,
            )
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


_sessions = _SessionStore()


# ── Tool definitions for the agent ───────────────────────────────────────────

def _make_tools(intel: Any, session: _Session) -> list:
    """
    Build LangChain Tool objects bound to a specific session.
    Each tool has a name, description, and callable. The agent decides
    when to invoke each based on the user's message and conversation state.
    """
    from langchain_core.tools import tool

    @tool
    def retrieve_from_documents(query: str) -> str:
        """
        Search the user's uploaded documents for information relevant to a query.
        Use this whenever the question requires looking up specific facts,
        numbers, names, or content from the documents.

        Returns a list of evidence snippets with citation IDs like [e0], [e1].
        Each snippet has a page number and a brief content excerpt.
        """
        from retrieval.router import plan_query
        plan = plan_query(
            query,
            tenant_id=session.tenant_id,
            user_ids=session.user_ids or None,
            visual_available=intel.s.embedding.visual_enabled,
        )
        evidence, _ = intel.retriever.retrieve(plan)

        if not evidence:
            return "No relevant evidence found in the indexed documents."

        # Stash evidence on the session so the final answer can reference it
        for e in evidence:
            session.accumulated_evidence[e.evidence_id] = e

        lines = []
        for e in evidence[:8]:
            snippet = (e.content or "")[:200].replace("\n", " ")
            lines.append(
                f"[{e.evidence_id}] (doc: {e.doc_title or e.doc_id[:8]}, "
                f"page {e.page_number}, {e.modality.value}): {snippet}"
            )
        return "\n".join(lines)

    @tool
    def list_documents() -> str:
        """
        List all documents available in the user's current workspace.
        Use this when the user asks "what documents do I have" or "what's
        available" or wants an overview of what they've uploaded.
        """
        # Query both indexes for distinct doc titles in this tenant
        try:
            from qdrant_client.http import models as qm
            results, _ = intel.dense_index._client.scroll(
                collection_name=intel.dense_index.collection,
                scroll_filter=qm.Filter(must=[
                    qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=session.tenant_id))
                ]),
                limit=200,
                with_payload=["doc_id", "doc_title"],
            )
            seen = {}
            for p in results:
                doc_id = p.payload.get("doc_id")
                title = p.payload.get("doc_title", "untitled")
                if doc_id and doc_id not in seen:
                    seen[doc_id] = title
            if not seen:
                return "No documents in the workspace yet. The user should upload some PDFs."
            return "Documents in workspace:\n" + "\n".join(
                f"- {title} (id: {did[:12]}...)" for did, title in seen.items()
            )
        except Exception as e:
            log.warning("list_documents failed: %s", e)
            return "Could not list documents."

    return [retrieve_from_documents, list_documents]


# ── The agent ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are DocIntel, a grounded document analyst chatbot.

Tools:
  - retrieve_from_documents(query): search the user's documents
  - list_documents(): see what documents are available

Rules:
  1. For ANY question about document content, you MUST call retrieve_from_documents first.
     Do NOT guess. Do NOT use prior knowledge.
  2. Every factual claim in your final answer MUST cite the evidence ID,
     like [e0] or [e1, e2]. Don't fabricate IDs — only use IDs returned by the tool.
  3. If the retrieved evidence is insufficient, say so clearly. Don't make things up.
  4. For follow-up questions like "what about that?", look at the chat history to
     resolve what "that" refers to, then retrieve.
  5. If the user asks something that doesn't need document lookup (e.g. "thanks",
     "explain that more"), answer from the chat history without calling tools.
  6. Be concise. One paragraph for simple questions, a few short paragraphs for analytical ones.
"""


@dataclass
class AgentReply:
    answer: str
    citations: list[dict]                 # [{evidence_id, page, modality, snippet, bbox}]
    session_id: str
    turns: int
    tool_calls: list[str]                 # which tools the agent called this turn


def chat(
    *,
    intel: Any,
    session_id: str,
    message: str,
    tenant_id: str = "default",
    user_ids: list[str] | None = None,
) -> AgentReply:
    """
    Send one user message to the agent. Returns the agent's reply.
    The session keeps history across calls.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
    from langgraph.prebuilt import create_react_agent

    session = _sessions.get_or_create(session_id, tenant_id, user_ids or [])
    tools = _make_tools(intel, session)

    llm = ChatGoogleGenerativeAI(
        model=intel.s.generation.model,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=intel.s.generation.temperature,
        max_output_tokens=intel.s.generation.max_tokens,
    )

    # LangGraph's prebuilt ReAct agent — handles the tool-calling loop
    agent = create_react_agent(llm, tools, prompt=_SYSTEM_PROMPT)

    # Build messages: system + prior history + new user message
    messages: list[Any] = []
    for turn in session.history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=message))

    # Invoke
    result = agent.invoke({"messages": messages})
    output_messages = result["messages"]

    # Extract final answer + tool calls made this turn
    final_answer = ""
    tool_calls_made: list[str] = []
    for m in output_messages[len(messages):]:  # only new messages from this turn
        if isinstance(m, AIMessage):
            if m.content:
                final_answer = m.content if isinstance(m.content, str) else str(m.content)
            for tc in (m.tool_calls or []):
                tool_calls_made.append(tc.get("name", "unknown"))

    # Persist history
    session.history.append({"role": "user", "content": message})
    session.history.append({"role": "assistant", "content": final_answer})

    # Build citations from accumulated evidence (referenced in final answer)
    import re
    cited_ids = set(re.findall(r"\[e(\d+)\]", final_answer))
    citations = []
    for eid in cited_ids:
        full_id = f"e{eid}"
        ev = session.accumulated_evidence.get(full_id)
        if ev:
            citations.append({
                "evidence_id": full_id,
                "page": ev.page_number,
                "modality": ev.modality.value,
                "snippet": (ev.content or "")[:200],
                "bbox": ev.bbox.model_dump(),
                "doc_title": ev.doc_title,
            })

    return AgentReply(
        answer=final_answer,
        citations=citations,
        session_id=session_id,
        turns=len(session.history) // 2,
        tool_calls=tool_calls_made,
    )


def reset_session(session_id: str) -> None:
    _sessions.reset(session_id)
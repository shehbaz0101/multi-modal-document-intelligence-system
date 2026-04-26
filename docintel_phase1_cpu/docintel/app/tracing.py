"""
Phase 3 — Langfuse tracing.

Setup:
  1. Sign up at https://cloud.langfuse.com (free tier)
  2. Create a project, grab public/secret keys
  3. Add to .env:
       OBS_TRACING_BACKEND=langfuse
       OBS_LANGFUSE_HOST=https://cloud.langfuse.com
       OBS_LANGFUSE_PUBLIC_KEY=pk-lf-...
       OBS_LANGFUSE_SECRET_KEY=sk-lf-...
  4. pip install langfuse
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

_langfuse: Any = None  # Langfuse client instance


def init_tracer(
    host: str | None,
    public_key: str | None,
    secret_key: str | None,
) -> bool:
    global _langfuse
    if not (host and public_key and secret_key):
        log.info("tracing: disabled (no Langfuse credentials)")
        return False
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
        )
        log.info("tracing: enabled (Langfuse @ %s)", host)
        return True
    except ImportError:
        log.warning("langfuse not installed — pip install langfuse")
        return False
    except Exception as e:
        log.warning("tracing init failed: %s", e)
        return False


@contextmanager
def trace_query(query: str, tenant_id: str, request_id: str = "-"):
    """
    Context manager that traces a full /ask call in Langfuse.

    Usage:
        with trace_query(query, tenant_id, rid) as t:
            t.event("retrieve_done", hits=5)
            t.update(output={"answer": text})
    """
    if _langfuse is None:
        yield _NullSpan()
        return

    try:
        trace = _langfuse.trace(
            name="docintel.ask",
            user_id=tenant_id,
            metadata={"request_id": request_id},
            input={"query": query},
        )
        wrapper = _TraceWrapper(trace)
        yield wrapper
        _langfuse.flush()
    except Exception as e:
        log.warning("trace_query error: %s", e)
        yield _NullSpan()


class _TraceWrapper:
    """Thin wrapper so callers don't import langfuse directly."""

    def __init__(self, trace: Any):
        self._trace = trace

    def event(self, name: str, **kwargs) -> None:
        try:
            self._trace.event(name=name, metadata=kwargs)
        except Exception:
            pass

    def update(self, **kwargs) -> None:
        try:
            self._trace.update(**kwargs)
        except Exception:
            pass


class _NullSpan:
    def event(self, name: str, **kwargs) -> None: pass
    def update(self, **kwargs) -> None: pass


def shutdown() -> None:
    if _langfuse is not None:
        try:
            _langfuse.flush()
        except Exception:
            pass
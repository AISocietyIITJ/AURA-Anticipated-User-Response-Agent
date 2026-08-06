"""
Shared planner helpers: the registered-agent registry and agent validation.

The set of "registered" agents is defined by the AgentTag nodes in Neo4j (the
same nodes the graph planner recommends from). Validation rejects any agent
name the LLM produces that is not part of this registry.
"""

from __future__ import annotations

import logging
import threading
import time

from app.graph.driver import session as graph_session

log = logging.getLogger(__name__)

# Module-level registry cache (thread-safe). Invalidated after CACHE_TTL seconds.
_CACHE: set[str] = set()
_CACHE_TIME = 0.0
_CACHE_TTL = 300.0
_LOCK = threading.Lock()


def get_registered_agents() -> set[str]:
    """Return the set of agent names registered in the project.

    Loads once from Neo4j AgentTag nodes and caches the result. Never raises:
    on a graph failure it returns the cached set, or an empty set if no cache
    exists yet (in which case LLM-generated agents cannot be validated and the
    planner falls back to the empty plan).
    """
    global _CACHE, _CACHE_TIME
    now = time.monotonic()
    with _LOCK:
        if _CACHE and now - _CACHE_TIME < _CACHE_TTL:
            return set(_CACHE)

    names: set[str] = set()
    try:
        with graph_session() as s:
            rec = s.run("MATCH (a:AgentTag) RETURN collect(a.name) AS names").single()
        if rec:
            names.update(n for n in (rec["names"] or []) if n)
    except Exception as exc:
        log.warning("Could not load agent registry from Neo4j: %s", exc)
        with _LOCK:
            if _CACHE:
                return set(_CACHE)

    if names:
        with _LOCK:
            _CACHE = names
            _CACHE_TIME = now
    return set(names)


def validate_agents(candidates: list[str] | None, registered: set[str]) -> list[str]:
    """Filter LLM-generated agents to those registered in the project.

    Order is preserved, duplicates are dropped, and unknown (hallucinated)
    names are rejected.
    """
    if not candidates:
        return []
    seen: set[str] = set()
    validated: list[str] = []
    for name in candidates:
        n = (name or "").strip()
        if not n or n in seen or n not in registered:
            continue
        seen.add(n)
        validated.append(n)
    return validated

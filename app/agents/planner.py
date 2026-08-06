"""
Planner Agent: recommends the next agent action(s).

The Neo4j graph is the PRIMARY decision maker. Given the current State
(intent, customer_tag, sentiment_bucket), it picks the agent_tags that
historically appeared in similar states AND were associated with positive
outcomes (OUTCOME_RESOLVED).

Backoff cascade if exact state has no data:
  1. (intent, customer_tag, sent_bucket)  -- exact
  2. (intent, customer_tag)                -- ignore sentiment
  3. (intent)                              -- intent only
  4. fallback: used_fallback=True, empty plan

Confidence + LLM fallback:
  Each graph match is scored (see app/agents/confidence.py). If the effective
  planner confidence is at or above the configured threshold
  (settings.planner_confidence_threshold), the graph recommendation is
  returned as-is. Otherwise the LLM is invoked (app/agents/llm_fallback.py) to
  generate a next-agent plan; every generated agent is validated against the
  registered-agent registry. If the LLM fails or nothing validates, the
  existing empty fallback is returned — the planner never crashes.
"""

from __future__ import annotations

import logging

from app.agents.confidence import MatchLevel, graph_confidence, planner_confidence, sort_tags
from app.agents.llm_fallback import generate_plan as llm_generate_plan
from app.agents.planner_utils import get_registered_agents
from app.core.config import settings
from app.core.schemas import PlannerSuggestion, State
from app.graph.driver import session as graph_session

log = logging.getLogger(__name__)


def _bucket(score: float) -> str:
    if score < -0.33:
        return "neg"
    if score > 0.33:
        return "pos"
    return "neu"


def _query_exact(intent: str, ctag: str, bucket: str) -> dict | None:
    cypher = """
    MATCH (st:State {key: $key})
    OPTIONAL MATCH (st)-[u:USED_AGENT_TAG]->(a:AgentTag)
    WITH st, collect({tag: a.name, count: u.count}) AS tags
    OPTIONAL MATCH (st)-[lo:LED_TO_OUTCOME]->(o:Outcome)
    WITH st, tags, collect({outcome: o.name, count: lo.count}) AS outcomes
    OPTIONAL MATCH (st)-[n:NEXT]->(nxt:State)
    WITH st, tags, outcomes,
         collect({next_key: nxt.key, count: n.count})[0..5] AS nexts
    RETURN tags, outcomes, nexts
    """
    key = f"{intent}|{ctag}|{bucket}"
    with graph_session() as s:
        rec = s.run(cypher, key=key).single()
    if not rec:
        return None
    tags = [t for t in (rec["tags"] or []) if t.get("tag")]
    if not tags:
        return None
    return {"tags": tags, "outcomes": rec["outcomes"] or [], "nexts": rec["nexts"] or []}


def _query_by_intent_ctag(intent: str, ctag: str) -> dict | None:
    cypher = """
    MATCH (st:State {intent: $intent, customer_tag: $ctag})
    OPTIONAL MATCH (st)-[u:USED_AGENT_TAG]->(a:AgentTag)
    WITH a.name AS tag, sum(u.count) AS count
    WHERE tag IS NOT NULL
    RETURN collect({tag: tag, count: count}) AS tags
    """
    with graph_session() as s:
        rec = s.run(cypher, intent=intent, ctag=ctag).single()
    if not rec or not rec["tags"]:
        return None
    return {"tags": rec["tags"], "outcomes": [], "nexts": []}


def _query_by_intent(intent: str) -> dict | None:
    cypher = """
    MATCH (st:State {intent: $intent})
    OPTIONAL MATCH (st)-[u:USED_AGENT_TAG]->(a:AgentTag)
    WITH a.name AS tag, sum(u.count) AS count
    WHERE tag IS NOT NULL
    RETURN collect({tag: tag, count: count}) AS tags
    """
    with graph_session() as s:
        rec = s.run(cypher, intent=intent).single()
    if not rec or not rec["tags"]:
        return None
    return {"tags": rec["tags"], "outcomes": [], "nexts": []}


def _resolved_share(outcomes: list[dict]) -> float:
    if not outcomes:
        return 0.0
    total = sum((o.get("count") or 0) for o in outcomes)
    if total == 0:
        return 0.0
    resolved = sum((o.get("count") or 0) for o in outcomes if o.get("outcome") == "OUTCOME_RESOLVED")
    return resolved / total


def _empty_suggestion(reason: str, rationale: str, used_llm: bool = False) -> PlannerSuggestion:
    return PlannerSuggestion(
        next_agent_tags=[],
        expected_outcome=None,
        confidence=0.0,
        rationale=rationale,
        used_fallback=True,
        used_llm_fallback=used_llm,
        fallback_reason=reason,
    )


def _graph_suggestion(
    res: dict,
    level: MatchLevel,
    top_k: int,
    rationale_parts: list[str],
) -> PlannerSuggestion:
    tags = sort_tags(res["tags"], top_k)
    conf = graph_confidence(res["tags"], top_k)
    effective_conf = planner_confidence(res["tags"], level, top_k)
    top = [t["tag"] for t in tags]

    outcomes = sorted(res["outcomes"], key=lambda o: o.get("count") or 0, reverse=True)
    expected_outcome = outcomes[0]["outcome"] if outcomes else None
    resolved_pct = _resolved_share(res["outcomes"])
    if outcomes:
        rationale_parts.append(
            f"resolved_share={resolved_pct:.2f}, top_outcome={expected_outcome}"
        )

    return PlannerSuggestion(
        next_agent_tags=top,
        expected_outcome=expected_outcome,
        confidence=float(conf),
        planner_confidence=float(effective_conf),
        rationale="; ".join(rationale_parts),
        used_fallback=False,
        used_llm_fallback=False,
        fallback_reason="",
    )


def _llm_fallback_suggestion(
    state: State,
    res: dict | None,
    effective_conf: float,
    rationale_parts: list[str],
    top_k: int,
) -> PlannerSuggestion:
    log.info(
        "Planner confidence %.2f below threshold %.2f. Invoking LLM fallback...",
        effective_conf,
        settings.planner_confidence_threshold,
    )
    agents = llm_generate_plan(
        state,
        graph_candidates=(res or {}).get("tags") or [],
        registered=get_registered_agents(),
        top_k=top_k,
    )
    if not agents:
        log.info("LLM fallback produced no usable plan; returning empty fallback.")
        return _empty_suggestion(
            reason="Planner confidence below threshold; LLM fallback returned no valid agents.",
            rationale="; ".join(rationale_parts) + "; LLM fallback returned no valid agents.",
            used_llm=True,
        )

    log.info("LLM generated plan: %s", " → ".join(agents))
    return PlannerSuggestion(
        next_agent_tags=agents,
        expected_outcome=None,
        confidence=float(effective_conf),
        planner_confidence=float(effective_conf),
        rationale="; ".join(rationale_parts) + f"; LLM fallback generated: {' -> '.join(agents)}",
        used_fallback=True,
        used_llm_fallback=True,
        fallback_reason="Planner confidence below threshold",
    )


def plan(state: State, top_k: int = 3) -> PlannerSuggestion:
    """Recommend the next agent action(s) for the given state.

    Graph is primary; the LLM is only invoked when graph confidence is below
    the configured threshold. Never raises.
    """
    intent = state.intent
    ctag = state.customer_tag
    bucket = _bucket(state.sentiment.score)

    try:
        return _plan(state, intent, ctag, bucket, top_k)
    except Exception as exc:
        log.exception("Planner failed: %s", exc)
        return _empty_suggestion(
            reason=f"Planner error: {exc}",
            rationale=f"Planner error: {exc}",
        )


def _plan(state: State, intent: str, ctag: str, bucket: str, top_k: int) -> PlannerSuggestion:
    rationale_parts: list[str] = []
    level = MatchLevel.NONE

    res = _query_exact(intent, ctag, bucket)
    if res:
        level = MatchLevel.EXACT
        rationale_parts.append(f"Matched state {intent}|{ctag}|{bucket}")
    else:
        res = _query_by_intent_ctag(intent, ctag)
        if res:
            level = MatchLevel.INTENT_CUSTOMER_TAG
            rationale_parts.append(f"Backed off to ({intent}, {ctag}) ignoring sentiment")
        else:
            res = _query_by_intent(intent)
            if res:
                level = MatchLevel.INTENT_ONLY
                rationale_parts.append(f"Backed off to intent={intent} only")

    if not res:
        effective_conf = planner_confidence([], MatchLevel.NONE, top_k)
        log.info("Planner confidence: %.2f", effective_conf)
        log.info("No graph match for (%s, %s, %s). Invoking LLM fallback...", intent, ctag, bucket)
        return _llm_fallback_suggestion(state, None, effective_conf, rationale_parts, top_k)

    effective_conf = planner_confidence(res["tags"], level, top_k)
    log.info("Planner confidence: %.2f (match=%s)", effective_conf, level.name)

    if effective_conf >= settings.planner_confidence_threshold:
        return _graph_suggestion(res, level, top_k, rationale_parts)

    return _llm_fallback_suggestion(state, res, effective_conf, rationale_parts, top_k)

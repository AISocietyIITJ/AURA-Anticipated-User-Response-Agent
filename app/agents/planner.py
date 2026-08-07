"""
Planner Agent: queries the Neo4j graph to recommend the next agent action(s).

Given the current State (intent, customer_tag, sentiment_bucket), it picks the
agent_tags that historically appeared in similar states AND were associated
with positive outcomes (OUTCOME_RESOLVED).

Backoff cascade if exact state has no data:
  1. (intent, customer_tag, sent_bucket)  -- exact
  2. (intent, customer_tag)                -- ignore sentiment
  3. (intent)                              -- intent only
  4. fallback: used_fallback=True, empty plan
"""

from __future__ import annotations

from app.core.schemas import PlannerSuggestion, State
from app.graph.driver import session as graph_session


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

CONFIDENCE_THRESHOLD = 0.4

def compute_confidence(res: dict) -> tuple[float, dict]:

    tags = sorted(
        res["tags"],
        key=lambda t: t.get("count") or 0,
        reverse=True,
    )
    if not tags:
        return 0.0, {
            "consensus": 0.0,
            "resolved_share": None,
        }
    total = sum(t.get("count") or 0 for t in tags)
    consensus = (
        tags[0]["count"] / total
        if total > 0
        else 0.0
    )
    outcomes = res.get("outcomes", [])
    if outcomes:
        resolved_share = _resolved_share(outcomes)

        confidence = (
            0.7 * resolved_share +
            0.3 * consensus
        )
    else:
        resolved_share = None

        confidence = consensus

    return confidence, {
        "consensus": consensus,
        "resolved_share": resolved_share,
    }


def plan(state: State, top_k: int = 3) -> PlannerSuggestion:
    intent = state.intent
    ctag = state.customer_tag
    bucket = _bucket(state.sentiment.score)

    used_fallback = False
    rationale_parts: list[str] = []

    res = _query_exact(intent, ctag, bucket)
    if res:
        confidence, stats = compute_confidence(res)
        if confidence >= CONFIDENCE_THRESHOLD:
            rationale_parts.append(f"Matched state {intent}|{ctag}|{bucket}")
        else:
            res=None
    else:
        res = _query_by_intent_ctag(intent, ctag)
        if res:
            confidence, stats = compute_confidence(res)
            if confidence >= CONFIDENCE_THRESHOLD:
                rationale_parts.append(f"Backed off to ({intent}, {ctag}) ignoring sentiment")
            else:
                res = None
        else:
            res = _query_by_intent(intent)
            if res:
                confidence, stats = compute_confidence(res)
                if confidence >= CONFIDENCE_THRESHOLD:
                    rationale_parts.append(f"Backed off to intent={intent} only")
                else:
                    res = None

    if not res:
        used_fallback = True
        return PlannerSuggestion(
            next_agent_tags=[],
            expected_outcome=None,
            confidence=0.0,
            rationale=f"No graph match for ({intent}, {ctag}, {bucket}); using LLM fallback.",
            used_fallback=True,
        )
    tags = sorted(res["tags"], key=lambda t: t.get("count") or 0, reverse=True)[:top_k]
    total = sum(t.get("count") or 0 for t in tags) or 1
    top = [t["tag"] for t in tags]
    #confidence = (tags[0]["count"] / total) if tags else 0.0

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
        confidence=float(confidence),
        rationale="; ".join(rationale_parts),
        used_fallback=used_fallback,
    )

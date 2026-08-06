"""
Planner tests: graph-first recommendation + LLM fallback.

Neo4j is mocked via a fake session injected into app.agents.planner.graph_session
and app.agents.planner_utils.graph_session. The LLM is mocked by patching
app.agents.llm_fallback._call_llm (or the planner's imported llm_generate_plan
for orchestration-level tests).
"""

from __future__ import annotations

import httpx
import pytest

from app.agents import llm_fallback as llm_fallback_module
from app.agents import planner as planner_module
from app.agents import planner_utils
from app.core.schemas import Sentiment, State

REGISTRY = [
    "history",
    "persona",
    "planner",
    "critic",
    "sentiment",
    "interpreter",
    "llm",
    "stt",
]


# --------------------------------------------------------------------------- #
# Graph mocks
# --------------------------------------------------------------------------- #
class FakeResult:
    def __init__(self, record):
        self._record = record

    def single(self):
        return self._record


class FakeSession:
    def __init__(self, handler):
        self._handler = handler

    def run(self, query, **params):
        return FakeResult(self._handler(query, params))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def exact_record(tags, outcomes=None):
    return {
        "tags": [{"tag": t, "count": c} for t, c in tags],
        "outcomes": outcomes or [],
        "nexts": [],
    }


def backoff_record(tags):
    return {"tags": [{"tag": t, "count": c} for t, c in tags], "outcomes": [], "nexts": []}


def make_handler(exact=None, intent_ctag=None, intent_only=None, registry=REGISTRY):
    def handler(query, params):
        if "collect(a.name)" in query:
            return {"names": list(registry)}
        if "key: $key" in query:
            return exact
        if "customer_tag: $ctag" in query:
            return intent_ctag
        return intent_only

    return handler


def make_state(**overrides):
    base = dict(
        conversation_id="c1",
        seq=1,
        text="I want my refund processed now",
        intent="Returns_and_Refunds",
        customer_tag="CUSTOMER_EXPRESSES_FRUSTRATION",
        sentiment=Sentiment(score=-0.7, label="negative"),
        history=[],
    )
    base.update(overrides)
    return State(**base)


@pytest.fixture(autouse=True)
def _reset_registry_cache():
    planner_utils._CACHE.clear()
    planner_utils._CACHE_TIME = 0.0
    yield


@pytest.fixture
def mock_graph(monkeypatch):
    def _mock(handler):
        session = FakeSession(handler)
        monkeypatch.setattr(planner_module, "graph_session", lambda: session)
        monkeypatch.setattr(planner_utils, "graph_session", lambda: session)
        return session

    return _mock


# --------------------------------------------------------------------------- #
# Graph-first behaviour
# --------------------------------------------------------------------------- #
def test_high_confidence_graph_path(mock_graph, monkeypatch):
    mock_graph(make_handler(exact=exact_record([("history", 5), ("persona", 3), ("planner", 2)])))

    def _boom(*args, **kwargs):
        raise AssertionError("LLM fallback should not be invoked on a high-confidence graph match")

    monkeypatch.setattr(planner_module, "llm_generate_plan", _boom)

    plan = planner_module.plan(make_state())

    assert plan.next_agent_tags == ["history", "persona", "planner"]
    assert plan.used_fallback is False
    assert plan.used_llm_fallback is False
    assert plan.planner_confidence >= 0.60
    assert plan.fallback_reason == ""


def test_successful_graph_lookup(mock_graph):
    mock_graph(
        make_handler(
            exact=exact_record(
                [("history", 5), ("persona", 3), ("planner", 2)],
                outcomes=[
                    {"outcome": "OUTCOME_RESOLVED", "count": 5},
                    {"outcome": "OUTCOME_NOT_RESOLVED", "count": 1},
                ],
            )
        )
    )

    plan = planner_module.plan(make_state())

    assert plan.next_agent_tags == ["history", "persona", "planner"]
    assert plan.expected_outcome == "OUTCOME_RESOLVED"
    assert plan.confidence == pytest.approx(0.5)
    assert plan.rationale


def test_backoff_to_intent_ctag(mock_graph):
    mock_graph(
        make_handler(
            exact=exact_record([]),
            intent_ctag=backoff_record([("history", 5), ("persona", 3)]),
        )
    )

    plan = planner_module.plan(make_state())

    assert plan.next_agent_tags == ["history", "persona"]
    assert plan.used_fallback is False
    assert "Backed off" in plan.rationale


# --------------------------------------------------------------------------- #
# LLM fallback behaviour
# --------------------------------------------------------------------------- #
def test_low_confidence_uses_llm_fallback(mock_graph, monkeypatch):
    mock_graph(make_handler(exact=exact_record([("history", 2), ("persona", 1)])))

    calls = {}

    def _stub(state, graph_candidates=None, registered=None, top_k=3):
        calls["state"] = state
        calls["candidates"] = graph_candidates
        return ["history", "persona"]

    monkeypatch.setattr(planner_module, "llm_generate_plan", _stub)

    plan = planner_module.plan(make_state())

    assert "state" in calls
    assert calls["candidates"] == [{"tag": "history", "count": 2}, {"tag": "persona", "count": 1}]
    assert plan.next_agent_tags == ["history", "persona"]
    assert plan.used_fallback is True
    assert plan.used_llm_fallback is True
    assert plan.planner_confidence < 0.60
    assert plan.fallback_reason == "Planner confidence below threshold"


def test_llm_fallback_valid_agents(mock_graph, monkeypatch):
    mock_graph(make_handler(exact=exact_record([("history", 2), ("persona", 1)])))
    monkeypatch.setattr(
        llm_fallback_module,
        "_call_llm",
        lambda prompt: '{"agents": ["history", "persona", "planner"]}',
    )

    plan = planner_module.plan(make_state())

    assert plan.next_agent_tags == ["history", "persona", "planner"]
    assert plan.used_llm_fallback is True


def test_hallucinated_agents_rejected(mock_graph, monkeypatch):
    mock_graph(make_handler(exact=exact_record([("history", 2), ("persona", 1)])))
    monkeypatch.setattr(
        llm_fallback_module,
        "_call_llm",
        lambda prompt: '{"agents": ["history", "NOT_A_REAL_AGENT", "history"]}',
    )

    plan = planner_module.plan(make_state())

    assert plan.next_agent_tags == ["history"]
    assert plan.used_llm_fallback is True


def test_invalid_llm_response(mock_graph, monkeypatch):
    mock_graph(make_handler(exact=exact_record([("history", 2), ("persona", 1)])))
    monkeypatch.setattr(llm_fallback_module, "_call_llm", lambda prompt: "this is not json")

    plan = planner_module.plan(make_state())

    assert plan.next_agent_tags == []
    assert plan.used_fallback is True
    assert plan.used_llm_fallback is True


def test_llm_timeout(mock_graph, monkeypatch):
    mock_graph(make_handler(exact=exact_record([("history", 2), ("persona", 1)])))

    def _raise(prompt):
        raise httpx.ConnectTimeout("connection timed out", request=None)

    monkeypatch.setattr(llm_fallback_module, "_call_llm", _raise)

    plan = planner_module.plan(make_state())

    assert plan.next_agent_tags == []
    assert plan.used_fallback is True
    assert plan.used_llm_fallback is True


# --------------------------------------------------------------------------- #
# No graph data
# --------------------------------------------------------------------------- #
def test_no_graph_data_uses_llm_fallback(mock_graph, monkeypatch):
    mock_graph(make_handler(exact=None, intent_ctag=None, intent_only=None))
    monkeypatch.setattr(
        llm_fallback_module,
        "_call_llm",
        lambda prompt: '{"agents": ["history", "persona"]}',
    )

    plan = planner_module.plan(make_state())

    assert plan.next_agent_tags == ["history", "persona"]
    assert plan.used_llm_fallback is True


def test_no_graph_data_llm_fails_returns_empty(mock_graph, monkeypatch):
    mock_graph(make_handler(exact=None, intent_ctag=None, intent_only=None))
    monkeypatch.setattr(llm_fallback_module, "_call_llm", lambda prompt: "garbage")

    plan = planner_module.plan(make_state())

    assert plan.next_agent_tags == []
    assert plan.used_fallback is True


# --------------------------------------------------------------------------- #
# Resilience
# --------------------------------------------------------------------------- #
def test_planner_never_crashes_on_graph_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("graph down")

    monkeypatch.setattr(planner_module, "graph_session", _raise)

    plan = planner_module.plan(make_state())

    assert plan.next_agent_tags == []
    assert plan.used_fallback is True
    assert "Planner error" in plan.fallback_reason

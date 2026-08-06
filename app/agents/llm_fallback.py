"""
LLM planner fallback.

Invoked only when the graph planner's confidence is below the configured
threshold. Uses the project's existing OpenAI-compatible LLM endpoint (same
base URL, auth and provider as the main chat / interpreter / sentiment) to
generate the most likely next-agent execution plan.

Inputs fed to the model:
  - original user query
  - detected intent
  - customer tag
  - sentiment (label + score)
  - previous graph candidates (may be empty)
  - available (registered) agent names

The model returns strict JSON: {"agents": ["history", "persona", ...]}. Every
returned agent is validated against the registered-agent registry; unregistered
agents are dropped. On any failure (timeout, provider error, invalid JSON,
parse error, no registered agents) an empty list is returned so the planner
falls back to its existing empty plan and never crashes.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from app.agents.planner_utils import validate_agents
from app.core.config import settings
from app.core.schemas import State

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a conversation-pipeline planner for a customer-care AI assistant. "
    "Given a customer message and its context, choose the sequence of downstream "
    "agent modules that should run next. Respond ONLY with a valid JSON object — "
    "no markdown, no explanation."
)

_USER_TMPL = """\
Decide the next agents to execute for this customer-care turn.

Context:
- Detected intent: {intent}
- Customer tag: {customer_tag}
- Sentiment: {sentiment_label} (score={sentiment_score:.2f})

Latest customer message:
"{text}"

Candidates already suggested by the historical graph (may be empty):
{candidates}

Available agents you may choose from:
{agents}

Return exactly this JSON structure:
{{
  "agents": ["<agent>", "<agent>", ...]
}}

Choose 1 to {top_k} agents, in execution order, using only the available agents listed above.
"""


def _build_prompt(
    state: State,
    graph_candidates: list[dict],
    agents: list[str],
    top_k: int,
) -> str:
    candidates = "\n".join(
        f"- {c.get('tag')} (historical count: {c.get('count') or 0})"
        for c in (graph_candidates or [])
    ) or "(none)"

    return _USER_TMPL.format(
        intent=state.intent,
        customer_tag=state.customer_tag,
        sentiment_label=state.sentiment.label,
        sentiment_score=state.sentiment.score,
        text=state.text,
        candidates=candidates,
        agents=", ".join(sorted(agents)),
        top_k=top_k,
    )


def _call_llm(prompt: str) -> str:
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    payload = {
        "model": settings.planner_llm_model or settings.llm_model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": settings.planner_llm_max_tokens,
        "temperature": settings.planner_llm_temperature,
    }
    with httpx.Client(timeout=settings.planner_llm_timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _parse_agents(text: str) -> list[str]:
    """Extract and parse the agent list from model output (tolerates fences)."""
    text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text.strip())
    data = json.loads(text)
    agents = data.get("agents")
    if not isinstance(agents, list):
        raise ValueError("'agents' must be a list")
    return [str(a) for a in agents]


def generate_plan(
    state: State,
    graph_candidates: list[dict] | None = None,
    registered: set[str] | None = None,
    top_k: int = 3,
) -> list[str]:
    """
    Generate a validated next-agent plan via the LLM.

    Returns an empty list when the LLM fails (timeout / provider error), the
    response cannot be parsed, or none of the produced agents are registered.
    """
    try:
        prompt = _build_prompt(
            state,
            graph_candidates or [],
            registered or set(),
            top_k,
        )
        raw = _call_llm(prompt)
        agents = _parse_agents(raw)
    except Exception as exc:
        log.warning("LLM planner fallback failed: %s", exc)
        return []

    validated = validate_agents(agents, registered or set())
    if not validated:
        log.warning("LLM planner fallback returned no registered agents: %s", agents)
    return validated

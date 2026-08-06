"""
LLM Response generator using an OpenAI-compatible local server.

Works with both Ollama (default for Windows) and vLLM. The two share an
OpenAI-compatible Chat Completions API at /v1/chat/completions, so the same
client code works for both — only base_url and model name change.

Inputs: State + PlannerSuggestion (or fallback path).
Output: a coaching suggestion the human agent can read on screen.

If the planner returns used_fallback=True (cold state), we skip the planner
guidance and ask the LLM to suggest a generic next move from the transcript.
"""

from __future__ import annotations

import httpx

from app.core.config import settings
from app.core.schemas import PlannerSuggestion, State


SYSTEM_PROMPT = """You are an AI customer care chatbot directly assisting a customer.
You should respond directly to the customer in a natural, helpful, and polite tone.
Keep your responses relatively short, conversational, and grounded in the customer's stated issue.
Do not provide coaching instructions, just provide the actual response you would send to the customer.
"""


def _build_user_prompt(state: State, plan: PlannerSuggestion) -> str:
    history = "\n".join(f"- {h}" for h in state.history[-4:]) or "(none yet)"
    if plan.used_fallback or not plan.next_agent_tags:
        guidance = "(no graph match — propose a sensible next move from the transcript alone.)"
    else:
        tags = ", ".join(plan.next_agent_tags)
        guidance = (
            f"Historical playbook for similar calls suggests next action(s): {tags}. "
            f"Expected outcome: {plan.expected_outcome or 'unknown'} "
            f"(planner_confidence={plan.confidence:.2f})."
        )

    return f"""Conversation context:
- Detected intent: {state.intent}
- Customer state: {state.customer_tag}
- Customer sentiment: {state.sentiment.label} (score={state.sentiment.score:.2f})

Recent transcript chunks:
{history}

Latest customer turn:
\"{state.text}\"

Planner guidance:
{guidance}

Write your direct response to the customer now."""


async def generate(state: State, plan: PlannerSuggestion) -> str:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(state, plan)},
        ],
        "max_tokens": 160,
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"].strip()

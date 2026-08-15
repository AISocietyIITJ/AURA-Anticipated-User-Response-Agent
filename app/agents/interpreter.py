"""
Interpreter Agent — LLM-based (Hugging Face router, OpenAI-compatible).

Uses the same LLM endpoint that's already working for the main chat.
Sends a structured prompt asking for JSON with intent + customer_tag.
No separate model or API endpoint needed.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from app.core.config import settings
from app.core.schemas import Interpretation

log = logging.getLogger(__name__)

INTENTS: list[str] = [
    "Returns_and_Refunds", "Technical_Issues", "Upgrades_and_Promotions",
    "Loyalty Program", "Delay Management", "Delivery Delays",
    "Product Feedback", "Cancellation Policies", "Booking Errors",
    "Churn Prediction", "Cross-Brand Mentions", "Brand Loyalty",
    "Service Complaints", "Price Sensitivity", "Feature Requests",
    "Account Issues", "Billing Issues", "Refund Status",
    "Order Tracking", "Subscription Issues", "Promotion Inquiry",
    "General Inquiry", "Complaint Resolution", "Onboarding",
]

CUSTOMER_TAGS: list[str] = [
    "CUSTOMER_EXPRESSES_SATISFACTION", "CUSTOMER_EXPRESSES_FRUSTRATION",
    "CUSTOMER_STATES_ISSUE", "CUSTOMER_REQUESTS_FINANCIAL_RELIEF",
    "CUSTOMER_EXPRESSES_CONFUSION", "CUSTOMER_OTHER",
    "CUSTOMER_OBJECTS_TO_POLICY", "CUSTOMER_REQUESTS_ESCALATION",
    "CUSTOMER_PROVIDES_INFO", "CUSTOMER_MENTIONS_COMPETITOR",
    "CUSTOMER_THREATENS_CHURN",
]

_SYSTEM = (
    "You are a classification engine. Respond ONLY with a valid JSON object — "
    "no markdown, no explanation."
)

_USER_TMPL = """\
Classify the following customer message.

Return exactly this JSON structure:
{{
  "intent": "<one of the intents below>",
  "intent_confidence": <float 0.0-1.0>,
  "customer_tag": "<one of the tags below>",
  "customer_tag_confidence": <float 0.0-1.0>
}}

Valid intents: {intents}
Valid customer tags: {tags}

Customer message:
\"{text}\"
"""


async def _call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def _extract_json(text: str) -> dict:
    """Extract JSON from model output, stripping any surrounding markdown fences."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in text: {text}")
    return json.loads(match.group(0))


async def interpret(text: str) -> Interpretation:
    text = (text or "").strip()
    if not text:
        return Interpretation(intent="General Inquiry", customer_tag="CUSTOMER_OTHER")

    prompt = _USER_TMPL.format(
        intents=", ".join(INTENTS),
        tags=", ".join(CUSTOMER_TAGS),
        text=text,
    )

    try:
        raw_out = await _call_llm(prompt)
        obj = _extract_json(raw_out)
        return Interpretation(
            intent=obj.get("intent", "General Inquiry"),
            customer_tag=obj.get("customer_tag", "CUSTOMER_OTHER"),
            intent_confidence=float(obj.get("intent_confidence", 0.0)),
            customer_tag_confidence=float(obj.get("customer_tag_confidence", 0.0)),
        )
    except Exception as e:
        log.error("Interpretation failed: %s (text=%r)", e, text)
        return Interpretation(
            intent="General Inquiry",
            customer_tag="CUSTOMER_OTHER",
            intent_confidence=0.0,
            customer_tag_confidence=0.0,
        )

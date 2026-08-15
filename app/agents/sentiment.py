"""
Sentiment Agent — LLM-based (Hugging Face router, OpenAI-compatible).

Uses the same LLM endpoint that's already working for the main chat.
Sends a structured prompt asking for JSON with sentiment score + label.
No separate model or API endpoint needed.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from app.core.config import settings
from app.core.schemas import Sentiment

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a sentiment analysis engine. Respond ONLY with a valid JSON object — "
    "no markdown, no explanation."
)

_USER_TMPL = """\
Analyse the sentiment of the following customer message.

Return exactly this JSON structure:
{{
  "label": "<negative | neutral | positive>",
  "score": <float from -1.0 (very negative) to 1.0 (very positive)>
}}

Customer message:
\"{text}\"
"""


async def _call_llm(prompt: str) -> str:
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 40,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in text: {text}")
    return json.loads(match.group(0))


async def analyze(text: str) -> Sentiment:
    text = (text or "").strip()
    if not text:
        return Sentiment(score=0.0, label="neutral")

    try:
        raw = await _call_llm(_USER_TMPL.format(text=text))
        data = _extract_json(raw)

        label = data.get("label", "neutral").lower()
        if label not in ("negative", "neutral", "positive"):
            label = "neutral"

        score = float(data.get("score", 0.0))
        score = max(-1.0, min(1.0, score))

        return Sentiment(score=score, label=label)

    except Exception as exc:
        log.warning("Sentiment analysis failed: %s — returning neutral", exc)
        return Sentiment(score=0.0, label="neutral")

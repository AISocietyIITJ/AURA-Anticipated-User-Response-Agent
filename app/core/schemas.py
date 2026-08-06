from typing import Optional
from pydantic import BaseModel, Field


class TranscriptChunk(BaseModel):
    conversation_id: str
    text: str
    seq: int = 0


class Interpretation(BaseModel):
    intent: str
    customer_tag: str
    intent_confidence: float = 0.0
    customer_tag_confidence: float = 0.0


class Sentiment(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0)
    label: str


class State(BaseModel):
    conversation_id: str
    seq: int
    text: str
    intent: str
    customer_tag: str
    sentiment: Sentiment
    history: list[str] = Field(default_factory=list)


class PlannerSuggestion(BaseModel):
    next_agent_tags: list[str]
    expected_outcome: Optional[str] = None
    confidence: float = 0.0
    rationale: str = ""
    used_fallback: bool = False
    # Effective planner confidence used for the fallback threshold decision.
    # Matches `confidence` on the graph path; reflects the (low) graph confidence
    # that triggered the LLM fallback otherwise.
    planner_confidence: float = 0.0
    used_llm_fallback: bool = False
    fallback_reason: str = ""


class CopilotResponse(BaseModel):
    state: State
    plan: PlannerSuggestion
    suggested_reply: str
    used_fallback: bool = False

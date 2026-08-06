"""
Planner confidence computation.

Kept in its own module so confidence logic is not scattered across the planner.
Two scores are produced for each graph match:

  graph_confidence(tags, top_k)
      Backward-compatible dominance score: the top candidate's share of the
      total count across the top-k candidates. This is the value historically
      returned in PlannerSuggestion.confidence.

  planner_confidence(tags, level, top_k)
      The *effective* confidence used for the fallback threshold decision.
      Combines three signals:
        - graph score           -> top-candidate dominance (graph_confidence)
        - occurrence frequency  -> how many historical occurrences exist
                                   (evidence_volume, capped at EVIDENCE_NORM)
        - match specificity     -> which backoff level matched
                                   (MatchLevel weight; exact > intent+tag > intent)
      planner_confidence = level * (TOP_SHARE_WEIGHT * graph_confidence
                                    + EVIDENCE_WEIGHT * evidence_volume)

Both signals are normalized to [0.0, 1.0]. Weight constants live here and are
documented in the README so they can be tuned in one place.
"""

from __future__ import annotations

from enum import Enum


# Relative weight of top-candidate dominance vs historical evidence volume.
# Balanced 50/50 so a single candidate with thin evidence (dominance 1.0) does
# not clear the confidence threshold on its own.
TOP_SHARE_WEIGHT = 0.5
EVIDENCE_WEIGHT = 0.5
# Total historical occurrences (sum of all edge counts) that saturates the
# evidence-volume signal at 1.0.
EVIDENCE_NORM = 10.0

DEFAULT_TOP_K = 3


class MatchLevel(float, Enum):
    """Backoff-cascade specificity. Higher = more specific graph match."""

    EXACT = 1.0
    INTENT_CUSTOMER_TAG = 0.9
    INTENT_ONLY = 0.75
    NONE = 0.0


def sort_tags(tags: list[dict], top_k: int) -> list[dict]:
    """Return the top-k tags by historical count, descending."""
    return sorted(tags, key=lambda t: t.get("count") or 0, reverse=True)[:top_k]


def graph_confidence(tags: list[dict], top_k: int = DEFAULT_TOP_K) -> float:
    """
    Relative strength of the top recommendation among the top-k candidates.

    Formula (unchanged from the original planner):
        confidence = top_count / sum(top-k counts)
    """
    top = sort_tags(tags, top_k)
    if not top:
        return 0.0
    total = sum(t.get("count") or 0 for t in top)
    if total <= 0:
        return 0.0
    return (top[0].get("count") or 0) / total


def evidence_volume(tags: list[dict]) -> float:
    """Occurrence frequency across *all* historical edges, capped at 1.0."""
    total = sum(t.get("count") or 0 for t in tags)
    return min(total / EVIDENCE_NORM, 1.0)


def planner_confidence(
    tags: list[dict],
    level: MatchLevel,
    top_k: int = DEFAULT_TOP_K,
) -> float:
    """
    Effective confidence used for the fallback threshold decision.

    Returns 0.0 when there is no data or the match level is NONE.
    """
    if not tags or level is MatchLevel.NONE:
        return 0.0
    share = graph_confidence(tags, top_k)
    volume = evidence_volume(tags)
    return float(level) * (TOP_SHARE_WEIGHT * share + EVIDENCE_WEIGHT * volume)

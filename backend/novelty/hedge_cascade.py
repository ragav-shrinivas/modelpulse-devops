"""
Hedge Cascade Detector
======================

PATENT NOVELTY:
    Hedging frequency in LLM outputs is a known proxy for model
    uncertainty. Prior work treats hedge count as a STATIC ratio.

    This module introduces a TEMPORAL HEDGE CASCADE: hedge ratios are
    computed per message in order, then their first and second
    derivatives are tracked.  An "early-warning cascade" fires when
    the second derivative (acceleration of hedging) exceeds a
    statistical threshold for K consecutive messages — indicating the
    model is *accelerating* toward uncertainty, not merely sitting at
    a high-uncertainty baseline.

    The detector also distinguishes:
        - epistemic hedges    ("I think", "maybe", "I'm not sure")
        - approximative hedges ("roughly", "about", "approximately")
        - probabilistic hedges ("likely", "probably", "could")
    so downstream alerting can route on hedge KIND, not just count.

    Output is suitable as a real-time monitoring signal that fires
    BEFORE the absolute hedge ratio would trigger a static threshold.
"""

from __future__ import annotations
import math
import re
from typing import Dict, List


EPISTEMIC = {
    "i think", "i believe", "i'm not sure", "im not sure", "not sure",
    "i'm uncertain", "im uncertain", "uncertain", "unclear",
    "from what i can tell", "as far as i know", "i don't know", "i dont know",
    "to my knowledge",
}
APPROXIMATIVE = {
    "roughly", "approximately", "about", "around", "near", "ish", "or so",
    "in the ballpark", "give or take",
}
PROBABILISTIC = {
    "maybe", "perhaps", "might", "could", "may", "likely", "probably",
    "possibly", "seems", "appears", "presumably", "plausibly",
}

HEDGE_LEXICON = {
    "epistemic": EPISTEMIC,
    "approximative": APPROXIMATIVE,
    "probabilistic": PROBABILISTIC,
}


def _count_hedges(text: str) -> Dict[str, int]:
    """Return per-category hedge counts."""
    low = text.lower()
    out = {k: 0 for k in HEDGE_LEXICON}
    for cat, lex in HEDGE_LEXICON.items():
        for phrase in lex:
            if " " in phrase:
                out[cat] += low.count(phrase)
            else:
                # word-boundary count for single tokens
                out[cat] += len(re.findall(rf"\b{re.escape(phrase)}\b", low))
    return out


def _hedge_ratio(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    if not words:
        return 0.0
    counts = _count_hedges(text)
    total = sum(counts.values())
    return min(1.0, total / max(len(words), 1) * 20)   # scale so 5% hedges → 1.0


def hedge_cascade(messages: List[str],
                  k_consecutive: int = 2,
                  sigma_mult: float = 1.5) -> Dict:
    """
    Detect a hedge cascade across a message sequence.

    Returns:
        per_message_ratio    list of [0,1] hedge ratios
        per_message_breakdown list of {epistemic, approximative, probabilistic}
        velocity             1st derivative
        acceleration         2nd derivative
        cascade_detected     bool
        cascade_start        index of first message in the cascade, or None
        dominant_category    which category dominates the cascade
    """
    if not messages:
        return _empty_result()

    ratios = [_hedge_ratio(m) for m in messages]
    breakdown = [_count_hedges(m) for m in messages]

    velocity = [round(ratios[i] - ratios[i - 1], 4)
                for i in range(1, len(ratios))]
    acceleration = [round(velocity[i] - velocity[i - 1], 4)
                    for i in range(1, len(velocity))]

    # Threshold = sigma_mult * std(acceleration). Need >=3 points for stable stats.
    cascade, start = False, None
    if len(acceleration) >= k_consecutive + 1:
        mu = sum(acceleration) / len(acceleration)
        var = sum((a - mu) ** 2 for a in acceleration) / len(acceleration)
        sigma = math.sqrt(var)
        thr = sigma_mult * sigma
        run = 0
        for i, a in enumerate(acceleration):
            if a > mu + thr and thr > 1e-6:
                run += 1
                if run >= k_consecutive:
                    cascade = True
                    start = i - k_consecutive + 2   # offset to message index
                    break
            else:
                run = 0

    # Dominant category over the whole conversation
    totals = {cat: 0 for cat in HEDGE_LEXICON}
    for b in breakdown:
        for cat, c in b.items():
            totals[cat] += c
    dominant = max(totals, key=totals.get) if any(totals.values()) else "none"

    insight = _make_insight(ratios, cascade, start, dominant)

    return {
        "per_message_ratio":     [round(r, 4) for r in ratios],
        "per_message_breakdown": breakdown,
        "velocity":              velocity,
        "acceleration":          acceleration,
        "cascade_detected":      cascade,
        "cascade_start_index":   start,
        "dominant_category":     dominant,
        "category_totals":       totals,
        "mean_ratio":            round(sum(ratios) / len(ratios), 4),
        "peak_ratio":            round(max(ratios), 4),
        "insight":               insight,
    }


def _empty_result() -> Dict:
    return {
        "per_message_ratio": [], "per_message_breakdown": [],
        "velocity": [], "acceleration": [],
        "cascade_detected": False, "cascade_start_index": None,
        "dominant_category": "none", "category_totals": {},
        "mean_ratio": 0.0, "peak_ratio": 0.0,
        "insight": "No messages to analyze.",
    }


def _make_insight(ratios, cascade, start, dominant):
    if cascade:
        return (f"Hedge cascade detected starting at message #{start}. "
                f"Dominant hedge type: {dominant}. Model is accelerating "
                f"toward uncertainty — investigate prompts after this point.")
    if ratios and max(ratios) > 0.6:
        return (f"Sustained high hedging detected (peak={max(ratios):.2f}). "
                f"Dominant type: {dominant}. Static-high regime, not a cascade.")
    if ratios and sum(ratios) / len(ratios) > 0.3:
        return f"Moderate hedging baseline, no cascade. Dominant: {dominant}."
    return "Low hedging across conversation — confident regime."

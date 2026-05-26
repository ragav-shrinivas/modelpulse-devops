"""
Multi-Scale Entropy Decomposition (MSED)
========================================

PATENT NOVELTY:
    Classical Shannon entropy collapses an entire signal into one scalar,
    losing the SCALE at which uncertainty manifests. MSED decomposes
    linguistic uncertainty into a cascade of orthogonal scales:

        L1 — TOKEN scale       (vocabulary / lexical uncertainty)
        L2 — PHRASE scale      (bigram / collocation uncertainty)
        L3 — SENTENCE scale    (length & syntactic-shape distribution)
        L4 — DISCOURSE scale   (cross-message turn-to-turn variation)

    Each level's entropy is computed independently AND a "scale-coupling
    coefficient" rho_{i,j} = corr(H_i over windows, H_j over windows) is
    reported.  High coupling at adjacent levels indicates a coherent
    confidence regime; decoupling indicates "scale-specific collapse"
    (e.g. model knows what to say at the word level but loses coherence
    across sentences).

    A composite MSED score is returned alongside the per-scale vector so
    downstream alerting can route on the SPECIFIC failure mode, not just
    "entropy is high".

No external dependencies — pure Python, deterministic, deployable on edge.
"""

from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


# ---------- helpers ----------

def _normalised_entropy(counts: Dict, n_classes: Optional[int] = None) -> float:
    """Shannon entropy normalised to [0, 1]."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    raw = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            raw -= p * math.log2(p)
    k = n_classes or len(counts)
    max_e = math.log2(k) if k > 1 else 1.0
    return raw / max_e if max_e > 0 else 0.0


def _tokens(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 4]


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


# ---------- per-scale entropies ----------

def _token_entropy(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    return _normalised_entropy(Counter(toks), n_classes=max(len(set(toks)), 1))


def _phrase_entropy(text: str) -> float:
    toks = _tokens(text)
    if len(toks) < 2:
        return 0.0
    bigrams = [f"{a}_{b}" for a, b in zip(toks[:-1], toks[1:])]
    return _normalised_entropy(Counter(bigrams), n_classes=max(len(set(bigrams)), 1))


def _sentence_entropy(text: str) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.0
    # Bucket sentence lengths into 10 bins of 5 words each
    buckets = Counter(min(len(s.split()) // 5, 9) for s in sents)
    return _normalised_entropy(buckets, n_classes=10)


def _discourse_entropy(messages: List[str]) -> float:
    """Variation in length + lexical novelty across consecutive messages."""
    if len(messages) < 2:
        return 0.0
    # Length CV
    lens = [len(_tokens(m)) for m in messages]
    if sum(lens) == 0:
        return 0.0
    mean = sum(lens) / len(lens)
    var = sum((l - mean) ** 2 for l in lens) / len(lens)
    cv = math.sqrt(var) / max(mean, 1)
    # Vocabulary turnover: jaccard distance between consecutive msgs
    jaccards = []
    for a, b in zip(messages[:-1], messages[1:]):
        sa, sb = set(_tokens(a)), set(_tokens(b))
        if sa or sb:
            jaccards.append(1 - len(sa & sb) / max(len(sa | sb), 1))
    turn_jd = sum(jaccards) / len(jaccards) if jaccards else 0.0
    # Weighted blend
    return min(1.0, 0.5 * min(1.0, cv) + 0.5 * turn_jd)


# ---------- scale-coupling ----------

def _scale_coupling(messages: List[str]) -> Dict[str, float]:
    """
    Compute per-message entropy at each scale, then Pearson-correlate scales.
    Returns rho_{L1L2}, rho_{L2L3}, and a global decoupling index.
    """
    if len(messages) < 3:
        return {"rho_token_phrase": 0.0, "rho_phrase_sentence": 0.0,
                "decoupling_index": 0.0}
    t = [_token_entropy(m) for m in messages]
    p = [_phrase_entropy(m) for m in messages]
    s = [_sentence_entropy(m) for m in messages]
    r_tp = _pearson(t, p)
    r_ps = _pearson(p, s)
    # Decoupling = 1 - mean coupling; high = inconsistent across scales = anomalous
    decoup = 1.0 - (abs(r_tp) + abs(r_ps)) / 2
    return {
        "rho_token_phrase":    round(r_tp, 4),
        "rho_phrase_sentence": round(r_ps, 4),
        "decoupling_index":    round(max(0.0, min(1.0, decoup)), 4),
    }


# ---------- main entry ----------

@dataclass
class MSEDResult:
    L1_token:       float
    L2_phrase:      float
    L3_sentence:    float
    L4_discourse:   float
    composite:      float
    weights:        Dict[str, float]
    coupling:       Dict[str, float]
    dominant_scale: str
    interpretation: str

    def to_dict(self) -> Dict:
        return asdict(self)


_DEFAULT_WEIGHTS = {
    "L1_token":     0.30,
    "L2_phrase":    0.30,
    "L3_sentence":  0.20,
    "L4_discourse": 0.20,
}


def multi_scale_entropy(messages: List[str],
                        weights: Optional[Dict[str, float]] = None) -> MSEDResult:
    """
    Compute MSED over a list of assistant messages.
    """
    if not messages:
        return MSEDResult(0, 0, 0, 0, 0, _DEFAULT_WEIGHTS,
                          {"rho_token_phrase": 0, "rho_phrase_sentence": 0,
                           "decoupling_index": 0},
                          "none", "No messages provided.")

    full = " ".join(messages)
    L1 = _token_entropy(full)
    L2 = _phrase_entropy(full)
    L3 = _sentence_entropy(full)
    L4 = _discourse_entropy(messages)

    w = weights or _DEFAULT_WEIGHTS
    composite = (
        w["L1_token"]     * L1 +
        w["L2_phrase"]    * L2 +
        w["L3_sentence"]  * L3 +
        w["L4_discourse"] * L4
    )
    composite = max(0.0, min(1.0, composite))

    coupling = _scale_coupling(messages)

    # Dominant failure scale (the one contributing most to composite)
    contributions = {
        "L1_token":     w["L1_token"]     * L1,
        "L2_phrase":    w["L2_phrase"]    * L2,
        "L3_sentence":  w["L3_sentence"]  * L3,
        "L4_discourse": w["L4_discourse"] * L4,
    }
    dominant = max(contributions, key=contributions.get)

    # Interpretation
    if composite < 0.3:
        intp = "Stable across all scales — coherent low-uncertainty regime."
    elif coupling["decoupling_index"] > 0.6:
        intp = (f"Scale decoupling detected (rho={coupling['decoupling_index']:.2f}). "
                f"Failure dominated by {dominant} — likely scale-specific collapse.")
    elif composite < 0.6:
        intp = f"Moderate multi-scale uncertainty; {dominant} contributes most."
    else:
        intp = (f"Severe coherent uncertainty across scales; "
                f"{dominant} leads. Consider retraining or temperature reduction.")

    return MSEDResult(
        L1_token=round(L1, 4),
        L2_phrase=round(L2, 4),
        L3_sentence=round(L3, 4),
        L4_discourse=round(L4, 4),
        composite=round(composite, 4),
        weights=w,
        coupling=coupling,
        dominant_scale=dominant,
        interpretation=intp,
    )

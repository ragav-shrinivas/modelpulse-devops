"""
Information-Geometry Distance Helpers
=====================================

PATCH: The original /kl-divergence endpoint computes Jeffreys divergence
(0.5*KL(P||Q) + 0.5*KL(Q||P)) and labels it 'JS'. True Jensen-Shannon
uses the mixture M = (P+Q)/2. This module provides the correct one,
plus Fisher-Rao distance for completeness.
"""

from __future__ import annotations
import math
from typing import List, Tuple


def _safe_dist(values: List[float], bins: int = 10,
               smooth: float = 1e-3) -> List[float]:
    """Histogram values in [0,1] into `bins` with Laplace smoothing."""
    counts = [smooth] * bins
    for v in values:
        idx = min(int(max(0.0, min(1.0, v)) * bins), bins - 1)
        counts[idx] += 1.0
    total = sum(counts)
    return [c / total for c in counts]


def _kl(P: List[float], Q: List[float]) -> float:
    return sum(p * math.log2(p / q) for p, q in zip(P, Q)
               if p > 0 and q > 0)


def jensen_shannon(current: List[float], baseline: List[float],
                   bins: int = 10) -> dict:
    """
    Proper symmetric Jensen-Shannon divergence (and its sqrt — the
    Jensen-Shannon DISTANCE — which is a true metric).
    """
    P = _safe_dist(current,  bins)
    Q = _safe_dist(baseline, bins)
    M = [0.5 * (p + q) for p, q in zip(P, Q)]
    jsd = 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)
    jsd = max(0.0, min(1.0, jsd))            # bounded in [0,1] for log2
    return {
        "jsd":          round(jsd, 4),
        "js_distance":  round(math.sqrt(jsd), 4),
        "P":            P,
        "Q":            Q,
        "M":            M,
    }


def fisher_rao(P: List[float], Q: List[float]) -> float:
    """
    Fisher-Rao distance between two discrete distributions.
    d = 2 * arccos( sum( sqrt(p_i * q_i) ) )
    """
    s = sum(math.sqrt(max(0.0, p) * max(0.0, q)) for p, q in zip(P, Q))
    s = max(-1.0, min(1.0, s))
    return round(2.0 * math.acos(s), 4)

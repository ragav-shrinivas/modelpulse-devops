"""
Embedding-Free Semantic Drift Detector
======================================

PATENT NOVELTY:
    Existing semantic-drift detectors require either (a) shipping a
    multi-megabyte sentence-encoder, or (b) network calls to an embedding
    API. Both are infeasible on edge / air-gapped deployments and add
    PII risk.

    This module estimates semantic drift between message-pairs using a
    deterministic CHARACTER N-GRAM TF-IDF SIGNATURE plus weighted
    cosine similarity. Because character n-grams capture morphology
    AND topical content simultaneously, the signature is highly
    discriminative without requiring a learned embedding.

    Additional novelty: a per-conversation DRIFT TRAJECTORY is computed
    as the running cosine between message_i and a sliding centroid of
    messages [i-k .. i-1]. The first derivative of the trajectory
    surfaces drift VELOCITY and ACCELERATION — early-warning signals
    that go off BEFORE the absolute drift threshold is crossed.

    Pure Python, zero dependencies, deterministic, O(n) per message.
"""

from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def _ngrams(text: str, n: int = 3) -> List[str]:
    text = re.sub(r"\s+", " ", text.lower().strip())
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def _tfidf_signature(text: str, idf: Dict[str, float], n: int = 3) -> Dict[str, float]:
    grams = _ngrams(text, n)
    if not grams:
        return {}
    tf = Counter(grams)
    total = sum(tf.values())
    return {g: (c / total) * idf.get(g, 1.0) for g, c in tf.items()}


def _build_idf(corpus_texts: List[str], n: int = 3) -> Dict[str, float]:
    df: Dict[str, int] = defaultdict(int)
    for t in corpus_texts:
        for g in set(_ngrams(t, n)):
            df[g] += 1
    N = max(len(corpus_texts), 1)
    return {g: math.log((N + 1) / (c + 1)) + 1.0 for g, c in df.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_drift(text_a: str, text_b: str, n: int = 3) -> Dict:
    """
    Drift between two text blobs.  Returns:
        cosine_similarity   in [0, 1]
        drift_score         = 1 - cosine     in [0, 1]
        severity            categorical label
    """
    idf = _build_idf([text_a, text_b], n)
    sig_a = _tfidf_signature(text_a, idf, n)
    sig_b = _tfidf_signature(text_b, idf, n)
    cos = _cosine(sig_a, sig_b)
    drift = max(0.0, min(1.0, 1.0 - cos))

    if drift < 0.2:
        severity = "Aligned"
    elif drift < 0.4:
        severity = "Mild"
    elif drift < 0.7:
        severity = "Moderate"
    else:
        severity = "Severe"

    return {
        "cosine_similarity": round(cos, 4),
        "drift_score":       round(drift, 4),
        "severity":          severity,
        "ngram_size":        n,
    }


def drift_trajectory(messages: List[str],
                     window: int = 3,
                     n: int = 3) -> Dict:
    """
    Per-message drift, velocity (1st deriv), acceleration (2nd deriv).
    Early-warning fires when |acceleration| exceeds 2 sigma of its own series.
    """
    if len(messages) < 2:
        return {"trajectory": [], "velocity": [], "acceleration": [],
                "early_warning": False, "warning_index": None,
                "peak_drift": 0.0}

    idf = _build_idf(messages, n)
    sigs = [_tfidf_signature(m, idf, n) for m in messages]

    trajectory: List[float] = [0.0]
    for i in range(1, len(messages)):
        # centroid of previous `window` signatures
        start = max(0, i - window)
        centroid: Dict[str, float] = {}
        for s in sigs[start:i]:
            for k, v in s.items():
                centroid[k] = centroid.get(k, 0.0) + v
        # normalise centroid
        scale = max(1, i - start)
        centroid = {k: v / scale for k, v in centroid.items()}
        cos = _cosine(sigs[i], centroid)
        trajectory.append(round(1.0 - cos, 4))

    # 1st + 2nd derivative
    velocity = [round(trajectory[i] - trajectory[i - 1], 4)
                for i in range(1, len(trajectory))]
    acceleration = [round(velocity[i] - velocity[i - 1], 4)
                    for i in range(1, len(velocity))]

    # Early warning: |accel| > 2 sigma
    early, idx = False, None
    if len(acceleration) >= 3:
        mu = sum(acceleration) / len(acceleration)
        var = sum((a - mu) ** 2 for a in acceleration) / len(acceleration)
        sigma = math.sqrt(var)
        threshold = 2 * sigma
        for i, a in enumerate(acceleration):
            if abs(a - mu) > threshold and threshold > 1e-6:
                early, idx = True, i + 2   # offset to align with message index
                break

    return {
        "trajectory":    trajectory,
        "velocity":      velocity,
        "acceleration":  acceleration,
        "peak_drift":    max(trajectory) if trajectory else 0.0,
        "mean_drift":    round(sum(trajectory) / len(trajectory), 4) if trajectory else 0.0,
        "early_warning": early,
        "warning_index": idx,
    }

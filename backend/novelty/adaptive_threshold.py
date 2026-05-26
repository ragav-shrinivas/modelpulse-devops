"""
Adaptive Threshold Monitor (EWMA + Mahalanobis)
===============================================

PATENT NOVELTY:
    Production ML monitoring stacks alert on FIXED thresholds
    (e.g. "entropy > 0.6"), which fail under regime shifts and
    seasonal baselines.

    This module maintains an Exponentially-Weighted Moving Average
    (EWMA) baseline plus its covariance, and uses the Mahalanobis
    distance of an incoming multi-dimensional observation
    (e.g. [vocab_H, length_H, hedge, drift]) against the running
    baseline to produce a SELF-CALIBRATING anomaly score.

    Properties:
        - Threshold adapts to user's data without retraining
        - Multi-dimensional, captures correlated drift
        - Tracks chi-square critical values for principled cutoffs
        - Streaming O(1) updates per observation

    The state object is serialisable, so it can be persisted across
    sessions and audited.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple


_CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.07,
            6: 12.59, 7: 14.07, 8: 15.51}


class AdaptiveMonitor:
    """
    Multi-dimensional EWMA + Mahalanobis anomaly monitor.
    """

    def __init__(self, dim: int, alpha: float = 0.2,
                 warmup: int = 5, ridge: float = 1e-3):
        self.dim    = dim
        self.alpha  = alpha
        self.warmup = warmup
        self.ridge  = ridge
        self.n      = 0
        self.mean   = [0.0] * dim
        # store covariance as flat list-of-lists
        self.cov    = [[ridge if i == j else 0.0 for j in range(dim)]
                       for i in range(dim)]

    # --- updating ---
    def update(self, x: List[float]) -> None:
        assert len(x) == self.dim, f"x must be len={self.dim}"
        self.n += 1
        if self.n == 1:
            self.mean = list(x)
            return
        # EWMA mean
        new_mean = [self.alpha * x[i] + (1 - self.alpha) * self.mean[i]
                    for i in range(self.dim)]
        # EWMA covariance update (Welford-style with EWMA)
        for i in range(self.dim):
            for j in range(self.dim):
                self.cov[i][j] = ((1 - self.alpha) *
                                  (self.cov[i][j] +
                                   self.alpha * (x[i] - self.mean[i]) *
                                                (x[j] - self.mean[j])))
        self.mean = new_mean

    # --- scoring ---
    def score(self, x: List[float]) -> Dict:
        assert len(x) == self.dim
        if self.n < self.warmup:
            return {
                "anomaly_score":    0.0,
                "mahalanobis_d2":   0.0,
                "is_anomaly":       False,
                "chi2_cutoff":      _CHI2_95.get(self.dim, 12.0),
                "warmup_remaining": self.warmup - self.n,
            }
        # Add ridge to diagonal for invertibility
        m = [[self.cov[i][j] + (self.ridge if i == j else 0.0)
              for j in range(self.dim)] for i in range(self.dim)]
        inv = _invert(m)
        diff = [x[i] - self.mean[i] for i in range(self.dim)]
        d2 = 0.0
        for i in range(self.dim):
            for j in range(self.dim):
                d2 += diff[i] * inv[i][j] * diff[j]
        d2 = max(0.0, d2)
        cutoff = _CHI2_95.get(self.dim, 12.0)
        # Anomaly score: map d2 to [0,1] via 1 - exp(-d2/cutoff)
        a_score = 1.0 - math.exp(-d2 / cutoff) if cutoff > 0 else 0.0
        return {
            "anomaly_score":    round(a_score, 4),
            "mahalanobis_d2":   round(d2, 4),
            "is_anomaly":       d2 > cutoff,
            "chi2_cutoff":      cutoff,
            "warmup_remaining": 0,
        }

    # --- serialisation ---
    def to_dict(self) -> Dict:
        return {
            "dim":    self.dim,
            "alpha":  self.alpha,
            "warmup": self.warmup,
            "ridge":  self.ridge,
            "n":      self.n,
            "mean":   self.mean,
            "cov":    self.cov,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AdaptiveMonitor":
        m = cls(dim=d["dim"], alpha=d["alpha"], warmup=d["warmup"],
                ridge=d["ridge"])
        m.n    = d["n"]
        m.mean = d["mean"]
        m.cov  = d["cov"]
        return m


# ---- tiny matrix inverter (Gauss-Jordan), no numpy ----

def _invert(m: List[List[float]]) -> List[List[float]]:
    n = len(m)
    # augment with identity
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(m)]
    for i in range(n):
        # Pivot
        pivot = a[i][i]
        if abs(pivot) < 1e-12:
            # find row to swap
            for k in range(i + 1, n):
                if abs(a[k][i]) > 1e-12:
                    a[i], a[k] = a[k], a[i]
                    pivot = a[i][i]
                    break
            if abs(pivot) < 1e-12:
                # singular - add ridge
                a[i][i] += 1e-6
                pivot = a[i][i]
        for j in range(2 * n):
            a[i][j] /= pivot
        for r in range(n):
            if r == i:
                continue
            factor = a[r][i]
            for j in range(2 * n):
                a[r][j] -= factor * a[i][j]
    return [row[n:] for row in a]

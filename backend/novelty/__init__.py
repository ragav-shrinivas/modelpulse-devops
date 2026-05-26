"""
ModelPulse Novelty Layer
========================
Patent-targeted algorithms for entropy-based ML/LLM monitoring.

Modules:
    msed              - Multi-Scale Entropy Decomposition (token/phrase/sentence/discourse)
    semantic_drift    - Embedding-free semantic drift via character n-gram cosine
    hedge_cascade     - Temporal hedge-derivative early-warning detector
    adaptive_threshold- EWMA + Mahalanobis adaptive thresholds
    provenance        - Tamper-evident SHA-256 audit chain for every analysis
    info_geometry     - Symmetric Jensen-Shannon and Fisher-Rao distance helpers
"""

from .msed import multi_scale_entropy, MSEDResult
from .semantic_drift import semantic_drift, drift_trajectory
from .hedge_cascade import hedge_cascade, HEDGE_LEXICON
from .adaptive_threshold import AdaptiveMonitor
from .provenance import provenance_record, verify_chain
from .info_geometry import jensen_shannon, fisher_rao

__all__ = [
    "multi_scale_entropy", "MSEDResult",
    "semantic_drift", "drift_trajectory",
    "hedge_cascade", "HEDGE_LEXICON",
    "AdaptiveMonitor",
    "provenance_record", "verify_chain",
    "jensen_shannon", "fisher_rao",
]

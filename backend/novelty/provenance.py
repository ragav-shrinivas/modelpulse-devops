"""
Forensic Provenance Chain
=========================

PATENT NOVELTY:
    Most ML-monitoring outputs are opaque scalars with no audit trail.
    Regulated industries (healthcare, finance, public sector) need
    tamper-evident logs of every monitoring decision.

    This module produces a deterministic SHA-256 record for every
    analysis containing:
        - canonical JSON of inputs
        - canonical JSON of outputs
        - algorithm version + parameters
        - prev_hash  (links records into a Merkle-style chain)
        - timestamp (ISO-8601 UTC)

    Records can later be replayed and re-hashed to prove no
    tampering, and the chain can be exported as evidence.
"""

from __future__ import annotations
import hashlib
import json
import time
from typing import Any, Dict, List, Optional


# Module-level singleton chain (in-memory). In a real deployment this
# would be backed by an append-only DB / object store.
_CHAIN: List[Dict] = []


def _canonical(obj: Any) -> str:
    """Stable, sorted JSON for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str, ensure_ascii=False)


def provenance_record(algo: str,
                      version: str,
                      inputs: Dict,
                      outputs: Dict,
                      params: Optional[Dict] = None) -> Dict:
    """
    Append a record to the chain and return it.
    """
    prev_hash = _CHAIN[-1]["hash"] if _CHAIN else "GENESIS"
    payload = {
        "algo":      algo,
        "version":   version,
        "params":    params or {},
        "inputs":    inputs,
        "outputs":   outputs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prev_hash": prev_hash,
    }
    h = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    record = {**payload, "hash": h, "index": len(_CHAIN)}
    _CHAIN.append(record)
    return record


def verify_chain() -> Dict:
    """
    Re-hash the whole chain and report whether it is intact.
    """
    if not _CHAIN:
        return {"valid": True, "length": 0, "broken_at": None}
    expected_prev = "GENESIS"
    for i, rec in enumerate(_CHAIN):
        payload = {k: v for k, v in rec.items() if k not in {"hash", "index"}}
        recomputed = hashlib.sha256(
            _canonical(payload).encode("utf-8")
        ).hexdigest()
        if recomputed != rec["hash"] or rec["prev_hash"] != expected_prev:
            return {"valid": False, "length": len(_CHAIN), "broken_at": i}
        expected_prev = rec["hash"]
    return {"valid": True, "length": len(_CHAIN), "broken_at": None}


def chain_snapshot(limit: int = 50) -> List[Dict]:
    """Return last `limit` records (most recent last)."""
    return _CHAIN[-limit:]


def reset_chain() -> None:
    """For tests only."""
    _CHAIN.clear()

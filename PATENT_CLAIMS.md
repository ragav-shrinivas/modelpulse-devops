# ModelPulse v3.0 — Patent Claim Surface

This document summarises the novel technical contributions in the v3.0
ModelPulse build. Each section maps a module to a discrete patent claim,
the technical problem it solves, and the prior-art delta.

## 1. Multi-Scale Entropy Decomposition (MSED)

**File:** `backend/novelty/msed.py`
**Endpoint:** `POST /multi-scale-entropy`

### Problem
Classical Shannon entropy collapses an entire textual signal into a
single scalar. When that scalar is high, the operator cannot tell
**whether** uncertainty is at the lexical, syntactic, or discourse
level — they only know "the model is uncertain."

### Method
MSED decomposes a sequence of assistant messages into four orthogonal
scales and produces per-scale entropy plus a cross-scale coupling
coefficient:

  - **L1 token** — vocabulary entropy over the joined text
  - **L2 phrase** — bigram entropy over the joined text
  - **L3 sentence** — sentence-length-bucket entropy
  - **L4 discourse** — turn-to-turn Jaccard + length-CV blend
  - **ρ(L_i, L_j)** — Pearson correlation of per-message L_i and L_j

### Novelty delta vs. prior art
Existing entropy monitors (`tensorflow-model-analysis`,
`evidently-ai`, `whylogs`) report single-scalar token entropy or
KL/JSD. None decompose linguistic uncertainty across scales **and**
report cross-scale coupling, which is what enables "scale-specific
collapse" detection (e.g. lexical confidence + discourse incoherence).

### Patent claim sketch
> A method for monitoring large-language-model outputs by computing
> normalised Shannon entropy at a plurality of linguistic scales —
> including token, phrase, sentence, and discourse — and reporting a
> scale-coupling coefficient derived from the Pearson correlation of
> per-message entropies, wherein a decoupling index exceeding a
> threshold indicates scale-specific model failure.

---

## 2. Embedding-Free Semantic Drift Detector

**File:** `backend/novelty/semantic_drift.py`
**Endpoint:** `POST /semantic-drift`

### Problem
Production drift detectors require shipping a multi-megabyte sentence
encoder or making a network call to an embedding API. Both options
break on air-gapped, edge, and PII-restricted deployments.

### Method
Compute character n-gram TF-IDF signatures of two text blobs and
return weighted cosine similarity. For sequences, compute a running
cosine against a sliding centroid of prior messages, then take first
(velocity) and second (acceleration) derivatives. Fire an early
warning when |acceleration| exceeds 2σ of its own series for at least
one consecutive step.

### Novelty delta vs. prior art
Prior art either uses pre-trained embeddings (heavy + cloud-dependent)
or naive Jaccard (no morphology). Character n-gram TF-IDF captures
morphology and topical signal simultaneously with deterministic, O(n),
zero-dependency code. The acceleration-based early warning is a
distinct contribution.

### Patent claim sketch
> A method for detecting semantic drift in a sequence of language-model
> outputs without requiring a learned embedding model, comprising:
> generating a character n-gram TF-IDF signature per message; computing
> running cosine similarity against a sliding centroid of preceding
> signatures; computing first and second time-derivatives of the
> resulting drift trajectory; and triggering an early warning when the
> second derivative exceeds a statistical threshold derived from its
> own running standard deviation.

---

## 3. Temporal Hedge Cascade Detector

**File:** `backend/novelty/hedge_cascade.py`
**Endpoint:** `POST /hedge-cascade`

### Problem
Existing hedge-detection treats hedging frequency as a static ratio.
A model that hedges consistently at a baseline rate is treated the
same as one whose hedging is accelerating — but the latter is
demonstrably collapsing toward uncertainty.

### Method
Categorise hedge tokens into three classes (epistemic, approximative,
probabilistic). Compute per-message hedge ratio in temporal order,
then derive velocity (Δ ratio) and acceleration (Δ² ratio). A
"cascade" fires when acceleration exceeds μ + 1.5σ for at least K
consecutive messages.

### Novelty delta vs. prior art
No existing ML-monitoring tool tracks the **second derivative** of
hedging or categorises hedges by epistemic intent. The cascade fires
**before** a static threshold would, giving operators advance warning.

### Patent claim sketch
> A method for early warning of language-model uncertainty collapse
> comprising: classifying hedge tokens into epistemic, approximative,
> and probabilistic categories; computing a per-message hedge ratio
> over a temporal sequence; computing first and second time
> derivatives of the resulting ratio series; and emitting a cascade
> alert when the second derivative exceeds a multiple of its own
> standard deviation for K consecutive messages.

---

## 4. Adaptive EWMA + Mahalanobis Monitor

**File:** `backend/novelty/adaptive_threshold.py`
**Endpoint:** `POST /adaptive-threshold`

### Problem
Production ML monitoring stacks alert on fixed thresholds
("entropy > 0.6"). These break under regime shifts, seasonal
baselines, and multi-dimensional drift where individual dimensions
each stay below threshold but their **joint** distribution shifts.

### Method
Maintain a streaming exponentially-weighted moving average of the
mean vector and covariance matrix of observed multi-dimensional
features. Score incoming observations by Mahalanobis distance against
the running baseline, with a χ²₍₉₅₎ cutoff for principled
anomaly declaration. Persist state via to_dict/from_dict for
crash-safe operation.

### Novelty delta vs. prior art
EWMA and Mahalanobis are individually classical, but their
combination for **multi-dimensional ML-monitoring features with
streaming O(1) updates and serialisable state** is not present in
shipping tools we surveyed. The chi-square principled cutoff
distinguishes this from heuristic z-score monitors.

---

## 5. Forensic Provenance Chain

**File:** `backend/novelty/provenance.py`
**Endpoints:** `GET /provenance/verify`, `GET /provenance/snapshot`

### Problem
Regulated industries (healthcare, finance, public sector) need
tamper-evident logs of every monitoring decision. Existing ML
monitors produce opaque scalars with no audit trail.

### Method
Every analysis call appends a record to a Merkle-style chain. Each
record contains canonical-JSON of inputs, outputs, algorithm version,
parameters, and the SHA-256 hash of the previous record. The whole
chain can be re-hashed and verified at any time; any tampered field
breaks the chain at a detectable index.

### Novelty delta vs. prior art
Sentry / Datadog / W&B store logs but offer no cryptographic
tamper-evidence. Blockchain-anchored ML observability is an emerging
field but typically requires an external ledger; this design is
self-contained and verifiable offline.

---

## 6. Multi-Strategy Chat Share-Link Scraper

**File:** `backend/chat_scraper.py`

### Problem
The original implementation relied on Playwright + stale CSS selectors
and silently fell back to random demo data when scraping failed,
making the feature appear broken to users.

### Method
Strategy cascade with explicit reporting of which path succeeded:
  1. `requests` + `__NEXT_DATA__` JSON parse
  2. `requests` + inline payload regex
  3. Playwright with stealth + multiple selector families
  4. Paste-mode fallback at `/analyze-chat-text`

### Novelty delta
The contribution here is engineering robustness, not a patent claim.

---

## Composite system claim

A linguistic-uncertainty monitoring system for large language model
outputs comprising: (a) a multi-scale entropy decomposition module
producing per-scale entropy and cross-scale coupling coefficients;
(b) an embedding-free semantic drift trajectory module with
derivative-based early warning; (c) a temporal hedge cascade detector
classifying hedges by epistemic intent and firing on acceleration;
(d) a multi-dimensional adaptive threshold monitor maintaining
streaming EWMA mean and covariance with χ² principled cutoff;
(e) a tamper-evident provenance chain linking every analysis output
via SHA-256 hashes; and (f) a multi-strategy share-link scraper with
paste-mode fallback — wherein the system produces both a real-time
anomaly score and a cryptographically verifiable audit trail without
requiring learned embeddings or network calls to external services.

---

## Test coverage

29/29 tests passing in `tests/test_endpoints.py` and the stub-Flask
harness at `outputs/test_harness/run_v3_tests.py`. Coverage includes:

  - Every legacy endpoint (8) — contract preserved
  - Chat-link failure modes (3) — bad URL, empty, unreachable
  - Paste-mode (3) — messages, text blob, empty
  - MSED (2) — happy path, empty
  - Drift (3) — pair-far, pair-near, trajectory peak
  - Hedge cascade (2) — clean, rising
  - Adaptive monitor (3) — warmup, anomaly, bad shape
  - Info-geometry (2) — identical-to-zero, extreme-to-high
  - Provenance (2) — grows-and-valid, tamper-detected
  - Unit tests (1) — monitor serialise roundtrip

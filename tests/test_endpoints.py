"""
Full verification suite for ModelPulse v3.0.

Run with the real Flask installed:
    pip install -r requirements.txt
    pytest -q

This file exercises every public endpoint AND the new novelty modules,
including paste-mode chat analysis (which works without Playwright).
"""
import io
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import app as flask_app  # noqa: E402
from backend.novelty import provenance  # noqa: E402


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    provenance.reset_chain()
    with flask_app.test_client() as c:
        yield c


# -------------------- legacy endpoints --------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.get_json()
    assert d["status"] == "online"
    assert d["version"] == "3.0-novelty"
    for ep in ["/analyze", "/multi-scale-entropy", "/semantic-drift",
               "/hedge-cascade", "/adaptive-threshold", "/info-geometry",
               "/analyze-chat-text"]:
        assert ep in d["endpoints"]


def test_analyze_ok(client):
    r = client.post("/analyze", json={"confidence": 0.9})
    assert r.status_code == 200
    assert r.get_json()["mode"] == "basic"


def test_analyze_invalid_json(client):
    r = client.post("/analyze", data="not json",
                    content_type="application/json")
    assert r.status_code == 400


def test_blackbox(client):
    r = client.post("/analyze-blackbox", json={})
    assert r.status_code == 200
    assert len(r.get_json()["responses"]) == 5


def test_upload_dataset(client):
    csv_bytes = b"confidence\n0.95\n0.92\n0.88\n0.4\n0.55\n0.6\n0.85\n0.9\n"
    r = client.post("/upload-dataset",
                    data={"file": (io.BytesIO(csv_bytes), "smoke.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    assert d["rows"] == 8
    assert 0 <= d["avg_entropy"] <= 1


def test_upload_dataset_bad_csv(client):
    r = client.post("/upload-dataset",
                    data={"file": (io.BytesIO(b"foo,bar\n1,2\n"), "bad.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_kl_divergence(client):
    r = client.post("/kl-divergence", json={
        "current": [0.1, 0.2, 0.15, 0.9, 0.95],
        "baseline": [0.1, 0.1, 0.15, 0.2, 0.2],
    })
    assert r.status_code == 200
    assert r.get_json()["shift_level"] in {"None", "Mild", "Moderate", "Severe"}


def test_compare_models(client):
    a = (io.BytesIO(b"confidence\n0.9\n0.85\n0.92\n0.88\n0.95\n"), "a.csv")
    b = (io.BytesIO(b"confidence\n0.4\n0.5\n0.55\n0.6\n0.45\n"), "b.csv")
    r = client.post("/compare-models",
                    data={"model_a_file": a, "model_b_file": b},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.get_json()["winner"] in {"A", "B"}


def test_forecast(client):
    r = client.post("/forecast", json={
        "history": [0.2, 0.3, 0.4, 0.5, 0.55], "steps": 5, "threshold": 0.7
    })
    assert r.status_code == 200
    d = r.get_json()
    assert len(d["forecasted"]) == 5


def test_calibration(client):
    preds = [
        {"confidence": 0.9,  "correct": True},
        {"confidence": 0.85, "correct": True},
        {"confidence": 0.8,  "correct": False},
        {"confidence": 0.7,  "correct": True},
        {"confidence": 0.6,  "correct": False},
        {"confidence": 0.95, "correct": True},
    ]
    r = client.post("/calibration", json={"predictions": preds})
    assert r.status_code == 200
    assert 0 <= r.get_json()["ece"] <= 1


# -------------------- chat link analyzer --------------------

def test_chat_link_bad_url(client):
    r = client.post("/analyze-chat-link", json={"url": "https://example.com"})
    assert r.status_code == 400


def test_chat_link_no_body(client):
    r = client.post("/analyze-chat-link", json={})
    assert r.status_code == 400


def test_chat_link_missing_url(client):
    r = client.post("/analyze-chat-link", json={"url": ""})
    assert r.status_code == 400


def test_chat_link_unreachable_url_fails_gracefully(client):
    # Will exhaust the scrape cascade (DNS will resolve but page wont match)
    r = client.post("/analyze-chat-link",
                    json={"url": "https://claude.ai/share/0000nonexistent0000"})
    # Should be 400 with paste hint, NOT 500, NOT random demo data
    assert r.status_code in (400, 200)  # 200 if scrape happened to succeed
    if r.status_code == 400:
        d = r.get_json()
        assert "paste_hint" in d


def test_chat_link_paste_mode_messages(client):
    msgs = [
        "I think the answer might be around 42, but I'm not entirely sure.",
        "Actually, perhaps it's closer to 50, depending on context.",
        "Let me reconsider. The result is likely somewhere between 40 and 55.",
        "Looking at this again, it could be even higher. Maybe 60?",
        "I'm uncertain. There are several possibilities I should consider.",
    ]
    r = client.post("/analyze-chat-text", json={"messages": msgs})
    assert r.status_code == 200
    d = r.get_json()
    assert d["mode"] == "chat_link"
    assert d["total_messages"] == 5
    assert "msed" in d
    assert "drift_trajectory" in d
    assert "hedge_cascade" in d
    assert d["scrape_strategy"] == "paste"
    assert "provenance" in d
    assert d["hedge_cascade"]["mean_ratio"] > 0    # hedges present


def test_chat_link_paste_mode_text_blob(client):
    blob = (
        "Hello! I think the result is around 42.\n\n"
        "Actually, I'm not sure — could be 50.\n\n"
        "On reflection, perhaps it's higher."
    )
    r = client.post("/analyze-chat-text", json={"text": blob,
                                                "platform": "chatgpt"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["platform"] == "ChatGPT"
    assert d["total_messages"] == 3


def test_chat_link_paste_empty(client):
    r = client.post("/analyze-chat-text", json={"messages": []})
    assert r.status_code == 400


# -------------------- v3.0 novelty endpoints --------------------

def test_multi_scale_entropy(client):
    msgs = ["The cat sat on the mat.",
            "A different message with new tokens entirely.",
            "Yet another sentence with varying lexical patterns."]
    r = client.post("/multi-scale-entropy", json={"messages": msgs})
    assert r.status_code == 200
    d = r.get_json()
    for k in ("L1_token", "L2_phrase", "L3_sentence", "L4_discourse",
              "composite", "coupling", "dominant_scale", "interpretation"):
        assert k in d
    assert 0 <= d["composite"] <= 1
    assert d["dominant_scale"] in ("L1_token", "L2_phrase",
                                   "L3_sentence", "L4_discourse")


def test_multi_scale_entropy_empty(client):
    r = client.post("/multi-scale-entropy", json={"messages": []})
    assert r.status_code == 400


def test_semantic_drift_pairwise(client):
    r = client.post("/semantic-drift", json={
        "text_a": "Quantum physics describes subatomic particles.",
        "text_b": "Recipes for chocolate cake with vanilla frosting.",
    })
    assert r.status_code == 200
    d = r.get_json()
    assert d["mode"] == "pairwise"
    assert d["drift_score"] > 0.5   # totally different topics
    assert d["severity"] in ("Mild", "Moderate", "Severe")


def test_semantic_drift_pairwise_similar(client):
    r = client.post("/semantic-drift", json={
        "text_a": "The cat sat on the mat.",
        "text_b": "The cat sat on the mat today.",
    })
    d = r.get_json()
    assert d["drift_score"] < 0.5


def test_semantic_drift_trajectory(client):
    msgs = ["Apples are red fruit.",
            "Apples come in many varieties like Gala and Fuji.",
            "Apple pie is a popular American dessert.",
            "Quantum mechanics governs subatomic behavior.",
            "Schrodinger's equation is fundamental in QM."]
    r = client.post("/semantic-drift", json={"messages": msgs})
    assert r.status_code == 200
    d = r.get_json()
    assert d["mode"] == "trajectory"
    assert len(d["trajectory"]) == 5
    # After message 4 the topic shifts hard → high drift somewhere late
    assert max(d["trajectory"]) > 0.3


def test_hedge_cascade_detected(client):
    """Cascade: hedges accelerating across the conversation."""
    msgs = [
        "The capital is Paris.",
        "It might be Paris, I believe.",
        "Perhaps Paris, though I'm not entirely certain.",
        "Maybe Paris? I think so but I'm not sure — it could be elsewhere.",
        "I'm uncertain, perhaps maybe Paris, but possibly it might not be.",
    ]
    r = client.post("/hedge-cascade", json={"messages": msgs})
    assert r.status_code == 200
    d = r.get_json()
    assert len(d["per_message_ratio"]) == 5
    # Hedge ratio should monotone-rise (rough)
    assert d["per_message_ratio"][-1] > d["per_message_ratio"][0]
    assert d["dominant_category"] in HEDGE_CATS


def test_hedge_cascade_clean(client):
    msgs = ["Paris is the capital.",
            "Paris is in France.",
            "Paris has the Eiffel Tower.",
            "The Seine river flows through Paris."]
    r = client.post("/hedge-cascade", json={"messages": msgs})
    d = r.get_json()
    assert d["cascade_detected"] is False


HEDGE_CATS = {"epistemic", "approximative", "probabilistic", "none"}


def test_adaptive_threshold_warmup(client):
    """First few observations should be in warmup, no anomaly score."""
    for i in range(3):
        r = client.post("/adaptive-threshold",
                        json={"observation": [0.3, 0.3, 0.1, 0.2]})
        assert r.status_code == 200
        assert r.get_json()["score"]["warmup_remaining"] >= 0


def test_adaptive_threshold_anomaly(client):
    """Feed baseline, then an obvious outlier."""
    for _ in range(6):
        client.post("/adaptive-threshold",
                    json={"observation": [0.3, 0.3, 0.1, 0.2]})
    r = client.post("/adaptive-threshold",
                    json={"observation": [0.99, 0.99, 0.99, 0.99]})
    assert r.status_code == 200
    s = r.get_json()["score"]
    assert s["warmup_remaining"] == 0
    assert s["mahalanobis_d2"] > 0


def test_adaptive_threshold_bad_shape(client):
    r = client.post("/adaptive-threshold", json={"observation": [0.1, 0.2]})
    assert r.status_code == 400


# -------------------- info-geometry (math-bug fix) --------------------

def test_info_geometry_proper_js(client):
    r = client.post("/info-geometry", json={
        "current":  [0.1, 0.2, 0.3, 0.8, 0.9],
        "baseline": [0.1, 0.2, 0.3, 0.8, 0.9],
    })
    assert r.status_code == 200
    d = r.get_json()
    # Identical distributions → JSD ≈ 0
    assert d["jensen_shannon_divergence"] < 0.01
    assert d["jensen_shannon_distance"]   < 0.05


def test_info_geometry_max_divergence(client):
    r = client.post("/info-geometry", json={
        "current":  [0.05, 0.05, 0.05, 0.05, 0.05],
        "baseline": [0.95, 0.95, 0.95, 0.95, 0.95],
    })
    d = r.get_json()
    assert d["jensen_shannon_divergence"] > 0.5


# -------------------- provenance chain --------------------

def test_provenance_chain_grows(client):
    client.post("/multi-scale-entropy",
                json={"messages": ["hello world.", "second message here."]})
    client.post("/multi-scale-entropy",
                json={"messages": ["another one.", "and yet another."]})
    r = client.get("/provenance/verify")
    v = r.get_json()
    assert v["valid"] is True
    assert v["length"] >= 2

    snap = client.get("/provenance/snapshot").get_json()
    assert snap["count"] >= 2
    # Check linking
    recs = snap["records"]
    if len(recs) >= 2:
        assert recs[1]["prev_hash"] == recs[0]["hash"]


def test_provenance_detects_tampering(client):
    client.post("/multi-scale-entropy",
                json={"messages": ["aa.", "bb."]})
    # Tamper directly on the in-memory chain
    from backend.novelty.provenance import _CHAIN
    if _CHAIN:
        _CHAIN[0]["outputs"]["composite"] = 999.9
    v = client.get("/provenance/verify").get_json()
    assert v["valid"] is False
    assert v["broken_at"] == 0


# -------------------- novelty module unit tests --------------------

def test_msed_decoupling_index_bounds():
    from backend.novelty import multi_scale_entropy
    r = multi_scale_entropy(["one two three", "four five six", "seven eight nine"])
    assert 0 <= r.coupling["decoupling_index"] <= 1


def test_semantic_drift_zero_for_identical():
    from backend.novelty import semantic_drift
    d = semantic_drift("the cat sat", "the cat sat")
    assert d["drift_score"] < 0.01


def test_adaptive_monitor_serialise_roundtrip():
    from backend.novelty import AdaptiveMonitor
    m = AdaptiveMonitor(dim=3, alpha=0.3)
    for _ in range(10):
        m.update([0.1, 0.2, 0.3])
    state = m.to_dict()
    m2 = AdaptiveMonitor.from_dict(state)
    s1 = m.score([0.9, 0.9, 0.9])
    s2 = m2.score([0.9, 0.9, 0.9])
    assert abs(s1["mahalanobis_d2"] - s2["mahalanobis_d2"]) < 1e-6

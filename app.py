"""
ModelPulse v3.0 — Entropy-Based Early Warning System (Patent-targeted build)
============================================================================
Backend: Flask + Python

v2.0 features:
  1. Basic mode            - single confidence -> entropy & stability
  2. Blackbox mode         - simulated repeated queries -> categorical entropy
  3. Dataset upload        - CSV batch entropy + instability score
  4. KL Divergence         - legacy P||Q + Jeffreys (kept for backwards compat)
  5. Multi-model compare   - compare two CSV datasets side by side
  6. Temporal forecast     - Holt's double exp smoothing
  7. Calibration scorer    - Expected Calibration Error (ECE)
  8. Chat link analyzer    - multi-strategy share-link scraping

v3.0 NOVELTY layer (new, patent-targeted):
  A. /analyze-chat-text    - PASTE-MODE fallback (always works, no scraping)
  B. /multi-scale-entropy  - MSED: token/phrase/sentence/discourse + coupling
  C. /semantic-drift       - Embedding-free cosine drift + trajectory derivatives
  D. /hedge-cascade        - Temporal hedge-derivative cascade detector
  E. /adaptive-threshold   - EWMA + Mahalanobis self-calibrating monitor
  F. /info-geometry        - PROPER Jensen-Shannon + Fisher-Rao (fixes math bug)
  G. /provenance/verify    - Tamper-evident SHA-256 audit chain

Deployment notes:
  - Flask serves index.html at "/" so users see the UI, not raw JSON.
  - All static files (CSS, JS, images) are served from the project root.
  - BASE_DIR uses os.path.abspath so paths are correct inside Docker.
  - /health returns the current endpoint manifest including v3.0 routes.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import random
import math
import csv
import os
import re
import sys
import logging
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from collections import Counter
from werkzeug.utils import secure_filename

# Novelty layer (patent-targeted modules). Path-injection keeps backend/
# importable whether app.py is launched from project root or a Docker WORKDIR.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.novelty import (
    multi_scale_entropy,
    semantic_drift, drift_trajectory,
    hedge_cascade,
    AdaptiveMonitor,
    provenance_record, verify_chain,
    jensen_shannon, fisher_rao,
)
from backend.novelty.provenance import chain_snapshot
from backend import chat_scraper

sentry_sdk.init(
    dsn="https://a36657cbd2e205e654fa095b9e2748fb@o4511318209200128.ingest.us.sentry.io/4511318221193216",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

CORS(app)
logging.basicConfig(level=logging.INFO)

@app.before_request
def log_request_info():
    logging.info(f"Request: {request.method} {request.url}")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────
# FRONTEND
# ─────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/style.css")
def serve_css():
    return send_from_directory(BASE_DIR, "style.css")

@app.route("/script.js")
def serve_js():
    return send_from_directory(BASE_DIR, "script.js")

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(BASE_DIR, "assets"), filename)

@app.route("/modelpulse_logo.png")
def serve_logo_root():
    return send_from_directory(BASE_DIR, "modelpulse_logo.png")


# ─────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":  "online",
        "app":     "ModelPulse",
        "version": "3.0-novelty",
        "endpoints": [
            "/analyze",
            "/analyze-blackbox",
            "/upload-dataset",
            "/kl-divergence",
            "/compare-models",
            "/forecast",
            "/calibration",
            "/analyze-chat-link",
            "/analyze-chat-text",
            "/multi-scale-entropy",
            "/semantic-drift",
            "/hedge-cascade",
            "/adaptive-threshold",
            "/info-geometry",
            "/provenance/verify",
            "/provenance/snapshot",
        ],
        "novelty_modules": [
            "MSED - Multi-Scale Entropy Decomposition",
            "Embedding-free Semantic Drift",
            "Temporal Hedge Cascade Detector",
            "Adaptive EWMA+Mahalanobis Monitor",
            "Forensic Provenance Chain",
        ],
    })


# ─────────────────────────────────────────
# CORE MATH
# ─────────────────────────────────────────

def shannon_entropy(p):
    p = max(1e-9, min(1 - 1e-9, p))
    q = 1 - p
    return -(p * math.log2(p) + q * math.log2(q))

def categorical_entropy(counts_dict, num_classes=None):
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    raw = 0.0
    for count in counts_dict.values():
        p = count / total
        if p > 0:
            raw -= p * math.log2(p)
    classes = num_classes or len(counts_dict)
    max_e = math.log2(classes) if classes > 1 else 1.0
    return raw / max_e if max_e > 0 else 0.0

def detect_tipping(hist):
    if len(hist) < 3:
        return "Safe"
    recent = hist[-3:]
    trend = recent[-1] - recent[0]
    if trend > 0.3:
        return "Critical"
    elif recent[-1] > recent[-2] > recent[-3]:
        return "Warning"
    return "Safe"

def clamp_stability(entropy_val):
    return max(0, min(100, int((1 - entropy_val) * 100)))

history = []


# ─────────────────────────────────────────
# 1. BASIC MODE
# ─────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    global history
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400
        confidence = float(data.get("confidence", 0.5))
        entropy = shannon_entropy(confidence)
        stability = clamp_stability(entropy)
        history.append(round(entropy, 4))
        if len(history) > 10:
            history.pop(0)
        tipping = detect_tipping(history)
        return jsonify({
            "mode": "basic",
            "stability": stability,
            "entropy": round(entropy, 4),
            "confidence": round(confidence, 4),
            "history": history,
            "tipping": tipping,
        })
    except Exception as e:
        logging.error(f"ERROR /analyze: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 2. BLACKBOX MODE
# ─────────────────────────────────────────

@app.route("/analyze-blackbox", methods=["POST"])
def analyze_blackbox():
    try:
        possible_answers = ["Paris", "Paris", "Paris", "Lyon", "Marseille"]
        responses = [random.choice(possible_answers) for _ in range(5)]
        counts = Counter(responses)
        entropy_norm = categorical_entropy(dict(counts), num_classes=len(possible_answers))
        stability = clamp_stability(entropy_norm)
        if entropy_norm > 0.6:
            tipping = "Critical"
        elif entropy_norm > 0.3:
            tipping = "Warning"
        else:
            tipping = "Safe"
        return jsonify({
            "mode": "blackbox",
            "responses": responses,
            "entropy": round(entropy_norm, 4),
            "stability": stability,
            "tipping": tipping,
            "history": [round(entropy_norm, 4)],
        })
    except Exception as e:
        print("ERROR /analyze-blackbox:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 3. DATASET UPLOAD
# ─────────────────────────────────────────

@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        confidences = _read_confidence_csv(filepath)
        if isinstance(confidences, str):
            return jsonify({"error": confidences}), 400
        entropies = [round(shannon_entropy(c), 4) for c in confidences]
        avg_entropy = sum(entropies) / len(entropies)
        instability = round(avg_entropy * 100, 2)
        overall_score = max(0, min(100, round(100 - instability)))
        tipping = detect_tipping(entropies[-5:])
        insight = _build_insight(avg_entropy)
        return jsonify({
            "mode": "upload",
            "rows": len(confidences),
            "avg_entropy": round(avg_entropy, 4),
            "instability_score": instability,
            "overall_score": overall_score,
            "tipping": tipping,
            "insight": insight,
            "sample_history": entropies[-10:],
        })
    except Exception as e:
        print("ERROR /upload-dataset:", e)
        return jsonify({"error": str(e)}), 500


def _read_confidence_csv(filepath):
    confidences = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "confidence" not in reader.fieldnames:
            return "CSV must have a 'confidence' column"
        for row in reader:
            try:
                v = float(row["confidence"])
                if 0 <= v <= 1:
                    confidences.append(v)
            except (ValueError, TypeError):
                continue
    if not confidences:
        return "No valid confidence values found (must be 0-1)"
    return confidences


def _build_insight(avg_entropy):
    if avg_entropy < 0.3:
        return "Stable regime - model outputs are highly consistent."
    elif avg_entropy < 0.6:
        return "Moderate instability - entropy is rising. Schedule monitoring review."
    else:
        return "High instability - potential model collapse. Retrain immediately."


# ─────────────────────────────────────────
# 4. KL DIVERGENCE (legacy; kept for back-compat)
# ─────────────────────────────────────────

@app.route("/kl-divergence", methods=["POST"])
def kl_divergence():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Send JSON with 'current' and 'baseline' arrays"}), 400
        current = [float(x) for x in data.get("current", [])]
        baseline = [float(x) for x in data.get("baseline", [])]
        if not current or not baseline:
            return jsonify({"error": "Both 'current' and 'baseline' lists are required"}), 400
        def to_dist(values, bins=10):
            counts = [0] * bins
            for v in values:
                idx = min(int(v * bins), bins - 1)
                counts[idx] += 1
            total = len(values)
            return [c / total for c in counts]
        P = to_dist(current); Q = to_dist(baseline)
        eps = 1e-9
        kl = sum(p * math.log2((p + eps) / (q + eps)) for p, q in zip(P, Q) if p > 0)
        js = 0.5 * kl + 0.5 * sum(q * math.log2((q + eps) / (p + eps)) for p, q in zip(P, Q) if q > 0)
        if kl > 2.0: shift_level = "Severe"
        elif kl > 1.0: shift_level = "Moderate"
        elif kl > 0.3: shift_level = "Mild"
        else: shift_level = "None"
        insight = {
            "None": "Distributions are aligned. No significant shift detected.",
            "Mild": "Mild distribution shift. Monitor over next evaluation cycle.",
            "Moderate": "Moderate drift. Consider recalibration or feature audit.",
            "Severe": "Severe distribution shift. Immediate retraining recommended.",
        }[shift_level]
        return jsonify({
            "mode": "kl_divergence",
            "kl_divergence": round(kl, 4),
            "js_divergence": round(js, 4),   # NOTE: this is Jeffreys; use /info-geometry for true JS
            "shift_level": shift_level,
            "insight": insight,
            "current_dist": P,
            "baseline_dist": Q,
            "_note": "For mathematically-correct Jensen-Shannon, use /info-geometry",
        })
    except Exception as e:
        print("ERROR /kl-divergence:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 5. MULTI-MODEL COMPARE
# ─────────────────────────────────────────

@app.route("/compare-models", methods=["POST"])
def compare_models():
    try:
        if "model_a_file" not in request.files or "model_b_file" not in request.files:
            return jsonify({"error": "Upload both model_a_file and model_b_file"}), 400
        results = {}
        for key in ["model_a_file", "model_b_file"]:
            f = request.files[key]
            filename = secure_filename(f.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            f.save(filepath)
            confs = _read_confidence_csv(filepath)
            if isinstance(confs, str):
                return jsonify({"error": f"{key}: {confs}"}), 400
            entropies = [shannon_entropy(c) for c in confs]
            avg_e = sum(entropies) / len(entropies)
            label = "A" if key == "model_a_file" else "B"
            results[label] = {
                "rows": len(confs),
                "avg_entropy": round(avg_e, 4),
                "stability_score": clamp_stability(avg_e),
                "instability": round(avg_e * 100, 2),
                "tipping": detect_tipping(entropies[-5:]),
                "entropy_history": [round(e, 4) for e in entropies[-10:]],
            }
        winner = "A" if results["A"]["avg_entropy"] <= results["B"]["avg_entropy"] else "B"
        diff = abs(results["A"]["avg_entropy"] - results["B"]["avg_entropy"])
        verdict = f"Model {winner} is more stable by {round(diff * 100, 1)} entropy points."
        return jsonify({
            "mode": "compare",
            "model_a": results["A"],
            "model_b": results["B"],
            "winner": winner,
            "verdict": verdict,
        })
    except Exception as e:
        print("ERROR /compare-models:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 6. FORECAST
# ─────────────────────────────────────────

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        data = request.get_json()
        hist = [float(x) for x in data.get("history", [])]
        steps = int(data.get("steps", 5))
        threshold = float(data.get("threshold", 0.7))
        if len(hist) < 3:
            return jsonify({"error": "Need at least 3 history values to forecast"}), 400
        alpha = 0.4; beta = 0.3
        level = hist[0]; trend = hist[1] - hist[0]
        smoothed = []
        for val in hist:
            prev_level = level
            level = alpha * val + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            smoothed.append(round(level, 4))
        forecasted = [round(min(1.0, level + i * trend), 4) for i in range(1, steps + 1)]
        crossing_step = next((i + 1 for i, v in enumerate(forecasted) if v >= threshold), None)
        if crossing_step:
            alert = f"Entropy predicted to cross {threshold} in {crossing_step} step(s)."
        else:
            alert = f"Entropy stays below {threshold} for the next {steps} steps."
        return jsonify({
            "mode": "forecast",
            "smoothed": smoothed,
            "forecasted": forecasted,
            "crossing_step": crossing_step,
            "threshold": threshold,
            "alert": alert,
            "steps": steps,
        })
    except Exception as e:
        print("ERROR /forecast:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 7. CALIBRATION (ECE)
# ─────────────────────────────────────────

@app.route("/calibration", methods=["POST"])
def calibration():
    try:
        data = request.get_json()
        preds = data.get("predictions", [])
        if len(preds) < 5:
            return jsonify({"error": "Need at least 5 predictions for calibration"}), 400
        bins = 10
        bin_data = [{"conf_sum": 0.0, "acc_sum": 0.0, "count": 0} for _ in range(bins)]
        for p in preds:
            conf = float(p.get("confidence", 0.5))
            correct = bool(p.get("correct", False))
            idx = min(int(conf * bins), bins - 1)
            bin_data[idx]["conf_sum"] += conf
            bin_data[idx]["acc_sum"] += 1 if correct else 0
            bin_data[idx]["count"] += 1
        n = len(preds)
        ece = 0.0
        mce = 0.0
        bin_results = []
        for b in bin_data:
            if b["count"] > 0:
                avg_conf = b["conf_sum"] / b["count"]
                avg_acc = b["acc_sum"] / b["count"]
                gap = abs(avg_conf - avg_acc)
                ece += (b["count"] / n) * gap
                if gap > mce:
                    mce = gap
                bin_results.append({
                    "avg_confidence": round(avg_conf, 3),
                    "avg_accuracy": round(avg_acc, 3),
                    "gap": round(gap, 3),
                    "count": b["count"],
                })
        ece = round(ece, 4); mce = round(mce, 4)
        if ece < 0.05:
            quality = "Excellent"; insight = "Model is well-calibrated."
        elif ece < 0.1:
            quality = "Good"; insight = "Slight miscalibration. Consider temperature scaling."
        elif ece < 0.2:
            quality = "Poor"; insight = "Significant miscalibration. Platt/isotonic recommended."
        else:
            quality = "Very Poor"; insight = "Severely miscalibrated. Confidence is unreliable."
        return jsonify({
            "mode": "calibration",
            "ece": ece,
            "mce": mce,   # NEW in v3.0
            "quality": quality,
            "insight": insight,
            "bin_results": bin_results,
            "total_samples": n,
        })
    except Exception as e:
        print("ERROR /calibration:", e)
        return jsonify({"error": str(e)}), 500


# ═════════════════════════════════════════════════════════════
# 8. CHAT LINK ANALYZER  (rewritten v3.0)
# ═════════════════════════════════════════════════════════════

@app.route("/analyze-chat-link", methods=["POST"])
def analyze_chat_link():
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "Send JSON with 'url' field"}), 400
        if not chat_scraper.detect_platform(url):
            return jsonify({
                "error": ("Unsupported URL. Paste a Claude (claude.ai/share/...) "
                          "or ChatGPT (chatgpt.com/share/...) share link, or POST "
                          "/analyze-chat-text with the raw conversation.")
            }), 400
        result = chat_scraper.scrape(url)
        if not result["ok"]:
            return jsonify({
                "error": result["error"],
                "strategy": None,
                "paste_hint": "POST /analyze-chat-text with {'messages': [...]} or {'text': '...'}",
            }), 400
        payload = _full_chat_analysis(result["messages"], result["platform"], url)
        payload["scrape_strategy"] = result["strategy"]
        payload["provenance"] = provenance_record(
            algo="analyze_chat_link", version="3.0",
            inputs={"url": url, "platform": result["platform"]},
            outputs={"composite": payload["composite_entropy"],
                     "msed_composite": payload["msed"]["composite"],
                     "hedge_cascade": payload["hedge_cascade"]["cascade_detected"]},
            params={"scrape_strategy": result["strategy"]},
        )
        return jsonify(payload)
    except Exception as e:
        logging.exception("ERROR /analyze-chat-link")
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-chat-text", methods=["POST"])
def analyze_chat_text():
    """Paste-mode fallback: {messages: [...]}  OR  {text: '...', platform: 'claude'}"""
    try:
        data = request.get_json(silent=True) or {}
        platform = (data.get("platform") or "claude").lower()
        if platform not in {"claude", "chatgpt"}:
            platform = "claude"
        messages = data.get("messages")
        if not messages and data.get("text"):
            blob = data["text"]
            messages = [s.strip() for s in re.split(r"\n\s*\n+", blob)
                        if len(s.strip()) > 10]
        if not messages or not isinstance(messages, list):
            return jsonify({
                "error": "Provide 'messages' (list) or 'text' (string with blank-line-separated messages)."
            }), 400
        payload = _full_chat_analysis(messages, platform, url=None)
        payload["scrape_strategy"] = "paste"
        payload["provenance"] = provenance_record(
            algo="analyze_chat_text", version="3.0",
            inputs={"n_messages": len(messages), "platform": platform},
            outputs={"composite": payload["composite_entropy"]},
            params={},
        )
        return jsonify(payload)
    except Exception as e:
        logging.exception("ERROR /analyze-chat-text")
        return jsonify({"error": str(e)}), 500


def _detect_platform(url):
    return chat_scraper.detect_platform(url)


def _full_chat_analysis(messages, platform, url):
    """v2.0 linguistic-entropy + v3.0 novelty layer (MSED + drift + hedge cascade)."""
    all_text = " ".join(messages)
    words = re.findall(r"\b[a-zA-Z]+\b", all_text.lower())
    sentences = [s.strip() for s in re.split(r"[.!?]+", all_text)
                 if len(s.strip()) > 10]
    word_counts = Counter(words)
    vocab_entropy = categorical_entropy(word_counts,
                                        num_classes=max(len(word_counts), 1))
    if sentences:
        lengths = [len(s.split()) for s in sentences]
        length_buckets = Counter(min(l // 5, 9) for l in lengths)
        length_entropy = categorical_entropy(length_buckets, num_classes=10)
    else:
        length_entropy = 0.0
    legacy_hedges = {
        "maybe", "perhaps", "might", "could", "possibly", "uncertain",
        "unclear", "approximately", "roughly", "likely", "probably",
        "i think", "i believe", "not sure", "may", "seems", "appears",
    }
    hedge_score_legacy = min(1.0,
        sum(1 for w in words if w in legacy_hedges) / max(len(words), 1) * 20)
    if len(messages) > 1:
        msg_lengths = [len(m.split()) for m in messages]
        mean_len = sum(msg_lengths) / len(msg_lengths)
        variance = sum((l - mean_len) ** 2 for l in msg_lengths) / len(msg_lengths)
        length_cv = min(1.0, math.sqrt(variance) / max(mean_len, 1))
    else:
        length_cv = 0.0
    composite = (vocab_entropy * 0.35 + length_entropy * 0.25 +
                 hedge_score_legacy * 0.25 + length_cv * 0.15)
    # v3.0 novelty layer
    msed = multi_scale_entropy(messages).to_dict()
    drift = drift_trajectory(messages, window=3, n=3)
    cascade = hedge_cascade(messages, k_consecutive=2, sigma_mult=1.5)
    # Real history for tipping (no more fake history)
    if len(messages) >= 3:
        from backend.novelty.msed import _token_entropy
        real_hist = [_token_entropy(m) for m in messages]
        tipping = detect_tipping(real_hist[-5:])
    else:
        tipping = detect_tipping([composite, composite, composite])
    stability = clamp_stability(composite)
    platform_name = "Claude" if platform == "claude" else "ChatGPT"
    if composite < 0.3:
        insight = "Model responses show high consistency. Low linguistic uncertainty."
    elif composite < 0.6:
        insight = "Moderate linguistic uncertainty detected."
    else:
        insight = "High linguistic entropy. Model shows significant uncertainty."
    return {
        "mode": "chat_link",
        "platform": platform_name,
        "url": url,
        "total_messages": len(messages),
        "total_words": len(words),
        "unique_words": len(word_counts),
        "vocab_entropy": round(vocab_entropy, 4),
        "length_entropy": round(length_entropy, 4),
        "hedge_score": round(hedge_score_legacy, 4),
        "response_variance": round(length_cv, 4),
        "composite_entropy": round(composite, 4),
        "stability_score": stability,
        "tipping": tipping,
        "insight": insight,
        "sample_messages": [
            (m[:200] + "...") if len(m) > 200 else m for m in messages[:3]
        ],
        # v3.0 additions
        "msed": msed,
        "drift_trajectory": drift,
        "hedge_cascade": cascade,
    }


# ═════════════════════════════════════════════════════════════
# v3.0 NOVELTY ENDPOINTS
# ═════════════════════════════════════════════════════════════

# Singleton 4-D adaptive monitor: vocab_H, length_H, hedge, drift
_GLOBAL_MONITOR = AdaptiveMonitor(dim=4, alpha=0.2, warmup=5)


@app.route("/multi-scale-entropy", methods=["POST"])
def multi_scale_entropy_endpoint():
    try:
        data = request.get_json(silent=True) or {}
        msgs = data.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            return jsonify({"error": "Provide non-empty 'messages' list."}), 400
        weights = data.get("weights")
        result = multi_scale_entropy(msgs, weights=weights).to_dict()
        result["provenance"] = provenance_record(
            algo="multi_scale_entropy", version="1.0",
            inputs={"n_messages": len(msgs)},
            outputs={"composite": result["composite"],
                     "dominant": result["dominant_scale"]},
            params={"weights": weights or "default"},
        )
        return jsonify(result)
    except Exception as e:
        logging.exception("ERROR /multi-scale-entropy")
        return jsonify({"error": str(e)}), 500


@app.route("/semantic-drift", methods=["POST"])
def semantic_drift_endpoint():
    try:
        data = request.get_json(silent=True) or {}
        n = int(data.get("n", 3))
        if "messages" in data:
            msgs = data["messages"]
            if not isinstance(msgs, list) or len(msgs) < 2:
                return jsonify({"error": "Provide 'messages' list of length >= 2."}), 400
            res = drift_trajectory(msgs, window=int(data.get("window", 3)), n=n)
            res["mode"] = "trajectory"
        elif "text_a" in data and "text_b" in data:
            res = semantic_drift(str(data["text_a"]), str(data["text_b"]), n=n)
            res["mode"] = "pairwise"
        else:
            return jsonify({"error": "Provide either ('text_a','text_b') or 'messages'."}), 400
        res["provenance"] = provenance_record(
            algo="semantic_drift", version="1.0",
            inputs={"mode": res["mode"]},
            outputs={k: v for k, v in res.items()
                     if isinstance(v, (int, float, bool, str))},
            params={"n": n},
        )
        return jsonify(res)
    except Exception as e:
        logging.exception("ERROR /semantic-drift")
        return jsonify({"error": str(e)}), 500


@app.route("/hedge-cascade", methods=["POST"])
def hedge_cascade_endpoint():
    try:
        data = request.get_json(silent=True) or {}
        msgs = data.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            return jsonify({"error": "Provide non-empty 'messages' list."}), 400
        k = int(data.get("k_consecutive", 2))
        s = float(data.get("sigma_mult", 1.5))
        res = hedge_cascade(msgs, k_consecutive=k, sigma_mult=s)
        res["provenance"] = provenance_record(
            algo="hedge_cascade", version="1.0",
            inputs={"n_messages": len(msgs)},
            outputs={"cascade_detected": res["cascade_detected"],
                     "dominant_category": res["dominant_category"],
                     "peak_ratio": res["peak_ratio"]},
            params={"k_consecutive": k, "sigma_mult": s},
        )
        return jsonify(res)
    except Exception as e:
        logging.exception("ERROR /hedge-cascade")
        return jsonify({"error": str(e)}), 500


@app.route("/adaptive-threshold", methods=["POST"])
def adaptive_threshold_endpoint():
    try:
        data = request.get_json(silent=True) or {}
        obs = data.get("observation")
        if (not isinstance(obs, list) or len(obs) != _GLOBAL_MONITOR.dim or
                not all(isinstance(x, (int, float)) for x in obs)):
            return jsonify({
                "error": f"'observation' must be a list of {_GLOBAL_MONITOR.dim} floats."
            }), 400
        score = _GLOBAL_MONITOR.score([float(x) for x in obs])
        if data.get("update", True):
            _GLOBAL_MONITOR.update([float(x) for x in obs])
        out = {
            "score": score,
            "monitor_state": {
                "n": _GLOBAL_MONITOR.n,
                "mean": [round(x, 4) for x in _GLOBAL_MONITOR.mean],
            },
        }
        out["provenance"] = provenance_record(
            algo="adaptive_threshold", version="1.0",
            inputs={"observation": obs},
            outputs={"anomaly_score": score["anomaly_score"],
                     "is_anomaly": score["is_anomaly"]},
            params={"dim": _GLOBAL_MONITOR.dim,
                    "alpha": _GLOBAL_MONITOR.alpha},
        )
        return jsonify(out)
    except Exception as e:
        logging.exception("ERROR /adaptive-threshold")
        return jsonify({"error": str(e)}), 500


@app.route("/provenance/verify", methods=["GET"])
def provenance_verify():
    return jsonify(verify_chain())


@app.route("/provenance/snapshot", methods=["GET"])
def provenance_snapshot():
    try:
        limit = int(request.args.get("limit", 50))
    except Exception:
        limit = 50
    snap = chain_snapshot(limit=limit)
    return jsonify({"records": snap, "count": len(snap)})


@app.route("/info-geometry", methods=["POST"])
def info_geometry_endpoint():
    """Proper Jensen-Shannon (corrects the math bug in legacy /kl-divergence)."""
    try:
        data = request.get_json(silent=True) or {}
        cur = [float(x) for x in data.get("current", [])]
        base = [float(x) for x in data.get("baseline", [])]
        if not cur or not base:
            return jsonify({"error": "Both 'current' and 'baseline' required."}), 400
        js = jensen_shannon(cur, base, bins=int(data.get("bins", 10)))
        fr = fisher_rao(js["P"], js["Q"])
        out = {
            "jensen_shannon_divergence": js["jsd"],
            "jensen_shannon_distance": js["js_distance"],
            "fisher_rao_distance": fr,
            "current_dist": js["P"],
            "baseline_dist": js["Q"],
            "mixture_dist": js["M"],
        }
        out["provenance"] = provenance_record(
            algo="info_geometry", version="1.0",
            inputs={"n_current": len(cur), "n_baseline": len(base)},
            outputs={"jsd": js["jsd"], "fisher_rao": fr},
            params={"bins": int(data.get("bins", 10))},
        )
        return jsonify(out)
    except Exception as e:
        logging.exception("ERROR /info-geometry")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

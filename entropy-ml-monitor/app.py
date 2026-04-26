"""
ModelPulse - Entropy-Based Early Warning System
Backend: Flask + Python
Features:
  1. Basic mode          - single confidence → entropy & stability
  2. Blackbox mode       - simulated repeated queries → categorical entropy
  3. Dataset upload      - CSV batch entropy analysis + instability score
  4. KL Divergence       - compare current vs baseline distribution
  5. Multi-model compare - compare two CSV datasets side by side
  6. Temporal forecast   - predict when entropy crosses critical threshold
  7. Calibration scorer  - Expected Calibration Error (ECE)
  8. Chat link analyzer  - scrape Claude/ChatGPT share links → linguistic entropy

Deployment notes:
  - Flask serves index.html at "/" so users see the UI, not raw JSON.
  - All static files (CSS, JS, images) are served from the project root.
  - BASE_DIR uses os.path.abspath so paths are correct inside Docker.
  - All existing API endpoints are untouched.
  - /health returns JSON for monitoring tools (replaces the old "/" JSON response).
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import random
import math
import csv
import os
import re
from collections import Counter
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────
# APP SETUP
# BASE_DIR = folder where app.py lives.
# Works correctly inside Docker regardless
# of the working directory at launch time.
# ─────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=BASE_DIR,   # serve static files from project root
    static_url_path=""        # no URL prefix → /style.css, /script.js work directly
)

CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────
# FRONTEND — serve UI at root
# ─────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend UI (index.html)."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css")
def serve_css():
    return send_from_directory(BASE_DIR, "style.css")


@app.route("/script.js")
def serve_js():
    return send_from_directory(BASE_DIR, "script.js")


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve logo and any other file inside /assets/."""
    return send_from_directory(os.path.join(BASE_DIR, "assets"), filename)


@app.route("/modelpulse_logo.png")
def serve_logo_root():
    """Fallback: serve logo if referenced from root (not /assets/)."""
    return send_from_directory(BASE_DIR, "modelpulse_logo.png")


# ─────────────────────────────────────────
# HEALTH CHECK  (JSON for monitoring)
# ─────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":    "online",
        "app":       "ModelPulse",
        "version":   "2.0",
        "endpoints": [
            "/analyze",
            "/analyze-blackbox",
            "/upload-dataset",
            "/kl-divergence",
            "/compare-models",
            "/forecast",
            "/calibration",
            "/analyze-chat-link"
        ]
    })


# ─────────────────────────────────────────
# CORE MATH UTILITIES
# ─────────────────────────────────────────

def shannon_entropy(p):
    """Binary Shannon entropy. Always returns value in [0, 1]."""
    p = max(1e-9, min(1 - 1e-9, p))
    q = 1 - p
    return -(p * math.log2(p) + q * math.log2(q))


def categorical_entropy(counts_dict, num_classes=None):
    """Shannon entropy over a categorical distribution, normalised to [0, 1]."""
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    raw = 0.0
    for count in counts_dict.values():
        p = count / total
        if p > 0:
            raw -= p * math.log2(p)
    classes = num_classes or len(counts_dict)
    max_e   = math.log2(classes) if classes > 1 else 1.0
    return raw / max_e if max_e > 0 else 0.0


def detect_tipping(hist):
    """Rolling-window tipping point detection."""
    if len(hist) < 3:
        return "Safe"
    recent = hist[-3:]
    trend  = recent[-1] - recent[0]
    if trend > 0.3:
        return "Critical"
    elif recent[-1] > recent[-2] > recent[-3]:
        return "Warning"
    return "Safe"


def clamp_stability(entropy_val):
    """Convert entropy [0,1] → stability score [0,100]."""
    return max(0, min(100, int((1 - entropy_val) * 100)))


# Global rolling history (basic mode only)
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
        entropy    = shannon_entropy(confidence)
        stability  = clamp_stability(entropy)

        history.append(round(entropy, 4))
        if len(history) > 10:
            history.pop(0)

        tipping = detect_tipping(history)

        return jsonify({
            "mode":       "basic",
            "stability":  stability,
            "entropy":    round(entropy, 4),
            "confidence": round(confidence, 4),
            "history":    history,
            "tipping":    tipping
        })
    except Exception as e:
        print("ERROR /analyze:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 2. BLACKBOX MODE
# ─────────────────────────────────────────

@app.route("/analyze-blackbox", methods=["POST"])
def analyze_blackbox():
    try:
        possible_answers = ["Paris", "Paris", "Paris", "Lyon", "Marseille"]
        responses        = [random.choice(possible_answers) for _ in range(5)]

        counts       = Counter(responses)
        entropy_norm = categorical_entropy(dict(counts), num_classes=len(possible_answers))
        stability    = clamp_stability(entropy_norm)

        if entropy_norm > 0.6:
            tipping = "Critical"
        elif entropy_norm > 0.3:
            tipping = "Warning"
        else:
            tipping = "Safe"

        return jsonify({
            "mode":      "blackbox",
            "responses": responses,
            "entropy":   round(entropy_norm, 4),
            "stability": stability,
            "tipping":   tipping,
            "history":   [round(entropy_norm, 4)]
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
        print("UPLOAD HIT")
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

        entropies     = [round(shannon_entropy(c), 4) for c in confidences]
        avg_entropy   = sum(entropies) / len(entropies)
        instability   = round(avg_entropy * 100, 2)
        overall_score = max(0, min(100, round(100 - instability)))
        tipping       = detect_tipping(entropies[-5:])
        insight       = _build_insight(avg_entropy)

        return jsonify({
            "mode":              "upload",
            "rows":              len(confidences),
            "avg_entropy":       round(avg_entropy, 4),
            "instability_score": instability,
            "overall_score":     overall_score,
            "tipping":           tipping,
            "insight":           insight,
            "sample_history":    entropies[-10:]
        })
    except Exception as e:
        print("ERROR /upload-dataset:", e)
        return jsonify({"error": str(e)}), 500


def _read_confidence_csv(filepath):
    """Return list of floats, or an error string."""
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
        return "No valid confidence values found (must be 0–1)"
    return confidences


def _build_insight(avg_entropy):
    if avg_entropy < 0.3:
        return "✅ Stable regime — model outputs are highly consistent. No action needed."
    elif avg_entropy < 0.6:
        return "⚠️ Moderate instability — entropy is rising. Schedule monitoring review."
    else:
        return "🔴 High instability — potential model collapse or distribution shift. Retrain immediately."


# ─────────────────────────────────────────
# 4. KL DIVERGENCE
# ─────────────────────────────────────────

@app.route("/kl-divergence", methods=["POST"])
def kl_divergence():
    """
    Body: { "current": [...], "baseline": [...] }
    Both are lists of floats in [0, 1].
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Send JSON with 'current' and 'baseline' arrays"}), 400

        current  = [float(x) for x in data.get("current",  [])]
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

        P   = to_dist(current)
        Q   = to_dist(baseline)
        eps = 1e-9

        kl = sum(p * math.log2((p + eps) / (q + eps)) for p, q in zip(P, Q) if p > 0)
        js = 0.5 * kl + 0.5 * sum(
            q * math.log2((q + eps) / (p + eps)) for p, q in zip(P, Q) if q > 0
        )

        if kl > 2.0:   shift_level = "Severe"
        elif kl > 1.0: shift_level = "Moderate"
        elif kl > 0.3: shift_level = "Mild"
        else:          shift_level = "None"

        insight = {
            "None":     "✅ Distributions are aligned. No significant shift detected.",
            "Mild":     "⚠️ Mild distribution shift. Monitor over next evaluation cycle.",
            "Moderate": "🟠 Moderate drift. Consider recalibration or feature audit.",
            "Severe":   "🔴 Severe distribution shift. Immediate retraining recommended."
        }[shift_level]

        return jsonify({
            "mode":          "kl_divergence",
            "kl_divergence": round(kl, 4),
            "js_divergence": round(js, 4),
            "shift_level":   shift_level,
            "insight":       insight,
            "current_dist":  P,
            "baseline_dist": Q
        })
    except Exception as e:
        print("ERROR /kl-divergence:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 5. MULTI-MODEL COMPARISON
# ─────────────────────────────────────────

@app.route("/compare-models", methods=["POST"])
def compare_models():
    """Multipart upload: model_a_file, model_b_file (both CSV)."""
    try:
        if "model_a_file" not in request.files or "model_b_file" not in request.files:
            return jsonify({"error": "Upload both model_a_file and model_b_file"}), 400

        results = {}
        for key in ["model_a_file", "model_b_file"]:
            f        = request.files[key]
            filename = secure_filename(f.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            f.save(filepath)

            confs = _read_confidence_csv(filepath)
            if isinstance(confs, str):
                return jsonify({"error": f"{key}: {confs}"}), 400

            entropies = [shannon_entropy(c) for c in confs]
            avg_e     = sum(entropies) / len(entropies)
            label     = "A" if key == "model_a_file" else "B"

            results[label] = {
                "rows":            len(confs),
                "avg_entropy":     round(avg_e, 4),
                "stability_score": clamp_stability(avg_e),
                "instability":     round(avg_e * 100, 2),
                "tipping":         detect_tipping(entropies[-5:]),
                "entropy_history": [round(e, 4) for e in entropies[-10:]]
            }

        winner  = "A" if results["A"]["avg_entropy"] <= results["B"]["avg_entropy"] else "B"
        diff    = abs(results["A"]["avg_entropy"] - results["B"]["avg_entropy"])
        verdict = f"Model {winner} is more stable by {round(diff * 100, 1)} entropy points."

        return jsonify({
            "mode":    "compare",
            "model_a": results["A"],
            "model_b": results["B"],
            "winner":  winner,
            "verdict": verdict
        })
    except Exception as e:
        print("ERROR /compare-models:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 6. TEMPORAL ENTROPY FORECAST
# ─────────────────────────────────────────

@app.route("/forecast", methods=["POST"])
def forecast():
    """
    Holt's double exponential smoothing.
    Body: { "history": [...], "steps": 5, "threshold": 0.7 }
    """
    try:
        data      = request.get_json()
        hist      = [float(x) for x in data.get("history", [])]
        steps     = int(data.get("steps", 5))
        threshold = float(data.get("threshold", 0.7))

        if len(hist) < 3:
            return jsonify({"error": "Need at least 3 history values to forecast"}), 400

        alpha = 0.4
        beta  = 0.3
        level = hist[0]
        trend = hist[1] - hist[0]

        smoothed = []
        for val in hist:
            prev_level = level
            level      = alpha * val + (1 - alpha) * (level + trend)
            trend      = beta  * (level - prev_level) + (1 - beta) * trend
            smoothed.append(round(level, 4))

        forecasted = [
            round(min(1.0, level + i * trend), 4)
            for i in range(1, steps + 1)
        ]

        crossing_step = next(
            (i + 1 for i, v in enumerate(forecasted) if v >= threshold),
            None
        )

        if crossing_step:
            alert = f"⚡ Entropy predicted to cross {threshold} in {crossing_step} step(s). Prepare to retrain."
        else:
            alert = f"✅ Entropy stays below {threshold} for the next {steps} steps."

        return jsonify({
            "mode":          "forecast",
            "smoothed":      smoothed,
            "forecasted":    forecasted,
            "crossing_step": crossing_step,
            "threshold":     threshold,
            "alert":         alert,
            "steps":         steps
        })
    except Exception as e:
        print("ERROR /forecast:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 7. CALIBRATION SCORER (ECE)
# ─────────────────────────────────────────

@app.route("/calibration", methods=["POST"])
def calibration():
    """
    Body: { "predictions": [{"confidence": 0.9, "correct": true}, ...] }
    """
    try:
        data  = request.get_json()
        preds = data.get("predictions", [])

        if len(preds) < 5:
            return jsonify({"error": "Need at least 5 predictions for calibration"}), 400

        bins     = 10
        bin_data = [{"conf_sum": 0.0, "acc_sum": 0.0, "count": 0} for _ in range(bins)]

        for p in preds:
            conf    = float(p.get("confidence", 0.5))
            correct = bool(p.get("correct", False))
            idx     = min(int(conf * bins), bins - 1)
            bin_data[idx]["conf_sum"] += conf
            bin_data[idx]["acc_sum"]  += 1 if correct else 0
            bin_data[idx]["count"]    += 1

        n           = len(preds)
        ece         = 0.0
        bin_results = []

        for b in bin_data:
            if b["count"] > 0:
                avg_conf = b["conf_sum"] / b["count"]
                avg_acc  = b["acc_sum"]  / b["count"]
                ece     += (b["count"] / n) * abs(avg_conf - avg_acc)
                bin_results.append({
                    "avg_confidence": round(avg_conf, 3),
                    "avg_accuracy":   round(avg_acc, 3),
                    "count":          b["count"]
                })

        ece = round(ece, 4)

        if ece < 0.05:
            quality = "Excellent"
            insight = "✅ Model is well-calibrated. Confidence scores are trustworthy."
        elif ece < 0.1:
            quality = "Good"
            insight = "🟡 Slight miscalibration. Consider temperature scaling."
        elif ece < 0.2:
            quality = "Poor"
            insight = "🟠 Significant miscalibration. Platt scaling or isotonic regression recommended."
        else:
            quality = "Very Poor"
            insight = "🔴 Severely miscalibrated. Model confidence is unreliable."

        return jsonify({
            "mode":          "calibration",
            "ece":           ece,
            "quality":       quality,
            "insight":       insight,
            "bin_results":   bin_results,
            "total_samples": n
        })
    except Exception as e:
        print("ERROR /calibration:", e)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 8. CHAT LINK ANALYZER
# ─────────────────────────────────────────

@app.route("/analyze-chat-link", methods=["POST"])
def analyze_chat_link():
    """
    Body: { "url": "https://claude.ai/share/..." }
    Requires: pip install playwright && playwright install chromium
    Falls back to demo mode if playwright is not installed.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Send JSON with 'url' field"}), 400

        url      = data.get("url", "").strip()
        platform = _detect_platform(url)

        if not platform:
            return jsonify({
                "error": (
                    "Unsupported URL. Paste a Claude (claude.ai/share/...) "
                    "or ChatGPT (chatgpt.com/share/...) share link."
                )
            }), 400

        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
            messages = _scrape_with_playwright(url, platform)
        except ImportError:
            return _demo_chat_analysis(url, platform)
        except Exception as scrape_err:
            print("SCRAPE ERROR:", scrape_err)
            return jsonify({
                "error": (
                    f"Could not scrape the page: {scrape_err}. "
                    "Run: pip install playwright && playwright install chromium"
                )
            }), 500

        if not messages:
            return jsonify({
                "error": (
                    "No assistant messages found. "
                    "Check the link is public and contains AI responses."
                )
            }), 400

        return jsonify(_compute_linguistic_entropy(messages, platform, url))

    except Exception as e:
        print("ERROR /analyze-chat-link:", e)
        return jsonify({"error": str(e)}), 500


def _detect_platform(url):
    if "claude.ai/share" in url:
        return "claude"
    if "chatgpt.com/share" in url or "chat.openai.com/share" in url:
        return "chatgpt"
    return None


def _scrape_with_playwright(url, platform):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx     = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)

        messages  = []
        selectors = (
            [
                "[data-testid='assistant-message'] .prose",
                ".assistant-message .prose",
                "[class*='assistant'] [class*='prose']",
                ".font-claude-message",
                "[class*='AssistantMessage']"
            ]
            if platform == "claude"
            else [
                "[data-message-author-role='assistant'] .markdown",
                ".agent-turn .markdown",
                "[class*='assistant'] .markdown",
                "div[data-message-author-role='assistant']"
            ]
        )

        for sel in selectors:
            els = page.query_selector_all(sel)
            if els:
                messages = [el.inner_text() for el in els if el.inner_text().strip()]
                break

        browser.close()
        return messages


def _compute_linguistic_entropy(messages, platform, url):
    all_text  = " ".join(messages)
    words     = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', all_text) if len(s.strip()) > 10]

    word_counts   = Counter(words)
    vocab_entropy = categorical_entropy(word_counts, num_classes=max(len(word_counts), 1))

    if sentences:
        lengths        = [len(s.split()) for s in sentences]
        length_buckets = Counter(min(l // 5, 9) for l in lengths)
        length_entropy = categorical_entropy(length_buckets, num_classes=10)
    else:
        length_entropy = 0.0

    hedge_words = {
        "maybe", "perhaps", "might", "could", "possibly", "uncertain",
        "unclear", "approximately", "roughly", "likely", "probably",
        "i think", "i believe", "not sure", "may", "seems", "appears"
    }
    hedge_score = min(
        1.0,
        sum(1 for w in words if w in hedge_words) / max(len(words), 1) * 20
    )

    if len(messages) > 1:
        msg_lengths = [len(m.split()) for m in messages]
        mean_len    = sum(msg_lengths) / len(msg_lengths)
        variance    = sum((l - mean_len) ** 2 for l in msg_lengths) / len(msg_lengths)
        length_cv   = min(1.0, math.sqrt(variance) / max(mean_len, 1))
    else:
        length_cv = 0.0

    composite = (
        vocab_entropy  * 0.35 +
        length_entropy * 0.25 +
        hedge_score    * 0.25 +
        length_cv      * 0.15
    )

    stability     = clamp_stability(composite)
    tipping       = detect_tipping([composite] * 3 + [composite + 0.01])
    platform_name = "Claude" if platform == "claude" else "ChatGPT"

    if composite < 0.3:
        insight = "✅ Model responses show high consistency and confidence. Low linguistic uncertainty."
    elif composite < 0.6:
        insight = "⚠️ Moderate linguistic uncertainty detected. Model hedges in several responses."
    else:
        insight = "🔴 High linguistic entropy. Model shows significant uncertainty across responses."

    return {
        "mode":              "chat_link",
        "platform":          platform_name,
        "url":               url,
        "total_messages":    len(messages),
        "total_words":       len(words),
        "unique_words":      len(word_counts),
        "vocab_entropy":     round(vocab_entropy, 4),
        "length_entropy":    round(length_entropy, 4),
        "hedge_score":       round(hedge_score, 4),
        "response_variance": round(length_cv, 4),
        "composite_entropy": round(composite, 4),
        "stability_score":   stability,
        "tipping":           tipping,
        "insight":           insight,
        "sample_messages":   [
            m[:200] + "..." if len(m) > 200 else m for m in messages[:3]
        ]
    }


def _demo_chat_analysis(url, platform):
    platform_name = "Claude" if platform == "claude" else "ChatGPT"
    return jsonify({
        "mode":              "chat_link",
        "platform":          platform_name,
        "url":               url,
        "demo_mode":         True,
        "demo_warning":      (
            "Playwright not installed. Showing demo data. "
            "Run: pip install playwright && playwright install chromium"
        ),
        "total_messages":    6,
        "total_words":       847,
        "unique_words":      312,
        "vocab_entropy":     round(random.uniform(0.3, 0.7), 4),
        "length_entropy":    round(random.uniform(0.2, 0.6), 4),
        "hedge_score":       round(random.uniform(0.1, 0.5), 4),
        "response_variance": round(random.uniform(0.1, 0.4), 4),
        "composite_entropy": round(random.uniform(0.25, 0.65), 4),
        "stability_score":   random.randint(35, 75),
        "tipping":           random.choice(["Safe", "Warning"]),
        "insight":           "⚠️ [DEMO] Moderate linguistic uncertainty detected in AI responses.",
        "sample_messages":   [
            "[DEMO] This is a simulated assistant message for demonstration purposes...",
            "[DEMO] The analysis would show real vocabulary entropy from actual messages...",
            "[DEMO] Install playwright to enable live scraping of shared conversations."
        ],
        "composite_entropy_chart": [round(random.uniform(0.2, 0.7), 4) for _ in range(6)]
    })


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    # debug=False for production / Docker
    app.run(host="0.0.0.0", port=5000, debug=False)
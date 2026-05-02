// ============================================================
// ModelPulse v2.0 — Full Frontend Logic
// ============================================================

const API = window.location.origin;
let chart = null, klChart = null, compareChart = null, forecastChart = null, calChart = null;

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    runLoader();
    setupSidebar();
    bindButtons();
    startLiveClock();
    pingBackend();
    setupFileDropZones();
    console.log("✅ ModelPulse v2.0 JS Loaded");
});

// ─────────────────────────────────────────────
// LOADER
// ─────────────────────────────────────────────
const LOADER_MSGS = [
    "Initializing entropy engine...",
    "Loading Shannon modules...",
    "Calibrating tipping detectors...",
    "Warming up forecasting engine...",
    "Launching chat analyzer...",
    "System ready ✅"
];

function runLoader() {
    const bar    = document.getElementById("loaderBar");
    const status = document.getElementById("loaderStatus");
    const loader = document.getElementById("loader");
    const appEl  = document.getElementById("app");
    let step = 0;

    const tick = () => {
        if (step >= LOADER_MSGS.length) {
            setTimeout(() => {
                loader.style.transition = "opacity 0.5s ease";
                loader.style.opacity = "0";
                setTimeout(() => {
                    loader.style.display = "none";
                    appEl.classList.remove("hidden");
                    appEl.style.display = "flex";
                }, 500);
            }, 200);
            return;
        }
        const pct = Math.round(((step + 1) / LOADER_MSGS.length) * 100);
        if (bar)    bar.style.width = pct + "%";
        if (status) status.textContent = LOADER_MSGS[step];
        step++;
        setTimeout(tick, 340);
    };
    tick();
}

// ─────────────────────────────────────────────
// SIDEBAR
// ─────────────────────────────────────────────
function setupSidebar() {
    const items = document.querySelectorAll(".menu-item");
    const pages = document.querySelectorAll(".page");

    items.forEach(item => {
        item.addEventListener("click", () => {
            items.forEach(i => i.classList.remove("active"));
            item.classList.add("active");

            pages.forEach(p => p.classList.add("hidden"));
            const page = document.getElementById(item.getAttribute("data-page"));
            if (page) page.classList.remove("hidden");

            const label = item.querySelector("span:not(.mi):not(.mbadge)")?.textContent?.trim() || "";
            setText("pageTitle", label);
            setText("breadcrumbSub", label);
        });
    });
}

// ─────────────────────────────────────────────
// BUTTON BINDING
// ─────────────────────────────────────────────
function bindButtons() {
    const bindings = {
        "btnBasic":       analyze,
        "btnBlackbox":    analyzeBlackbox,
        "btnUpload":      uploadDataset,
        "btnAnalyzeChat": analyzeChatLink,
        "btnKL":          computeKL,
        "btnCompare":     compareModels,
        "btnForecast":    runForecast,
        "btnCalibration": scoreCalibration,
    };

    for (const [id, fn] of Object.entries(bindings)) {
        const btn = document.getElementById(id);
        if (btn) btn.addEventListener("click", fn);
        else console.error("❌ Button not found:", id);
    }
}

// ─────────────────────────────────────────────
// FILE DROP ZONES
// ─────────────────────────────────────────────
function setupFileDropZones() {
    setupDrop("dropZone",  "datasetFile", "dropText");
    setupDrop(null,        "modelAFile",  "dropTextA");
    setupDrop(null,        "modelBFile",  "dropTextB");
}

function setupDrop(zoneId, inputId, textId) {
    const input = document.getElementById(inputId);
    const label = zoneId ? document.getElementById(zoneId) : input?.closest("label");
    const textEl = textId ? document.getElementById(textId) : null;

    if (input) {
        input.addEventListener("change", () => {
            const name = input.files[0]?.name;
            if (name && textEl) textEl.textContent = name;
        });
    }

    if (label) {
        label.addEventListener("click", (e) => {
            if (e.target !== input) { e.preventDefault(); input?.click(); }
        });
        label.addEventListener("dragover", e => { e.preventDefault(); label.style.borderColor = "var(--accent)"; });
        label.addEventListener("dragleave", () => { label.style.borderColor = ""; });
        label.addEventListener("drop", e => {
            e.preventDefault();
            label.style.borderColor = "";
            const file = e.dataTransfer.files[0];
            if (file && input) {
                const dt = new DataTransfer();
                dt.items.add(file);
                input.files = dt.files;
                if (textEl) textEl.textContent = file.name;
            }
        });
    }
}

// ─────────────────────────────────────────────
// LIVE CLOCK
// ─────────────────────────────────────────────
function startLiveClock() {
    const el = document.getElementById("liveTime");
    if (!el) return;
    const upd = () => el.textContent = new Date().toLocaleTimeString("en-US", { hour12: false });
    upd(); setInterval(upd, 1000);
}

// ─────────────────────────────────────────────
// BACKEND PING
// ─────────────────────────────────────────────
async function pingBackend() {
    const dot = document.getElementById("backendDot");
    const lbl = document.getElementById("backendLabel");
    try {
        const r = await fetch(`${API}/`);
        if (r.ok) {
            dot?.classList.add("online");
            if (lbl) lbl.textContent = "Backend online";
        } else throw new Error();
    } catch {
        dot?.classList.add("offline");
        if (lbl) lbl.textContent = "Backend offline";
    }
}

// ─────────────────────────────────────────────
// 1. BASIC MODE
// ─────────────────────────────────────────────
async function analyze() {
    const conf = parseFloat(document.getElementById("inputData")?.value);
    if (isNaN(conf) || conf < 0 || conf > 1) return showError("Enter a value between 0.0 and 1.0");

    setLoading("btnBasic", true);
    try {
        const data = await post(`${API}/analyze`, { confidence: conf });
        updateDashboard(data);
    } catch (e) { showError(e.message); }
    finally { setLoading("btnBasic", false); }
}

// ─────────────────────────────────────────────
// 2. BLACKBOX MODE
// ─────────────────────────────────────────────
async function analyzeBlackbox() {
    setLoading("btnBlackbox", true);
    try {
        const data = await post(`${API}/analyze-blackbox`, {});
        updateDashboard(data);

        const chipList = document.getElementById("responses");
        if (chipList && data.responses) {
            chipList.innerHTML = "";
            data.responses.forEach(r => {
                const c = document.createElement("span");
                c.className = "chip";
                c.textContent = r;
                chipList.appendChild(c);
            });
            const info = makeChip(`Entropy: ${data.entropy}`, "var(--accent)", "rgba(0,200,255,0.08)");
            chipList.appendChild(info);
        }
    } catch (e) { showError(e.message); }
    finally { setLoading("btnBlackbox", false); }
}

// ─────────────────────────────────────────────
// 3. UPLOAD DATASET
// ─────────────────────────────────────────────
async function uploadDataset() {
    const file = document.getElementById("datasetFile")?.files[0];
    if (!file) return showError("Select a CSV file first.");
    if (!file.name.endsWith(".csv")) return showError("Only .csv files supported.");

    setLoading("btnUpload", true);
    try {
        const fd = new FormData();
        fd.append("file", file);

        const res = await fetch(`${API}/upload-dataset`, { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) return showError(data.error || "Upload failed");

        // Switch to results page
        document.querySelectorAll(".page").forEach(p => p.classList.add("hidden"));
        document.getElementById("resultsPage")?.classList.remove("hidden");
        document.querySelectorAll(".menu-item").forEach(i => i.classList.remove("active"));
        document.querySelector('[data-page="resultsPage"]')?.classList.add("active");
        setText("pageTitle", "Results"); setText("breadcrumbSub", "Results");

        setText("res_rows",        data.rows);
        setText("res_entropy",     data.avg_entropy?.toFixed(4));
        setText("res_instability", data.instability_score?.toFixed(2) + "%");
        setText("res_status",      data.tipping);
        setText("res_insight",     data.insight);

        const score = data.overall_score ?? Math.max(0, Math.round(100 - data.instability_score));
        animateValue("res_mainScore", 0, score, 900);
        if (data.sample_history?.length) updateChart(data.sample_history);
    } catch (e) { showError(e.message); }
    finally { setLoading("btnUpload", false); }
}

// ─────────────────────────────────────────────
// 4. CHAT LINK ANALYZER
// ─────────────────────────────────────────────
async function analyzeChatLink() {
    const url = document.getElementById("chatLinkInput")?.value?.trim();
    if (!url) return showError("Paste a Claude or ChatGPT share link.");

    const isValid = url.includes("claude.ai/share") || url.includes("chatgpt.com/share") || url.includes("chat.openai.com/share");
    if (!isValid) return showError("URL must be a Claude (claude.ai/share/...) or ChatGPT (chatgpt.com/share/...) share link.");

    setLoading("btnAnalyzeChat", true);
    document.getElementById("chatLinkResults")?.classList.add("hidden");

    try {
        const data = await post(`${API}/analyze-chat-link`, { url });

        // Show results section
        document.getElementById("chatLinkResults")?.classList.remove("hidden");

        animateValue("cl_score", 0, data.stability_score ?? 0, 900);
        setText("cl_platform",  data.platform);
        setText("cl_msgs",      data.total_messages);
        setText("cl_words",     data.total_words);
        setText("cl_vocab",     data.vocab_entropy?.toFixed(4));
        setText("cl_length",    data.length_entropy?.toFixed(4));
        setText("cl_hedge",     data.hedge_score?.toFixed(4));
        setText("cl_variance",  data.response_variance?.toFixed(4));
        setText("cl_composite", data.composite_entropy?.toFixed(4));
        setText("cl_tipping",   data.tipping);
        setText("cl_insight",   data.insight);

        // Sample messages
        const samplesEl = document.getElementById("cl_samples");
        if (samplesEl && data.sample_messages) {
            samplesEl.innerHTML = "";
            data.sample_messages.forEach((msg, i) => {
                const div = document.createElement("div");
                div.className = "sample-msg";
                div.innerHTML = `<span class="sample-num">${i + 1}</span><span>${escHtml(msg)}</span>`;
                samplesEl.appendChild(div);
            });
        }

        // Demo warning
        const warn = document.getElementById("chatDemoWarning");
        if (warn) warn.classList.toggle("hidden", !data.demo_mode);

        // Apply tipping class
        applyTippingClass("chatLinkResults", data.tipping);

    } catch (e) { showError(e.message); }
    finally { setLoading("btnAnalyzeChat", false); }
}

// ─────────────────────────────────────────────
// 5. KL DIVERGENCE
// ─────────────────────────────────────────────
async function computeKL() {
    const currentRaw  = document.getElementById("klCurrent")?.value?.trim();
    const baselineRaw = document.getElementById("klBaseline")?.value?.trim();

    if (!currentRaw || !baselineRaw) return showError("Paste values in both fields.");

    const parseList = s => s.split(/[\s,]+/).map(parseFloat).filter(v => !isNaN(v) && v >= 0 && v <= 1);
    const current  = parseList(currentRaw);
    const baseline = parseList(baselineRaw);

    if (current.length < 3)  return showError("Current needs at least 3 values.");
    if (baseline.length < 3) return showError("Baseline needs at least 3 values.");

    setLoading("btnKL", true);
    try {
        const data = await post(`${API}/kl-divergence`, { current, baseline });

        document.getElementById("klResults")?.classList.remove("hidden");
        animateValue("kl_value", 0, Math.round(data.kl_divergence * 100), 900);
        // Show actual value after animation
        setTimeout(() => setText("kl_value", data.kl_divergence?.toFixed(4)), 950);
        setText("kl_kl",    data.kl_divergence?.toFixed(4));
        setText("kl_js",    data.js_divergence?.toFixed(4));
        setText("kl_shift", data.shift_level);
        setText("kl_insight", data.insight);

        // Distribution chart
        if (data.current_dist && data.baseline_dist) {
            renderKLChart(data.current_dist, data.baseline_dist);
        }
    } catch (e) { showError(e.message); }
    finally { setLoading("btnKL", false); }
}

function renderKLChart(P, Q) {
    const ctx = document.getElementById("klChart");
    if (!ctx) return;
    if (klChart) { klChart.destroy(); klChart = null; }
    const labels = P.map((_, i) => `${(i * 0.1).toFixed(1)}–${((i + 1) * 0.1).toFixed(1)}`);
    klChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                { label: "Current (P)", data: P, backgroundColor: "rgba(0,200,255,0.5)", borderColor: "#00c8ff", borderWidth: 1 },
                { label: "Baseline (Q)", data: Q, backgroundColor: "rgba(255,184,48,0.4)", borderColor: "#ffb830", borderWidth: 1 }
            ]
        },
        options: { ...chartBaseOptions(), scales: { ...darkScales() } }
    });
}

// ─────────────────────────────────────────────
// 6. MODEL COMPARISON
// ─────────────────────────────────────────────
async function compareModels() {
    const fileA = document.getElementById("modelAFile")?.files[0];
    const fileB = document.getElementById("modelBFile")?.files[0];

    if (!fileA) return showError("Upload Model A CSV.");
    if (!fileB) return showError("Upload Model B CSV.");

    setLoading("btnCompare", true);
    try {
        const fd = new FormData();
        fd.append("model_a_file", fileA);
        fd.append("model_b_file", fileB);

        const res  = await fetch(`${API}/compare-models`, { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) return showError(data.error || "Compare failed");

        document.getElementById("compareResults")?.classList.remove("hidden");

        animateValue("cmp_scoreA", 0, data.model_a.stability_score, 900);
        animateValue("cmp_scoreB", 0, data.model_b.stability_score, 900);
        setText("cmp_entropyA",  data.model_a.avg_entropy?.toFixed(4));
        setText("cmp_rowsA",     data.model_a.rows);
        setText("cmp_tippingA",  data.model_a.tipping);
        setText("cmp_entropyB",  data.model_b.avg_entropy?.toFixed(4));
        setText("cmp_rowsB",     data.model_b.rows);
        setText("cmp_tippingB",  data.model_b.tipping);
        setText("cmp_winner",    "Model " + data.winner);
        setText("cmp_verdict",   data.verdict);

        // Highlight winner
        document.getElementById("compareCardA")?.classList.toggle("winner-glow", data.winner === "A");
        document.getElementById("compareCardB")?.classList.toggle("winner-glow", data.winner === "B");

        // Comparison chart
        renderCompareChart(data.model_a.entropy_history, data.model_b.entropy_history);
    } catch (e) { showError(e.message); }
    finally { setLoading("btnCompare", false); }
}

function renderCompareChart(histA, histB) {
    const ctx = document.getElementById("compareChart");
    if (!ctx) return;
    if (compareChart) { compareChart.destroy(); compareChart = null; }
    const len = Math.max(histA.length, histB.length);
    const labels = Array.from({ length: len }, (_, i) => "T" + (i + 1));
    compareChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                { label: "Model A", data: histA, borderColor: "#00c8ff", backgroundColor: "rgba(0,200,255,0.1)", tension: 0.4, fill: true, pointRadius: 3 },
                { label: "Model B", data: histB, borderColor: "#ffb830", backgroundColor: "rgba(255,184,48,0.08)", tension: 0.4, fill: true, pointRadius: 3 }
            ]
        },
        options: { ...chartBaseOptions(), scales: { ...darkScales(), y: { ...darkScales().y, min: 0, max: 1 } } }
    });
}

// ─────────────────────────────────────────────
// 7. FORECAST
// ─────────────────────────────────────────────
async function runForecast() {
    const raw       = document.getElementById("forecastInput")?.value?.trim();
    const steps     = parseInt(document.getElementById("forecastSteps")?.value)     || 5;
    const threshold = parseFloat(document.getElementById("forecastThreshold")?.value) || 0.7;

    if (!raw) return showError("Paste entropy history values.");
    const history = raw.split(/[\s,]+/).map(parseFloat).filter(v => !isNaN(v));
    if (history.length < 3) return showError("Need at least 3 history values.");

    setLoading("btnForecast", true);
    try {
        const data = await post(`${API}/forecast`, { history, steps, threshold });

        document.getElementById("forecastResults")?.classList.remove("hidden");

        const cross = data.crossing_step;
        setText("fc_crossing", cross ? String(cross) : "Never");
        setText("fc_alert", data.alert);

        // Forecast chips
        const vals = document.getElementById("fc_values");
        if (vals) {
            vals.innerHTML = "";
            data.forecasted.forEach((v, i) => {
                const chip = makeChip(
                    `T+${i + 1}: ${v}`,
                    v >= threshold ? "var(--critical)" : v >= threshold * 0.75 ? "var(--warning)" : "var(--safe)",
                    "rgba(255,255,255,0.04)"
                );
                vals.appendChild(chip);
            });
        }

        // Forecast chart
        renderForecastChart(data.smoothed, data.forecasted, threshold);
    } catch (e) { showError(e.message); }
    finally { setLoading("btnForecast", false); }
}

function renderForecastChart(smoothed, forecasted, threshold) {
    const ctx = document.getElementById("forecastChart");
    if (!ctx) return;
    if (forecastChart) { forecastChart.destroy(); forecastChart = null; }

    const histLabels = smoothed.map((_, i) => "T" + (i + 1));
    const fcLabels   = forecasted.map((_, i) => "T+" + (i + 1));
    const allLabels  = [...histLabels, ...fcLabels];
    const histData   = [...smoothed, ...Array(forecasted.length).fill(null)];
    const fcData     = [...Array(smoothed.length).fill(null), ...forecasted];
    const threshLine = Array(allLabels.length).fill(threshold);

    forecastChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: allLabels,
            datasets: [
                { label: "History (smoothed)", data: histData,  borderColor: "#00c8ff", backgroundColor: "rgba(0,200,255,0.1)", tension: 0.4, fill: true,  pointRadius: 4 },
                { label: "Forecast",           data: fcData,    borderColor: "#ffb830", backgroundColor: "rgba(255,184,48,0.08)", tension: 0.4, fill: false, borderDash: [5, 3], pointRadius: 4 },
                { label: `Threshold (${threshold})`, data: threshLine, borderColor: "rgba(255,59,48,0.5)", borderDash: [6, 4], pointRadius: 0, fill: false }
            ]
        },
        options: { ...chartBaseOptions(), scales: { ...darkScales(), y: { ...darkScales().y, min: 0, max: 1 } } }
    });
}

function useCurrentHistory() {
    // Pull history from the last basic analysis (stored in window)
    if (window._lastHistory?.length) {
        document.getElementById("forecastInput").value = window._lastHistory.join(", ");
    } else {
        showError("Run Basic mode a few times first to build history.");
    }
}

// ─────────────────────────────────────────────
// 8. CALIBRATION
// ─────────────────────────────────────────────
async function scoreCalibration() {
    const raw = document.getElementById("calibrationInput")?.value?.trim();
    if (!raw) return showError("Paste prediction JSON.");

    let preds;
    try {
        preds = JSON.parse(raw);
        if (!Array.isArray(preds)) throw new Error();
    } catch {
        return showError("Invalid JSON. Must be an array like [{\"confidence\": 0.9, \"correct\": true}, ...]");
    }

    if (preds.length < 5) return showError("Need at least 5 predictions.");

    setLoading("btnCalibration", true);
    try {
        const data = await post(`${API}/calibration`, { predictions: preds });

        document.getElementById("calibrationResults")?.classList.remove("hidden");

        setText("cal_ece",     data.ece?.toFixed(4));
        setText("cal_quality", data.quality);
        setText("cal_samples", data.total_samples);
        setText("cal_insight", data.insight);
        animateValue("cal_ece", 0, Math.round(data.ece * 100), 900);
        setTimeout(() => setText("cal_ece", data.ece?.toFixed(4)), 950);

        if (data.bin_results) renderCalChart(data.bin_results);
    } catch (e) { showError(e.message); }
    finally { setLoading("btnCalibration", false); }
}

function renderCalChart(bins) {
    const ctx = document.getElementById("calChart");
    if (!ctx) return;
    if (calChart) { calChart.destroy(); calChart = null; }

    const labels = bins.map((_, i) => `Bin ${i + 1}`);
    const confs  = bins.map(b => b.avg_confidence);
    const accs   = bins.map(b => b.avg_accuracy);

    calChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                { label: "Confidence", data: confs, backgroundColor: "rgba(0,200,255,0.5)", borderColor: "#00c8ff", borderWidth: 1 },
                { label: "Accuracy",   data: accs,  backgroundColor: "rgba(0,232,122,0.4)", borderColor: "#00e87a", borderWidth: 1 }
            ]
        },
        options: { ...chartBaseOptions(), scales: { ...darkScales(), y: { ...darkScales().y, min: 0, max: 1 } } }
    });
}

function loadCalibrationDemo() {
    const demo = Array.from({ length: 20 }, (_, i) => ({
        confidence: parseFloat((0.4 + (i / 20) * 0.55).toFixed(2)),
        correct:    Math.random() > 0.4
    }));
    document.getElementById("calibrationInput").value = JSON.stringify(demo, null, 2);
}

// ─────────────────────────────────────────────
// DASHBOARD UI UPDATE
// ─────────────────────────────────────────────
function updateDashboard(data) {
    setText("stability",  data.stability + "%");
    setText("entropy",    typeof data.entropy === "number" ? data.entropy.toFixed(4) : String(data.entropy));
    setText("confidence", data.confidence !== undefined ? data.confidence.toFixed(4) : (data.stability / 100).toFixed(2));

    const t = data.tipping || "Safe";
    setText("tippingStatus", t.toUpperCase());

    const badge = document.getElementById("tippingBadge");
    if (badge) badge.style.color = t === "Critical" ? "var(--critical)" : t === "Warning" ? "var(--warning)" : "var(--safe)";

    const msgs = { Safe: "✅ Stable. No intervention required.", Warning: "⚠️ Entropy rising. Monitor closely.", Critical: "🔴 Critical drift. Retrain or rollback immediately." };
    setText("tippingMsg", msgs[t] || "Awaiting analysis...");

    applyTippingClass("tippingCard", t);

    const fill = document.getElementById("tippingBarFill");
    if (fill) {
        fill.style.width = t === "Critical" ? "100%" : t === "Warning" ? "55%" : "20%";
        fill.style.background = t === "Critical" ? "var(--critical)" : t === "Warning" ? "var(--warning)" : "linear-gradient(90deg, var(--safe), var(--accent))";
    }

    drawRing(data.stability || 0);
    animateValue("mainScore", 0, Math.max(0, Math.min(100, data.stability || 0)), 800);

    if (data.history?.length) updateChart(data.history);

    // Store history for forecast
    if (data.history) window._lastHistory = data.history;
}

// ─────────────────────────────────────────────
// ENTROPY CHART
// ─────────────────────────────────────────────
function updateChart(pts) {
    const ctx = document.getElementById("chart");
    if (!ctx) return;
    if (chart) { chart.destroy(); chart = null; }
    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: pts.map((_, i) => "T" + (i + 1)),
            datasets: [{
                label: "Entropy",
                data: pts,
                tension: 0.45,
                fill: true,
                borderColor: "#00c8ff",
                backgroundColor: "rgba(0,200,255,0.08)",
                pointBackgroundColor: "#00c8ff",
                pointRadius: 4,
                pointHoverRadius: 7,
                borderWidth: 2
            }]
        },
        options: { ...chartBaseOptions(), scales: { ...darkScales(), y: { ...darkScales().y, min: 0, max: 1 } } }
    });
}

// ─────────────────────────────────────────────
// RING CANVAS
// ─────────────────────────────────────────────
function drawRing(score) {
    const canvas = document.getElementById("ringCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const cx = canvas.width / 2, cy = canvas.height / 2, r = 52;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 7; ctx.stroke();

    const pct = Math.max(0, Math.min(100, score)) / 100;
    const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
    grad.addColorStop(0, "#0071e3"); grad.addColorStop(1, "#00c8ff");
    ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * pct);
    ctx.strokeStyle = grad; ctx.lineWidth = 7; ctx.lineCap = "round";
    ctx.shadowColor = "rgba(0,200,255,0.5)"; ctx.shadowBlur = 12; ctx.stroke();
}

// ─────────────────────────────────────────────
// CHART HELPERS
// ─────────────────────────────────────────────
function chartBaseOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600 },
        plugins: { legend: { labels: { color: "rgba(232,234,240,0.6)", font: { size: 11 } } } }
    };
}

function darkScales() {
    const axis = { ticks: { color: "rgba(232,234,240,0.4)", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" } };
    return { x: axis, y: { ...axis } };
}

// ─────────────────────────────────────────────
// TIPPING CLASS
// ─────────────────────────────────────────────
function applyTippingClass(id, tipping) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove("safe", "warning", "critical");
    const map = { Safe: "safe", Warning: "warning", Critical: "critical" };
    if (map[tipping]) el.classList.add(map[tipping]);
}

// ─────────────────────────────────────────────
// ANIMATED COUNTER
// ─────────────────────────────────────────────
function animateValue(id, start, end, dur) {
    const el = document.getElementById(id);
    if (!el) return;
    let t0 = null;
    const step = ts => {
        if (!t0) t0 = ts;
        const p = Math.min((ts - t0) / dur, 1);
        el.textContent = Math.floor(p * (end - start) + start);
        if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

// ─────────────────────────────────────────────
// SYNTHETIC CSV
// ─────────────────────────────────────────────
function generateSyntheticData() {
    const rows    = parseInt(document.getElementById("synthRows")?.value) || 50;
    const pattern = document.getElementById("synthPattern")?.value || "stable";
    const values  = [];
    const noise   = () => (Math.random() - 0.5) * 0.05;

    for (let i = 0; i < rows; i++) {
        const t = i / rows;
        let v;
        switch (pattern) {
            case "stable":   v = 0.85 + noise(); break;
            case "drifting": v = 0.9 - t * 0.6 + noise(); break;
            case "collapse": v = t < 0.7 ? 0.85 + noise() : 0.15 + noise(); break;
            default:         v = Math.random();
        }
        values.push(Math.max(0.001, Math.min(0.999, parseFloat(v.toFixed(4)))));
    }
    downloadCSV("confidence\n" + values.join("\n"), `modelpulse_${pattern}_${rows}rows.csv`);
}

function downloadSampleCSV() {
    downloadCSV("confidence\n0.91\n0.87\n0.54\n0.32\n0.78\n0.65\n0.23\n0.88\n0.71\n0.45", "modelpulse_sample.csv");
}

function downloadCSV(content, filename) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([content], { type: "text/csv" }));
    a.download = filename;
    a.click();
}

// ─────────────────────────────────────────────
// THEME / AUTH
// ─────────────────────────────────────────────
function toggleTheme()  { document.body.classList.toggle("light-mode"); }
function showLogin()    { alert("Login coming soon."); }
function showRegister() { alert("Register coming soon."); }
function logout()       { setText("userStatus", "Guest"); alert("Logged out."); }

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────
async function post(url, body) {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
    return data;
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val ?? "–";
    else console.warn("setText: not found →", id);
}

function showError(msg) { alert("⚠️ " + msg); console.error("ERR:", msg); }

function setLoading(id, loading) {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.disabled = loading;
    btn.style.opacity = loading ? "0.55" : "1";
    if (loading) { btn._orig = btn.textContent; btn.textContent = "Loading..."; }
    else { btn.textContent = btn._orig || btn.getAttribute("data-label") || "Run"; }
}

function makeChip(text, color = "var(--text)", bg = "var(--surface2)") {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = text;
    c.style.color = color;
    c.style.background = bg;
    return c;
}

function escHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
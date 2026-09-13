/**
 * app.js — SIMATA Bug Classifier Frontend Logic
 * ================================================
 * Menangani:
 *   1. Debounced input listener pada textarea Log Temuan
 *   2. Fetch POST ke /api/predict
 *   3. Update DOM: badge rekomendasi, dropdown, probability bars
 *   4. Health check dan API connection status
 *   5. Loading states & error handling
 */

(function () {
  "use strict";

  // ================================================================
  // Constants
  // ================================================================
  const API_PREDICT_URL = "/api/predict";
  const API_HEALTH_URL = "/api/health";
  const DEBOUNCE_MS = 500;
  const MIN_TEXT_LENGTH = 10; // Minimum karakter sebelum kirim ke API
  const HEALTH_CHECK_INTERVAL = 30_000; // 30 detik

  // Severity CSS class mapping
  const SEVERITY_CLASSES = ["severity-fatal", "severity-mayor", "severity-minor", "severity-kosmetik"];
  const SEVERITY_MAP = {
    Fatal: "severity-fatal",
    Mayor: "severity-mayor",
    Minor: "severity-minor",
    Kosmetik: "severity-kosmetik",
  };

  // ================================================================
  // DOM Elements
  // ================================================================
  const elTextarea = document.getElementById("input-log-temuan");
  const elSelect = document.getElementById("select-kategori");
  const elAiLoading = document.getElementById("ai-loading");
  // [REMARKED] Badge rekomendasi & probability bars di-comment karena HTML-nya juga di-comment
  // const elAiRecommendation = document.getElementById("ai-recommendation");
  // const elAiPrediction = document.getElementById("ai-prediction-text");
  // const elAiConfidence = document.getElementById("ai-confidence-text");
  // const elProbContainer = document.getElementById("probability-container");
  const elDropdownHint = document.getElementById("dropdown-hint");
  const elTanggal = document.getElementById("input-tanggal");

  // ================================================================
  // State
  // ================================================================
  let debounceTimer = null;
  let abortController = null;
  let isApiConnected = false;
  let lastPrediction = null;
  let userOverride = false; // True jika user manual mengubah dropdown

  // ================================================================
  // Initialization
  // ================================================================
  function init() {
    // Set tanggal hari ini
    if (elTanggal) {
      const today = new Date();
      const yyyy = today.getFullYear();
      const mm = String(today.getMonth() + 1).padStart(2, "0");
      const dd = String(today.getDate()).padStart(2, "0");
      elTanggal.value = `${yyyy}-${mm}-${dd}`;
    }

    // Attach event listeners
    elTextarea.addEventListener("input", onTextareaInput);
    elSelect.addEventListener("change", onDropdownChange);

    // Initial health check
    checkApiHealth();

    // Periodic health check
    setInterval(checkApiHealth, HEALTH_CHECK_INTERVAL);
  }

  // ================================================================
  // API Health Check
  // ================================================================
  async function checkApiHealth() {
    try {
      const res = await fetch(API_HEALTH_URL, { method: "GET" });
      if (res.ok) {
        const data = await res.json();
        setApiStatus(data.models_loaded === true);
      } else {
        setApiStatus(false);
      }
    } catch {
      setApiStatus(false);
    }
  }

  function setApiStatus(connected) {
    isApiConnected = connected;
  }

  // ================================================================
  // Textarea Input Handler (Debounced)
  // ================================================================
  function onTextareaInput() {
    userOverride = false;

    // Cancel pending request
    if (debounceTimer) clearTimeout(debounceTimer);
    if (abortController) {
      abortController.abort();
      abortController = null;
    }

    const text = elTextarea.value.trim();

    if (text.length < MIN_TEXT_LENGTH) {
      hideRecommendation();
      return;
    }

    // Show loading after a brief moment
    debounceTimer = setTimeout(() => {
      requestPrediction(text);
    }, DEBOUNCE_MS);
  }

  // ================================================================
  // Dropdown Manual Change
  // ================================================================
  function onDropdownChange() {
    // If user manually changes dropdown after AI recommendation,
    // mark as override so we don't auto-change it back
    if (lastPrediction && elSelect.value !== lastPrediction) {
      userOverride = true;
      elDropdownHint.textContent = "Anda memilih kategori secara manual.";
      elDropdownHint.classList.add("ai-active");
    }
  }

  // ================================================================
  // Request Prediction from API
  // ================================================================
  async function requestPrediction(text) {
    if (!isApiConnected) {
      return;
    }

    // Show loading state
    showLoading();

    // Abort previous in-flight request
    if (abortController) abortController.abort();
    abortController = new AbortController();

    try {
      const res = await fetch(API_PREDICT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: abortController.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      hideLoading();
      showRecommendation(data);
    } catch (err) {
      if (err.name === "AbortError") {
        // Request was cancelled — do nothing
        return;
      }
      console.error("Prediction error:", err);
      hideLoading();
      hideRecommendation();
    }
  }

  // ================================================================
  // Show / Hide Loading
  // ================================================================
  function showLoading() {
    elAiLoading.classList.add("visible");
    // [REMARKED] elAiRecommendation.classList.remove("visible");
    // [REMARKED] elProbContainer.classList.remove("visible");
  }

  function hideLoading() {
    elAiLoading.classList.remove("visible");
  }

  // ================================================================
  // Show Recommendation
  // ================================================================
  function showRecommendation(data) {
    const { prediction, confidence, probabilities } = data;
    lastPrediction = prediction;

    /* [REMARKED] Badge rekomendasi AI
    const severityClass = SEVERITY_MAP[prediction] || "";
    SEVERITY_CLASSES.forEach((cls) => elAiRecommendation.classList.remove(cls));
    if (severityClass) elAiRecommendation.classList.add(severityClass);
    elAiRecommendation.classList.add("visible");

    elAiPrediction.textContent = prediction;
    elAiConfidence.textContent = `${(confidence * 100).toFixed(1)}%`;
    */

    // --- Auto-select dropdown (only if user hasn't manually overridden) ---
    if (!userOverride) {
      elSelect.value = prediction;
      applySeverityStyle(prediction);
      elDropdownHint.textContent = "Otomatis dipilih berdasarkan rekomendasi AI.";
      elDropdownHint.classList.add("ai-active");
    }

    /* [REMARKED] Probability Bars
    if (probabilities) {
      elProbContainer.classList.add("visible");
      updateProbabilityBars(probabilities);
    }
    */
  }

  // ================================================================
  // Hide Recommendation
  // ================================================================
  function hideRecommendation() {
    /* [REMARKED] Badge & probability bars
    elAiRecommendation.classList.remove("visible");
    SEVERITY_CLASSES.forEach((cls) => elAiRecommendation.classList.remove(cls));
    elProbContainer.classList.remove("visible");
    */

    // Reset dropdown styling
    SEVERITY_CLASSES.forEach((cls) => elSelect.classList.remove(cls));

    // Reset hint
    elDropdownHint.textContent = "Silakan pilih melalui dropdown jika rekomendasi tidak sesuai.";
    elDropdownHint.classList.remove("ai-active");

    lastPrediction = null;
  }

  // ================================================================
  // Severity Styling on Dropdown
  // ================================================================
  function applySeverityStyle(prediction) {
    SEVERITY_CLASSES.forEach((cls) => elSelect.classList.remove(cls));
    const cls = SEVERITY_MAP[prediction];
    if (cls) elSelect.classList.add(cls);
  }

  /* [REMARKED] Update Probability Bars
  // ================================================================
  // Update Probability Bars
  // ================================================================
  function updateProbabilityBars(probabilities) {
    const categories = ["Fatal", "Mayor", "Minor", "Kosmetik"];
    categories.forEach((cat) => {
      const key = cat.toLowerCase();
      const prob = probabilities[cat] || 0;
      const pct = (prob * 100).toFixed(1);

      const barEl = document.getElementById(`prob-bar-${key}`);
      const valEl = document.getElementById(`prob-val-${key}`);

      if (barEl) {
        // Use requestAnimationFrame for smooth animation
        requestAnimationFrame(() => {
          barEl.style.width = `${pct}%`;
        });
      }
      if (valEl) {
        valEl.textContent = `${pct}%`;
      }
    });
  }
  */

  // ================================================================
  // Start
  // ================================================================
  document.addEventListener("DOMContentLoaded", init);
})();

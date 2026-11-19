/**
 * Fake News Detection System - Frontend JavaScript Integration
 * Connects UI to Flask REST API endpoints and Cloud AI Server
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Elements ---
  const newsInput = document.getElementById("news-input");
  const charCount = document.getElementById("char-count");
  const wordCount = document.getElementById("word-count");
  const detectionForm = document.getElementById("detection-form");
  const clearBtn = document.getElementById("clear-btn");
  const analyzeBtn = document.getElementById("analyze-btn");
  const errorBanner = document.getElementById("error-banner");
  const errorMessage = document.getElementById("error-message");
  const loadingIndicator = document.getElementById("loading-indicator");
  const resultContainer = document.getElementById("result-container");
  const resultCard = document.getElementById("result-card");

  // Engine Selector Elements
  const enginePills = document.querySelectorAll(".engine-pill");
  const engineRadios = document.querySelectorAll('input[name="detection_engine"]');
  const btnCloudSettings = document.getElementById("btn-cloud-settings");
  const cloudConfigModal = document.getElementById("cloud-config-modal");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const modalBackdrop = document.getElementById("modal-backdrop");
  const geminiApiKeyInput = document.getElementById("gemini-api-key");
  const btnSaveApiKey = document.getElementById("btn-save-api-key");
  const btnClearApiKey = document.getElementById("btn-clear-api-key");
  const modalKeyStatus = document.getElementById("modal-key-status");

  // Sample Buttons
  const btnSampleReal = document.getElementById("btn-sample-real");
  const btnSampleFake = document.getElementById("btn-sample-fake");

  // Result Elements
  const resultIcon = document.getElementById("result-icon");
  const resultTitle = document.getElementById("result-title");
  const confidencePercent = document.getElementById("confidence-percent");
  const confidenceTitle = document.getElementById("confidence-title");
  const confidenceBarFill = document.getElementById("confidence-bar-fill");
  const scoreTypeNote = document.getElementById("score-type-note");
  const resModelName = document.getElementById("res-model-name");
  const resTimestamp = document.getElementById("res-timestamp");
  const resLength = document.getElementById("res-length");
  const resCleanedWords = document.getElementById("res-cleaned-words");
  const resExcerpt = document.getElementById("res-excerpt");
  const btnAnalyzeAnother = document.getElementById("btn-analyze-another");
  const btnCopyResult = document.getElementById("btn-copy-result");

  // Cloud Evidence Elements
  const cloudEvidenceBox = document.getElementById("cloud-evidence-box");
  const resExplanation = document.getElementById("res-explanation");
  const resSignalsList = document.getElementById("res-signals-list");

  // Model Info Elements
  const metricModelName = document.getElementById("metric-model-name");
  const metricTrainSize = document.getElementById("metric-train-size");
  const metricTestSize = document.getElementById("metric-test-size");
  const metricTrainedAt = document.getElementById("metric-trained-at");
  const metricAccuracy = document.getElementById("metric-accuracy");
  const metricPrecision = document.getElementById("metric-precision");
  const metricRecall = document.getElementById("metric-recall");
  const metricF1 = document.getElementById("metric-f1");
  const datasetAlertBanner = document.getElementById("dataset-alert-banner");
  const datasetWarningText = document.getElementById("dataset-warning-text");

  // Separate Class Metrics
  const fakePrecision = document.getElementById("fake-precision");
  const fakeRecall = document.getElementById("fake-recall");
  const fakeF1 = document.getElementById("fake-f1");
  const fakeSupport = document.getElementById("fake-support");
  const realPrecision = document.getElementById("real-precision");
  const realRecall = document.getElementById("real-recall");
  const realF1 = document.getElementById("real-f1");
  const realSupport = document.getElementById("real-support");

  // Confusion Matrix Elements
  const cmTN = document.getElementById("cm-tn");
  const cmFP = document.getElementById("cm-fp");
  const cmFN = document.getElementById("cm-fn");
  const cmTP = document.getElementById("cm-tp");
  const comparisonTbody = document.getElementById("comparison-tbody");

  // History Elements
  const historyTbody = document.getElementById("history-tbody");
  const historyCount = document.getElementById("history-count");
  const btnClearHistory = document.getElementById("btn-clear-history");

  // Navigation & Status
  const navToggle = document.getElementById("nav-toggle");
  const navMenu = document.getElementById("nav-menu");
  const navLinks = document.querySelectorAll(".nav-link");
  const systemStatus = document.getElementById("system-status");

  // --- Real & Fake Demonstration Samples ---
  const SAMPLE_REAL =
    "Senate passes bipartisan infrastructure package after months of negotiations. The United States Senate approved a trillion dollar bipartisan infrastructure bill on Tuesday, providing significant federal funding for roads, bridges, public transit, clean water systems, and broadband internet access across the country.";

  const SAMPLE_FAKE =
    "SHOCKING: Secret government lab exposed developing weather control weapons. Whistleblowers just leaked from a classified underground base that secret weather control machines are causing natural disasters to manipulate the stock market and control world leaders. The mainstream media is completely silent!";

  // --- Helper: Get Stored Cloud API Key ---
  function getStoredApiKey() {
    return localStorage.getItem("verinews_gemini_api_key") || "";
  }

  // --- Helper: Get Active Detection Engine ---
  function getActiveEngine() {
    const checked = document.querySelector('input[name="detection_engine"]:checked');
    return checked ? checked.value : "cloud";
  }

  // --- Engine Selector UI Logic ---
  engineRadios.forEach((radio) => {
    radio.addEventListener("change", () => {
      enginePills.forEach((pill) => pill.classList.remove("active"));
      const parentLabel = radio.closest(".engine-pill");
      if (parentLabel) parentLabel.classList.add("active");
    });
  });

  // --- Cloud Modal Logic ---
  if (btnCloudSettings && cloudConfigModal) {
    btnCloudSettings.addEventListener("click", () => {
      geminiApiKeyInput.value = getStoredApiKey();
      updateModalKeyStatus();
      cloudConfigModal.classList.remove("hidden");
    });

    const closeModal = () => cloudConfigModal.classList.add("hidden");
    if (btnCloseModal) btnCloseModal.addEventListener("click", closeModal);
    if (modalBackdrop) modalBackdrop.addEventListener("click", closeModal);

    if (btnSaveApiKey) {
      btnSaveApiKey.addEventListener("click", () => {
        const keyVal = geminiApiKeyInput.value.trim();
        if (keyVal) {
          localStorage.setItem("verinews_gemini_api_key", keyVal);
          alert("Gemini Cloud AI API key saved successfully! The Cloud AI Server will now be utilized for deep verification.");
        } else {
          localStorage.removeItem("verinews_gemini_api_key");
        }
        updateModalKeyStatus();
        closeModal();
      });
    }

    if (btnClearApiKey) {
      btnClearApiKey.addEventListener("click", () => {
        localStorage.removeItem("verinews_gemini_api_key");
        geminiApiKeyInput.value = "";
        updateModalKeyStatus();
        alert("API key cleared. System will use the Live Cloud Knowledge Server.");
      });
    }
  }

  function updateModalKeyStatus() {
    if (!modalKeyStatus) return;
    const key = getStoredApiKey();
    if (key) {
      modalKeyStatus.style.backgroundColor = "#eff6ff";
      modalKeyStatus.style.borderColor = "#bfdbfe";
      modalKeyStatus.style.color = "#1e40af";
      modalKeyStatus.innerHTML = '<i class="fa-solid fa-circle-check"></i> <span>Custom Gemini Cloud AI Key active.</span>';
    } else {
      modalKeyStatus.style.backgroundColor = "#f0fdf4";
      modalKeyStatus.style.borderColor = "#bbf7d0";
      modalKeyStatus.style.color = "#166534";
      modalKeyStatus.innerHTML = '<i class="fa-solid fa-circle-check"></i> <span>Live Cloud Knowledge Server active (Zero-config mode).</span>';
    }
  }

  // --- Helper: Update Textarea Counters ---
  function updateCounters() {
    const text = newsInput.value || "";
    const chars = text.length;
    const words = text.trim() === "" ? 0 : text.trim().split(/\s+/).length;
    charCount.textContent = chars.toLocaleString();
    wordCount.textContent = words.toLocaleString();
  }

  // --- Helper: Error Banner ---
  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove("hidden");
  }

  function hideError() {
    errorBanner.classList.add("hidden");
  }

  // --- Event: Text Input Counter ---
  newsInput.addEventListener("input", () => {
    updateCounters();
    if (errorBanner && !errorBanner.classList.contains("hidden")) {
      hideError();
    }
  });

  // --- Event: Sample Buttons ---
  btnSampleReal.addEventListener("click", () => {
    newsInput.value = SAMPLE_REAL;
    updateCounters();
    hideError();
    resultContainer.classList.add("hidden");
    newsInput.focus();
  });

  btnSampleFake.addEventListener("click", () => {
    newsInput.value = SAMPLE_FAKE;
    updateCounters();
    hideError();
    resultContainer.classList.add("hidden");
    newsInput.focus();
  });

  // --- Event: Clear Form ---
  clearBtn.addEventListener("click", () => {
    newsInput.value = "";
    updateCounters();
    hideError();
    resultContainer.classList.add("hidden");
    newsInput.focus();
  });

  // --- Event: Submit Form for Analysis ---
  detectionForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();

    const rawText = newsInput.value.trim();

    // Client-side Validation
    if (!rawText) {
      showError("Please enter a news headline or article. Input cannot be empty.");
      newsInput.focus();
      return;
    }

    if (rawText.length < 15) {
      showError(`Input is too short (${rawText.length} characters). Please enter at least 15 characters for a reliable prediction.`);
      newsInput.focus();
      return;
    }

    const words = rawText.split(/\s+/);
    if (words.length < 3) {
      showError(`Input contains too few words (${words.length}). Please enter at least 3 words.`);
      newsInput.focus();
      return;
    }

    // Display Loading State
    loadingIndicator.classList.remove("hidden");
    resultContainer.classList.add("hidden");
    analyzeBtn.disabled = true;

    const activeEngine = getActiveEngine();
    const loadingText = activeEngine === "cloud"
      ? '<i class="fa-solid fa-spinner fa-spin"></i> Querying Cloud Server...'
      : '<i class="fa-solid fa-spinner fa-spin"></i> Analysing...';
    analyzeBtn.innerHTML = loadingText;

    try {
      const payload = {
        news: rawText,
        engine: activeEngine,
        api_key: getStoredApiKey() || null
      };

      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok || data.status !== "success") {
        throw new Error(data.message || "Failed to analyze the news article. Please try again.");
      }

      // Render Prediction Results
      renderPredictionResult(data, rawText);

      // Refresh Session History
      loadHistory();
    } catch (err) {
      showError(err.message || "Unable to connect to the prediction server. Please make sure the backend is running.");
    } finally {
      loadingIndicator.classList.add("hidden");
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Analyse News';
    }
  });

  // --- Render Prediction Result ---
  function renderPredictionResult(data, originalText) {
    const isReal = data.prediction === "REAL";

    // Theme Card
    resultCard.classList.remove("is-real", "is-fake");
    resultCard.classList.add(isReal ? "is-real" : "is-fake");

    // Icons and Titles
    if (isReal) {
      resultIcon.className = "fa-solid fa-circle-check";
      resultTitle.textContent = "Classified as REAL NEWS";
      confidenceTitle.textContent = "Authenticity Confidence";
    } else {
      resultIcon.className = "fa-solid fa-triangle-exclamation";
      resultTitle.textContent = "Classified as FAKE NEWS";
      confidenceTitle.textContent = "Misinformation Risk";
    }

    // Handle Confidence Display Gracefully
    if (data.confidence !== null && data.confidence !== undefined && !isNaN(Number(data.confidence))) {
      const conf = Number(data.confidence);
      confidencePercent.textContent = `${conf.toFixed(1)}%`;
      confidenceBarFill.style.width = "0%";
      setTimeout(() => {
        confidenceBarFill.style.width = `${Math.min(100, Math.max(0, conf))}%`;
      }, 50);
      scoreTypeNote.textContent = data.score_label || "Authenticity Probability Score";
    } else {
      confidencePercent.textContent = "N/A";
      confidenceBarFill.style.width = "0%";
      scoreTypeNote.textContent = data.score_label || "Probability estimate not available";
    }

    // Analysis Metadata
    resModelName.textContent = data.model_name || "Cloud AI / Machine Learning Engine";
    resTimestamp.textContent = data.analysis_time || new Date().toLocaleTimeString();

    const charLen = originalText.length;
    const wordLen = originalText.trim().split(/\s+/).length;
    resLength.textContent = `${charLen} chars (${wordLen} words)`;

    const cleanWordCount = data.input_metrics && data.input_metrics.cleaned_word_count !== undefined
      ? data.input_metrics.cleaned_word_count
      : "--";
    resCleanedWords.textContent = `${cleanWordCount} n-gram tokens`;

    // Excerpt Snippet
    const excerpt = originalText.length > 220
      ? originalText.substring(0, 220) + "..."
      : originalText;
    resExcerpt.textContent = `"${excerpt}"`;

    // Render Cloud Fact-Checking Evidence & Explanation
    if (data.explanation && cloudEvidenceBox) {
      resExplanation.textContent = data.explanation;

      if (data.key_signals && Array.isArray(data.key_signals) && data.key_signals.length > 0) {
        resSignalsList.innerHTML = "";
        data.key_signals.forEach((signal) => {
          const li = document.createElement("li");
          li.textContent = signal;
          resSignalsList.appendChild(li);
        });
      }
      cloudEvidenceBox.classList.remove("hidden");
    } else if (cloudEvidenceBox) {
      cloudEvidenceBox.classList.add("hidden");
    }

    // Reveal Result Container & Smooth Scroll
    resultContainer.classList.remove("hidden");
    resultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // --- Event: Analyse Another Article ---
  btnAnalyzeAnother.addEventListener("click", () => {
    newsInput.value = "";
    updateCounters();
    hideError();
    resultContainer.classList.add("hidden");
    newsInput.scrollIntoView({ behavior: "smooth" });
    newsInput.focus();
  });

  // --- Event: Copy Result Summary ---
  btnCopyResult.addEventListener("click", () => {
    const title = resultTitle.textContent;
    const conf = confidencePercent.textContent;
    const model = resModelName.textContent;
    const time = resTimestamp.textContent;
    const expl = resExplanation ? `\nEvidence: ${resExplanation.textContent}` : "";
    const summaryText = `[VeriNews AI Analysis Result]\nVerdict: ${title}\nConfidence: ${conf}\nModel: ${model}\nAnalyzed at: ${time}${expl}\nNote: Generated by Fake News Detection System.`;

    navigator.clipboard.writeText(summaryText).then(() => {
      const originalHtml = btnCopyResult.innerHTML;
      btnCopyResult.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
      btnCopyResult.classList.add("btn-primary");
      setTimeout(() => {
        btnCopyResult.innerHTML = originalHtml;
        btnCopyResult.classList.remove("btn-primary");
      }, 2000);
    }).catch(() => {
      alert("Unable to copy to clipboard.");
    });
  });

  // --- Load Model Metrics from /model-info ---
  async function loadModelInfo() {
    try {
      const res = await fetch("/model-info");
      const data = await res.json();

      if (data.status === "success" && data.metrics) {
        const m = data.metrics;

        // Meta Bar
        metricModelName.textContent = m.selected_model || "Logistic Regression";
        metricTrainedAt.textContent = m.trained_at || "Recent";

        if (m.dataset) {
          metricTrainSize.textContent = `${m.dataset.training_samples.toLocaleString()} samples (80%)`;
          metricTestSize.textContent = `${m.dataset.testing_samples.toLocaleString()} samples (20%)`;
        }

        // Demonstration Dataset Alert
        if (m.is_sample_dataset && datasetAlertBanner) {
          datasetAlertBanner.classList.remove("hidden");
          if (m.sample_dataset_warning && datasetWarningText) {
            datasetWarningText.textContent = m.sample_dataset_warning;
          }
        } else if (datasetAlertBanner) {
          datasetAlertBanner.classList.add("hidden");
        }

        // 4 Key Overall Performance Metrics
        if (m.performance) {
          metricAccuracy.textContent = `${m.performance.accuracy.toFixed(1)}%`;
          metricPrecision.textContent = `${m.performance.precision.toFixed(1)}%`;
          metricRecall.textContent = `${m.performance.recall.toFixed(1)}%`;
          metricF1.textContent = `${m.performance.f1_score.toFixed(1)}%`;
        }

        // Separate Class Breakdown (FAKE vs REAL)
        if (m.class_metrics) {
          const fc = m.class_metrics.fake_class;
          const rc = m.class_metrics.real_class;

          if (fc && fakePrecision && fakeRecall && fakeF1) {
            fakePrecision.textContent = `${fc.precision.toFixed(1)}%`;
            fakeRecall.textContent = `${fc.recall.toFixed(1)}%`;
            fakeF1.textContent = `${fc.f1_score.toFixed(1)}%`;
            if (fakeSupport) fakeSupport.textContent = `Support: ${fc.support} samples`;
          }

          if (rc && realPrecision && realRecall && realF1) {
            realPrecision.textContent = `${rc.precision.toFixed(1)}%`;
            realRecall.textContent = `${rc.recall.toFixed(1)}%`;
            realF1.textContent = `${rc.f1_score.toFixed(1)}%`;
            if (realSupport) realSupport.textContent = `Support: ${rc.support} samples`;
          }
        }

        // Confusion Matrix
        if (m.confusion_matrix) {
          cmTN.textContent = m.confusion_matrix.true_negatives_fake_correct;
          cmFP.textContent = m.confusion_matrix.false_positives_fake_as_real;
          cmFN.textContent = m.confusion_matrix.false_negatives_real_as_fake;
          cmTP.textContent = m.confusion_matrix.true_positives_real_correct;
        }

        // Algorithm Comparison Table
        if (m.all_models_comparison && comparisonTbody) {
          comparisonTbody.innerHTML = "";
          const activeModel = m.selected_model;

          for (const [modelName, metrics] of Object.entries(m.all_models_comparison)) {
            const tr = document.createElement("tr");
            if (modelName === activeModel) {
              tr.classList.add("selected-row");
            }

            const isCurrent = modelName === activeModel;
            tr.innerHTML = `
              <td>
                <strong>${modelName}</strong>
                ${isCurrent ? ' <span class="badge badge-real" style="padding: 0.15rem 0.5rem; font-size: 0.7rem;"><i class="fa-solid fa-check"></i> Deployed</span>' : ''}
              </td>
              <td>${metrics.accuracy.toFixed(2)}%</td>
              <td>${metrics.precision.toFixed(2)}%</td>
              <td>${metrics.recall.toFixed(2)}%</td>
              <td><strong>${metrics.f1_score.toFixed(2)}%</strong></td>
            `;
            comparisonTbody.appendChild(tr);
          }
        }
      } else {
        metricModelName.textContent = "Pending Model Training";
        metricAccuracy.textContent = "N/A";
        metricPrecision.textContent = "N/A";
        metricRecall.textContent = "N/A";
        metricF1.textContent = "N/A";
      }
    } catch (err) {
      console.warn("Could not retrieve model metrics:", err);
      metricModelName.textContent = "Offline / Error";
    }
  }

  // --- Load Isolated Per-User History from /history ---
  async function loadHistory() {
    try {
      const res = await fetch("/history");
      const data = await res.json();

      let history = [];
      if (data.status === "success" && Array.isArray(data.history)) {
        history = data.history;
      }

      historyCount.textContent = history.length;

      if (history.length === 0) {
        historyTbody.innerHTML = `
          <tr id="history-empty-row">
            <td colspan="5" class="empty-history-cell">
              <i class="fa-solid fa-clipboard-question"></i>
              <p>No news articles analyzed yet in this session.</p>
              <span>Use the "Analyse Your News" section above to run your first detection!</span>
            </td>
          </tr>
        `;
        return;
      }

      historyTbody.innerHTML = "";
      history.forEach((item, index) => {
        const isReal = item.prediction === "REAL";
        const badgeClass = isReal ? "history-badge real" : "history-badge fake";
        const iconClass = isReal ? "fa-solid fa-check" : "fa-solid fa-triangle-exclamation";
        const confDisplay = typeof item.confidence === "number" && !isNaN(item.confidence)
          ? `${item.confidence.toFixed(1)}%`
          : "N/A";

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>#${index + 1}</strong></td>
          <td><span style="color: var(--text-main); font-weight: 500;">${escapeHtml(item.preview)}</span></td>
          <td>
            <span class="${badgeClass}">
              <i class="${iconClass}"></i> ${item.prediction}
            </span>
          </td>
          <td><strong>${confDisplay}</strong></td>
          <td style="color: var(--text-muted); font-size: 0.825rem;">${item.timestamp}</td>
        `;
        historyTbody.appendChild(tr);
      });
    } catch (err) {
      console.warn("Could not retrieve session history:", err);
    }
  }

  // --- Event: Clear Isolated Session History ---
  btnClearHistory.addEventListener("click", async () => {
    if (confirm("Are you sure you want to clear your prediction history for this session?")) {
      try {
        await fetch("/history/clear", { method: "POST" });
        loadHistory();
      } catch (err) {
        alert("Failed to clear session history.");
      }
    }
  });

  // --- Check Backend Health ---
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (data.status === "healthy" && systemStatus) {
        systemStatus.innerHTML = `
          <span class="status-dot"></span>
          <span class="status-text">Cloud &amp; Local Active (${data.local_model_loaded ? "Model Ready" : "Awaiting Model"})</span>
        `;
      }
    } catch {
      if (systemStatus) {
        systemStatus.innerHTML = `
          <span class="status-dot" style="background-color: #ef4444; box-shadow: 0 0 6px #ef4444;"></span>
          <span class="status-text" style="color: #ef4444;">Backend Disconnected</span>
        `;
      }
    }
  }

  // --- Mobile Navigation Toggle ---
  navToggle.addEventListener("click", () => {
    navMenu.classList.toggle("open");
  });

  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      navMenu.classList.remove("open");
      navLinks.forEach((l) => l.classList.remove("active"));
      link.classList.add("active");
    });
  });

  // --- Helper: Escape HTML ---
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // --- Initial Setup on Page Load ---
  updateCounters();
  checkHealth();
  loadModelInfo();
  loadHistory();
});

/**
 * Spam SMS / Email Classifier - Vercel Web Application Client
 * Features:
 *  - 0ms Client-Side Mathematical Inference Engine (TF-IDF + Multinomial Naive Bayes)
 *  - Real-time Serverless Vercel API fallback & latency benchmarking
 *  - Word Signal Explainability & Keyword Extraction
 *  - Preloaded Benchmark Test Cases (13 examples)
 *  - Raw Email Parsing (Header extraction & text cleaning)
 *  - Batch CSV / Multi-line Classifier
 *  - Decision Threshold Tuning & Sensitivity Curve
 */

let MODEL_DATA = null;
let CURRENT_THRESHOLD = 0.50;

// Default benchmark examples from assignment and validation suite
const BENCHMARK_EXAMPLES = [
  { id: 1, text: "Congratulations! You have won a prize. Claim now.", trueLabel: "Spam" },
  { id: 2, text: "Are we still meeting at 5 pm?", trueLabel: "Ham" },
  { id: 3, text: "URGENT: claim your free reward today!", trueLabel: "Spam" },
  { id: 4, text: "Please send me the notes from class.", trueLabel: "Ham" },
  { id: 5, text: "Your account has been selected for a bonus.", trueLabel: "Spam" },
  { id: 6, text: "I will call you when I reach home.", trueLabel: "Ham" },
  { id: 7, text: "Limited offer! Click the link to get a discount.", trueLabel: "Spam" },
  { id: 8, text: "Can you share the assignment file?", trueLabel: "Ham" },
  { id: 9, text: "You have been selected for a cash reward.", trueLabel: "Spam" },
  { id: 10, text: "Thanks, see you tomorrow.", trueLabel: "Ham" },
  { id: 11, text: "Hey, can you review the pull request on GitHub before tonight?", trueLabel: "Ham" },
  { id: 12, text: "IMPORTANT: Your parcel delivery failed. Update your card details at http://scam-delivery.link", trueLabel: "Spam" },
  { id: 13, text: "Your Google verification code is 482910. Do not share this with anyone.", trueLabel: "Ham" }
];

// Load model export on startup
async function initApp() {
  setupTabs();
  setupEventListeners();
  renderBenchmarkList();

  try {
    const res = await fetch("/model_export.json");
    if (!res.ok) throw new Error("Could not load /model_export.json");
    MODEL_DATA = await res.json();
    MODEL_DATA._regex = new RegExp(MODEL_DATA.token_pattern || "\\b\\w+\\b|[!?$£]", "gi");
    
    document.getElementById("engine-status").innerHTML = 
      `<span class="status-dot"></span> Vercel Engine Ready (${MODEL_DATA.vocab_size.toLocaleString()} n-grams)`;
    
    // Auto-evaluate first benchmark
    runClassification(BENCHMARK_EXAMPLES[0].text);
  } catch (err) {
    console.warn("Client model load failed, falling back to /api/predict:", err);
    document.getElementById("engine-status").innerHTML = 
      `<span class="status-dot" style="background:#f59e0b"></span> Vercel API Mode (Online)`;
  }
}

// Tab navigation
function setupTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");
    });
  });
}

// Pure client-side zero-latency inference
function predictClientSide(text, threshold = CURRENT_THRESHOLD) {
  if (!MODEL_DATA) return null;

  const t0 = performance.now();
  const lower = text.toLowerCase();
  const matches = lower.match(MODEL_DATA._regex) || [];
  
  // Unigrams and bigrams
  const terms = [...matches];
  for (let i = 0; i < matches.length - 1; i++) {
    terms.push(`${matches[i]} ${matches[i + 1]}`);
  }

  const tf = {};
  for (const t of terms) {
    if (MODEL_DATA.vocabulary[t]) {
      tf[t] = (tf[t] || 0) + 1;
    }
  }

  // L2 Norm of TF * IDF
  let normSq = 0;
  for (const t in tf) {
    const idf = MODEL_DATA.vocabulary[t][0];
    const val = tf[t] * idf;
    normSq += val * val;
  }
  const norm = normSq > 0 ? Math.sqrt(normSq) : 1.0;

  let jllHam = MODEL_DATA.class_log_prior[0];
  let jllSpam = MODEL_DATA.class_log_prior[1];
  const contributions = [];

  for (const t in tf) {
    const [idf, diff, hamLp, spamLp] = MODEL_DATA.vocabulary[t];
    const weight = (tf[t] * idf) / norm;
    jllHam += weight * hamLp;
    jllSpam += weight * spamLp;

    contributions.push({
      term: t,
      weight: weight,
      spamSignal: diff * weight,
      favors: diff > 0 ? "Spam" : "Ham"
    });
  }

  // LogSumExp
  const maxJll = Math.max(jllHam, jllSpam);
  const sumExp = Math.exp(jllHam - maxJll) + Math.exp(jllSpam - maxJll);
  const logSum = maxJll + Math.log(sumExp);

  const probHam = Math.exp(jllHam - logSum);
  const probSpam = Math.exp(jllSpam - logSum);

  contributions.sort((a, b) => Math.abs(b.spamSignal) - Math.abs(a.spamSignal));

  const isSpam = probSpam >= threshold;
  const t1 = performance.now();

  return {
    label: isSpam ? "Spam" : "Ham",
    confidence: isSpam ? probSpam : probHam,
    hamProb: probHam,
    spamProb: probSpam,
    latencyMs: +(t1 - t0).toFixed(2),
    engine: "Browser Edge (0ms)",
    contributions: contributions.slice(0, 10)
  };
}

// Serverless API call
async function predictServerSide(text, threshold = CURRENT_THRESHOLD) {
  const t0 = performance.now();
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, threshold, explain: true })
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  const t1 = performance.now();
  
  return {
    label: data.predicted_label,
    confidence: data.confidence,
    hamProb: data.ham_probability,
    spamProb: data.spam_probability,
    latencyMs: +(t1 - t0).toFixed(2),
    engine: "Vercel Serverless Function",
    contributions: data.explanation || []
  };
}

// Master classification handler
async function runClassification(text) {
  if (!text || !text.trim()) return;

  const mode = document.querySelector('input[name="engine-select"]:checked')?.value || "edge";
  const btn = document.getElementById("classify-btn");
  if (btn) btn.disabled = true;

  try {
    let result = null;
    if (mode === "edge" && MODEL_DATA) {
      result = predictClientSide(text, CURRENT_THRESHOLD);
    } else {
      try {
        result = await predictServerSide(text, CURRENT_THRESHOLD);
      } catch (e) {
        console.warn("API failed, using edge engine:", e);
        result = predictClientSide(text, CURRENT_THRESHOLD);
      }
    }

    displayResult(result);
  } catch (err) {
    alert("Classification failed: " + err.message);
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Display result in UI
function displayResult(res) {
  const box = document.getElementById("result-box");
  const badge = document.getElementById("result-badge");
  const confText = document.getElementById("confidence-text");
  const fillSpam = document.getElementById("spam-bar-fill");
  const fillHam = document.getElementById("ham-bar-fill");
  const spamPct = document.getElementById("spam-prob-pct");
  const hamPct = document.getElementById("ham-prob-pct");
  const metaText = document.getElementById("latency-meta");
  const signalContainer = document.getElementById("signal-chips");

  box.className = `result-box ${res.label.toLowerCase()}`;
  badge.className = `result-badge ${res.label.toLowerCase()}`;
  badge.innerText = res.label;

  confText.innerText = `${(res.confidence * 100).toFixed(2)}% Confidence`;

  const sPct = (res.spamProb * 100).toFixed(1);
  const hPct = (res.hamProb * 100).toFixed(1);

  fillSpam.style.width = `${sPct}%`;
  fillHam.style.width = `${hPct}%`;
  spamPct.innerText = `${sPct}%`;
  hamPct.innerText = `${hPct}%`;

  metaText.innerText = `Engine: ${res.engine} • Latency: ${res.latencyMs} ms • Decision Threshold: ${CURRENT_THRESHOLD.toFixed(2)}`;

  // Signals
  signalContainer.innerHTML = "";
  if (res.contributions && res.contributions.length > 0) {
    res.contributions.forEach(item => {
      const chip = document.createElement("span");
      const favorsClass = item.favors.toLowerCase();
      const signalVal = item.spamSignal !== undefined ? item.spamSignal : item.spam_signal;
      const sign = signalVal > 0 ? "+" : "";
      chip.className = `signal-chip ${favorsClass}`;
      chip.innerHTML = `<strong>${item.term}</strong> (${sign}${signalVal?.toFixed(2) || "0"})`;
      signalContainer.appendChild(chip);
    });
  } else {
    signalContainer.innerHTML = `<span style="font-size:0.8rem; color:#94a3b8">No distinctive keywords detected</span>`;
  }
}

// Benchmark list rendering
function renderBenchmarkList() {
  const container = document.getElementById("benchmark-list");
  if (!container) return;
  container.innerHTML = "";

  BENCHMARK_EXAMPLES.forEach(item => {
    const el = document.createElement("div");
    el.className = "quick-pill";
    el.innerHTML = `
      <div style="flex:1; margin-right: 0.5rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
        <span style="font-weight:700; color:#64748b; margin-right:4px;">#${item.id}</span>
        <span>${item.text}</span>
      </div>
      <span class="pill-tag ${item.trueLabel.toLowerCase()}">${item.trueLabel}</span>
    `;
    el.addEventListener("click", () => {
      document.getElementById("message-input").value = item.text;
      runClassification(item.text);
    });
    container.appendChild(el);
  });
}

// Setup Event Listeners
function setupEventListeners() {
  const classifyBtn = document.getElementById("classify-btn");
  const msgInput = document.getElementById("message-input");
  const clearBtn = document.getElementById("clear-btn");
  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdVal = document.getElementById("threshold-val");

  classifyBtn?.addEventListener("click", () => runClassification(msgInput.value));
  clearBtn?.addEventListener("click", () => {
    msgInput.value = "";
    msgInput.focus();
  });

  thresholdSlider?.addEventListener("input", (e) => {
    CURRENT_THRESHOLD = parseFloat(e.target.value);
    if (thresholdVal) thresholdVal.innerText = CURRENT_THRESHOLD.toFixed(2);
    if (msgInput.value.trim()) runClassification(msgInput.value);
  });

  // Email Parser Tab
  const parseEmailBtn = document.getElementById("parse-email-btn");
  parseEmailBtn?.addEventListener("click", () => {
    const raw = document.getElementById("raw-email-input")?.value || "";
    parseAndClassifyEmail(raw);
  });

  // Batch Processor Tab
  const batchBtn = document.getElementById("run-batch-btn");
  batchBtn?.addEventListener("click", runBatchProcessing);

  // Live typing debounce optional
  let debounceTimer;
  msgInput?.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (msgInput.value.trim().length > 3) {
        runClassification(msgInput.value);
      }
    }, 350);
  });
}

// Email Parser logic
function parseAndClassifyEmail(rawEmail) {
  if (!rawEmail.trim()) return;

  let subject = "";
  let sender = "";
  let body = rawEmail;

  // Simple RFC 822 header extraction
  const subjectMatch = rawEmail.match(/^Subject:\s*(.*)$/im);
  const fromMatch = rawEmail.match(/^From:\s*(.*)$/im);
  if (subjectMatch) subject = subjectMatch[1].trim();
  if (fromMatch) sender = fromMatch[1].trim();

  // Extract body after blank line if headers present
  const headerSplit = rawEmail.split(/\r?\n\r?\n/);
  if (headerSplit.length > 1 && (subjectMatch || fromMatch)) {
    body = headerSplit.slice(1).join("\n");
  }

  // Strip basic HTML tags
  const cleanBody = body.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
  const fullTextToScore = `${subject} ${cleanBody}`.trim();

  document.getElementById("email-subject-val").innerText = subject || "(None detected)";
  document.getElementById("email-sender-val").innerText = sender || "(None detected)";
  document.getElementById("email-body-val").innerText = cleanBody.slice(0, 300) + (cleanBody.length > 300 ? "..." : "");

  const result = predictClientSide(fullTextToScore, CURRENT_THRESHOLD);
  const badge = document.getElementById("email-result-badge");
  badge.className = `result-badge ${result.label.toLowerCase()}`;
  badge.innerText = `${result.label} (${(result.confidence * 100).toFixed(1)}%)`;
  document.getElementById("email-result-container").style.display = "block";
}

// Batch Processing
function runBatchProcessing() {
  const text = document.getElementById("batch-input")?.value || "";
  const lines = text.split("\n").map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return alert("Please enter at least one message.");

  const tbody = document.getElementById("batch-results-tbody");
  tbody.innerHTML = "";
  let spamCount = 0;
  let hamCount = 0;

  lines.forEach((line, idx) => {
    const res = predictClientSide(line, CURRENT_THRESHOLD);
    if (res.label === "Spam") spamCount++; else hamCount++;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="padding:0.5rem; border:1px solid #e2e8f0;">#${idx + 1}</td>
      <td style="padding:0.5rem; border:1px solid #e2e8f0; max-width:350px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${line}</td>
      <td style="padding:0.5rem; border:1px solid #e2e8f0; font-weight:700;" class="${res.label === 'Spam' ? 'text-red' : 'text-green'}">
        <span class="pill-tag ${res.label.toLowerCase()}">${res.label}</span>
      </td>
      <td style="padding:0.5rem; border:1px solid #e2e8f0;">${(res.confidence * 100).toFixed(1)}%</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById("batch-summary").innerText = 
    `Processed ${lines.length} items • ${spamCount} Spam (${((spamCount / lines.length) * 100).toFixed(1)}%) • ${hamCount} Ham (${((hamCount / lines.length) * 100).toFixed(1)}%)`;
  document.getElementById("batch-results-table").style.display = "table";
}

// Initialize on DOM load
window.addEventListener("DOMContentLoaded", initApp);

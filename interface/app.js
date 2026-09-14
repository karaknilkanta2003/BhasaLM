// Indic Transformer LM Playground Client Logic

const PRESETS = {
  hindi: [
    {
      label: "Transitive Extrema (लंबाई)",
      text: "अमित, राहुल से लंबा है। राहुल, विकास से लंबा है। इनमें से सबसे लंबा कौन है? उत्तर:"
    },
    {
      label: "Quantitative Comparison (फल)",
      text: "कविता के पास 32 सेब हैं और अंजलि के पास 47 सेब हैं। किसके पास अधिक सेब हैं? उत्तर:"
    },
    {
      label: "Multi-Hop Inequality (वजन)",
      text: "दीपक, नीरज से भारी है और नीरज, प्रमोद से भारी है। क्या दीपक, प्रमोद से भारी है? उत्तर:"
    },
    {
      label: "Open Causal Prose (साहित्य)",
      text: "भारत एक विशाल और सुंदर देश है जहां विभिन्न संस्कृतियों के लोग"
    }
  ],
  assamese: [
    {
      label: "Transitive Extrema (ওখ-চাপৰ)",
      text: "ৰাহুল ৰমেনতকৈ ওখ। ৰমেন বিকাশতকৈ ওখ। আটাইতকৈ ওখ কোন? উত্তৰ:"
    },
    {
      label: "Quantitative Comparison (টকা)",
      text: "প্ৰণৱৰ ওচৰত ২৮ টকা আছে আৰু ভাস্কৰৰ ওচৰত ১৪ টকা আছে। কাৰ ওচৰত বেছি ধন আছে? উত্তৰ:"
    },
    {
      label: "Multi-Hop Verification (ওখ নে?)",
      text: "মানস দীপাংকৰতকৈ ওখ আৰু দীপাংকৰ মৃদুলতকৈ ওখ। মানস মৃদুলতকৈ ওখ নে? উত্তৰ:"
    },
    {
      label: "Open Causal Prose (সাহিত্য)",
      text: "অসম প্ৰাকৃতিক সৌন্দৰ্যৰে ভৰপূৰ এখন মনোৰম ৰাজ্য য'ত"
    }
  ]
};

let currentLang = "hindi";
let currentModelType = "finetuned";

document.addEventListener("DOMContentLoaded", () => {
  initLanguageSelector();
  initModelTypeSelector();
  initSliders();
  initPresets();
  initGenerateButton();
  checkStatus();
});

// Check hardware status
async function checkStatus() {
  const statusEl = document.getElementById("device-status");
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    statusEl.textContent = `Accelerator: ${data.device.toUpperCase()} (Online)`;
  } catch (err) {
    statusEl.textContent = "Accelerator: CPU (Connected)";
  }
}

// 1. Language selector
function initLanguageSelector() {
  const buttons = document.querySelectorAll("#lang-selector .pill-btn");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentLang = btn.dataset.lang;
      initPresets();
      
      const placeholder = currentLang === "hindi" 
        ? "Type your Hindi prompt here..." 
        : "Type your Assamese prompt here...";
      document.getElementById("prompt-input").placeholder = placeholder;
    });
  });
}

// 2. Model Checkpoint type
function initModelTypeSelector() {
  const radios = document.querySelectorAll('input[name="model_type"]');
  radios.forEach(radio => {
    radio.addEventListener("change", (e) => {
      document.querySelectorAll(".radio-card").forEach(c => c.classList.remove("active"));
      e.target.closest(".radio-card").classList.add("active");
      currentModelType = e.target.value;
    });
  });
}

// 3. Sliders
function initSliders() {
  const tempSlider = document.getElementById("temp-slider");
  const tempVal = document.getElementById("temp-val");
  tempSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    let desc = "Balanced";
    if (val === 0.0) desc = "Greedy (Deductive)";
    else if (val < 0.5) desc = "Low Entropy";
    else if (val > 1.0) desc = "Creative";
    tempVal.textContent = `${val.toFixed(1)} (${desc})`;
  });

  const tokensSlider = document.getElementById("max-tokens-slider");
  const tokensVal = document.getElementById("max-tokens-val");
  tokensSlider.addEventListener("input", (e) => {
    tokensVal.textContent = `${e.target.value} tokens`;
  });

  document.getElementById("clear-btn").addEventListener("click", () => {
    document.getElementById("prompt-input").value = "";
    document.getElementById("prompt-info").textContent = "Ready for prompt";
  });
}

// 4. Presets
function initPresets() {
  const container = document.getElementById("preset-container");
  container.innerHTML = "";
  const list = PRESETS[currentLang] || [];

  list.forEach(preset => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = preset.label;
    chip.addEventListener("click", () => {
      const input = document.getElementById("prompt-input");
      input.value = preset.text;
      input.focus();
      document.getElementById("prompt-info").textContent = `${input.value.length} characters`;
    });
    container.appendChild(chip);
  });
}

// 5. Generate action
function initGenerateButton() {
  const btn = document.getElementById("generate-btn");
  const input = document.getElementById("prompt-input");

  input.addEventListener("input", () => {
    document.getElementById("prompt-info").textContent = `${input.value.length} characters`;
  });

  btn.addEventListener("click", async () => {
    const promptText = input.value.trim();
    if (!promptText) {
      input.focus();
      return;
    }

    const temperature = parseFloat(document.getElementById("temp-slider").value);
    const maxTokens = parseInt(document.getElementById("max-tokens-slider").value, 10);

    btn.disabled = true;
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="spin">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M12 6v6l4 2"></path>
      </svg>
      <span>Inferring Model...</span>
    `;

    const displayArea = document.getElementById("output-display");
    displayArea.innerHTML = `
      <div style="color: var(--text-muted); display: flex; align-items: center; gap: 0.75rem;">
        <span class="status-dot"></span>
        <span>Running causal forward pass across 6 Transformer blocks...</span>
      </div>
    `;

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonBody({
          lang: currentLang,
          model_type: currentModelType,
          prompt: promptText,
          temperature: temperature,
          max_tokens: maxTokens
        })
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      // Render response
      renderOutput(data);

    } catch (err) {
      displayArea.innerHTML = `
        <div style="color: #ef4444;">
          <strong>Inference Error:</strong> ${err.message}
        </div>
      `;
    } finally {
      btn.disabled = false;
      btn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        <span>Generate Completion</span>
      `;
    }
  });
}

function jsonBody(obj) {
  return JSON.stringify(obj);
}

function renderOutput(data) {
  const displayArea = document.getElementById("output-display");
  const latencyPill = document.getElementById("latency-pill");
  const tokensPill = document.getElementById("tokens-pill");
  const modelTag = document.getElementById("active-model-tag");
  const diagBar = document.getElementById("diagnostic-bar");
  const candidateList = document.getElementById("candidate-list");
  const inspection = document.getElementById("token-inspection");
  const tokenChips = document.getElementById("token-chips");

  latencyPill.style.display = "inline-block";
  latencyPill.textContent = `${data.latency_ms} ms`;

  tokensPill.style.display = "inline-block";
  tokensPill.textContent = `${data.num_tokens} tokens`;

  if (modelTag) {
    const langLabel = currentLang === "hindi" ? "Hindi" : "Assamese";
    const typeLabel = currentModelType === "finetuned" ? "Reasoning SFT" : "Pretrained Base";
    modelTag.textContent = `${langLabel} · ${typeLabel}`;
  }

  const completion = data.completion ? data.completion.trim() : "";
  
  if (data.emitted_eos_immediately || !completion) {
    displayArea.innerHTML = `
      <div class="empty-eos-notice">
        <strong>⚠️ Model emitted &lt;/s&gt; (End-of-Sequence) immediately.</strong>
        <small>In this ~25M parameter model, when prediction confidence is low or mode-collapsed, it predicts EOS to stop generating rather than halluncinating. Try adjusting prompt wording, increasing temperature (e.g., 0.8 - 1.2), or switching to the <strong>Pretrained Base</strong> model.</small>
      </div>
      <div class="full-context-box">
        <span class="context-label">Prompt Checked:</span>
        <span class="result-prompt">${escapeHtml(data.prompt)}</span>
      </div>
    `;
  } else {
    displayArea.innerHTML = `
      <div class="result-answer-box">
        <span class="result-tag">Generated Completion:</span>
        <div class="result-text-main">${escapeHtml(completion)}</div>
      </div>
      <div class="full-context-box">
        <span class="context-label">Full Continuity (Prompt + Generated):</span>
        <span class="result-prompt">${escapeHtml(data.prompt)}</span>
        <span class="result-generated"> ${escapeHtml(completion)}</span>
      </div>
    `;
  }

  // Diagnostic Probabilities
  if (data.top_candidates && data.top_candidates.length > 0) {
    diagBar.style.display = "block";
    candidateList.innerHTML = "";
    data.top_candidates.forEach(cand => {
      const item = document.createElement("div");
      item.className = "prob-item";
      const isEos = cand.is_eos;
      item.innerHTML = `
        <span class="prob-name" title="${escapeHtml(cand.piece)}">${escapeHtml(cand.piece)}</span>
        <div class="prob-bar-container">
          <div class="prob-bar-fill ${isEos ? 'eos-fill' : ''}" style="width: ${Math.max(2, cand.prob)}%;"></div>
        </div>
        <span class="prob-val">${cand.prob}%</span>
      `;
      candidateList.appendChild(item);
    });
  } else {
    diagBar.style.display = "none";
  }

  // Render tokens
  if (data.token_pieces && data.token_pieces.length > 0) {
    inspection.style.display = "block";
    tokenChips.innerHTML = "";
    data.token_pieces.forEach(p => {
      const pill = document.createElement("span");
      pill.className = "token-pill";
      pill.textContent = p;
      tokenChips.appendChild(pill);
    });
  } else {
    inspection.style.display = "none";
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

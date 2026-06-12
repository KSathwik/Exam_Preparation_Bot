/* Exam Prep Bot — Frontend logic */

const API = "/api";
let documentReady = false;

document.addEventListener("DOMContentLoaded", () => {
  setupUpload();
  checkAPIStatus();
  loadStats();
  setupQueryInput();
});

/* ── Upload ──────────────────────────────────────────────────────── */

function setupUpload() {
  const area = document.getElementById("uploadArea");
  const input = document.getElementById("fileInput");

  area.addEventListener("click", () => input.click());
  area.addEventListener("dragover", (e) => { e.preventDefault(); area.classList.add("dragover"); });
  area.addEventListener("dragleave", () => area.classList.remove("dragover"));
  area.addEventListener("drop", (e) => {
    e.preventDefault();
    area.classList.remove("dragover");
    if (e.dataTransfer.files[0]) { input.files = e.dataTransfer.files; uploadFile(); }
  });
  input.addEventListener("change", uploadFile);
}

async function uploadFile() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) return;

  const form = new FormData();
  form.append("file", file);
  const s = document.getElementById("uploadStatus");
  s.innerHTML = '<div class="status info"><div class="spinner"></div>Uploading…</div>';

  try {
    const res = await fetch(`${API}/documents/upload`, { method: "POST", body: form });
    const data = await res.json();
    if (data.success) {
      s.innerHTML = `<div class="status success">&#10003; ${esc(data.message)}</div>`;
      documentReady = true;
      loadStats();
    } else {
      s.innerHTML = `<div class="status error">&#10007; ${esc(data.detail || data.error || "Upload failed")}</div>`;
    }
  } catch (err) {
    s.innerHTML = `<div class="status error">&#10007; ${esc(err.message)}</div>`;
  }
}

/* ── Chat ────────────────────────────────────────────────────────── */

function setupQueryInput() {
  document.getElementById("queryInput").addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuery(); }
  });
}

async function sendQuery() {
  const input = document.getElementById("queryInput");
  const query = input.value.trim();
  if (!query) return;
  if (!documentReady) { alert("Please upload a document first"); return; }

  addMessage(query, "user");
  input.value = "";
  const loadId = addMessage("Thinking…", "assistant");

  try {
    const res = await fetch(`${API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    removeMessage(loadId);

    if (data.success) {
      addMessage(data.answer, "assistant");
      displayResponse(data);
    } else {
      addMessage("Error: " + (data.detail || data.error || "Unknown error"), "assistant");
    }
  } catch (err) {
    removeMessage(loadId);
    addMessage("Error: " + err.message, "assistant");
  }
}

function addMessage(content, role) {
  const box = document.getElementById("messagesBox");
  const div = document.createElement("div");
  const id = "msg-" + Date.now();
  div.id = id;
  div.className = "message " + role;
  div.innerHTML =
    `<div class="message-content">${esc(content)}</div>` +
    `<div class="message-time">${new Date().toLocaleTimeString()}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return id;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function displayResponse(data) {
  document.getElementById("responsePanel").style.display = "block";
  document.getElementById("answerText").textContent = data.answer;
  document.getElementById("intentValue").textContent = data.query_intent || "-";
  document.getElementById("confidenceValue").textContent =
    data.overall_confidence != null ? (data.overall_confidence * 100).toFixed(0) + "%" : "-";
  document.getElementById("riskValue").textContent = data.hallucination_risk || "-";
  document.getElementById("timeValue").textContent =
    data.response_time_seconds != null ? data.response_time_seconds.toFixed(2) : "-";

  const srcDiv = document.getElementById("sourcesDiv");
  if (data.sources && data.sources.length) {
    let html = '<div class="sources-title">Sources</div>';
    data.sources.forEach((s) => {
      const page = s.page_number ?? s.page ?? "?";
      const quote = s.quoted_text ?? s.quote ?? "";
      html += `<div class="source-item">
        <div class="source-page">Page ${page}</div>
        <div class="source-quote">"${esc(quote.substring(0, 150))}…"</div>
      </div>`;
    });
    srcDiv.innerHTML = html;
  } else {
    srcDiv.innerHTML = "";
  }
}

/* ── Status / Stats ──────────────────────────────────────────────── */

async function checkAPIStatus() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    const el = document.getElementById("apiStatus");
    el.className = "status success";
    el.innerHTML = `&#10003; API is operational (v${data.version || "1.0.0"})`;
  } catch {
    const el = document.getElementById("apiStatus");
    el.className = "status error";
    el.innerHTML = "&#10007; API is unavailable";
  }
}

async function loadStats() {
  try {
    const res = await fetch(`${API}/documents/stats`);
    const data = await res.json();
    const vs = data.vector_store || {};
    document.getElementById("stats").innerHTML = `
      <div style="color:#666;line-height:1.8">
        <p><strong>Total Chunks:</strong> ${vs.total_vectors || 0}</p>
        <p><strong>Embedding Model:</strong> ${data.embedding_model || vs.embedding_model || "-"}</p>
        <p><strong>Vector Dimension:</strong> ${vs.embedding_dimension || "-"}</p>
      </div>`;
  } catch {
    document.getElementById("stats").innerHTML = '<div class="status error">Unable to load statistics</div>';
  }
}

/* ── Util ────────────────────────────────────────────────────────── */

function esc(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

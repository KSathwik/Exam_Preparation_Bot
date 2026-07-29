/* Exam Prep Bot — Frontend logic */

const API = "/api";
const API_KEY_STORAGE_KEY = "examPrepApiKey";
let documentReady = false;

document.addEventListener("DOMContentLoaded", () => {
  setupUpload();
  setupApiKey();
  checkAPIStatus();
  loadStats();
  loadDocumentSummary();
  setupQueryInput();
});

/* ── API Key ─────────────────────────────────────────────────────── */

function getApiKey() {
  // A manually-saved key (this browser) takes precedence over the key the
  // server embedded into the page, so a manual entry can still override it.
  return localStorage.getItem(API_KEY_STORAGE_KEY) || window.__APP_API_KEY__ || "";
}

function authHeaders(extra) {
  const key = getApiKey();
  return key ? { ...extra, "X-API-Key": key } : { ...extra };
}

function setupApiKey() {
  const input = document.getElementById("apiKeyInput");
  const stored = localStorage.getItem(API_KEY_STORAGE_KEY);
  if (stored) {
    input.value = stored;
    showApiKeyStatus(true, false);
  } else if (window.__APP_API_KEY__) {
    input.placeholder = "Using server-configured key — enter one here to override";
    showApiKeyStatus(true, true);
  } else {
    showApiKeyStatus(false, false);
  }
  input.addEventListener("keypress", (e) => {
    if (e.key === "Enter") { e.preventDefault(); saveApiKey(); }
  });
}

function saveApiKey() {
  const input = document.getElementById("apiKeyInput");
  const key = input.value.trim();
  if (!key) {
    localStorage.removeItem(API_KEY_STORAGE_KEY);
    showApiKeyStatus(!!window.__APP_API_KEY__, !!window.__APP_API_KEY__);
    loadStats();
  loadDocumentSummary();
    return;
  }
  localStorage.setItem(API_KEY_STORAGE_KEY, key);
  showApiKeyStatus(true, false);
  loadStats();
  loadDocumentSummary();
}

function showApiKeyStatus(hasKey, isAuto) {
  const el = document.getElementById("apiKeyStatus");
  if (isAuto) {
    el.innerHTML = '<div class="status success">&#10003; Using this server\'s configured key automatically</div>';
  } else if (hasKey) {
    el.innerHTML = '<div class="status success">&#10003; Key saved for this browser</div>';
  } else {
    el.innerHTML = '<div class="status info">No key set — uploads and questions will fail</div>';
  }
}

/* ── Upload ──────────────────────────────────────────────────────── */

function setupUpload() {
  const area = document.getElementById("uploadArea");
  const input = document.getElementById("fileInput");

  area.addEventListener("click", () => input.click());
  area.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
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
    const res = await fetch(`${API}/documents/upload`, { method: "POST", headers: authHeaders(), body: form });
    const data = await res.json();
    if (data.success) {
      s.innerHTML = `<div class="status success">&#10003; ${esc(data.message)}</div>`;
      documentReady = true;
      loadStats();
  loadDocumentSummary();
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

function buildWsUrl() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const key = encodeURIComponent(getApiKey());
  return `${protocol}//${location.host}${API}/ws?api_key=${key}`;
}

function sendQuery() {
  const input = document.getElementById("queryInput");
  const query = input.value.trim();
  if (!query) return;
  if (!documentReady) { alert("Please upload a document first"); return; }

  addMessage(query, "user");
  input.value = "";

  const sendBtn = document.getElementById("sendBtn");
  sendBtn.disabled = true;
  input.disabled = true;

  const msgId = addMessage("", "assistant", { streaming: true });
  let answerText = "";
  let settled = false;

  const finishStreaming = () => {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  };

  let ws;
  try {
    ws = new WebSocket(buildWsUrl());
  } catch (err) {
    setMessageContent(msgId, "Error: " + err.message, false);
    finishStreaming();
    return;
  }

  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({ query }));
  });

  ws.addEventListener("message", (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }

    if (data.type === "intent") {
      setMessageIntent(msgId, data.intent, data.confidence);
    } else if (data.type === "chunk") {
      answerText += data.text;
      setMessageContent(msgId, answerText, true);
    } else if (data.type === "complete") {
      settled = true;
      answerText = data.answer ?? answerText;
      setMessageContent(msgId, answerText, false);
      displayResponse(data);
      if (data.format_type === "out_of_scope") {
        showNoMatchSuggestions(msgId);
      }
      finishStreaming();
      ws.close();
    } else if (data.type === "error") {
      settled = true;
      setMessageContent(msgId, "Error: " + (data.message || "Unknown error"), false);
      finishStreaming();
      ws.close();
    }
  });

  ws.addEventListener("close", () => {
    if (!settled) {
      setMessageContent(msgId, answerText || "Error: connection closed unexpectedly", false);
      finishStreaming();
    }
  });
}

function renderMessageContent(text, streaming) {
  const cursor = streaming ? '<span class="stream-cursor" aria-hidden="true"></span>' : "";
  return esc(text) + cursor;
}

function addMessage(content, role, opts) {
  const options = opts || {};
  const box = document.getElementById("messagesBox");
  const div = document.createElement("div");
  const id = "msg-" + Date.now() + "-" + Math.random().toString(36).slice(2, 7);
  div.id = id;
  div.className = "message " + role;
  div.innerHTML =
    `<div class="message-content">${renderMessageContent(content, !!options.streaming)}</div>` +
    `<div class="message-time">${new Date().toLocaleTimeString()}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return id;
}

function setMessageContent(id, text, streaming) {
  const el = document.getElementById(id);
  if (!el) return;
  const contentEl = el.querySelector(".message-content");
  if (!contentEl) return;
  contentEl.innerHTML = renderMessageContent(text, streaming);
  const box = document.getElementById("messagesBox");
  box.scrollTop = box.scrollHeight;
}

function setMessageIntent(id, intent, confidence) {
  if (!intent) return;
  const el = document.getElementById(id);
  if (!el) return;
  let label = el.querySelector(".message-intent");
  if (!label) {
    label = document.createElement("div");
    label.className = "message-intent";
    el.insertBefore(label, el.firstChild);
  }
  const pct = confidence != null ? ` · ${Math.round(confidence * 100)}%` : "";
  label.textContent = intent + pct;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

/* ── "No match" graceful suggestions ────────────────────────────────
   Uploaded files are saved on disk (and indexed) as "<uuid>_<original
   name>" — strip that prefix so suggestions show the name the user
   actually recognizes. */
function cleanDocName(name) {
  return (name || "").replace(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_/i, "");
}

async function fetchLoadedDocumentNames() {
  try {
    const res = await fetch(`${API}/documents/list`, { headers: authHeaders() });
    if (!res.ok) return [];
    const data = await res.json();
    const names = (data.documents || []).map((d) => cleanDocName(d.file_name));
    return [...new Set(names)];
  } catch {
    return [];
  }
}

async function showNoMatchSuggestions(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add("message-nomatch");

  const box = document.createElement("div");
  box.className = "message-suggestions";
  el.appendChild(box);

  const names = await fetchLoadedDocumentNames();

  if (names.length === 0) {
    box.innerHTML =
      "<p>No documents uploaded yet.</p>" +
      '<button type="button" class="btn btn-primary suggestion-chip" id="gotoUploadBtn">Upload a document to get started</button>';
    document.getElementById("gotoUploadBtn").addEventListener("click", () => {
      const area = document.getElementById("uploadArea");
      area.scrollIntoView({ behavior: "smooth", block: "center" });
      area.classList.add("pulse");
      setTimeout(() => area.classList.remove("pulse"), 1500);
    });
    return;
  }

  box.innerHTML = "<p>Try asking about one of your loaded documents:</p><div class=\"suggestion-chips\"></div>";
  const chipsDiv = box.querySelector(".suggestion-chips");
  names.forEach((name) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion-chip";
    chip.textContent = name;
    chip.title = `Ask about ${name}`;
    chip.addEventListener("click", () => {
      const input = document.getElementById("queryInput");
      input.value = `What does "${name}" cover?`;
      input.focus();
    });
    chipsDiv.appendChild(chip);
  });
}

function displayResponse(data) {
  document.getElementById("responsePanel").style.display = "block";
  document.getElementById("answerText").textContent = data.answer;
  document.getElementById("intentValue").textContent = data.query_intent || data.intent || "-";
  const confidence = data.overall_confidence ?? data.confidence;
  document.getElementById("confidenceValue").textContent =
    confidence != null ? (confidence * 100).toFixed(0) + "%" : "-";
  document.getElementById("riskValue").textContent = data.hallucination_risk || "-";
  const responseTime = data.response_time_seconds ?? data.response_time;
  document.getElementById("timeValue").textContent =
    responseTime != null ? responseTime.toFixed(2) : "-";

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
    const res = await fetch(`${API}/documents/stats`, { headers: authHeaders() });
    if (res.status === 401) {
      document.getElementById("stats").innerHTML = '<div class="status error">Set your API key above to load statistics</div>';
      return;
    }
    const data = await res.json();
    const vs = data.vector_store || {};
    document.getElementById("stats").innerHTML = `
      <div class="about">
        <p><strong>Total Chunks:</strong> ${vs.total_vectors || 0}</p>
        <p><strong>Embedding Model:</strong> ${data.embedding_model || vs.embedding_model || "-"}</p>
        <p><strong>Vector Dimension:</strong> ${vs.embedding_dimension || "-"}</p>
      </div>`;
  } catch {
    document.getElementById("stats").innerHTML = '<div class="status error">Unable to load statistics</div>';
  }
}

async function loadDocumentSummary() {
  const el = document.getElementById("docSummary");
  if (!el) return;
  try {
    const res = await fetch(`${API}/documents/list`, { headers: authHeaders() });
    if (!res.ok) {
      el.textContent = "";
      return;
    }
    const data = await res.json();
    const count = data.total_documents || 0;
    el.textContent = count === 0
      ? "No documents loaded yet — upload one to get started."
      : `${count} document${count === 1 ? "" : "s"} loaded.`;
  } catch {
    el.textContent = "";
  }
}

/* ── Util ────────────────────────────────────────────────────────── */

function esc(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

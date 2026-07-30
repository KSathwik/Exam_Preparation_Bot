/* Message rendering + WebSocket streaming. One request streams at a time
   (the composer disables itself while streaming), so a single closure-level
   "current stream" is enough — no need to key state by message id. */

import { state, setSessionId, newConversationId, setDocumentIds } from "./state.js";
import { buildWsUrl, listDocuments } from "./api.js";
import { renderMarkdown, escapeText, renderMathIn } from "./markdown.js";

let messagesBox;
let jumpBtn;
let pinned = true;
let current = null; // { ws, row, bubbleEl, metaEl, query, draftText, finalText, stage, settled }

export function initChat() {
  messagesBox = document.getElementById("messagesBox");
  jumpBtn = document.getElementById("jumpToLatestBtn");

  messagesBox.addEventListener("scroll", () => {
    const threshold = 80;
    pinned = messagesBox.scrollHeight - messagesBox.scrollTop - messagesBox.clientHeight < threshold;
    jumpBtn.hidden = pinned;
  });

  jumpBtn.addEventListener("click", () => {
    pinned = true;
    scrollToBottom();
    jumpBtn.hidden = true;
  });
}

function scrollToBottom() {
  messagesBox.scrollTop = messagesBox.scrollHeight;
}

function reflectScroll() {
  if (pinned) {
    scrollToBottom();
    jumpBtn.hidden = true;
  } else {
    jumpBtn.hidden = false;
  }
}

export function clearMessages() {
  if (current && current.ws) stopStreaming();
  messagesBox.innerHTML = "";
  pinned = true;
}

/* ── Row builders ──────────────────────────────────────────────────── */

function actionButton(label, title) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "message-action-btn";
  btn.title = title;
  btn.setAttribute("aria-label", title);
  btn.textContent = label;
  return btn;
}

function wireCopyButton(btn, getText) {
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(getText());
      btn.classList.add("copied");
      const original = btn.textContent;
      btn.textContent = "✓";
      setTimeout(() => {
        btn.classList.remove("copied");
        btn.textContent = original;
      }, 1200);
    } catch {
      /* clipboard unavailable — silently ignore */
    }
  });
}

function buildUserRow(text) {
  const row = document.createElement("div");
  row.className = "message-row user";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  row.appendChild(bubble);

  const actions = document.createElement("div");
  actions.className = "message-actions";
  const copyBtn = actionButton("⧉", "Copy");
  wireCopyButton(copyBtn, () => text);
  const editBtn = actionButton("✎", "Edit");
  editBtn.addEventListener("click", () => {
    const input = document.getElementById("queryInput");
    input.value = text;
    input.focus();
    input.dispatchEvent(new Event("input"));
  });
  actions.append(copyBtn, editBtn);
  row.appendChild(actions);

  row.dataset.role = "user";
  row.dataset.text = text;
  return row;
}

function thinkingBubbleHtml(label) {
  return (
    '<span class="thinking-dots"><span></span><span></span><span></span></span>' +
    `<span class="thinking-label">${escapeText(label || "Thinking…")}</span>`
  );
}

function riskBadgeClass(risk) {
  if (risk === "low") return "success";
  if (risk === "high") return "error";
  return "info";
}

function confidenceBadgeClass(confidence) {
  if (confidence == null) return "info";
  if (confidence >= 0.7) return "success";
  if (confidence >= 0.4) return "info";
  return "error";
}

function renderSourcesHtml(sources) {
  if (!sources || !sources.length) return "";
  const items = sources
    .map((s) => {
      const page = s.page ?? s.page_number ?? "?";
      const quote = (s.quote ?? s.quoted_text ?? "").slice(0, 150);
      return `<div class="source-item"><div class="source-page">Page ${escapeText(String(page))}</div><div class="source-quote">"${escapeText(quote)}…"</div></div>`;
    })
    .join("");
  return `<details class="sources-disclosure"><summary>Sources (${sources.length})</summary>${items}</details>`;
}

function buildAssistantRow(query) {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.dataset.role = "assistant";
  row.dataset.query = query;

  const intentEl = document.createElement("div");
  intentEl.className = "message-intent";
  intentEl.hidden = true;
  row.appendChild(intentEl);

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerHTML = thinkingBubbleHtml();
  row.appendChild(bubble);

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.hidden = true;
  row.appendChild(meta);

  const suggestions = document.createElement("div");
  suggestions.className = "message-suggestions";
  row.appendChild(suggestions);

  const studyChips = document.createElement("div");
  studyChips.className = "study-chips";
  row.appendChild(studyChips);

  const actions = document.createElement("div");
  actions.className = "message-actions";
  const copyBtn = actionButton("⧉", "Copy");
  wireCopyButton(copyBtn, () => row.dataset.finalText || bubble.textContent || "");
  const regenBtn = actionButton("↻", "Regenerate");
  regenBtn.addEventListener("click", () => regenerate(row));
  actions.append(copyBtn, regenBtn);
  row.appendChild(actions);

  return { row, intentEl, bubble, meta, suggestions, studyChips };
}

const STUDY_TOOLS = [
  { label: "Summarize", build: (q) => `Summarize the answer to: "${q}"` },
  { label: "Explain simpler", build: (q) => `Explain this more simply: "${q}"` },
  { label: "Explain in detail", build: (q) => `Explain this in more detail: "${q}"` },
];

function populateStudyChips(container, query) {
  if (!container) return;
  if (!query) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = "";
  for (const tool of STUDY_TOOLS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion-chip";
    chip.textContent = tool.label;
    chip.addEventListener("click", () => {
      if (state.streaming) return;
      sendQuery(tool.build(query));
    });
    container.appendChild(chip);
  }
}

/* ── Static (non-streaming) rendering — used for conversation history ── */

export function renderHistory(messages) {
  clearMessages();
  let lastUserQuery = "";
  for (const m of messages) {
    if (m.role === "user") {
      messagesBox.appendChild(buildUserRow(m.content));
      lastUserQuery = m.content;
    } else {
      const { row, intentEl, bubble, studyChips } = buildAssistantRow(lastUserQuery);
      if (m.intent) {
        intentEl.hidden = false;
        intentEl.textContent = m.intent;
      }
      bubble.innerHTML = renderMarkdown(m.content);
      renderMathIn(bubble);
      row.dataset.finalText = m.content;
      populateStudyChips(studyChips, lastUserQuery);
      messagesBox.appendChild(row);
    }
  }
  reflectScroll();
}

export function addSystemMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row system";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  messagesBox.appendChild(row);
  reflectScroll();
}

/* ── "No match" graceful suggestions ─────────────────────────────────
   Uploaded files are saved on disk (and indexed) as "<uuid>_<original
   name>" — strip that prefix so suggestions show the name the user
   actually recognizes. */
function cleanDocName(name) {
  return (name || "").replace(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_/i, "");
}

async function showNoMatchSuggestions(suggestionsEl) {
  // Group by display name -> every document_id sharing that name (the same
  // file can legitimately be uploaded more than once, producing separate
  // ids) so a chip click can scope retrieval to exactly the document it
  // names, not just mention the filename in freeform query text — a plain
  // text mention doesn't reliably out-rank other documents in a global
  // search (see the "confusing between uploads" fix).
  const byName = new Map();
  try {
    const data = await listDocuments();
    for (const d of data.documents || []) {
      const name = cleanDocName(d.file_name);
      if (!byName.has(name)) byName.set(name, []);
      byName.get(name).push(d.document_id);
    }
  } catch {
    /* byName stays empty */
  }

  if (byName.size === 0) {
    suggestionsEl.innerHTML =
      "<p>No documents uploaded yet.</p>" +
      '<button type="button" class="suggestion-chip" id="gotoUploadBtn">Upload a document to get started</button>';
    document.getElementById("gotoUploadBtn").addEventListener("click", () => {
      document.getElementById("fileInput").click();
    });
    return;
  }

  const chips = [...byName.keys()]
    .map((name) => `<button type="button" class="suggestion-chip" data-name="${escapeText(name)}">${escapeText(name)}</button>`)
    .join("");
  suggestionsEl.innerHTML = `<p>Try asking about one of your loaded documents:</p><div class="suggestion-chips">${chips}</div>`;
  suggestionsEl.querySelectorAll(".suggestion-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      setDocumentIds(byName.get(chip.dataset.name) || []);
      const input = document.getElementById("queryInput");
      input.value = `What does "${chip.dataset.name}" cover?`;
      input.focus();
      input.dispatchEvent(new Event("input"));
    });
  });
}

/* ── Streaming ─────────────────────────────────────────────────────── */

// Draft + always-on reflection means two full LLM round-trips per turn — a
// generous ceiling so this only fires for a genuinely stalled/dropped
// connection, never a slow-but-alive one. Whatever the exact cause (network
// hiccup, a proxy killing an idle connection, a hung provider call), the
// composer must never be left permanently disabled with no way to recover.
const STREAM_TIMEOUT_MS = 120_000;

function clearStreamTimeout(c) {
  if (c.timeoutId) {
    clearTimeout(c.timeoutId);
    c.timeoutId = null;
  }
}

function setComposerStreaming(streaming) {
  state.streaming = streaming;
  const sendBtn = document.getElementById("sendBtn");
  const input = document.getElementById("queryInput");
  sendBtn.classList.toggle("stopping", streaming);
  sendBtn.textContent = streaming ? "■" : "➤";
  sendBtn.title = streaming ? "Stop generating" : "Send";
  input.disabled = streaming;
}

export function stopStreaming() {
  if (!current) return;
  const c = current;
  c.settled = true;
  clearStreamTimeout(c);
  try {
    c.ws.close();
  } catch {
    /* ignore */
  }
  if (!c.finalText && !c.draftText) {
    c.bubble.textContent = "Stopped.";
  } else {
    c.bubble.innerHTML = renderMarkdown(c.draftText || c.finalText) + '<div class="response-time">Stopped</div>';
  }
  setComposerStreaming(false);
  current = null;
}

function regenerate(row) {
  const query = row.dataset.query;
  if (!query || state.streaming) return;
  sendQuery(query, { reuseRow: row });
}

export function sendQuery(text, opts) {
  const options = opts || {};
  const query = text.trim();
  if (!query || state.streaming) return;

  if (!state.sessionId) {
    setSessionId(newConversationId());
  }
  const isFirstMessage = messagesBox.children.length === 0;

  let built;
  if (options.reuseRow) {
    const bubble = options.reuseRow.querySelector(".message-bubble");
    bubble.innerHTML = thinkingBubbleHtml();
    const meta = options.reuseRow.querySelector(".message-meta");
    meta.hidden = true;
    meta.innerHTML = "";
    const suggestions = options.reuseRow.querySelector(".message-suggestions");
    suggestions.innerHTML = "";
    const studyChips = options.reuseRow.querySelector(".study-chips");
    studyChips.innerHTML = "";
    const intentEl = options.reuseRow.querySelector(".message-intent");
    intentEl.hidden = true;
    built = { row: options.reuseRow, intentEl, bubble, meta, suggestions, studyChips };
  } else {
    messagesBox.appendChild(buildUserRow(query));
    built = buildAssistantRow(query);
    messagesBox.appendChild(built.row);
  }
  reflectScroll();

  setComposerStreaming(true);

  const c = {
    row: built.row,
    bubble: built.bubble,
    intentEl: built.intentEl,
    meta: built.meta,
    suggestions: built.suggestions,
    studyChips: built.studyChips,
    query,
    draftText: "",
    finalText: "",
    stage: null,
    settled: false,
    ws: null,
    timeoutId: null,
  };
  current = c;

  let ws;
  try {
    ws = new WebSocket(buildWsUrl());
  } catch (err) {
    c.bubble.textContent = "Error: " + err.message;
    setComposerStreaming(false);
    current = null;
    return;
  }
  c.ws = ws;
  c.timeoutId = setTimeout(() => {
    if (c.settled || current !== c) return;
    c.settled = true;
    try {
      ws.close();
    } catch {
      /* ignore */
    }
    const shown = c.finalText || c.draftText;
    c.bubble.innerHTML =
      (shown ? renderMarkdown(shown) : "") +
      '<div class="response-time">Timed out — please try again.</div>';
    setComposerStreaming(false);
    current = null;
  }, STREAM_TIMEOUT_MS);

  if (isFirstMessage && !options.reuseRow) {
    window.dispatchEvent(new CustomEvent("chat:turn-started", { detail: { sessionId: state.sessionId, query } }));
  }

  ws.addEventListener("open", () => {
    ws.send(
      JSON.stringify({
        query,
        session_id: state.sessionId,
        device_id: state.deviceId,
        document_ids: state.documentIds,
      })
    );
  });

  ws.addEventListener("message", (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }

    if (data.type === "intent") {
      c.intentEl.hidden = false;
      const pct = data.confidence != null ? ` · ${Math.round(data.confidence * 100)}%` : "";
      c.intentEl.textContent = data.intent + pct;
    } else if (data.type === "status" && (data.stage === "retrieving" || data.stage === "drafting")) {
      // No chunks have started yet — update the thinking indicator's label
      // in place instead of leaving the user staring at unlabeled dots
      // during what can be a real 10-30s LLM call.
      if (!c.stage) {
        const label = c.bubble.querySelector(".thinking-label");
        if (label) label.textContent = data.message;
      }
    } else if (data.type === "status" && data.stage === "reflecting") {
      c.bubble.classList.add("answer-refining");
      if (!c.row.querySelector(".reflecting-pill")) {
        const pill = document.createElement("div");
        pill.className = "reflecting-pill";
        pill.textContent = data.message || "Reviewing answer…";
        c.bubble.after(pill);
      }
    } else if (data.type === "chunk") {
      const pill = c.row.querySelector(".reflecting-pill");
      if (data.stage === "final" && c.stage !== "final") {
        c.stage = "final";
        c.finalText = "";
        c.bubble.classList.remove("answer-refining");
        if (pill) pill.remove();
      } else if (!c.stage) {
        c.stage = "draft";
      }
      if (c.stage === "final") c.finalText += data.text;
      else c.draftText += data.text;
      const shown = c.stage === "final" ? c.finalText : c.draftText;
      c.bubble.innerHTML = renderMarkdown(shown) + '<span class="stream-cursor" aria-hidden="true"></span>';
      reflectScroll();
    } else if (data.type === "complete") {
      c.settled = true;
      clearStreamTimeout(c);
      const finalAnswer = data.answer ?? c.finalText ?? c.draftText;
      c.row.dataset.finalText = finalAnswer;
      c.bubble.classList.remove("answer-refining");
      const pill = c.row.querySelector(".reflecting-pill");
      if (pill) pill.remove();
      c.bubble.innerHTML = renderMarkdown(finalAnswer);
      renderMathIn(c.bubble);

      if (data.format_type === "greeting") {
        // A canned small-talk reply — the intent label (shown before we knew
        // this was small talk) plus confidence/risk/citation badges and
        // study-tool chips are all meaningless noise on "Hello!".
        c.intentEl.hidden = true;
        c.meta.hidden = true;
        c.meta.innerHTML = "";
      } else {
        c.meta.hidden = false;
        const confidenceBadge = `<span class="badge ${confidenceBadgeClass(data.confidence)}">${data.confidence != null ? Math.round(data.confidence * 100) + "% confidence" : "confidence n/a"}</span>`;
        const riskBadge = `<span class="badge ${riskBadgeClass(data.hallucination_risk)}">${data.hallucination_risk || "unknown"} risk</span>`;
        const timeLabel = data.response_time != null ? `<span class="response-time">${data.response_time.toFixed(2)}s</span>` : "";
        c.meta.innerHTML = confidenceBadge + riskBadge + timeLabel + renderSourcesHtml(data.sources);
      }

      if (data.format_type === "out_of_scope") {
        c.row.classList.add("message-nomatch");
        showNoMatchSuggestions(c.suggestions);
      } else if (data.format_type !== "greeting") {
        populateStudyChips(c.studyChips, c.query);
      }

      setComposerStreaming(false);
      reflectScroll();
      ws.close();
      current = null;

      if (isFirstMessage && !options.reuseRow) {
        window.dispatchEvent(new CustomEvent("chat:turn-completed", { detail: { sessionId: state.sessionId } }));
      }
    } else if (data.type === "error") {
      c.settled = true;
      clearStreamTimeout(c);
      c.bubble.textContent = "Error: " + (data.message || "Unknown error");
      setComposerStreaming(false);
      ws.close();
      current = null;
    }
  });

  ws.addEventListener("close", () => {
    if (!c.settled) {
      c.settled = true;
      clearStreamTimeout(c);
      c.bubble.textContent = c.draftText || c.finalText || "Error: connection closed unexpectedly";
      setComposerStreaming(false);
      current = null;
    }
  });
}

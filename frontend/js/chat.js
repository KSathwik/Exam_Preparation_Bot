/* Message rendering + WebSocket streaming. One request streams at a time
   (the composer disables itself while streaming), so a single closure-level
   "current stream" is enough — no need to key state by message id. */

import { state, setSessionId, newConversationId, setDocumentIds, setDomainPreset } from "./state.js";
import { buildWsUrl, listDocuments } from "./api.js";
import { renderMarkdown, escapeText, renderMathIn } from "./markdown.js";
import { buildResearchControl } from "./continueResearch.js";

let messagesBox;
let jumpBtn;
let pinned = true;
let current = null; // { ws, row, bubbleEl, metaEl, query, finalText, settled }

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

  messagesBox.addEventListener("click", (e) => {
    const chip = e.target.closest(".goto-upload-chip");
    if (chip) {
      const fileInput = document.getElementById("fileInput");
      if (fileInput) fileInput.click();
    }
  });

  initPresetSelector();
}

function initPresetSelector() {
  const bar = document.getElementById("presetSelectorBar");
  if (!bar) return;
  const pills = bar.querySelectorAll(".preset-pill");
  pills.forEach((pill) => {
    const isSelected = pill.dataset.preset === (state.domainPreset || "general");
    pill.classList.toggle("active", isSelected);
    pill.addEventListener("click", () => {
      pills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      setDomainPreset(pill.dataset.preset);
    });
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

// Used by the sidebar AI Hub (see aiHub.js), which isn't tied to any one
// message row — it needs "whatever the student was just asking about" in
// the active conversation, so the row DOM itself (dataset.text, set in
// buildUserRow) is the simplest source of truth, no extra state to thread.
export function getLastUserQuery() {
  const rows = messagesBox.querySelectorAll(".message-row.user");
  const last = rows[rows.length - 1];
  return last ? last.dataset.text || "" : "";
}

export function clearMessages() {
  if (current && current.ws) stopStreaming();
  messagesBox.innerHTML = "";
  pinned = true;
}

/* ── Empty state ───────────────────────────────────────────────────── */

const EXAMPLE_PROMPTS = [
  "Summarize key concepts & definitions",
  "Extract main insights & technical details",
  "Walk me through this process step by step",
  "Quiz me or generate practice questions",
];

export function showEmptyStateIfNeeded() {
  if (messagesBox.children.length > 0) return;
  const wrap = document.createElement("div");
  wrap.className = "empty-state";
  wrap.innerHTML =
    '<div class="empty-state-icon">✦</div>' +
    "<h2>Ready when you are</h2>" +
    "<p>Upload your knowledge base or documents, then ask anything about them.</p>" +
    '<div class="empty-state-chips">' +
    EXAMPLE_PROMPTS.map((p) => `<button type="button" class="suggestion-chip">${escapeText(p)}</button>`).join("") +
    "</div>";
  wrap.querySelectorAll(".suggestion-chip").forEach((chip) => {
    chip.addEventListener("click", () => sendQuery(chip.textContent));
  });
  messagesBox.appendChild(wrap);
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
    if (state.streaming) return;
    state.editingUserRow = row;
    const input = document.getElementById("queryInput");
    input.value = text;
    input.focus();
    input.select();
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

// Study chips must track the actual most-recent assistant reply, not just
// "the last .message-row in the DOM" — a pure CSS :last-child selector
// silently breaks the moment anything else (e.g. an upload's system
// message) gets appended after the answer, which is a completely normal
// mid-conversation flow. Tracked explicitly via a class toggle instead.
function markLatestAssistantRow(row) {
  const previous = messagesBox.querySelector(".message-row.assistant.is-latest-assistant");
  if (previous && previous !== row) previous.classList.remove("is-latest-assistant");
  row.classList.add("is-latest-assistant");
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
  // Only ever revealed once a response has actually finished (see the
  // "complete" WS handler and renderHistory below) — never while streaming.
  const researchControl = buildResearchControl(() => row.dataset.query);
  actions.append(copyBtn, regenBtn, researchControl);
  row.appendChild(actions);

  return { row, intentEl, bubble, meta, suggestions, studyChips, researchControl };
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
      const { row, intentEl, bubble, studyChips, researchControl } = buildAssistantRow(lastUserQuery);
      const isGreeting = m.format_type === "greeting";
      if (m.intent && !isGreeting) {
        intentEl.hidden = false;
        intentEl.textContent = m.intent;
      }
      bubble.innerHTML = renderMarkdown(m.content);
      renderMathIn(bubble);
      row.dataset.finalText = m.content;
      // A replayed greeting gets the same treatment as a live one (see the
      // "complete" WS handler) — intent label, study chips, and "Open in"
      // are meaningless noise on "Hello!".
      if (!isGreeting) {
        populateStudyChips(studyChips, lastUserQuery);
        researchControl.hidden = false; // history entries are already-complete responses
      }
      messagesBox.appendChild(row);
      markLatestAssistantRow(row);
    }
  }
  reflectScroll();
}

export function showHistorySkeleton() {
  clearMessages();
  messagesBox.innerHTML =
    '<div class="history-skeleton" aria-hidden="true">' +
    '<div class="skeleton-row user"></div><div class="skeleton-row assistant"></div>' +
    '<div class="skeleton-row user"></div><div class="skeleton-row assistant"></div>' +
    "</div>";
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
  // text mention doesn't reliably out-rank other documents in this
  // conversation. Scoped to this conversation's own documents (server-side
  // authoritative scope — see resolve_document_scope), never every document
  // ever uploaded across every conversation.
  const byName = new Map();
  try {
    const data = await listDocuments(state.sessionId);
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
      '<button type="button" class="suggestion-chip goto-upload-chip">Upload a document to get started</button>';
    return;
  }

  suggestionsEl.innerHTML = '<p>Try asking about one of your loaded documents:</p><div class="suggestion-chips"></div>';
  const chipsEl = suggestionsEl.querySelector(".suggestion-chips");
  // Built via createElement + dataset/textContent (real DOM property
  // assignment) rather than string-interpolated into innerHTML — a filename
  // is fully attacker-controlled at upload time, and escaping it into an
  // HTML attribute string is one missed character away from an XSS
  // breakout. This is immune to that class of bug structurally.
  for (const name of byName.keys()) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion-chip";
    chip.dataset.name = name;
    chip.textContent = name;
    chip.addEventListener("click", () => {
      setDocumentIds(byName.get(name) || []);
      const input = document.getElementById("queryInput");
      input.value = `What does "${name}" cover?`;
      input.focus();
      input.dispatchEvent(new Event("input"));
    });
    chipsEl.appendChild(chip);
  }
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
  // A live region re-announcing every ~40-character chunk floods a screen
  // reader with partial-word fragments. Suppress announcements while
  // streaming; restoring "polite" right as the turn settles means the
  // final, complete answer still gets announced once, cleanly.
  messagesBox.setAttribute("aria-live", streaming ? "off" : "polite");
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
  if (!c.finalText) {
    c.bubble.textContent = "Stopped.";
  } else {
    c.bubble.innerHTML = renderMarkdown(c.finalText) + '<div class="response-time">Stopped</div>';
  }
  setComposerStreaming(false);
  current = null;
}

function regenerate(row) {
  const query = row.dataset.query;
  if (!query || state.streaming) return;
  sendQuery(query, { reuseRow: row });
}

/* Resets an existing assistant row back to a fresh "thinking" state so it
   can be streamed into again in place — shared by Regenerate and
   edit-and-resend, which both replace a row's content rather than
   appending a new one. */
function resetAssistantRowForStreaming(row, query) {
  const bubble = row.querySelector(".message-bubble");
  bubble.innerHTML = thinkingBubbleHtml();
  const meta = row.querySelector(".message-meta");
  meta.hidden = true;
  meta.innerHTML = "";
  const suggestions = row.querySelector(".message-suggestions");
  suggestions.innerHTML = "";
  const studyChips = row.querySelector(".study-chips");
  studyChips.innerHTML = "";
  const intentEl = row.querySelector(".message-intent");
  intentEl.hidden = true;
  const researchControl = row.querySelector(".research-control");
  researchControl.hidden = true; // hide again until the re-run response completes
  row.dataset.query = query;
  return { row, intentEl, bubble, meta, suggestions, studyChips, researchControl };
}

export function sendQuery(text, opts) {
  const options = opts || {};
  const query = text.trim();
  if (!query || state.streaming) return;

  const editRow = state.editingUserRow;
  state.editingUserRow = null;

  if (!state.sessionId) {
    setSessionId(newConversationId());
  }

  messagesBox.querySelector(".empty-state")?.remove();
  const isFirstMessage = messagesBox.children.length === 0;

  let built;
  if (options.reuseRow) {
    built = resetAssistantRowForStreaming(options.reuseRow, query);
  } else if (editRow && editRow.isConnected) {
    // Edit-and-resend: update the edited message in place, matching
    // ChatGPT/Claude's edit behavior — never append a duplicate of the old
    // exchange below it. Normally reuses the paired assistant reply right
    // after it, but that pairing isn't guaranteed (a dropped connection, a
    // reload mid-turn, or editing a message that never got a reply yet can
    // all leave a user message with no assistant sibling) — insert a fresh
    // reply row directly after the edited message in that case, rather than
    // falling through to the generic append path below, which would leave
    // the stale original text visible above a brand-new duplicate exchange.
    editRow.dataset.text = query;
    editRow.querySelector(".message-bubble").textContent = query;
    if (editRow.nextElementSibling?.dataset.role === "assistant") {
      built = resetAssistantRowForStreaming(editRow.nextElementSibling, query);
    } else {
      built = buildAssistantRow(query);
      editRow.after(built.row);
      markLatestAssistantRow(built.row);
    }
  } else {
    messagesBox.appendChild(buildUserRow(query));
    built = buildAssistantRow(query);
    messagesBox.appendChild(built.row);
    markLatestAssistantRow(built.row);
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
    researchControl: built.researchControl,
    query,
    finalText: "",
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
    c.bubble.innerHTML =
      (c.finalText ? renderMarkdown(c.finalText) : "") +
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
        domain_preset: state.domainPreset,
        top_k: state.topK,
        temperature: state.temperature,
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
    } else if (data.type === "chunk") {
      // The only stream that ever arrives is the final, already-reflected
      // answer — drafting and reflection both happen silently server-side
      // first (see OrchestratorAgent.run), so there's nothing to distinguish
      // here and no risk of the shown text ever getting replaced mid-stream.
      //
      // Plain escaped text during the stream, not a full markdown/DOMPurify
      // pass — re-parsing the ENTIRE accumulated answer on every ~40-char
      // chunk (dozens of times for a long answer) is wasted O(n²) work for
      // an intermediate view that gets fully replaced on "complete" anyway
      // (below). Markdown/code/math rendering only has to happen once, on
      // the final text.
      c.finalText += data.text;
      c.bubble.innerHTML =
        escapeText(c.finalText).replace(/\n/g, "<br>") + '<span class="stream-cursor" aria-hidden="true"></span>';
      reflectScroll();
    } else if (data.type === "complete") {
      c.settled = true;
      clearStreamTimeout(c);
      const finalAnswer = data.answer ?? c.finalText;
      c.row.dataset.finalText = finalAnswer;
      c.bubble.innerHTML = renderMarkdown(finalAnswer);
      renderMathIn(c.bubble);

      if (data.format_type === "greeting") {
        // A canned small-talk reply — the intent label (shown before we knew
        // this was small talk) plus confidence/risk/citation badges, study-
        // tool chips, and "Continue Research" are all meaningless noise on
        // "Hello!".
        c.intentEl.hidden = true;
        c.meta.hidden = true;
        c.meta.innerHTML = "";
        c.researchControl.hidden = true;
      } else {
        // Smart Visibility: only ever reveal "Continue Research" once a
        // response has actually finished — never while streaming/thinking.
        c.researchControl.hidden = false;
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
      c.bubble.textContent = c.finalText || "Error: connection closed unexpectedly";
      setComposerStreaming(false);
      current = null;
    }
  });
}

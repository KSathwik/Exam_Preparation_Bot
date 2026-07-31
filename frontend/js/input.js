/* Composer: auto-growing textarea, keyboard shortcuts, char count, attach/
   drag-drop upload, and the send/stop toggle button. */

import { state, addDocumentId } from "./state.js";
import { uploadDocument } from "./api.js";
import { sendQuery, stopStreaming, addSystemMessage } from "./chat.js";

const MAX_CHARS = 1000;
const WARN_AT = 800;

export function initInput() {
  const form = document.getElementById("composerForm");
  const input = document.getElementById("queryInput");
  const charCount = document.getElementById("charCount");
  const sendBtn = document.getElementById("sendBtn");
  const attachBtn = document.getElementById("attachBtn");
  const fileInput = document.getElementById("fileInput");
  const chatColumn = document.getElementById("chatColumn");

  const autoGrow = () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  };

  const updateCharCount = () => {
    const len = input.value.length;
    if (len > WARN_AT) {
      charCount.hidden = false;
      charCount.textContent = `${len}/${MAX_CHARS}`;
      charCount.classList.toggle("warn", len > MAX_CHARS);
    } else {
      charCount.hidden = true;
    }
    // Sending past the limit used to just silently no-op inside the submit
    // handler below — no shake, no error, nothing. While streaming, sendBtn
    // doubles as the Stop button and must stay enabled regardless of
    // leftover text length (see setComposerStreaming in chat.js).
    if (!state.streaming) {
      sendBtn.disabled = len === 0 || len > MAX_CHARS;
    }
  };

  input.addEventListener("input", () => {
    autoGrow();
    updateCharCount();
    // Cleared without sending — no longer editing that message (see chat.js
    // sendQuery), so a fresh unrelated question doesn't get misapplied to it.
    if (!input.value) state.editingUserRow = null;
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.streaming) {
      stopStreaming();
      return;
    }
    const text = input.value.trim();
    if (!text || text.length > MAX_CHARS) {
      // Enter key still submits the form even with sendBtn disabled
      // (requestSubmit() doesn't check a specific button's state) — this
      // used to silently no-op here with zero feedback. A brief shake on
      // the composer plus the already-visible over-limit count is enough
      // signal without a full toast system.
      if (text.length > MAX_CHARS) {
        const composerInner = input.closest(".composer-inner");
        composerInner.classList.remove("shake");
        // Force reflow so re-adding the class restarts the animation even
        // if the user hits Enter again before the previous shake finished.
        void composerInner.offsetWidth;
        composerInner.classList.add("shake");
      }
      return;
    }
    sendQuery(text);
    input.value = "";
    autoGrow();
    updateCharCount();
  });

  attachBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) handleUpload(file);
    fileInput.value = "";
  });

  chatColumn.addEventListener("dragover", (e) => {
    e.preventDefault();
    chatColumn.classList.add("dragover");
  });
  chatColumn.addEventListener("dragleave", (e) => {
    if (e.target === chatColumn) chatColumn.classList.remove("dragover");
  });
  chatColumn.addEventListener("drop", (e) => {
    e.preventDefault();
    chatColumn.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (!file) return;
    // accept=".pdf,.docx" on the file picker only constrains that dialog —
    // a drop bypasses it entirely, so a wrong file type used to only get
    // caught after a full upload round-trip to the server.
    if (!/\.(pdf|docx)$/i.test(file.name)) {
      addSystemMessage(`✗ Unsupported file type — only PDF or DOCX is accepted.`);
      return;
    }
    handleUpload(file);
  });

  updateCharCount(); // starts disabled — the composer opens with empty input
}

async function handleUpload(file) {
  addSystemMessage(`Uploading ${file.name}…`);
  try {
    const data = await uploadDocument(file, state.sessionId, state.deviceId);
    if (data.success) {
      addDocumentId(data.document_id);
      addSystemMessage(`✓ ${data.message}`);
    } else {
      addSystemMessage(`✗ ${data.detail || data.error || "Upload failed"}`);
    }
  } catch (err) {
    addSystemMessage(`✗ ${err.message}`);
  }
}

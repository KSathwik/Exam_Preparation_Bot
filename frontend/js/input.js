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
  };

  input.addEventListener("input", () => {
    autoGrow();
    updateCharCount();
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
    if (!text || text.length > MAX_CHARS) return;
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
    if (file) handleUpload(file);
  });
}

async function handleUpload(file) {
  addSystemMessage(`Uploading ${file.name}…`);
  try {
    const data = await uploadDocument(file);
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

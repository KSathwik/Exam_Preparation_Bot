/* Chat history sidebar: list/switch/rename/delete conversations, New Chat.
   Decoupled from chat.js via window CustomEvents (chat:turn-started /
   chat:turn-completed) rather than a direct import, so there's no import
   cycle between the two modules. */

import { state, setSessionId, newConversationId, loadPersistedSessionId, clearDocumentIds } from "./state.js";
import { listConversations, getConversation, renameConversation, deleteConversation } from "./api.js";
import { renderHistory, clearMessages, stopStreaming, showHistorySkeleton, showEmptyStateIfNeeded } from "./chat.js";

let listEl;
let searchInput;

function deriveTitle(text, limit = 50) {
  const trimmed = (text || "").trim();
  return trimmed.length <= limit ? trimmed : trimmed.slice(0, limit).trimEnd() + "…";
}

function closeAllMenus() {
  listEl.querySelectorAll(".conversation-menu").forEach((m) => m.remove());
}

function buildItem(convo) {
  const li = document.createElement("li");
  li.className = "conversation-item";
  li.dataset.sessionId = convo.session_id;
  li.tabIndex = 0;
  li.setAttribute("role", "button");
  li.setAttribute("aria-label", `Open conversation: ${convo.title || "New conversation"}`);
  if (convo.session_id === state.sessionId) li.classList.add("active");
  if (convo.pending) li.dataset.pending = "true";

  const title = document.createElement("span");
  title.className = "conversation-title";
  title.textContent = convo.title || "New conversation";
  li.appendChild(title);

  const menuBtn = document.createElement("button");
  menuBtn.type = "button";
  menuBtn.className = "conversation-menu-btn";
  menuBtn.setAttribute("aria-label", "Conversation options");
  menuBtn.textContent = "⋮";
  li.appendChild(menuBtn);

  li.addEventListener("click", (e) => {
    if (e.target === menuBtn) return;
    switchConversation(convo.session_id);
  });
  li.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target === li) {
      e.preventDefault();
      switchConversation(convo.session_id);
    }
  });

  menuBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const existing = li.querySelector(".conversation-menu");
    closeAllMenus();
    if (existing) return; // toggle off

    const menu = document.createElement("div");
    menu.className = "conversation-menu";

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.textContent = "Rename";
    renameBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      closeAllMenus();
      startRename(li, convo);
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "danger";
    deleteBtn.textContent = "Delete";
    deleteBtn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      closeAllMenus();
      if (!confirm("Delete this conversation? This cannot be undone.")) return;

      // Optimistic: reflect the delete immediately rather than waiting on
      // the network round-trip before the UI updates.
      const wasActive = state.sessionId === convo.session_id;
      state.conversations = state.conversations.filter((c) => c.session_id !== convo.session_id);
      applySearchFilter();
      if (wasActive) startNewChat();

      try {
        await deleteConversation(convo.session_id);
      } catch {
        // Roll back — the delete didn't actually happen server-side.
        await refreshConversationList();
      }
    });

    menu.append(renameBtn, deleteBtn);
    li.appendChild(menu);
  });

  return li;
}

function startRename(li, convo) {
  const titleEl = li.querySelector(".conversation-title");
  const input = document.createElement("input");
  input.type = "text";
  input.className = "conversation-title-input";
  input.value = convo.title || "";
  titleEl.replaceWith(input);
  input.focus();
  input.select();

  // Escape doesn't blur the input, so it stays focused/in the DOM while
  // refreshConversationList()'s network round-trip is in flight — if the
  // user then clicks elsewhere, the still-attached blur listener would fire
  // commit() with the very text Escape was meant to discard. `settled`
  // makes cancel-then-commit and commit-then-cancel both no-ops the second
  // time, regardless of which fires first.
  let settled = false;

  const commit = async () => {
    if (settled) return;
    settled = true;
    const newTitle = input.value.trim();
    if (newTitle && newTitle !== convo.title) {
      // Optimistic: reflect the new title immediately, don't wait on the
      // network round-trip first.
      convo.title = newTitle;
      applySearchFilter();
      try {
        await renameConversation(convo.session_id, newTitle);
      } catch {
        await refreshConversationList(); // roll back — the rename didn't stick server-side
        return;
      }
    }
    refreshConversationList();
  };

  const cancel = () => {
    if (settled) return;
    settled = true;
    refreshConversationList();
  };

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); input.blur(); }
    if (e.key === "Escape") { e.preventDefault(); cancel(); }
  });
  input.addEventListener("blur", commit, { once: true });
}

function renderList(conversations) {
  listEl.innerHTML = "";
  closeAllMenus();
  if (conversations.length === 0 && searchInput && searchInput.value.trim()) {
    const empty = document.createElement("li");
    empty.className = "conversation-empty-msg";
    empty.textContent = "No matching conversations";
    listEl.appendChild(empty);
    return;
  }
  for (const convo of conversations) {
    listEl.appendChild(buildItem(convo));
  }
}

function applySearchFilter() {
  const term = (searchInput?.value || "").trim().toLowerCase();
  const filtered = term
    ? state.conversations.filter((c) => (c.title || "").toLowerCase().includes(term))
    : state.conversations;
  renderList(filtered);
}

export async function refreshConversationList() {
  try {
    const data = await listConversations(state.deviceId);
    state.conversations = data.conversations || [];
  } catch {
    return;
  }
  applySearchFilter();
}

export function highlightActive() {
  listEl.querySelectorAll(".conversation-item").forEach((li) => {
    li.classList.toggle("active", li.dataset.sessionId === state.sessionId);
  });
}

async function switchConversation(sessionId) {
  if (sessionId === state.sessionId) return;
  stopStreaming();
  setSessionId(sessionId);
  // No document-to-conversation link is persisted yet, so there's no way to
  // recover which document(s) belonged to this (possibly older) conversation
  // — reset rather than carry over whatever was scoped in the previous one.
  clearDocumentIds();
  highlightActive();
  closeSidebarOnMobile();
  showHistorySkeleton();
  try {
    const data = await getConversation(sessionId);
    renderHistory(data.messages || []);
    if (!data.messages || data.messages.length === 0) showEmptyStateIfNeeded();
  } catch {
    clearMessages();
    showEmptyStateIfNeeded();
  }
}

export function startNewChat() {
  stopStreaming();
  clearMessages();
  setSessionId(newConversationId());
  clearDocumentIds();
  highlightActive();
  closeSidebarOnMobile();
  showEmptyStateIfNeeded();
}

function closeSidebarOnMobile() {
  const sidebar = document.getElementById("sidebar");
  if (!sidebar.classList.contains("sidebar-open")) return;
  sidebar.classList.remove("sidebar-open");
  document.getElementById("sidebarBackdrop").classList.remove("visible");
  document.getElementById("chatColumn").removeAttribute("inert");
  document.getElementById("sidebarToggleBtn").focus();
}

export async function initSidebar() {
  listEl = document.getElementById("conversationList");
  searchInput = document.getElementById("conversationSearch");
  document.getElementById("newChatBtn").addEventListener("click", startNewChat);
  searchInput.addEventListener("input", applySearchFilter);

  document.getElementById("sidebarToggleBtn").addEventListener("click", () => {
    const sidebar = document.getElementById("sidebar");
    sidebar.classList.add("sidebar-open");
    document.getElementById("sidebarBackdrop").classList.add("visible");
    // Keyboard/screen-reader users could otherwise Tab straight through the
    // open drawer into the (visually hidden-behind-it) chat column — trap
    // focus in the drawer while it's open and move focus into it.
    document.getElementById("chatColumn").setAttribute("inert", "");
    sidebar.querySelector("button, [href], input, [tabindex]")?.focus();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && document.getElementById("sidebar").classList.contains("sidebar-open")) {
      closeSidebarOnMobile();
    }
  });
  document.getElementById("sidebarBackdrop").addEventListener("click", closeSidebarOnMobile);
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".conversation-item")) closeAllMenus();
  });

  window.addEventListener("chat:turn-started", (e) => {
    const { sessionId, query } = e.detail;
    if (state.conversations.some((c) => c.session_id === sessionId)) return;
    const placeholder = document.createElement("li");
    placeholder.className = "conversation-item active";
    placeholder.dataset.sessionId = sessionId;
    placeholder.dataset.pending = "true";
    const title = document.createElement("span");
    title.className = "conversation-title";
    title.textContent = deriveTitle(query);
    placeholder.appendChild(title);
    listEl.insertBefore(placeholder, listEl.firstChild);
  });

  window.addEventListener("chat:turn-completed", () => {
    refreshConversationList();
  });

  const persisted = loadPersistedSessionId();
  if (persisted) {
    setSessionId(persisted);
    try {
      const data = await getConversation(persisted);
      renderHistory(data.messages || []);
      if (!data.messages || data.messages.length === 0) showEmptyStateIfNeeded();
    } catch {
      setSessionId(newConversationId());
      showEmptyStateIfNeeded();
    }
  } else {
    setSessionId(newConversationId());
    showEmptyStateIfNeeded();
  }

  await refreshConversationList();
}

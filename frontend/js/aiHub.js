/* Sidebar "AI Hub": collapsible shortcuts to external AI assistants and
   knowledge sources, grouped into two clearly labeled sub-sections. Each
   group is a disclosure the student opens on demand — collapsed by default
   so "Recent Chats" (the section used on every visit) gets the sidebar's
   free space instead of splitting it with two lists that are used rarely by
   comparison. Configuration-driven — the only thing that changes to add a
   new provider is researchProviders.js. */

import { providersByCategory, openResearchProvider } from "./researchProviders.js";
import { getLastUserQuery } from "./chat.js";

const COLLAPSE_KEY_PREFIX = "examPrepAiHubCollapsed:";

function isCollapsed(groupKey) {
  try {
    const stored = localStorage.getItem(COLLAPSE_KEY_PREFIX + groupKey);
    // No stored preference yet -> collapsed, so a first-time visitor gets
    // the maximum room for Recent Chats; once they open a group, their
    // choice persists across reloads.
    return stored === null ? true : stored === "true";
  } catch {
    return true;
  }
}

function setCollapsed(groupKey, collapsed) {
  try {
    localStorage.setItem(COLLAPSE_KEY_PREFIX + groupKey, String(collapsed));
  } catch {
    /* ignore */
  }
}

function buildGroup(groupKey, title, icon, providers) {
  const section = document.createElement("div");
  section.className = "ai-hub-group";

  const itemsId = `aiHubItems-${groupKey}`;
  const header = document.createElement("button");
  header.type = "button";
  header.className = "ai-hub-group-header";
  header.setAttribute("aria-controls", itemsId);
  header.innerHTML =
    `<span aria-hidden="true">${icon}</span><span>${title}</span>` +
    `<span class="ai-hub-chevron" aria-hidden="true">▾</span>`;

  // Two nested elements, not one: the outer .ai-hub-group-items is the grid
  // container the 1fr/0fr collapse transition animates (that trick needs
  // exactly one grid row, i.e. exactly one child); .ai-hub-group-items-inner
  // holds the actual provider buttons and is what clips them via overflow
  // while the outer row's height animates to 0.
  const items = document.createElement("div");
  items.className = "ai-hub-group-items";
  items.id = itemsId;
  const itemsInner = document.createElement("div");
  itemsInner.className = "ai-hub-group-items-inner";
  items.appendChild(itemsInner);

  const applyCollapsed = (collapsed) => {
    section.classList.toggle("collapsed", collapsed);
    header.setAttribute("aria-expanded", String(!collapsed));
  };
  applyCollapsed(isCollapsed(groupKey));

  header.addEventListener("click", () => {
    const collapsed = !section.classList.contains("collapsed");
    applyCollapsed(collapsed);
    setCollapsed(groupKey, collapsed);
  });

  section.appendChild(header);

  for (const provider of providers) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ai-hub-item";
    btn.innerHTML =
      `<span class="ai-hub-item-icon" aria-hidden="true">${provider.icon}</span><span>${provider.label}</span>`;
    btn.title = `Open ${provider.label} in a new tab`;
    btn.setAttribute("aria-label", `Open ${provider.label} in a new tab`);
    // A plain, always-tabbable button — no custom menu/roving-tabindex
    // semantics needed, unlike the transient per-message popup (see
    // continueResearch.js): once its group is open, native Tab/Enter/Space
    // behavior is already fully accessible.
    btn.addEventListener("click", () => openResearchProvider(provider, getLastUserQuery()));
    itemsInner.appendChild(btn);
  }
  section.appendChild(items);
  return section;
}

export function initAiHub() {
  const mount = document.getElementById("sidebarAiHub");
  if (!mount) return;
  mount.innerHTML = "";
  mount.appendChild(buildGroup("ai", "AI Hub", "🤖", providersByCategory("ai")));
  mount.appendChild(buildGroup("knowledge", "Knowledge", "📖", providersByCategory("knowledge")));
}

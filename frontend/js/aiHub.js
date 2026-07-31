/* Sidebar "AI Hub": always-visible (never a dropdown/modal) shortcuts to
   external AI assistants and knowledge sources, grouped into two clearly
   labeled sub-sections per the product requirement to keep "AI assistants"
   and "knowledge sources" visually distinct. Configuration-driven — the only
   thing that changes to add a new provider is researchProviders.js. */

import { providersByCategory, openResearchProvider } from "./researchProviders.js";
import { getLastUserQuery } from "./chat.js";

function buildGroup(title, icon, providers) {
  const section = document.createElement("div");
  section.className = "ai-hub-group";

  const header = document.createElement("div");
  header.className = "ai-hub-group-header";
  header.innerHTML = `<span aria-hidden="true">${icon}</span><span>${title}</span>`;
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
    // continueResearch.js): this list is permanently on-screen, so native
    // Tab/Enter/Space behavior is already fully accessible.
    btn.addEventListener("click", () => openResearchProvider(provider, getLastUserQuery()));
    section.appendChild(btn);
  }
  return section;
}

export function initAiHub() {
  const mount = document.getElementById("sidebarAiHub");
  if (!mount) return;
  mount.innerHTML = "";
  mount.appendChild(buildGroup("AI Hub", "🤖", providersByCategory("ai")));
  mount.appendChild(buildGroup("Knowledge", "📖", providersByCategory("knowledge")));
}

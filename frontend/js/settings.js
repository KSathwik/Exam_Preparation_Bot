/* Settings modal: appearance + about, everything else developer-facing
   (API key override, index/model internals) stays behind the collapsed
   "Developer options" disclosure — regular students never see it. */

import { getStoredThemeChoice, applyTheme } from "./theme.js";
import { getApiKey, setApiKey, getStats } from "./api.js";

function highlightThemeControl(choice) {
  document.querySelectorAll("#themeControl button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.themeChoice === choice);
  });
}

async function refreshStats() {
  const el = document.getElementById("settingsStats");
  el.textContent = "Loading…";
  try {
    const data = await getStats();
    const vs = data.vector_store || {};
    el.innerHTML = `
      <strong>${vs.total_vectors || 0}</strong> chunks indexed<br>
      Embedding model: <strong>${data.embedding_model || vs.embedding_model || "-"}</strong><br>
      Vector dimension: <strong>${vs.embedding_dimension || "-"}</strong>`;
  } catch {
    el.textContent = "Set an API key to view this information.";
  }
}

export function initSettings() {
  const modal = document.getElementById("settingsModal");
  const openBtn = document.getElementById("settingsBtn");
  const closeBtn = document.getElementById("closeSettingsBtn");
  const apiKeyInput = document.getElementById("apiKeyInput");
  const saveKeyBtn = document.getElementById("saveKeyBtn");
  const devOptions = document.getElementById("devOptions");

  openBtn.addEventListener("click", () => {
    highlightThemeControl(getStoredThemeChoice());
    apiKeyInput.value = getApiKey();
    modal.showModal();
  });

  closeBtn.addEventListener("click", () => modal.close());
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.close();
  });

  document.querySelectorAll("#themeControl button").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyTheme(btn.dataset.themeChoice);
      highlightThemeControl(btn.dataset.themeChoice);
    });
  });

  // Only fetch/show model+index internals once the user actually opens
  // Developer options — never on a plain Settings visit.
  devOptions.addEventListener("toggle", () => {
    if (devOptions.open) refreshStats();
  });

  saveKeyBtn.addEventListener("click", () => {
    setApiKey(apiKeyInput.value.trim());
    if (devOptions.open) refreshStats();
  });
}

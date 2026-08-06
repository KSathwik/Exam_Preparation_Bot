/* Settings modal: appearance + about, everything else developer-facing
   (API key override, index/model internals) stays behind the collapsed
   "Developer options" disclosure — regular students never see it. */

import { getStoredThemeChoice, applyTheme } from "./theme.js";
import { getApiKey, setApiKey, getStats } from "./api.js";
import { state, setTopK, setTemperature } from "./state.js";

function highlightThemeControl(choice) {
  document.querySelectorAll("#themeControl button").forEach((btn) => {
    const isActive = btn.dataset.themeChoice === choice;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-pressed", String(isActive));
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

  const topKSlider = document.getElementById("topKSlider");
  const topKValue = document.getElementById("topKValue");
  const tempSlider = document.getElementById("tempSlider");
  const tempValue = document.getElementById("tempValue");

  openBtn.addEventListener("click", () => {
    highlightThemeControl(getStoredThemeChoice());
    apiKeyInput.value = getApiKey();
    if (topKSlider && topKValue) {
      topKSlider.value = state.topK;
      topKValue.textContent = String(state.topK);
    }
    if (tempSlider && tempValue) {
      tempSlider.value = state.temperature;
      tempValue.textContent = Number(state.temperature).toFixed(2);
    }
    modal.showModal();
  });

  if (topKSlider && topKValue) {
    topKSlider.addEventListener("input", () => {
      const val = parseInt(topKSlider.value, 10);
      topKValue.textContent = String(val);
      setTopK(val);
    });
  }

  if (tempSlider && tempValue) {
    tempSlider.addEventListener("input", () => {
      const val = parseFloat(tempSlider.value);
      tempValue.textContent = val.toFixed(2);
      setTemperature(val);
    });
  }

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

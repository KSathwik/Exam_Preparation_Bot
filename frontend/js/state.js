/* Shared in-memory + localStorage-backed state. No framework — every other
   module reads/writes this same object so there's one source of truth for
   "which conversation / device / streaming status are we in." */

const DEVICE_ID_KEY = "examPrepDeviceId";
const SESSION_ID_KEY = "examPrepSessionId";
const PRESET_KEY = "examPrepDomainPreset";
const TOP_K_KEY = "examPrepTopK";
const TEMP_KEY = "examPrepTemperature";

function readDeviceId() {
  try {
    let id = localStorage.getItem(DEVICE_ID_KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(DEVICE_ID_KEY, id);
    }
    return id;
  } catch {
    return crypto.randomUUID();
  }
}

function readStorageValue(key, fallback) {
  try {
    const val = localStorage.getItem(key);
    return val !== null ? val : fallback;
  } catch {
    return fallback;
  }
}

export const state = {
  deviceId: readDeviceId(),
  sessionId: null,
  ws: null,
  streaming: false,
  conversations: [],
  documentIds: [],
  editingUserRow: null,
  domainPreset: readStorageValue(PRESET_KEY, "general"),
  topK: parseInt(readStorageValue(TOP_K_KEY, "4"), 10) || 4,
  temperature: parseFloat(readStorageValue(TEMP_KEY, "0.3")) || 0.3,
};

export function setDomainPreset(preset) {
  state.domainPreset = preset;
  try {
    localStorage.setItem(PRESET_KEY, preset);
  } catch {}
}

export function setTopK(val) {
  const num = parseInt(val, 10) || 4;
  state.topK = num;
  try {
    localStorage.setItem(TOP_K_KEY, String(num));
  } catch {}
}

export function setTemperature(val) {
  const num = parseFloat(val) || 0.3;
  state.temperature = num;
  try {
    localStorage.setItem(TEMP_KEY, String(num));
  } catch {}
}

export function addDocumentId(id) {
  if (id && !state.documentIds.includes(id)) state.documentIds.push(id);
}

export function setDocumentIds(ids) {
  state.documentIds = [...new Set((ids || []).filter(Boolean))];
}

export function clearDocumentIds() {
  state.documentIds = [];
}

export function setSessionId(id) {
  state.sessionId = id;
  try {
    if (id) localStorage.setItem(SESSION_ID_KEY, id);
    else localStorage.removeItem(SESSION_ID_KEY);
  } catch {
    /* ignore */
  }
}

export function loadPersistedSessionId() {
  try {
    return localStorage.getItem(SESSION_ID_KEY) || null;
  } catch {
    return null;
  }
}

export function newConversationId() {
  return crypto.randomUUID();
}

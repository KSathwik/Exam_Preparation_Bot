/* Shared in-memory + localStorage-backed state. No framework — every other
   module reads/writes this same object so there's one source of truth for
   "which conversation / device / streaming status are we in." */

const DEVICE_ID_KEY = "examPrepDeviceId";
const SESSION_ID_KEY = "examPrepSessionId";

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

export const state = {
  deviceId: readDeviceId(),
  sessionId: null,
  ws: null,
  streaming: false,
  conversations: [],
  // Documents uploaded during this conversation (in-memory only, not
  // persisted — there's no document-to-conversation link in the DB yet).
  // Sent with every question so retrieval prefers what was actually
  // uploaded here instead of searching every document ever uploaded,
  // which risks blending unrelated documents into one answer.
  documentIds: [],
};

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

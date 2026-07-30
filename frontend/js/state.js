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
  // Optional narrowing hint only — the server resolves the conversation's
  // real document set from the DB (documents.session_id, set at upload
  // time) as the authoritative scope, so losing this on reload/tab-switch
  // is safe. Populated on upload and by suggestion-chip clicks to narrow a
  // question to one specific document among possibly several uploaded in
  // this same conversation; never used to *expand* scope beyond it.
  documentIds: [],
  // Set by clicking Edit on a past user message; read (and cleared) by the
  // next sendQuery() call so it regenerates that message+reply pair in
  // place instead of appending a new one. Never persisted — reset if the
  // composer is cleared without sending (see input.js) or the row is no
  // longer in the DOM (conversation switched/cleared).
  editingUserRow: null,
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

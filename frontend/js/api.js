/* Thin fetch wrappers around the backend REST/WebSocket surface. Nothing in
   here touches the DOM. */

const API = "/api";
const API_KEY_STORAGE_KEY = "examPrepApiKey";

export function getApiKey() {
  // A manually-saved key (this browser) takes precedence over the key the
  // server embedded into the page, so a manual entry can still override it.
  try {
    return localStorage.getItem(API_KEY_STORAGE_KEY) || window.__APP_API_KEY__ || "";
  } catch {
    return window.__APP_API_KEY__ || "";
  }
}

export function setApiKey(key) {
  try {
    if (key) localStorage.setItem(API_KEY_STORAGE_KEY, key);
    else localStorage.removeItem(API_KEY_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

export function authHeaders(extra) {
  const key = getApiKey();
  return key ? { ...extra, "X-API-Key": key } : { ...(extra || {}) };
}

export function buildWsUrl() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const key = encodeURIComponent(getApiKey());
  return `${protocol}//${location.host}${API}/ws?api_key=${key}`;
}

async function asJson(res) {
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* no body */
  }
  if (!res.ok) {
    const message = (data && (data.detail || data.error)) || res.statusText || "Request failed";
    throw new Error(message);
  }
  return data;
}

export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/documents/upload`, { method: "POST", headers: authHeaders(), body: form });
  return asJson(res);
}

export async function listDocuments() {
  const res = await fetch(`${API}/documents/list`, { headers: authHeaders() });
  return asJson(res);
}

export async function getStats() {
  const res = await fetch(`${API}/documents/stats`, { headers: authHeaders() });
  return asJson(res);
}

export async function listConversations(deviceId) {
  const res = await fetch(`${API}/conversations?device_id=${encodeURIComponent(deviceId)}`, {
    headers: authHeaders(),
  });
  return asJson(res);
}

export async function getConversation(sessionId) {
  const res = await fetch(`${API}/conversations/${encodeURIComponent(sessionId)}`, {
    headers: authHeaders(),
  });
  return asJson(res);
}

export async function renameConversation(sessionId, title) {
  const res = await fetch(`${API}/conversations/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ title }),
  });
  return asJson(res);
}

export async function deleteConversation(sessionId) {
  const res = await fetch(`${API}/conversations/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  return asJson(res);
}

/* Markdown → sanitize → innerHTML pipeline for assistant messages, plus math
   rendering. User messages never go through this — they're always rendered
   as plain escaped text (see escapeText), since they're the user's own input
   and don't need markdown/math/code handling. */

let configured = false;

function configureMarked() {
  if (configured) return;
  configured = true;
  if (!window.marked) return;

  if (window.markedHighlight && window.hljs) {
    window.marked.use(
      window.markedHighlight.markedHighlight({
        langPrefix: "hljs language-",
        highlight(code, lang) {
          const language = window.hljs.getLanguage(lang) ? lang : "plaintext";
          return window.hljs.highlight(code, { language }).value;
        },
      })
    );
  }
  window.marked.setOptions({ gfm: true, breaks: false });

  if (window.DOMPurify) {
    window.DOMPurify.addHook("afterSanitizeAttributes", (node) => {
      if (node.tagName === "A") {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener noreferrer");
      }
    });
  }
}

export function escapeText(text) {
  const d = document.createElement("div");
  d.textContent = text ?? "";
  return d.innerHTML;
}

export function renderMarkdown(text) {
  configureMarked();
  if (!window.marked) return escapeText(text);
  const html = window.marked.parse(text ?? "");
  if (!window.DOMPurify) return html;
  return window.DOMPurify.sanitize(html, {
    FORBID_TAGS: ["script", "style", "iframe"],
    FORBID_ATTR: ["onerror", "onload", "onclick", "onmouseover"],
  });
}

export function renderMathIn(el) {
  if (!window.renderMathInElement || !el) return;
  try {
    window.renderMathInElement(el, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  } catch {
    /* best-effort — a malformed expression shouldn't break the message */
  }
}

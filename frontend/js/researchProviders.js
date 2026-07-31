/* Configuration-driven provider definitions for the AI Hub (sidebar section)
   and the per-message "Open in" popup. Add a new provider (Stack Overflow,
   MDN, arXiv, Google Scholar, YouTube, Khan Academy, Microsoft Copilot, ...)
   by appending an entry here — nothing else in the app needs to change.

   Each entry is { id, label, icon, category, homeUrl, buildUrl? }.
   `category` is "ai" (an AI assistant) or "knowledge" (a reference source) —
   used to group providers into the sidebar's two sub-sections; the flatter
   per-message popup ignores it and just lists everything in order.

   `buildUrl(query)` is only present when a provider has a *verified* query
   parameter that actually carries the question over (tested by hand — see
   below); when it's absent, opening the provider just opens `homeUrl`
   untouched, per the requirement to never guess at unsupported behavior.

   Verified by manual testing (logged-in session, prefill only, no send):
     - Claude and Perplexity DO carry the query over via `?q=`.
     - ChatGPT and Gemini silently strip/ignore `?q=` — homepage-only.
   Microsoft Copilot sits behind a sign-in wall in every environment
   available for testing, so it isn't wired in as an active provider yet —
   see the Future-Proof Design note above for how to add it (and others)
   once its prefill behavior can actually be verified. */

export const RESEARCH_PROVIDERS = [
  {
    id: "chatgpt",
    label: "ChatGPT",
    icon: "💬",
    category: "ai",
    homeUrl: "https://chatgpt.com/",
  },
  {
    id: "claude",
    label: "Claude",
    icon: "🧠",
    category: "ai",
    homeUrl: "https://claude.ai/new",
    buildUrl: (query) => `https://claude.ai/new?q=${encodeURIComponent(query)}`,
  },
  {
    id: "gemini",
    label: "Gemini",
    icon: "✨",
    category: "ai",
    homeUrl: "https://gemini.google.com/app",
  },
  {
    id: "perplexity",
    label: "Perplexity",
    icon: "🔍",
    category: "ai",
    homeUrl: "https://www.perplexity.ai/",
    buildUrl: (query) => `https://www.perplexity.ai/search?q=${encodeURIComponent(query)}`,
  },
  {
    id: "wikipedia",
    label: "Wikipedia",
    icon: "🌐",
    category: "knowledge",
    homeUrl: "https://en.wikipedia.org/",
    // No fulltext param: Wikipedia's own Special:Search redirects straight to
    // an exact/near-match article when one exists, and falls back to a full
    // search-results page otherwise — exactly the desired behavior, for free.
    buildUrl: (query) => `https://en.wikipedia.org/wiki/Special:Search?search=${encodeURIComponent(query)}`,
  },
];

export function providersByCategory(category) {
  return RESEARCH_PROVIDERS.filter((p) => p.category === category);
}

/* Opens in a new, unrelated tab (target="_blank" + noopener/noreferrer) —
   never touches app state, so the current conversation is always exactly as
   the student left it when they come back. */
export function openResearchProvider(provider, query) {
  const url = (provider.buildUrl && query ? provider.buildUrl(query) : null) || provider.homeUrl;
  window.open(url, "_blank", "noopener,noreferrer");
}

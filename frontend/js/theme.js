/* Appearance: System / Light / Dark. "System" means no explicit [data-theme]
   attribute is set at all, so the CSS `@media (prefers-color-scheme: dark)`
   block keeps tracking OS changes live; Light/Dark set an explicit
   [data-theme] override that wins regardless of OS preference. */

const THEME_KEY = "examPrepTheme";
const media = window.matchMedia("(prefers-color-scheme: dark)");

function systemPrefersDark() {
  return media.matches;
}

export function getStoredThemeChoice() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    return stored === "light" || stored === "dark" ? stored : "system";
  } catch {
    return "system";
  }
}

function effectiveTheme(choice) {
  return choice === "system" ? (systemPrefersDark() ? "dark" : "light") : choice;
}

function syncHighlightJsTheme(effective) {
  const light = document.getElementById("hljsLightTheme");
  const dark = document.getElementById("hljsDarkTheme");
  if (!light || !dark) return;
  light.disabled = effective === "dark";
  dark.disabled = effective === "light";
}

export function applyTheme(choice) {
  if (choice === "system") {
    document.documentElement.removeAttribute("data-theme");
    try {
      localStorage.removeItem(THEME_KEY);
    } catch {
      /* ignore */
    }
  } else {
    document.documentElement.setAttribute("data-theme", choice);
    try {
      localStorage.setItem(THEME_KEY, choice);
    } catch {
      /* ignore */
    }
  }
  syncHighlightJsTheme(effectiveTheme(choice));
}

export function initTheme() {
  const choice = getStoredThemeChoice();
  syncHighlightJsTheme(effectiveTheme(choice));
  media.addEventListener("change", () => {
    if (getStoredThemeChoice() === "system") syncHighlightJsTheme(effectiveTheme("system"));
  });
}

import { initTheme } from "./theme.js";
import { initChat } from "./chat.js";
import { initSidebar } from "./sidebar.js";
import { initInput } from "./input.js";
import { initSettings } from "./settings.js";

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initChat();
  initInput();
  initSettings();
  initSidebar();
});

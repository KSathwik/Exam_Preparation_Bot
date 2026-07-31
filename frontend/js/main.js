import { initTheme } from "./theme.js";
import { initChat } from "./chat.js";
import { initSidebar } from "./sidebar.js";
import { initInput } from "./input.js";
import { initSettings } from "./settings.js";
import { initAiHub } from "./aiHub.js";

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initChat();
  initInput();
  initSettings();
  initSidebar();
  initAiHub();
});

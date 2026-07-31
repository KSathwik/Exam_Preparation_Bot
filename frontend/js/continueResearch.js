/* "Continue Research" trigger + anchored dropdown, one per assistant message.
   Mirrors the sidebar's conversation-menu pattern (small popup, closed via a
   document-level click listener) — except the menu itself is appended to
   <body> and positioned with `position:fixed` against the trigger's on-screen
   coordinates, rather than `position:absolute` inside the message row: the
   row lives inside #messagesBox, which is `overflow-y:auto` and would
   otherwise clip the popup the moment the trigger is near the bottom of the
   visible scroll area. */

import { RESEARCH_PROVIDERS, openResearchProvider } from "./researchProviders.js";

let openControl = null; // { trigger, menu } for whichever menu is currently open

function closeOpenMenu() {
  if (!openControl) return;
  const { trigger, menu } = openControl;
  menu.remove();
  trigger.setAttribute("aria-expanded", "false");
  openControl = null;
}

document.addEventListener("click", (e) => {
  if (openControl && !e.target.closest(".research-menu") && !e.target.closest(".research-control")) {
    closeOpenMenu();
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && openControl) {
    const { trigger } = openControl;
    closeOpenMenu();
    trigger.focus();
  }
});
// The menu's position is computed once, at open time, against the trigger's
// current viewport coordinates — rather than tracking it live, just close on
// scroll/resize (capture:true so this fires for #messagesBox's own scrolling,
// which doesn't bubble). Simple, and matches how transient popovers here
// (e.g. the conversation kebab menu) already behave — no drag-following.
document.addEventListener("scroll", () => closeOpenMenu(), true);
window.addEventListener("resize", () => closeOpenMenu());

function focusItem(items, index) {
  items[(index + items.length) % items.length].focus();
}

function positionMenu(menu, trigger) {
  const rect = trigger.getBoundingClientRect();
  const menuRect = menu.getBoundingClientRect();
  const margin = 8;
  let top = rect.bottom + 4;
  if (top + menuRect.height > window.innerHeight - margin) {
    top = rect.top - menuRect.height - 4;
  }
  let left = rect.left;
  if (left + menuRect.width > window.innerWidth - margin) {
    left = window.innerWidth - menuRect.width - margin;
  }
  menu.style.top = `${Math.max(margin, top)}px`;
  menu.style.left = `${Math.max(margin, left)}px`;
}

function openMenu(trigger, getQuery) {
  const menu = document.createElement("div");
  menu.className = "research-menu";
  menu.setAttribute("role", "menu");
  menu.setAttribute("aria-label", "Continue with");
  menu.style.visibility = "hidden"; // measured once appended, then positioned+revealed

  const header = document.createElement("div");
  header.className = "research-menu-header";
  header.textContent = "Continue with…";
  header.setAttribute("aria-hidden", "true"); // decorative — menu's own aria-label already says this
  menu.appendChild(header);

  const items = RESEARCH_PROVIDERS.map((provider) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "research-menu-item";
    item.setAttribute("role", "menuitem");
    item.innerHTML =
      `<span class="research-menu-icon" aria-hidden="true">${provider.icon}</span>` +
      `<span>${provider.label}</span>`;
    item.addEventListener("click", (e) => {
      e.stopPropagation();
      openResearchProvider(provider, getQuery());
      closeOpenMenu();
      trigger.focus();
    });
    return item;
  });

  items.forEach((item, i) => {
    item.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); focusItem(items, i + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); focusItem(items, i - 1); }
      else if (e.key === "Home") { e.preventDefault(); focusItem(items, 0); }
      else if (e.key === "End") { e.preventDefault(); focusItem(items, items.length - 1); }
    });
  });

  menu.append(...items);
  document.body.appendChild(menu);
  positionMenu(menu, trigger);
  menu.style.visibility = "";

  trigger.setAttribute("aria-expanded", "true");
  openControl = { trigger, menu };
  items[0].focus();
}

/* getQuery is a function (not a plain string) so the control always reflects
   the row's current query — e.g. after Regenerate re-runs a row in place. */
export function buildResearchControl(getQuery) {
  const wrap = document.createElement("div");
  wrap.className = "research-control";
  wrap.hidden = true; // shown by the caller once the response has finished streaming

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "message-action-btn research-trigger";
  trigger.title = "Open in";
  trigger.setAttribute("aria-label", "Open in — continue with another AI or search the web");
  trigger.setAttribute("aria-haspopup", "menu");
  trigger.setAttribute("aria-expanded", "false");
  trigger.textContent = "↗";

  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    const wasOpen = openControl && openControl.trigger === trigger;
    closeOpenMenu();
    if (!wasOpen) openMenu(trigger, getQuery);
  });

  wrap.appendChild(trigger);
  return wrap;
}

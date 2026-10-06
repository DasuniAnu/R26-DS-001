// Page-data scraping. The video-id/title/description/transcript extraction
// here is the same technique already proven working in the False Content
// project's own chrome_extension/content.js — reused as-is since it's the
// only piece the False Content backend actually needs (Tamil and Sinhala's
// YouTube routes only need the video URL and do their own audio download +
// transcription server-side, so they don't need any of this page data).

function textFrom(selector) {
  const element = document.querySelector(selector);
  return element ? element.innerText.trim() : "";
}

function videoIdFromLocation() {
  const url = new URL(window.location.href);
  if (url.hostname.includes("youtu.be")) {
    return url.pathname.replace("/", "");
  }
  if (url.pathname.startsWith("/shorts/")) {
    return url.pathname.split("/")[2] || "";
  }
  return url.searchParams.get("v") || "";
}

function collectDescription() {
  const selectors = [
    "ytd-watch-metadata #description-inline-expander",
    "ytd-watch-metadata #description",
    "#description.ytd-video-secondary-info-renderer"
  ];
  for (const selector of selectors) {
    const text = textFrom(selector);
    if (text) return text;
  }
  return "";
}

function collectTranscript() {
  const selectors = [
    "ytd-transcript-segment-renderer",
    "yt-formatted-string.segment-text",
    "#segments-container ytd-transcript-segment-renderer"
  ];
  const parts = [];
  for (const selector of selectors) {
    document.querySelectorAll(selector).forEach((node) => {
      const text = node.innerText.trim();
      if (text && !parts.includes(text)) parts.push(text);
    });
    if (parts.length) break;
  }
  return parts.join(" ");
}

function collectYouTubePageData() {
  return {
    url: window.location.href,
    video_id: videoIdFromLocation(),
    title: textFrom("h1.ytd-watch-metadata") || document.title.replace(" - YouTube", ""),
    description: collectDescription(),
    transcript: collectTranscript()
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === "COLLECT_YOUTUBE_DATA") {
    sendResponse({ ok: true, data: collectYouTubePageData() });
  }
});

// =============================================================
// Everything below is new, additive-only page behavior:
//   - auto-run analysis on every video, using saved settings, so the user
//     doesn't have to reopen the popup and click Check on each video
//   - a small on-page status panel (isolated in a Shadow DOM, so none of
//     its CSS can leak into or collide with YouTube's own styles)
//   - flagged-comment labeling, which only ever ADDS elements or toggles a
//     class on a wrapper it creates — it never removes or rewrites any
//     node YouTube itself rendered, and every DOM lookup is wrapped so an
//     unmatched/changed YouTube layout just means this feature quietly
//     does nothing, never an error that could affect the page.
// =============================================================

function storageKey(videoId) {
  return `analysis_${videoId}`;
}

function pct(value) {
  return `${Number(value || 0).toFixed(1)}%`;
}

// ── On-page status panel ────────────────────────────────────
let panelHost = null;
let panelBody = null;

function ensurePanel() {
  if (panelHost && document.body.contains(panelHost)) return panelBody;

  panelHost = document.createElement("div");
  panelHost.id = "truthlense-panel-host";
  panelHost.style.cssText = "position:fixed;top:88px;right:16px;z-index:2147483000;";
  const shadow = panelHost.attachShadow({ mode: "open" });

  const style = document.createElement("style");
  style.textContent = `
    .panel { width: 240px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
      box-shadow: 0 10px 30px rgba(15,23,42,0.18); font-family: Arial, sans-serif; overflow: hidden; }
    .head { display:flex; align-items:center; justify-content:space-between;
      background: linear-gradient(135deg,#3b82f6,#1d4ed8); color:#fff; padding:8px 10px; font-size:12px; font-weight:700; }
    .head button { background: transparent; border: none; color: #fff; cursor: pointer; font-size: 13px; opacity: .85; }
    .body { padding: 10px 12px; }
    .row { display:flex; align-items:center; justify-content:space-between; font-size:11px; color:#334155; padding:4px 0; }
    .badge { border-radius:999px; padding:2px 8px; font-size:10px; font-weight:700; }
    .badge.safe { background:#dcfce7; color:#16a34a; }
    .badge.hate { background:#fee2e2; color:#dc2626; }
    .badge.neutral { background:#eff6ff; color:#2563eb; }
    .loading { font-size: 11px; color: #64748b; font-style: italic; }
    .empty { font-size: 11px; color: #94a3b8; }
    .warn { font-size: 10px; color: #92400e; background: #fffbeb; border: 1px solid #fde68a;
      border-radius: 6px; padding: 5px 7px; margin-bottom: 6px; font-weight: 600; }
    .minimized .body { display: none; }
  `;

  const wrapper = document.createElement("div");
  wrapper.className = "panel";
  wrapper.innerHTML = `
    <div class="head">
      <span>🛡️ TruthLenseAI</span>
      <button id="toggle" title="Minimize">–</button>
    </div>
    <div class="body"><div class="empty">No checks enabled for this video.</div></div>
  `;
  shadow.appendChild(style);
  shadow.appendChild(wrapper);

  shadow.getElementById("toggle").addEventListener("click", () => {
    wrapper.classList.toggle("minimized");
  });

  document.documentElement.appendChild(panelHost);
  panelBody = wrapper.querySelector(".body");
  return panelBody;
}

function renderPanel(state) {
  const body = ensurePanel();
  if (!state) {
    body.innerHTML = `<div class="empty">No checks enabled for this video.</div>`;
    return;
  }

  const rows = [];

  if (state.tamil) rows.push(renderPanelRow("Tamil Hate Speech", state.tamil, (d) => {
    const hate = d.verdict === "HATE SPEECH DETECTED";
    return { cls: hate ? "hate" : "safe", text: hate ? "Detected" : "Safe" };
  }));

  if (state.tamilComments) rows.push(renderPanelRow("Tamil Comments", state.tamilComments, (d) => {
    const hate = d.hate_count > 0;
    return { cls: hate ? "hate" : "safe", text: hate ? `${d.hate_count} flagged` : "Clean" };
  }));

  if (state.sinhala) rows.push(renderPanelRow("Sinhala Hate Speech", state.sinhala, (d) => {
    const hate = (d.summary || {}).hate_segments > 0;
    return { cls: hate ? "hate" : "safe", text: hate ? "Detected" : "Safe" };
  }));

  if (state.sinhala) rows.push(renderPanelRow("Audio Deepfake", state.sinhala, (d) => {
    const fake = (d.summary || {}).fake_segments > 0;
    return { cls: fake ? "hate" : "safe", text: fake ? "Detected" : "Authentic" };
  }));

  if (state.sinhalaComments) rows.push(renderPanelRow("Sinhala Comments", state.sinhalaComments, (d) => {
    const hate = (d.summary || {}).hate_comments > 0;
    return { cls: hate ? "hate" : "safe", text: hate ? `${d.summary.hate_comments} flagged` : "Clean" };
  }));

  if (state.falseContent) rows.push(renderPanelRow("False Content", state.falseContent, (d) => {
    const isFalse = String(d.label || d.predicted_label || "").toLowerCase() === "false";
    return { cls: isFalse ? "hate" : "safe", text: isFalse ? "False" : "Not False" };
  }));

  const languageWarning = (state.tamil && state.tamil.languageWarning) || (state.sinhala && state.sinhala.languageWarning);
  const warningHtml = languageWarning ? `<div class="warn">⚠️ ${languageWarning}</div>` : "";

  body.innerHTML = (warningHtml + rows.join("")) || `<div class="empty">No checks enabled for this video.</div>`;
}

function renderPanelRow(label, moduleState, extract) {
  if (!moduleState || moduleState.status === "pending" || moduleState.status === "running") {
    return `<div class="row"><span>${label}</span><span class="loading">running…</span></div>`;
  }
  if (moduleState.status === "error") {
    return `<div class="row"><span>${label}</span><span class="badge neutral">error</span></div>`;
  }
  const { cls, text } = extract(moduleState.data);
  return `<div class="row"><span>${label}</span><span class="badge ${cls}">${text}</span></div>`;
}

// ── Flagged-comment labeling ────────────────────────────────
// Additive only: wraps a matched comment's own body node with a sibling
// overlay + a small label, using a class toggle to dim/reveal. Nothing
// belonging to YouTube is ever removed or rewritten.
function normalizeForMatch(text) {
  return String(text || "").replace(/\s+/g, " ").trim().toLowerCase();
}

function findCommentNodes() {
  // Defensive: try both the current and older YouTube comment renderers.
  const selectors = ["ytd-comment-view-model", "ytd-comment-renderer"];
  for (const selector of selectors) {
    const nodes = document.querySelectorAll(selector);
    if (nodes.length) return Array.from(nodes);
  }
  return [];
}

function commentBodyEl(node) {
  return node.querySelector("#content-text") || node.querySelector("yt-attributed-string#content-text");
}

function commentAuthorEl(node) {
  return node.querySelector("#author-text");
}

function labelFlaggedComment(node, confidenceText) {
  if (node.dataset.truthlenseFlagged === "1") return; // already labeled
  node.dataset.truthlenseFlagged = "1";

  const bodyEl = commentBodyEl(node);
  const authorEl = commentAuthorEl(node);
  if (!bodyEl) return;

  try {
    if (authorEl) {
      const tag = document.createElement("span");
      tag.textContent = "⚠ Hate Speech Detected";
      tag.style.cssText = "display:inline-block;margin-left:8px;background:#fee2e2;color:#dc2626;" +
        "font-size:10px;font-weight:700;border-radius:999px;padding:1px 8px;vertical-align:middle;";
      authorEl.appendChild(tag);
    }

    bodyEl.style.filter = "blur(4px)";
    bodyEl.style.userSelect = "none";
    bodyEl.style.transition = "filter .15s";

    const revealRow = document.createElement("div");
    revealRow.style.cssText = "font-size:11px;color:#dc2626;margin-top:4px;cursor:pointer;font-weight:600;";
    revealRow.textContent = `🚫 Comment hidden — flagged as hate speech (${confidenceText}). Click to show anyway.`;
    revealRow.addEventListener("click", () => {
      const hidden = bodyEl.style.filter !== "none";
      bodyEl.style.filter = hidden ? "none" : "blur(4px)";
      revealRow.textContent = hidden
        ? "🙈 Hide again"
        : `🚫 Comment hidden — flagged as hate speech (${confidenceText}). Click to show anyway.`;
    });
    bodyEl.insertAdjacentElement("afterend", revealRow);
  } catch (e) {
    // Never let a DOM-shape surprise here affect the rest of the page.
    console.debug("TruthLenseAI: comment labeling skipped for one node", e);
  }
}

function applyCommentLabels(flaggedTexts) {
  if (!flaggedTexts || !flaggedTexts.length) return;
  const normalizedFlagged = flaggedTexts.map((f) => ({ norm: normalizeForMatch(f.text), confidence: f.confidenceText }));

  try {
    for (const node of findCommentNodes()) {
      const bodyEl = commentBodyEl(node);
      if (!bodyEl) continue;
      const bodyNorm = normalizeForMatch(bodyEl.innerText);
      const match = normalizedFlagged.find((f) => f.norm && (bodyNorm === f.norm || bodyNorm.includes(f.norm) || f.norm.includes(bodyNorm)));
      if (match) labelFlaggedComment(node, match.confidence);
    }
  } catch (e) {
    console.debug("TruthLenseAI: comment scan skipped", e);
  }
}

function flaggedTextsFromState(state) {
  const out = [];
  if (state.tamilComments && state.tamilComments.status === "done") {
    for (const c of state.tamilComments.data.hate_comments || []) {
      out.push({ text: c.text, confidenceText: `${c.confidence}%` });
    }
  }
  if (state.sinhalaComments && state.sinhalaComments.status === "done") {
    const results = state.sinhalaComments.data.comments || [];
    for (const c of results) {
      if (c.is_offensive) out.push({ text: c.text, confidenceText: pct(c.confidence * 100) });
    }
  }
  return out;
}

// Comments load lazily as the user scrolls, so re-scan on DOM changes too,
// not just once when the analysis finishes.
let lastFlaggedTexts = [];
const commentObserver = new MutationObserver(() => {
  if (lastFlaggedTexts.length) applyCommentLabels(lastFlaggedTexts);
});

function watchComments() {
  const container = document.querySelector("#comments") || document.body;
  commentObserver.disconnect();
  commentObserver.observe(container, { childList: true, subtree: true });
}

// ── Self-healing after Chrome recycles the extension ────────
// Manifest V3 extensions can be killed and restarted by Chrome at any time
// (most often under memory pressure) even while a YouTube tab stays open.
// When that happens, this already-injected content script becomes
// "orphaned" — its chrome.* calls throw "Extension context invalidated"
// forever, since there is no way for an orphaned script to reconnect to
// the new extension instance. The only real fix is a fresh page load,
// which injects a new content script bound to the current extension.
// Rather than surface a console error and silently do nothing (the
// previous behavior — confusing, and only noticed when a check never
// starts), detect this proactively and reload automatically.
function isExtensionContextValid() {
  try {
    return !!(chrome && chrome.runtime && chrome.runtime.id);
  } catch (e) {
    return false;
  }
}

let reconnecting = false;
function reconnectPage() {
  if (reconnecting) return;
  reconnecting = true;
  try {
    const body = ensurePanel();
    body.innerHTML = '<div class="empty">⚠️ Extension was reloaded by Chrome — refreshing this page to reconnect...</div>';
  } catch (e) {
    // The panel itself may be unreachable at this point; the reload below
    // is what actually matters.
  }
  setTimeout(() => location.reload(), 900);
}

// ── Storage-driven live updates ─────────────────────────────
let watchedVideoId = null;

async function refreshFromStorage() {
  if (!watchedVideoId) return;
  if (!isExtensionContextValid()) return reconnectPage();
  try {
    const stored = await chrome.storage.local.get(storageKey(watchedVideoId));
    const state = stored[storageKey(watchedVideoId)];
    renderPanel(state);
    if (state) {
      lastFlaggedTexts = flaggedTextsFromState(state);
      if (lastFlaggedTexts.length) {
        applyCommentLabels(lastFlaggedTexts);
        watchComments();
      }
    }
  } catch (e) {
    if (String(e.message || e).includes("Extension context invalidated")) reconnectPage();
  }
}

try {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local" || !watchedVideoId) return;
    if (changes[storageKey(watchedVideoId)]) refreshFromStorage();
  });
} catch (e) {
  // Context was already dead before this script finished initializing.
  reconnectPage();
}

// ── Auto-run on every video, using saved settings ───────────
async function maybeAutoStart(videoId, pageData) {
  if (!isExtensionContextValid()) return reconnectPage();
  try {
    const stored = await chrome.storage.local.get(["settings", storageKey(videoId)]);
    const settings = stored.settings;
    if (!settings || (settings.hateLang === "off" && !settings.falseContent)) return; // nothing enabled
    if (stored[storageKey(videoId)]) return; // already started/finished for this video

    chrome.runtime.sendMessage({
      type: "START_ANALYSIS",
      payload: { videoId, tabUrl: pageData.url, pageData, settings }
    });
  } catch (e) {
    if (String(e.message || e).includes("Extension context invalidated")) reconnectPage();
  }
}

async function onVideoReady() {
  if (!isExtensionContextValid()) return reconnectPage();
  const data = collectYouTubePageData();
  if (!data.video_id) return;
  watchedVideoId = data.video_id;
  ensurePanel();
  await refreshFromStorage();
  await maybeAutoStart(data.video_id, data);
}

// YouTube is a single-page app; it fires this custom event on in-app
// navigation between videos without a full page reload.
document.addEventListener("yt-navigate-finish", () => {
  setTimeout(onVideoReady, 500); // let the new page's metadata render first
});

// Fallback in case yt-navigate-finish isn't available on some YouTube
// versions: poll the URL for changes. Cheap, and only acts on a real change.
// This same tick also doubles as the proactive health check — it runs
// every 1.5s regardless of navigation, so a Chrome-recycled extension
// gets detected and the page reloaded within ~1.5s, not whenever the user
// next happens to click something.
let lastUrl = location.href;
setInterval(() => {
  if (!isExtensionContextValid()) {
    reconnectPage();
    return;
  }
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    setTimeout(onVideoReady, 500);
  }
}, 1500);

if (document.readyState === "complete") {
  onVideoReady();
} else {
  window.addEventListener("load", onVideoReady);
}

const pageStatusEl = document.getElementById("pageStatus");
const videoInfoEl = document.getElementById("videoInfo");
const videoTitleEl = document.getElementById("videoTitle");
const hateSpeechEnabledEl = document.getElementById("hateSpeechEnabled");
const hateSpeechBodyEl = document.getElementById("hateSpeechBody");
const includeCommentsEl = document.getElementById("includeComments");
const falseContentEl = document.getElementById("falseContent");
const videoDeepfakeEl = document.getElementById("videoDeepfake");
const checkButton = document.getElementById("check");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

let currentVideoId = null;
let currentTabUrl = null;
let currentPageData = null;

function activeTab() {
  return chrome.tabs.query({ active: true, currentWindow: true }).then((tabs) => tabs[0]);
}

function collectPageData(tabId) {
  return chrome.tabs.sendMessage(tabId, { type: "COLLECT_YOUTUBE_DATA" });
}

// The "Hate Speech Analysis" checkbox and the Tamil/Sinhala radio pair are
// two separate controls in the UI, but background.js only ever needs one
// value — "off" | "tamil" | "sinhala" — so that's still the only thing
// written to settings. Unticking the module checkbox is what maps to "off".
function getHateLang() {
  if (!hateSpeechEnabledEl.checked) return "off";
  const checked = document.querySelector('input[name="hateLang"]:checked');
  return checked ? checked.value : "tamil"; // module on but no language picked yet — default to Tamil
}

function currentSettings() {
  return {
    hateLang: getHateLang(),
    includeComments: includeCommentsEl.checked,
    falseContent: falseContentEl.checked,
    videoDeepfake: videoDeepfakeEl.checked
  };
}

function applySettings(settings) {
  const hateLang = settings.hateLang || "off";
  const enabled = hateLang !== "off";
  hateSpeechEnabledEl.checked = enabled;
  hateSpeechBodyEl.hidden = !enabled;
  const radio = document.querySelector(`input[name="hateLang"][value="${enabled ? hateLang : "tamil"}"]`);
  if (radio) radio.checked = true;
  includeCommentsEl.checked = !!settings.includeComments;
  falseContentEl.checked = !!settings.falseContent;
  videoDeepfakeEl.checked = !!settings.videoDeepfake;
}

function saveSettings() {
  chrome.storage.local.set({ settings: currentSettings() });
}

let isOnYouTube = false;

function updateCheckButtonState() {
  const settings = currentSettings();
  const nothingSelected = settings.hateLang === "off" && !settings.falseContent && !settings.videoDeepfake;
  checkButton.disabled = !isOnYouTube || nothingSelected;
  checkButton.textContent = nothingSelected ? "Choose a check above" : "Check This Video";
}

// ── Result rendering ──────────────────────────────────────
function pct(value) {
  return `${Number(value).toFixed(1)}%`;
}

function formatSec(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// If a module has been "running" far longer than any real analysis should
// take, the background service worker most likely got shut down mid-request
// by Chrome (a Manifest V3 platform limit, not something this extension can
// fully prevent) — the backend may well have finished on its own with no
// one left to receive the response. Rather than spin forever with no way
// out, offer a manual retry once that's clearly happened.
const STUCK_AFTER_MS = 6 * 60 * 1000; // 6 minutes — real runs finish in ~1-2

function moduleCard(title, moduleKey, moduleState, renderBody, videoStartedAt) {
  const card = document.createElement("div");
  card.className = "result-card";
  const heading = document.createElement("div");
  heading.className = "title";
  heading.textContent = title;
  card.appendChild(heading);

  // Older stored runs (from before this retry feature existed) won't have
  // a per-module startedAt — fall back to the whole video's start time so
  // those don't stay stuck with no way out either.
  const referenceStart = (moduleState && moduleState.startedAt) || videoStartedAt;
  const isStuck = moduleState && (moduleState.status === "pending" || moduleState.status === "running") &&
    referenceStart && (Date.now() - referenceStart > STUCK_AFTER_MS);

  if (isStuck) {
    const stuckMsg = document.createElement("div");
    stuckMsg.className = "result-warning";
    stuckMsg.textContent = "⚠️ Taking much longer than expected — the browser may have paused this in the background. The check may have actually finished on the server already.";
    card.appendChild(stuckMsg);
    const retryBtn = document.createElement("button");
    retryBtn.className = "retry-btn";
    retryBtn.textContent = "Retry this check";
    retryBtn.addEventListener("click", async () => {
      retryBtn.disabled = true;
      retryBtn.textContent = "Retrying...";
      await chrome.runtime.sendMessage({ type: "RETRY_MODULE", payload: { videoId: currentVideoId, moduleKey } });
      await loadAndRenderState();
    });
    card.appendChild(retryBtn);
  } else if (!moduleState || moduleState.status === "pending" || moduleState.status === "running") {
    const loading = document.createElement("div");
    loading.className = "result-loading";
    loading.textContent = "Running... this can take a few minutes on first use.";
    card.appendChild(loading);
  } else if (moduleState.status === "error") {
    const err = document.createElement("div");
    err.className = "result-error";
    err.textContent = moduleState.error || "Something went wrong.";
    card.appendChild(err);
  } else if (moduleState.status === "done") {
    renderBody(card, moduleState.data);
    if (moduleState.languageWarning) {
      const warn = document.createElement("div");
      warn.className = "result-warning";
      warn.textContent = `⚠️ ${moduleState.languageWarning}`;
      card.appendChild(warn);
    }
  }
  return card;
}

function badge(text, cls) {
  const span = document.createElement("span");
  span.className = `badge ${cls}`;
  span.textContent = text;
  return span;
}

function meta(text) {
  const div = document.createElement("div");
  div.className = "result-meta";
  div.innerHTML = text;
  return div;
}

// A short list of flagged instances (timestamp/text + confidence), capped
// so the popup doesn't turn into an unreadable wall of text — the rest are
// still available by scrolling the panel injected on the YouTube page.
function instanceList(items, formatLine, limit = 3) {
  const wrap = document.createElement("div");
  wrap.className = "instance-list";
  items.slice(0, limit).forEach((item) => {
    const row = document.createElement("div");
    row.className = "instance-row";
    row.innerHTML = formatLine(item);
    wrap.appendChild(row);
  });
  if (items.length > limit) {
    const more = document.createElement("div");
    more.className = "instance-more";
    more.textContent = `+ ${items.length - limit} more`;
    wrap.appendChild(more);
  }
  return wrap;
}

// Flagged comments: shown as a plain count (not "N / total"), with an
// expandable list of which comments were actually flagged, not just a
// number.
function commentsDetail(count, items, formatLine) {
  const wrap = document.createElement("div");
  if (!count) return wrap;
  const details = document.createElement("details");
  details.className = "comment-details";
  const summary = document.createElement("summary");
  summary.textContent = `View flagged comment${count === 1 ? "" : "s"}`;
  details.appendChild(summary);
  const list = document.createElement("div");
  list.className = "instance-list";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "instance-row";
    row.innerHTML = formatLine(item);
    list.appendChild(row);
  });
  details.appendChild(list);
  wrap.appendChild(details);
  return wrap;
}

// ── Summary strip: one compact stat per enabled check, at a glance ──
function summaryCard(icon, label, valueText, cls) {
  const card = document.createElement("div");
  card.className = `summary-card ${cls}`;
  card.innerHTML = `<div class="summary-icon">${icon}</div><div class="summary-value">${valueText}</div><div class="summary-label">${label}</div>`;
  return card;
}

function buildSummaryStrip(state) {
  const cards = [];

  if (state.tamil && state.tamil.status === "done") {
    const isHate = state.tamil.data.verdict === "HATE SPEECH DETECTED";
    cards.push(summaryCard("🔊", "Hate Speech", isHate ? `${state.tamil.data.hate_chunks} segment${state.tamil.data.hate_chunks === 1 ? "" : "s"}` : "Safe", isHate ? "hate" : "safe"));
  }
  if (state.sinhala && state.sinhala.status === "done") {
    const summary = state.sinhala.data.summary || {};
    const isHate = summary.hate_segments > 0;
    cards.push(summaryCard("🔊", "Hate Speech", isHate ? `${summary.hate_segments} segment${summary.hate_segments === 1 ? "" : "s"}` : "Safe", isHate ? "hate" : "safe"));
    const isFake = summary.fake_segments > 0;
    cards.push(summaryCard("🎭", "Deepfake", isFake ? `${summary.fake_segments} segment${summary.fake_segments === 1 ? "" : "s"}` : "Authentic", isFake ? "hate" : "safe"));
  }
  const commentsState = state.tamilComments || state.sinhalaComments;
  if (commentsState && commentsState.status === "done") {
    const count = state.tamilComments ? state.tamilComments.data.hate_count : state.sinhalaComments.data.summary.hate_comments;
    cards.push(summaryCard("💬", "Comments", count ? `${count} flagged` : "Clean", count ? "hate" : "safe"));
  }
  if (state.falseContent && state.falseContent.status === "done") {
    const isFalse = String(state.falseContent.data.label || state.falseContent.data.predicted_label || "").toLowerCase() === "false";
    cards.push(summaryCard("📰", "False Content", isFalse ? "False" : "Not False", isFalse ? "hate" : "safe"));
  }
  if (state.videoDeepfake && state.videoDeepfake.status === "done") {
    const isFake = state.videoDeepfake.data.verdict === "FAKE";
    cards.push(summaryCard("🎬", "Video Deepfake", isFake ? "Fake detected" : "Real", isFake ? "hate" : "safe"));
  }

  if (!cards.length) return null;
  const strip = document.createElement("div");
  strip.className = "summary-strip";
  cards.forEach((c) => strip.appendChild(c));
  return strip;
}

function renderResults(state) {
  resultsEl.innerHTML = "";
  if (!state) return;

  const strip = buildSummaryStrip(state);
  if (strip) resultsEl.appendChild(strip);

  if (state.tamil) {
    resultsEl.appendChild(moduleCard("Tamil Hate Speech", "tamil", state.tamil, (card, data) => {
      const isHate = data.verdict === "HATE SPEECH DETECTED";
      card.appendChild(badge(isHate ? "🔴 HATE SPEECH DETECTED" : "🟢 VIDEO IS SAFE", isHate ? "hate" : "safe"));
      if (data.hate_instances && data.hate_instances.length) {
        card.appendChild(instanceList(data.hate_instances, (i) =>
          `<b>${i.timestamp}</b> · ${pct(i.confidence)} — "${i.sentence}"`
        ));
      }
    }, state.startedAt));
  }

  if (state.tamilComments) {
    resultsEl.appendChild(moduleCard("Tamil Comments", "tamilComments", state.tamilComments, (card, data) => {
      const hasHate = data.hate_count > 0;
      card.appendChild(badge(hasHate ? `${data.hate_count} flagged` : "Clean", hasHate ? "hate" : "safe"));
      card.appendChild(commentsDetail(data.hate_count, data.hate_comments, (c) =>
        `${pct(c.confidence)} — "${c.text}"`
      ));
    }, state.startedAt));
  }

  if (state.sinhala) {
    resultsEl.appendChild(moduleCard("Sinhala Hate Speech + Deepfake", "sinhala", state.sinhala, (card, data) => {
      const summary = data.summary || {};
      const hasHate = summary.hate_segments > 0;
      const hasFake = summary.fake_segments > 0;
      card.appendChild(badge(hasHate ? "🔴 Hate Speech" : "🟢 No Hate Speech", hasHate ? "hate" : "safe"));
      card.appendChild(badge(hasFake ? "🟡 Deepfake Detected" : "🟢 Audio Authentic", hasFake ? "hate" : "safe"));
      const hateSegments = (data.segments || []).filter((s) => s.hate_speech);
      if (hateSegments.length) {
        card.appendChild(instanceList(hateSegments, (s) =>
          `<b>${s.start_fmt}–${s.end_fmt}</b> · ${pct(s.hate_confidence * 100)} — "${(s.transcript || "").slice(0, 80)}"`
        ));
      }
      const fakeSegments = (data.segments || []).filter((s) => !s.audio_authentic);
      if (fakeSegments.length) {
        card.appendChild(instanceList(fakeSegments, (s) =>
          `<b>${s.start_fmt}–${s.end_fmt}</b> · ${pct(s.authenticity_confidence * 100)} suspected deepfake`
        ));
      }
    }, state.startedAt));
  }

  if (state.sinhalaComments) {
    resultsEl.appendChild(moduleCard("Sinhala Comments", "sinhalaComments", state.sinhalaComments, (card, data) => {
      const summary = data.summary || {};
      const hasHate = summary.hate_comments > 0;
      const flagged = (data.comments || []).filter((c) => c.is_offensive);
      card.appendChild(badge(hasHate ? `${summary.hate_comments} flagged` : "Clean", hasHate ? "hate" : "safe"));
      card.appendChild(commentsDetail(summary.hate_comments, flagged, (c) =>
        `${pct(c.confidence * 100)} — <b>${c.author}</b>: "${c.text}"`
      ));
    }, state.startedAt));
  }

  if (state.falseContent) {
    resultsEl.appendChild(moduleCard("Sinhala False Content", "falseContent", state.falseContent, (card, data) => {
      const label = data.label || data.predicted_label || "Unknown";
      const isFalse = String(label).toLowerCase() === "false";
      card.appendChild(badge(isFalse ? "🔴 False" : "🟢 Not False", isFalse ? "hate" : "safe"));
      card.appendChild(meta(`Confidence: ${pct((data.confidence || 0) * 100)}`));
      if (data.warnings && data.warnings.length) {
        card.appendChild(meta(data.warnings.join(" ")));
      }
    }, state.startedAt));
  }

  if (state.videoDeepfake) {
    resultsEl.appendChild(moduleCard("Video Deepfake Detection", "videoDeepfake", state.videoDeepfake, (card, data) => {
      const isFake = data.verdict === "FAKE";
      card.appendChild(badge(isFake ? "🔴 LIKELY DEEPFAKE" : "🟢 LIKELY REAL", isFake ? "hate" : "safe"));
      card.appendChild(meta(`Fake content: ${pct(data.fake_percentage || 0)} of the video`));
      if (data.fake_segments && data.fake_segments.length) {
        card.appendChild(instanceList(data.fake_segments, (s) =>
          `<b>${formatSec(s.start_time)}–${formatSec(s.end_time)}</b> · ${pct((s.peak_confidence || s.confidence || 0) * 100)} confidence`
        ));
      }
    }, state.startedAt));
  }
}

// ── Storage sync (so results appear even if the popup was reopened) ──
function storageKey(videoId) {
  return `analysis_${videoId}`;
}

async function loadAndRenderState() {
  if (!currentVideoId) return;
  const stored = await chrome.storage.local.get(storageKey(currentVideoId));
  const state = stored[storageKey(currentVideoId)];
  renderResults(state);
  if (state && state.status === "running") {
    statusEl.textContent = "Analysis running in the background — you can close this popup.";
  } else if (state && state.status === "done") {
    statusEl.textContent = "Analysis complete.";
  } else {
    statusEl.textContent = "";
  }
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local" || !currentVideoId) return;
  if (changes[storageKey(currentVideoId)]) {
    loadAndRenderState();
  }
});

// ── Init ───────────────────────────────────────────────────
async function init() {
  const stored = await chrome.storage.local.get("settings");
  applySettings(stored.settings || {});

  hateSpeechEnabledEl.addEventListener("change", () => {
    const enabled = hateSpeechEnabledEl.checked;
    hateSpeechBodyEl.hidden = !enabled;
    if (enabled && !document.querySelector('input[name="hateLang"]:checked')) {
      document.querySelector('input[name="hateLang"][value="tamil"]').checked = true;
    }
    saveSettings();
    updateCheckButtonState();
  });
  document.querySelectorAll('input[name="hateLang"]').forEach((el) => {
    el.addEventListener("change", saveSettings);
  });
  includeCommentsEl.addEventListener("change", saveSettings);
  falseContentEl.addEventListener("change", () => {
    saveSettings();
    updateCheckButtonState();
  });
  videoDeepfakeEl.addEventListener("change", () => {
    saveSettings();
    updateCheckButtonState();
  });

  const tab = await activeTab();
  if (!tab || !tab.url || !(tab.url.includes("youtube.com/watch") || tab.url.includes("youtu.be/") || tab.url.includes("youtube.com/shorts"))) {
    pageStatusEl.textContent = "Open a YouTube video to check it.";
    updateCheckButtonState();
    return;
  }
  isOnYouTube = true;

  try {
    const response = await collectPageData(tab.id);
    if (!response || !response.ok) throw new Error("Could not read this page.");
    currentPageData = response.data;
    currentVideoId = response.data.video_id;
    currentTabUrl = response.data.url;
    pageStatusEl.textContent = "";
    videoInfoEl.hidden = false;
    videoTitleEl.textContent = response.data.title || "This video";
    updateCheckButtonState();
    await loadAndRenderState();
  } catch (error) {
    pageStatusEl.textContent = "Could not read this YouTube page — try reloading it.";
    isOnYouTube = false;
    updateCheckButtonState();
  }
}

checkButton.addEventListener("click", async () => {
  const settings = currentSettings();
  saveSettings();
  statusEl.textContent = "Starting analysis in the background...";
  resultsEl.innerHTML = "";

  await chrome.runtime.sendMessage({
    type: "START_ANALYSIS",
    payload: {
      videoId: currentVideoId,
      tabUrl: currentTabUrl,
      pageData: currentPageData,
      settings
    }
  });

  await loadAndRenderState();
});

init();

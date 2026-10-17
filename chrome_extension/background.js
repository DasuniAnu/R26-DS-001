// Runs the actual (multi-minute) YouTube analyses. This lives in the
// background service worker, not the popup, because Chrome tears down a
// popup's JS the instant it loses focus — which would silently abort an
// in-flight fetch. Progress and results are written to chrome.storage.local
// keyed by video id, so the popup can be closed and reopened at any time
// without losing anything; it just re-reads storage.

const GATEWAY_URL = "http://127.0.0.1:8000";
const FALSE_CONTENT_URL = "http://127.0.0.1:5002";
const DEEPFAKE_BACKEND_URL = "http://127.0.0.1:5003";

function storageKey(videoId) {
  return `analysis_${videoId}`;
}

async function getState(videoId) {
  const stored = await chrome.storage.local.get(storageKey(videoId));
  return stored[storageKey(videoId)] || null;
}

async function setState(videoId, state) {
  await chrome.storage.local.set({ [storageKey(videoId)]: state });
}

async function patchModule(videoId, moduleKey, patch) {
  const state = (await getState(videoId)) || {};
  state[moduleKey] = { ...(state[moduleKey] || {}), ...patch };
  await setState(videoId, state);
  return state;
}

async function markOverallStatus(videoId) {
  const state = await getState(videoId);
  if (!state) return;
  const moduleKeys = ["tamil", "tamilComments", "sinhala", "sinhalaComments", "falseContent", "videoDeepfake"];
  const active = moduleKeys.filter((key) => state[key]);
  const allSettled = active.every((key) => state[key].status === "done" || state[key].status === "error");
  state.status = allSettled ? "done" : "running";
  await setState(videoId, state);
}

// ── Service-worker keepalive ─────────────────────────────────
// Manifest V3 service workers are supposed to stay alive while a fetch()
// is in flight, but Chrome also enforces a hard maximum lifetime that can
// still kill one mid-request on a long YouTube analysis (several minutes).
// If that happens, the backend finishes the work and responds, but there's
// no JS left running to receive it, so the stored status stays "running"
// forever even though the server-side job actually succeeded. A recurring
// alarm is the documented way to keep resetting the idle-shutdown timer;
// it can't override the hard cap, which is why runModule() below also
// stamps a startedAt time so the popup can offer a manual retry if a
// module is ever stuck unreasonably long.
const KEEPALIVE_ALARM = "truthlense-keepalive";
chrome.alarms.create(KEEPALIVE_ALARM, { periodInMinutes: 1 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === KEEPALIVE_ALARM) {
    // Touching storage is enough to count as activity; nothing to do here.
  }
});

async function runModule(videoId, moduleKey, url, body) {
  await patchModule(videoId, moduleKey, { status: "running", data: null, error: null, languageWarning: null, startedAt: Date.now() });
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `Request failed (HTTP ${response.status})`);
    }
    await patchModule(videoId, moduleKey, { status: "done", data: payload, error: null });
  } catch (error) {
    await patchModule(videoId, moduleKey, { status: "error", data: null, error: error.message });
  }
  await markOverallStatus(videoId);
}

// ── Wrong-language safeguard ─────────────────────────────────
// Tamil and Sinhala use entirely separate Unicode script blocks. Both
// modules already transcribe the video's audio internally as part of
// their normal analysis — this just inspects that real transcript
// afterwards for a script mismatch (e.g. Sinhala audio run through the
// Tamil module because of a misclick). It never changes the manual
// selection or the actual result — it only adds a warning alongside it.
const TAMIL_RANGE = [0x0b80, 0x0bff];
const SINHALA_RANGE = [0x0d80, 0x0dff];
const MISMATCH_MIN_CHARS = 40;
const MISMATCH_THRESHOLD = 0.15;

function scriptFractions(text) {
  const clean = String(text || "").replace(/\s/g, "");
  if (!clean.length) return { tamil: 0, sinhala: 0, total: 0 };
  let tamilCount = 0;
  let sinhalaCount = 0;
  for (const ch of clean) {
    const code = ch.codePointAt(0);
    if (code >= TAMIL_RANGE[0] && code <= TAMIL_RANGE[1]) tamilCount++;
    else if (code >= SINHALA_RANGE[0] && code <= SINHALA_RANGE[1]) sinhalaCount++;
  }
  return { tamil: tamilCount / clean.length, sinhala: sinhalaCount / clean.length, total: clean.length };
}

function detectLanguageMismatch(selectedLang, transcriptText) {
  const { tamil, sinhala, total } = scriptFractions(transcriptText);
  if (total < MISMATCH_MIN_CHARS) return null; // not enough transcript to judge either way
  if (selectedLang === "tamil" && sinhala >= MISMATCH_THRESHOLD) {
    return "This audio looks like Sinhala, not Tamil — try the Sinhala check instead.";
  }
  if (selectedLang === "sinhala" && tamil >= MISMATCH_THRESHOLD) {
    return "This audio looks like Tamil, not Sinhala — try the Tamil check instead.";
  }
  return null;
}

async function checkTamilLanguageMismatch(videoId) {
  const state = await getState(videoId);
  const moduleState = state && state.tamil;
  if (!moduleState || moduleState.status !== "done") return;
  const transcript = (moduleState.data.chunks || []).map((c) => c.transcript || "").join(" ");
  const warning = detectLanguageMismatch("tamil", transcript);
  if (warning) await patchModule(videoId, "tamil", { languageWarning: warning });
}

async function checkSinhalaLanguageMismatch(videoId) {
  const state = await getState(videoId);
  const moduleState = state && state.sinhala;
  if (!moduleState || moduleState.status !== "done") return;
  const transcript = (moduleState.data.segments || []).map((s) => s.transcript || "").join(" ");
  const warning = detectLanguageMismatch("sinhala", transcript);
  if (warning) await patchModule(videoId, "sinhala", { languageWarning: warning });
}

// One function per possible module, shared between a fresh run and a
// manual retry of just one stuck module, so there's a single place that
// knows each module's URL/body — no logic duplicated between the two.
function runOneModule(moduleKey, videoId, tabUrl, pageData) {
  switch (moduleKey) {
    case "tamil":
      return runModule(videoId, "tamil", `${GATEWAY_URL}/analyze_youtube`, {
        youtube_url: tabUrl, use_lexicon: true
      }).then(() => checkTamilLanguageMismatch(videoId));
    case "tamilComments":
      return runModule(videoId, "tamilComments", `${GATEWAY_URL}/analyze_youtube_comments`, {
        youtube_url: tabUrl, use_lexicon: true
      });
    case "sinhala":
      return runModule(videoId, "sinhala", `${GATEWAY_URL}/api/youtube/analyze`, {
        url: tabUrl
      }).then(() => checkSinhalaLanguageMismatch(videoId));
    case "sinhalaComments":
      return runModule(videoId, "sinhalaComments", `${GATEWAY_URL}/api/youtube/comments`, {
        url: tabUrl
      });
    case "falseContent":
      return runModule(videoId, "falseContent", `${FALSE_CONTENT_URL}/api/predict-youtube`, {
        url: tabUrl, video_id: videoId, page_data: pageData
      });
    case "videoDeepfake":
      // Much heavier than the other modules (downloads the full video and
      // runs frame-by-frame analysis) — genuinely takes minutes, not
      // seconds, so this is the module most likely to hit the "stuck,
      // retry" path further down, through no fault of its own.
      return runModule(videoId, "videoDeepfake", `${DEEPFAKE_BACKEND_URL}/analyze_youtube`, {
        url: tabUrl
      });
    default:
      return Promise.resolve();
  }
}

async function startAnalysis({ videoId, tabUrl, pageData, settings }) {
  const initialState = {
    status: "running",
    startedAt: Date.now(),
    videoUrl: tabUrl,
    pageData,
    settings
  };
  if (settings.hateLang === "tamil") {
    initialState.tamil = { status: "pending" };
    if (settings.includeComments) initialState.tamilComments = { status: "pending" };
  }
  if (settings.hateLang === "sinhala") {
    initialState.sinhala = { status: "pending" };
    if (settings.includeComments) initialState.sinhalaComments = { status: "pending" };
  }
  if (settings.falseContent) {
    initialState.falseContent = { status: "pending" };
  }
  if (settings.videoDeepfake) {
    initialState.videoDeepfake = { status: "pending" };
  }
  await setState(videoId, initialState);

  // Jobs run concurrently and each writes its own result independently —
  // one module failing (e.g. Sinhala's ASR model still downloading) never
  // blocks or hides the others' results.
  const jobs = Object.keys(initialState)
    .filter((key) => ["tamil", "tamilComments", "sinhala", "sinhalaComments", "falseContent", "videoDeepfake"].includes(key))
    .map((key) => runOneModule(key, videoId, tabUrl, pageData));

  await Promise.allSettled(jobs);
}

async function retryModule({ videoId, moduleKey }) {
  const state = await getState(videoId);
  if (!state) return;
  await runOneModule(moduleKey, videoId, state.videoUrl, state.pageData);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === "START_ANALYSIS") {
    startAnalysis(message.payload).catch((error) => {
      console.error("TruthLenseAI analysis failed to start:", error);
    });
    sendResponse({ ok: true });
    return true;
  }
  if (message && message.type === "RETRY_MODULE") {
    retryModule(message.payload).catch((error) => {
      console.error("TruthLenseAI retry failed:", error);
    });
    sendResponse({ ok: true });
    return true;
  }
});

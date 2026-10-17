const API_URL = "http://127.0.0.1:5000/api/predict-youtube";

const statusEl = document.getElementById("status");
const checkButton = document.getElementById("check");
const resultEl = document.getElementById("result");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const fieldsEl = document.getElementById("fields");
const reviewEl = document.getElementById("review");
const evidenceEl = document.getElementById("evidence");
const probabilitiesEl = document.getElementById("probabilities");
const warningsEl = document.getElementById("warnings");
const CLASS_ORDER = ["False", "Not False"];

function setStatus(message) {
  statusEl.textContent = message;
}

function activeTab() {
  return chrome.tabs.query({ active: true, currentWindow: true }).then((tabs) => tabs[0]);
}

function collectPageData(tabId) {
  return chrome.tabs.sendMessage(tabId, { type: "COLLECT_YOUTUBE_DATA" });
}

function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function renderResult(result) {
  resultEl.hidden = false;
  const label = result.label || result.predicted_label || "Unknown";
  labelEl.textContent = label;
  labelEl.className = `label ${String(label).toLowerCase().replace(/\s+/g, "-")}`;
  confidenceEl.textContent = `Confidence: ${percent(result.confidence || 0)}`;

  const used = result.used_fields || {};
  fieldsEl.textContent = `Used: title=${!!used.title}, description=${!!used.description}, transcript=${!!used.transcript}`;
  reviewEl.textContent = `Review: ${result.review_recommendation || "review_optional"}`;
  evidenceEl.textContent = (result.evidence_signals || [])
    .map((signal) => `${signal.name}=${signal.status}`)
    .join("; ");

  probabilitiesEl.innerHTML = "";
  const probabilities = result.class_probabilities || {};
  CLASS_ORDER.forEach((label) => {
    const item = document.createElement("li");
    item.textContent = `${label}: ${percent(probabilities[label])}`;
    probabilitiesEl.appendChild(item);
  });

  warningsEl.textContent = (result.warnings || []).join(" ");
}

async function checkVideo() {
  checkButton.disabled = true;
  resultEl.hidden = true;
  setStatus("Collecting YouTube page data...");

  try {
    const tab = await activeTab();
    if (!tab || !tab.url || !tab.url.includes("youtube")) {
      throw new Error("Open a YouTube video tab first.");
    }

    const response = await collectPageData(tab.id);
    if (!response || !response.ok) {
      throw new Error("Could not read the YouTube page.");
    }

    setStatus("Sending to local Sinhala model...");
    const apiResponse = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: response.data.url,
        video_id: response.data.video_id,
        page_data: response.data
      })
    });

    const payload = await apiResponse.json();
    if (!apiResponse.ok) {
      throw new Error(payload.error || "Local model API returned an error.");
    }

    setStatus("Prediction complete.");
    renderResult(payload);
  } catch (error) {
    setStatus(error.message);
  } finally {
    checkButton.disabled = false;
  }
}

checkButton.addEventListener("click", checkVideo);

/* Tamil Hate Speech Detector — popup script
   Reads state from chrome.storage.local (written by content.js)
   and updates the UI.
*/

function pct(val) {
  return (val * 100).toFixed(1) + '%';
}

function labelClass(prediction) {
  return prediction === 'hate' ? 'val-hate' : 'val-safe';
}

function renderStats(data) {
  // ── Video section ─────────────────────────────────
  const vr = data.videoResult;
  if (vr) {
    document.getElementById('transcription').textContent =
      vr.transcription || '[no speech detected]';

    const textEl = document.getElementById('textResult');
    textEl.textContent = `${vr.text_prediction} (${pct(vr.text_confidence)})`;
    textEl.className = `row-value ${labelClass(vr.text_prediction)}`;

    const audioEl = document.getElementById('audioResult');
    audioEl.textContent = `${vr.audio_prediction} (${pct(vr.audio_confidence)})`;
    audioEl.className = `row-value ${labelClass(vr.audio_prediction)}`;

    const badge = document.getElementById('fusionBadge');
    if (vr.fusion_prediction === 'hate') {
      badge.className = 'badge badge-hate';
      badge.textContent = `⚠ HATE  ${pct(vr.fusion_confidence)}`;
    } else {
      badge.className = 'badge badge-safe';
      badge.textContent = `✓ SAFE  ${pct(vr.fusion_confidence)}`;
    }
  }

  // ── Comments section ──────────────────────────────
  const cs = data.commentStats || { total: 0, hate: 0, safe: 0 };
  document.getElementById('totalCount').textContent = cs.total;
  document.getElementById('hateCount').textContent  = cs.hate;
  document.getElementById('safeCount').textContent  = cs.safe;

  // ── Last updated ──────────────────────────────────
  if (data.lastUpdated) {
    document.getElementById('lastUpdated').textContent =
      'Updated ' + data.lastUpdated;
  }

  // ── Toggle state ──────────────────────────────────
  const toggle = document.getElementById('toggleSwitch');
  toggle.checked = data.isEnabled !== false;
}

function loadAndRender() {
  chrome.storage.local.get(
    ['videoResult', 'commentStats', 'lastUpdated', 'isEnabled'],
    renderStats
  );
}

// Initial load
loadAndRender();

// Refresh button
document.getElementById('refreshBtn').addEventListener('click', loadAndRender);

// Toggle switch
document.getElementById('toggleSwitch').addEventListener('change', (e) => {
  const isEnabled = e.target.checked;
  chrome.storage.local.set({ isEnabled });
  // content.js listens to storage changes and reacts
});

// Auto-refresh every 5 seconds while popup is open
setInterval(loadAndRender, 5000);

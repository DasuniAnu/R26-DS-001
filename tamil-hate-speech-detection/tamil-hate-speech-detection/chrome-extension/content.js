/* Tamil Hate Speech Detector — content script
   Anutthara S.A.D | IT22217554 | SLIIT
   Runs on every YouTube page.
*/
console.log('%c[THSD] Tamil Hate Speech Detector LOADED ✓', 'color:green;font-size:14px;font-weight:bold');

(function () {
  'use strict';

  const API = 'http://localhost:5000';
  const CHUNK_SECONDS = 30;
  const LOG = (msg) => console.log('[THSD]', msg);

  let enabled = true;
  let audioCapturing = false;
  let audioCtx = null;

  const processedEls = new WeakSet();
  let commentStats = { total: 0, hate: 0, safe: 0 };
  let videoResult = null;

  // ─────────────────────────────────────────────────────
  // WAV encoder  (float32 PCM → WAV Blob, no ffmpeg needed)
  // ─────────────────────────────────────────────────────
  function float32ToWav(samples, sampleRate) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buf);

    const wr = (off, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
    };

    wr(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    wr(8, 'WAVE');
    wr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);              // PCM
    view.setUint16(22, 1, true);              // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    wr(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let off = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      off += 2;
    }
    return new Blob([buf], { type: 'audio/wav' });
  }

  // ─────────────────────────────────────────────────────
  // VIDEO AUDIO — banner
  // ─────────────────────────────────────────────────────
  function showVideoBanner(result) {
    let banner = document.getElementById('thsd-video-banner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'thsd-video-banner';
      Object.assign(banner.style, {
        position: 'fixed', top: '0', left: '0', right: '0',
        zIndex: '2147483647', padding: '10px 20px',
        fontFamily: '-apple-system, Roboto, Arial, sans-serif',
        fontSize: '13px', fontWeight: '600',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        gap: '12px', boxShadow: '0 2px 10px rgba(0,0,0,0.35)',
        lineHeight: '1.4'
      });
      document.body.prepend(banner);
    }

    const tc = (result.text_confidence * 100).toFixed(1);
    const ac = (result.audio_confidence * 100).toFixed(1);
    const fc = (result.fusion_confidence * 100).toFixed(1);

    if (result.fusion_prediction === 'hate') {
      banner.style.background = '#dc2626';
      banner.style.color = '#fff';
      banner.innerHTML = `
        <span>⚠️ HATE SPEECH DETECTED IN VIDEO</span>
        <span style="opacity:.85;font-weight:400">
          Text: ${result.text_prediction} (${tc}%) &nbsp;|&nbsp;
          Audio: ${result.audio_prediction} (${ac}%) &nbsp;|&nbsp;
          Fusion: ${fc}% hate
        </span>
        <span style="cursor:pointer;margin-left:8px;opacity:.7"
              onclick="document.getElementById('thsd-video-banner').remove()">✕</span>
      `;
    } else {
      banner.style.background = '#16a34a';
      banner.style.color = '#fff';
      banner.innerHTML = `
        <span>✅ Video audio appears safe (${fc}% confidence)</span>
        <span style="cursor:pointer;margin-left:8px;opacity:.7"
              onclick="document.getElementById('thsd-video-banner').remove()">✕</span>
      `;
    }
  }

  // ─────────────────────────────────────────────────────
  // VIDEO AUDIO — send 30-second chunk to Flask
  // ─────────────────────────────────────────────────────
  function sendAudioChunk(pcm, sampleRate) {
    const wav = float32ToWav(pcm, sampleRate);
    const form = new FormData();
    form.append('audio', wav, 'chunk.wav');

    fetch(`${API}/predict_fusion_video`, { method: 'POST', body: form })
      .then(r => r.json())
      .then(result => {
        videoResult = result;
        showVideoBanner(result);
        saveStats();
        LOG(`Video chunk analysed — fusion: ${result.fusion_prediction} (${(result.fusion_confidence * 100).toFixed(1)}%)`);
      })
      .catch(e => LOG(`Video API error: ${e.message}`));
  }

  // ─────────────────────────────────────────────────────
  // VIDEO AUDIO — capture via Web Audio API
  // ─────────────────────────────────────────────────────
  function startAudioCapture() {
    if (audioCapturing || !enabled) return;

    const video = document.querySelector('video');
    if (!video) {
      setTimeout(startAudioCapture, 2000);
      return;
    }

    try {
      const stream = video.captureStream ? video.captureStream()
                   : video.mozCaptureStream ? video.mozCaptureStream()
                   : null;
      if (!stream) { LOG('captureStream not available'); return; }

      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) { LOG('No audio tracks in stream'); return; }

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const audioStream = new MediaStream(audioTracks);
      const source = audioCtx.createMediaStreamSource(audioStream);

      // ScriptProcessorNode collects PCM samples
      const bufSize = 4096;
      const processor = audioCtx.createScriptProcessor(bufSize, 1, 1);
      const accumulated = [];

      const capturedSampleRate = audioCtx.sampleRate;
      processor.onaudioprocess = (e) => {
        if (!enabled || !audioCtx) return;
        const data = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < data.length; i++) accumulated.push(data[i]);

        const needed = Math.floor(capturedSampleRate * CHUNK_SECONDS);
        if (accumulated.length >= needed) {
          const chunk = new Float32Array(accumulated.splice(0, needed));
          sendAudioChunk(chunk, capturedSampleRate);
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
      audioCapturing = true;
      LOG('Audio capture started');
    } catch (e) {
      LOG(`Audio capture failed: ${e.message}`);
    }
  }

  // ─────────────────────────────────────────────────────
  // COMMENTS — multi-selector element finder
  // ─────────────────────────────────────────────────────
  const COMMENT_SELECTORS = [
    // Desktop YouTube (www.youtube.com)
    'ytd-comment-renderer #content-text',
    '#content-text.ytd-comment-renderer',
    '#comments #content-text',
    'ytd-comment-view-model #content-text',
    // Mobile YouTube (m.youtube.com)
    'ytm-comment-renderer .comment-text',
    'ytm-comment-renderer span',
    '.comment-renderer-text',
    // Fallback — works on both
    '#content-text',
  ];

  function findCommentElements() {
    for (const sel of COMMENT_SELECTORS) {
      const els = document.querySelectorAll(sel);
      if (els.length > 0) {
        console.log(`[THSD] Selector "${sel}" → ${els.length} elements`);
        return Array.from(els);
      }
    }
    return [];
  }

  // ─────────────────────────────────────────────────────
  // COMMENTS — hide hate, insert warning div
  // ─────────────────────────────────────────────────────
  function processComments() {
    if (!enabled) return;

    const commentEls = findCommentElements();
    console.log(`[THSD] Comments found: ${commentEls.length}`);

    // batch is a plain array; elMap is parallel array (same index = same comment)
    const batch  = [];
    const elMap  = [];

    commentEls.forEach(el => {
      if (processedEls.has(el)) return;
      const text = el.innerText.trim();
      if (!text) return;            // ← don't mark empty/loading elements as done
      processedEls.add(el);        // ← only mark when text actually exists
      batch.push(text);
      elMap.push(el);
    });

    console.log(`[THSD] New comments to analyse: ${batch.length}`);
    if (batch.length === 0) return;

    fetch(`${API}/predict_batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: batch })
    })
      .then(r => r.json())
      .then(data => {
        commentStats.total += data.total;
        commentStats.hate  += data.hate_count;
        commentStats.safe  += data.safe_count;

        data.results.forEach((result, idx) => {
          const el = elMap[idx];     // ← index-based lookup, never mismatches
          if (!el) return;

          const thread = el.closest('ytd-comment-thread-renderer');
          if (!thread) return;

          if (result.label === 'hate') {
            thread.style.display = 'none';

            const warning = document.createElement('div');
            Object.assign(warning.style, {
              background: '#fef2f2', border: '1px solid #fca5a5',
              borderRadius: '8px', padding: '10px 16px', margin: '6px 0',
              fontSize: '13px', color: '#dc2626',
              fontFamily: '-apple-system, Roboto, Arial, sans-serif',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px'
            });

            const msg = document.createElement('span');
            msg.textContent = '⚠️ Hate speech detected — comment hidden by Tamil Hate Speech Detector';

            const showBtn = document.createElement('button');
            showBtn.textContent = 'Show anyway';
            Object.assign(showBtn.style, {
              background: '#fff', border: '1px solid #fca5a5',
              borderRadius: '4px', padding: '3px 10px',
              fontSize: '12px', color: '#dc2626',
              cursor: 'pointer', whiteSpace: 'nowrap', flexShrink: '0'
            });

            let revealed = false;
            showBtn.addEventListener('click', () => {
              if (!revealed) {
                thread.style.display = '';
                showBtn.textContent = 'Hide again';
                msg.textContent = '⚠️ Hate speech detected — comment shown below';
                warning.style.background = '#fff7ed';
                warning.style.borderColor = '#fdba74';
                warning.style.color = '#c2410c';
                revealed = true;
              } else {
                thread.style.display = 'none';
                showBtn.textContent = 'Show anyway';
                msg.textContent = '⚠️ Hate speech detected — comment hidden by Tamil Hate Speech Detector';
                warning.style.background = '#fef2f2';
                warning.style.borderColor = '#fca5a5';
                warning.style.color = '#dc2626';
                revealed = false;
              }
            });

            warning.appendChild(msg);
            warning.appendChild(showBtn);
            thread.before(warning);
          }
        });

        saveStats();
        LOG(`Comments: ${data.total} sent, ${data.hate_count} hidden`);
      })
      .catch(e => LOG(`Comment API error: ${e.message}`));
  }

  // ─────────────────────────────────────────────────────
  // COMMENTS — observer + periodic scan + scroll trigger
  // ─────────────────────────────────────────────────────
  function startCommentObserver() {
    // 1. Wait 5 s then scan (YouTube loads comments lazily after page render)
    setTimeout(processComments, 5000);

    // 2. Scan every 3 s to catch comments loaded as user scrolls
    setInterval(processComments, 3000);

    // 3. MutationObserver on the comments container for real-time additions
    const observer = new MutationObserver(() => processComments());

    function attachObserver() {
      const section = document.querySelector('ytd-comments');
      if (section) {
        observer.observe(section, { childList: true, subtree: true });
        LOG('MutationObserver attached to ytd-comments');
      } else {
        setTimeout(attachObserver, 1500);
      }
    }
    attachObserver();

    // 4. Scroll detection — YouTube renders comments only after the user
    //    scrolls past the video player (~500 px down the page)
    let scrollTriggered = false;
    window.addEventListener('scroll', () => {
      if (scrollTriggered) return;
      const scrollY = window.scrollY || document.documentElement.scrollTop;
      if (scrollY > 500) {
        scrollTriggered = true;
        LOG('User scrolled past video — scanning comments in 2 s');
        setTimeout(processComments, 2000);
      }
    }, { passive: true });
  }

  // ─────────────────────────────────────────────────────
  // STORAGE helpers
  // ─────────────────────────────────────────────────────
  function saveStats() {
    chrome.storage.local.set({
      commentStats,
      videoResult,
      lastUpdated: new Date().toLocaleTimeString(),
      isEnabled: enabled
    });
  }

  // React to popup toggle
  chrome.storage.onChanged.addListener((changes) => {
    if ('isEnabled' in changes) {
      enabled = changes.isEnabled.newValue;
      if (!enabled) {
        const banner = document.getElementById('thsd-video-banner');
        if (banner) banner.remove();
        if (audioCtx) { audioCtx.close(); audioCtx = null; audioCapturing = false; }
        LOG('Extension disabled');
      } else {
        init();
        LOG('Extension enabled');
      }
    }
  });

  // ─────────────────────────────────────────────────────
  // INIT — called on page load and YouTube SPA navigation
  // ─────────────────────────────────────────────────────
  function init() {
    if (!enabled) return;

    const isVideoPage = window.location.pathname === '/watch';

    if (isVideoPage) {
      startCommentObserver();
      // Give the video element time to appear before capturing
      setTimeout(startAudioCapture, 4000);
    }
  }

  // Load saved enabled state, then initialise
  chrome.storage.local.get(['isEnabled'], (res) => {
    enabled = res.isEnabled !== false; // default true
    init();
  });

  // YouTube is a SPA — re-run on every navigation
  document.addEventListener('yt-navigate-finish', () => {
    audioCapturing = false;
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    commentStats = { total: 0, hate: 0, safe: 0 };
    videoResult = null;
    saveStats(); // clear popup immediately on page change
    init();
  });

})();

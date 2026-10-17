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

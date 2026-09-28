# =============================================================
# YOUTUBE ANALYSIS UTILITY
# Downloads YouTube audio, splits into 10s segments for analysis
# =============================================================

import os
import uuid
import tempfile
import re
import numpy as np
import librosa
import yt_dlp
import requests
from urllib.parse import urlparse, parse_qs

SAMPLE_RATE    = 16000
SEGMENT_DURATION = 10                          # seconds
SEGMENT_SAMPLES  = SAMPLE_RATE * SEGMENT_DURATION  # 160,000
MAX_VIDEO_DURATION = 600                       # 10 minutes

TEMP_DIR = os.path.join(tempfile.gettempdir(), "yt_hate_detect")
os.makedirs(TEMP_DIR, exist_ok=True)

def format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"

def get_video_id(url: str) -> str:
    """Extract and validate a video ID from common YouTube URL formats."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        video_id = None

        if host == "youtu.be":
            video_id = parsed.path.lstrip("/").split("/")[0]
        elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
            if parsed.path == '/watch':
                video_id = parse_qs(parsed.query).get("v", [None])[0]
            elif parsed.path.startswith(('/embed/', '/v/', '/shorts/', '/live/')):
                video_id = parsed.path.strip('/').split('/')[1]

        if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            return video_id
    except Exception:
        pass
    return None

def get_video_metadata(url: str) -> dict:
    """Fetch video info without downloading."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return {
            "title":     info.get("title", "Unknown"),
            "duration":  info.get("duration", 0),
            "channel":   info.get("channel", info.get("uploader", "Unknown")),
            "thumbnail": info.get("thumbnail", ""),
            "url":       url,
        }
    except Exception as e:
        raise ValueError(f"Invalid URL or video unavailable. Details: {str(e)}")

def download_youtube_audio(url: str):
    """Download best audio track as WAV. Returns (filepath, metadata)."""
    uid = uuid.uuid4().hex
    out_template = os.path.join(TEMP_DIR, uid + ".%(ext)s")
    
    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            meta = {
                "title":     info.get("title", "Unknown"),
                "duration":  info.get("duration", 0),
                "channel":   info.get("channel", info.get("uploader", "Unknown")),
                "thumbnail": info.get("thumbnail", ""),
                "url":       url,
            }
            filepath = os.path.join(TEMP_DIR, uid + ".wav")
            if not os.path.exists(filepath):
                if "requested_downloads" in info and len(info["requested_downloads"]) > 0:
                    filepath = info["requested_downloads"][0]["filepath"]
        return filepath, meta
    except Exception as e:
        raise RuntimeError(f"Audio download failed. Make sure FFmpeg is installed or try again later. Details: {str(e)}")

import torch

_vad_model = None
_vad_utils = None

def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
    return _vad_model, _vad_utils

def analyze_youtube_video(url: str):
    metadata = get_video_metadata(url)
    duration = metadata["duration"]
    if duration <= 0:
        raise ValueError("Could not determine video duration or video has no audio.")
    if duration > MAX_VIDEO_DURATION:
        raise ValueError(
            f"Video too long ({format_time(duration)}). "
            f"Max allowed: {format_time(MAX_VIDEO_DURATION)} (10 min)."
        )

    filepath, _ = download_youtube_audio(url)
    if not filepath or not os.path.exists(filepath):
        raise RuntimeError("Audio extraction failed. Verify FFmpeg is accessible.")

    try:
        try:
            audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio. Verify FFmpeg and audio codecs. Details: {str(e)}")

        total_samples = len(audio)
        if total_samples == 0:
            raise RuntimeError("Audio file is empty.")

        # Keep the original waveform for ASR. VAD masking below is useful for
        # audio classification, but it can remove phonetic detail needed by
        # Whisper to produce an accurate transcript.
        asr_audio = audio.copy()

        # --- VAD MASKING: Keep only speech, zero out music/noise ---
        try:
            model, utils = _load_vad()
            get_speech_timestamps = utils[0]
            audio_tensor = torch.from_numpy(audio)
            
            # Get timestamps of speech
            speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=SAMPLE_RATE)
            
            # Create a boolean mask and apply it
            mask = np.zeros(len(audio), dtype=bool)
            for ts in speech_timestamps:
                mask[ts['start']:ts['end']] = True
                
            audio = audio * mask
        except Exception as e:
            print(f"[VAD Warning] Failed to apply VAD mask, falling back to raw audio. Details: {e}")
        # -----------------------------------------------------------

        segments = []
        for start in range(0, total_samples, SEGMENT_SAMPLES):
            end = min(start + SEGMENT_SAMPLES, total_samples)
            seg = audio[start:end].copy()
            asr_seg = asr_audio[start:end].copy()

            if len(seg) < SEGMENT_SAMPLES:
                seg = np.pad(seg, (0, SEGMENT_SAMPLES - len(seg)), mode="constant")
            if len(asr_seg) < SEGMENT_SAMPLES:
                asr_seg = np.pad(asr_seg, (0, SEGMENT_SAMPLES - len(asr_seg)), mode="constant")

            peak = np.max(np.abs(seg))
            if peak > 0:
                seg = seg / peak

            start_t = start / SAMPLE_RATE
            end_t   = min(end / SAMPLE_RATE, float(duration))
            segments.append({
                "start_time": round(start_t, 2),
                "end_time":   round(end_t, 2),
                "start_fmt":  format_time(start_t),
                "end_fmt":    format_time(end_t),
                "audio":      seg.astype(np.float32),
                "asr_audio":  asr_seg.astype(np.float32),
            })
        return metadata, segments
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass

def _normalise_comments(raw_comments, max_comments: int | None = None) -> list:
    """Convert YouTube API and yt-dlp comment objects to one response shape."""
    comments = []
    for comment in raw_comments:
        text = (comment.get("text") or "").strip()
        if not text:
            continue
        comments.append({
            "author": comment.get("author", "Unknown"),
            "text": text,
            "likes": comment.get("likes", comment.get("like_count", 0)) or 0,
        })
        if max_comments is not None and len(comments) >= max_comments:
            break
    return comments


def _fetch_comments_from_api(video_id: str, api_key: str, max_comments: int | None = None) -> list:
    """Fetch top-level comments using the official YouTube Data API, with paging."""
    comments = []
    page_token = None

    while max_comments is None or len(comments) < max_comments:
        page_size = 100 if max_comments is None else min(100, max_comments - len(comments))
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": page_size,
            "key": api_key,
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(
            "https://www.googleapis.com/youtube/v3/commentThreads",
            params=params,
            timeout=20,
        )
        if response.status_code != 200:
            detail = response.json().get("error", {}).get("message", response.text)
            raise RuntimeError(f"YouTube Data API returned {response.status_code}: {detail}")

        data = response.json()
        raw_comments = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = (snippet.get("textOriginal") or "").strip()
            if text:
                raw_comments.append({
                    "author": snippet.get("authorDisplayName", "Unknown"),
                    "text": text,
                    "likes": snippet.get("likeCount", 0),
                })
        remaining = None if max_comments is None else max_comments - len(comments)
        comments.extend(_normalise_comments(raw_comments, remaining))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return comments if max_comments is None else comments[:max_comments]


def fetch_youtube_comments(url: str, max_comments: int | None = None) -> tuple[list, str]:
    """Fetch public top-level comments via the Data API or yt-dlp fallback."""
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Please provide a valid YouTube video URL.")

    api_error = None
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if api_key:
        try:
            comments = _fetch_comments_from_api(video_id, api_key, max_comments)
            if comments:
                return comments, "YouTube Data API"
            raise ValueError("No public comments are available for this video.")
        except ValueError:
            raise
        except Exception as exc:
            api_error = str(exc)

    try:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "getcomments": True,
        }
        if max_comments is not None:
            opts["extractor_args"] = {"youtube": {"max_comments": [str(max_comments)]}}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

        comments = _normalise_comments(info.get("comments") or [], max_comments)
        if comments:
            return comments, "yt-dlp"

        message = "No public comments are available for this video."
        if api_error:
            message += f" The configured YouTube API could not be used: {api_error}"
        raise ValueError(message)
    except ValueError:
        raise
    except Exception as exc:
        message = f"Could not retrieve YouTube comments: {exc}"
        if api_error:
            message += f" The configured YouTube API also failed: {api_error}"
        message += " Add a valid YOUTUBE_API_KEY for the most reliable comment retrieval."
        raise ValueError(message) from exc

import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import re
import html
import wave
import io
from datetime import datetime

from sinhala_page import render_sinhala_tabs

# Points at the integration gateway (gateway/app.py) instead of the Tamil
# backend directly. The gateway forwards every non-/api/ route to this same
# Tamil backend unchanged, so behavior is identical to standalone.
API_URL = "http://127.0.0.1:8000"

# The false-content detector's own Flask+Jinja2 app (unmodified UI), running
# directly on its own port rather than through the gateway — its /predict
# route name collides with Tamil's, so it is kept fully separate instead of
# sharing the gateway's path-based routing.
FALSE_CONTENT_URL = "http://127.0.0.1:5002"

# The video deepfake detector's own Streamlit app (unmodified UI), running
# directly on its own port — same reasoning as False Content above: it's a
# fully separate Streamlit process, not something that fits the gateway's
# path-based routing, so it's embedded via iframe instead.
VIDEO_DEEPFAKE_URL = "http://127.0.0.1:8502"


def build_html_report(data):
    """Builds a self-contained, styled HTML report for a YouTube analysis run."""
    video = data.get("video")
    comments = data.get("comments")

    info_rows = [("Source URL", data["url"]), ("Report generated", data["generated"])]
    if data.get("video_title"):
        info_rows = [
            ("Title", data["video_title"]),
            ("Channel", data.get("video_channel", "Unknown channel")),
            ("Duration", data.get("video_duration", "-")),
        ] + info_rows
    info_table = "".join(
        f'<tr><td class="label">{label}</td><td class="value">{value}</td></tr>'
        for label, value in info_rows
    )

    video_section = ""
    if video:
        total = video["total_segments"]
        flagged = video["flagged_segments"]
        safe_segments = max(total - flagged, 0)
        hate_pct = round(flagged / total * 100, 1) if total else 0.0

        if flagged > 0:
            assessment = (
                f'<strong>{flagged}</strong> of {total} segment(s) were flagged as offensive by the detection model. '
                f'Review the flagged timestamps below alongside the original video before making a final decision.'
            )
        else:
            assessment = (
                'No segments were flagged as offensive by the detection model. '
                'Review timestamps alongside the original video before drawing conclusions.'
            )

        if video["flagged_content"]:
            flagged_html = "".join(
                f'<div class="flagged-item"><div class="ts">Timestamp: {item["timestamp"]}</div>'
                f'<div class="txt">"{item["text"]}"</div></div>'
                for item in video["flagged_content"]
            )
        else:
            flagged_html = '<p class="empty">No offensive content was detected in this video.</p>'

        window_labels = {"HATE": "Offensive", "SAFE": "Not Offensive"}
        window_pill_cls = {"HATE": "verdict-hate", "SAFE": "verdict-safe"}
        timeline_rows = "".join(
            f'<tr><td>{i}</td><td>{w["timestamp"]}</td>'
            f'<td><span class="verdict-pill {window_pill_cls.get(w["verdict"], "")}">'
            f'{window_labels.get(w["verdict"], w["verdict"])}</span></td>'
            f'<td>{w["confidence"]}%</td><td>"{w["transcript"]}"</td></tr>'
            for i, w in enumerate(video.get("all_windows", []), start=1)
        )
        timeline_table = (
            f'<table class="timeline-table"><thead><tr><th>#</th><th>Time</th><th>Result</th>'
            f'<th>Confidence</th><th>Transcript</th></tr></thead><tbody>{timeline_rows}</tbody></table>'
            if timeline_rows else '<p class="empty">No segments were analyzed.</p>'
        )

        video_section = f"""
        <div class="section">
          <h2>Video Information</h2>
          <table class="info-table">{info_table}</table>

          <h2>Analysis Overview</h2>
          <div class="stat-row">
            <div class="stat-card stat-blue"><div class="lbl">Segments Analyzed</div><div class="num">{total}</div></div>
            <div class="stat-card stat-red"><div class="lbl">Offensive Segments</div><div class="num">{flagged} <span class="pct">({hate_pct}%)</span></div></div>
            <div class="stat-card stat-green"><div class="lbl">Clean Segments</div><div class="num">{safe_segments}</div></div>
          </div>
          <div class="assessment-box"><strong>Assessment:</strong> {assessment}</div>
          <p class="note">For easy analysis, audio is split into 30-second frames — e.g. the 0s&ndash;30s frame is referred to as one "segment" below.</p>

          <h2>Detected Content</h2>
          {flagged_html}

          <h2>Segment Timeline</h2>
          {timeline_table}
        </div>
        """

    comments_section = ""
    if comments:
        analyzed = comments["analyzed"]
        harmful = comments["harmful"]
        safe = comments["safe"]
        flagged_comments = comments.get("flagged", [])
        if flagged_comments:
            flagged_comments_html = "".join(
                f'<div class="flagged-item"><div class="ts">Confidence: {item["confidence"]}%</div>'
                f'<div class="txt">"{item["text"]}"</div></div>'
                for item in flagged_comments
            )
        else:
            flagged_comments_html = '<p class="empty">No potentially offensive comments were detected.</p>'

        comments_section = f"""
        <div class="section">
          <h2>Comments Analysis</h2>
          <div class="stat-row">
            <div class="stat-card stat-blue"><div class="lbl">Analyzed Comments</div><div class="num">{analyzed}</div></div>
            <div class="stat-card stat-red"><div class="lbl">Offensive</div><div class="num">{harmful}</div></div>
            <div class="stat-card stat-green"><div class="lbl">Safe</div><div class="num">{safe}</div></div>
          </div>
          <h2>Detected Hate Comments</h2>
          {flagged_comments_html}
        </div>
        """

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>YouTube Analysis Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:#f8fafc; margin:0; padding:32px; color:#1e293b; }}
  .container {{ max-width: 860px; margin: 0 auto; }}
  .header {{ text-align:center; padding: 10px 20px 24px; border-bottom:1px solid #e2e8f0; margin-bottom:24px; }}
  .header h1 {{ margin:0 0 6px; font-size:1.6rem; color:#0f172a; }}
  .header .subtitle {{ color:#64748b; font-size:0.9rem; }}
  .section {{ background:white; border-radius:14px; padding:24px 28px; margin-bottom:20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); }}
  .section h2 {{ margin:22px 0 10px; font-size:1rem; color:#0f172a; }}
  .section h2:first-child {{ margin-top:0; }}
  .result-badge {{ display:inline-block; padding:8px 18px; border-radius:999px; font-weight:700; font-size:0.95rem; }}
  .result-harmful {{ background:#fef2f2; color:#dc2626; border:2px solid #dc2626; }}
  .result-safe {{ background:#f0fdf4; color:#16a34a; border:2px solid #16a34a; }}
  .info-table {{ width:100%; border-collapse:collapse; }}
  .info-table td {{ padding:10px 4px; font-size:0.88rem; border-bottom:1px solid #e2e8f0; }}
  .info-table td.label {{ color:#64748b; font-weight:600; width:170px; }}
  .info-table td.value {{ color:#111827; word-break:break-word; }}
  .stat-row {{ display:flex; gap:16px; margin: 4px 0 18px; flex-wrap:wrap; }}
  .stat-card {{ flex:1; min-width:140px; background:#f8fafc; border-radius:10px; padding:14px 16px; border-left:4px solid #94a3b8; }}
  .stat-card.stat-blue {{ border-left-color:#1d4ed8; }}
  .stat-card.stat-red {{ border-left-color:#dc2626; }}
  .stat-card.stat-green {{ border-left-color:#16a34a; }}
  .stat-card .lbl {{ font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:.04em; }}
  .stat-card .num {{ font-size:1.5rem; font-weight:800; margin-top:4px; }}
  .stat-card .pct {{ font-size:0.9rem; font-weight:600; color:#64748b; }}
  .assessment-box {{ background:#eff6ff; border-left:4px solid #1d4ed8; border-radius:8px; padding:12px 16px; font-size:0.85rem; color:#1e3a5f; margin-bottom:18px; }}
  .flagged-item {{ background:#fff7f7; border-left:4px solid #dc2626; border-radius:8px; padding:10px 14px; margin-bottom:8px; }}
  .flagged-item .ts {{ font-size:0.75rem; color:#64748b; font-weight:600; }}
  .flagged-item .txt {{ font-size:0.9rem; color:#111827; font-style:italic; margin-top:4px; }}
  .timeline-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  .timeline-table th {{ text-align:left; padding:9px 8px; background:#0f172a; color:white; font-size:0.68rem; text-transform:uppercase; letter-spacing:.04em; }}
  .timeline-table th:first-child {{ border-radius:6px 0 0 0; }}
  .timeline-table th:last-child {{ border-radius:0 6px 0 0; }}
  .timeline-table td {{ padding:9px 8px; border-bottom:1px solid #e2e8f0; vertical-align:top; }}
  .timeline-table tr:nth-child(even) td {{ background:#f8fafc; }}
  .verdict-pill {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.72rem; font-weight:700; white-space:nowrap; }}
  .verdict-hate {{ background:#fef2f2; color:#dc2626; }}
  .verdict-safe {{ background:#f0fdf4; color:#16a34a; }}
  .empty {{ color:#64748b; font-size:0.9rem; }}
  .note {{ color:#64748b; font-size:0.78rem; margin:-8px 0 18px; }}
  .footer {{ text-align:center; font-size:0.75rem; color:#94a3b8; margin-top:24px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🛡️ YouTube Analysis Report</h1>
    <div class="subtitle">Tamil hate speech detection assessment</div>
  </div>
  {video_section}
  {comments_section}
  <div class="footer">Generated by Tamil Hate Speech Detector v1.0</div>
</div>
</body>
</html>"""


def render_yt_video_preview(video_info):
    """Renders the thumbnail/title/description card. Returns (title, channel, duration_str) or None."""
    if not (video_info and "error" not in video_info):
        return None
    duration_sec = video_info.get("duration") or 0
    hh, rem = divmod(int(duration_sec), 3600)
    mm, ss = divmod(rem, 60)
    duration_str = f"{hh:02d}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"

    col_thumb, col_info = st.columns([1, 3])
    with col_thumb:
        if video_info.get("thumbnail"):
            st.image(video_info["thumbnail"], use_container_width=True)
    with col_info:
        st.markdown(f"**{video_info.get('title', 'Untitled video')}**")
        st.caption(f"{video_info.get('uploader', 'Unknown channel')}  ·  Duration: {duration_str}")
        description = (video_info.get("description") or "").strip()
        if description:
            if len(description) > 220:
                st.markdown(description[:220] + "…")
                with st.expander("Show full description"):
                    st.markdown(description)
            else:
                st.markdown(description)
    st.markdown("---")
    return video_info.get("title", "Untitled video"), video_info.get("uploader", "Unknown channel"), duration_str


def render_yt_video_results(yt_result):
    """Renders the video summary/detected content/window expander. Returns report_data['video'] or None."""
    if not yt_result:
        return None
    if "error" in yt_result:
        st.error(f"Error: {yt_result['error']}")
        return None

    hate_instances = yt_result.get("hate_instances", [])
    verdict = yt_result.get("verdict")

    if verdict == "HATE SPEECH DETECTED":
        st.markdown("""
        <div style="background:#fef2f2;border:2px solid #dc2626;border-radius:10px;padding:14px 18px;margin-bottom:10px;">
          <div style="font-size:16px;font-weight:800;color:#dc2626;">🔴 Offensive Speech Detected</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#f0fdf4;border:2px solid #16a34a;border-radius:10px;padding:14px 18px;margin-bottom:10px;">
          <div style="font-size:16px;font-weight:800;color:#16a34a;">🟢 Not Offensive Content Detected</div>
        </div>
        """, unsafe_allow_html=True)

    window_style = {
        "HATE": ("🔴", "Offensive Speech Detected"),
        "SAFE": ("🟢", "Not Offensive"),
    }
    asr_badge = {"Sarvam AI": "S", "ElevenLabs": "E", "Google STT": "G", "Whisper": "W"}
    all_windows = []
    for c in yt_result.get("chunks", []):
        w_verdict = c["verdict"]
        w_conf = round(c.get("worst_confidence", 0.0) * 100, 1)
        w_transcript = c.get("transcript", "") or "(no speech detected)"
        all_windows.append({
            "timestamp": f"{c['start_time']} - {c['end_time']}",
            "verdict": w_verdict,
            "confidence": w_conf,
            "transcript": w_transcript,
            "asr_model": c.get("asr_model", ""),
        })

    if all_windows:
        summary_rows = "".join(
            f'<div style="font-family:monospace;font-size:13px;padding:4px 0;">'
            f'{w["timestamp"]} &nbsp; '
            f'{window_style.get(w["verdict"], ("⚪", w["verdict"]))[0]} '
            f'{window_style.get(w["verdict"], ("⚪", w["verdict"]))[1]}</div>'
            for w in all_windows
        )
        st.markdown(f'<div style="background:#f8fafc;border-radius:10px;padding:12px 16px;margin-bottom:14px;">{summary_rows}</div>', unsafe_allow_html=True)

    overall_result = "Offensive" if verdict == "HATE SPEECH DETECTED" else "Not Offensive"
    total_segments   = yt_result.get('total_chunks', 0)
    flagged_segments = yt_result.get('hate_chunks', 0)
    offensive_rate   = round(flagged_segments / total_segments * 100, 1) if total_segments else 0.0

    st.markdown("#### Video Summary")
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card"><div class="num num-blue">{total_segments}</div><div class="lbl">Segments Analyzed</div></div>
      <div class="stat-card"><div class="num num-red">{flagged_segments}</div><div class="lbl">Offensive Content Detected</div></div>
      <div class="stat-card"><div class="num num-red">{flagged_segments}/{total_segments} <span style="font-size:14px;font-weight:600;color:#64748b;">({offensive_rate}%)</span></div><div class="lbl">Offensive Segments Ratio</div></div>
    </div>
    """, unsafe_allow_html=True)

    if hate_instances:
        with st.expander(f"Detected Content ({len(hate_instances)})"):
            for inst in hate_instances:
                full_transcript = inst.get("full_transcript") or inst["sentence"]
                worst_sentence = inst["sentence"]
                if worst_sentence and worst_sentence in full_transcript:
                    shown_transcript = full_transcript.replace(
                        worst_sentence,
                        f'<mark style="background:#fecaca;padding:0 2px;border-radius:3px;">{worst_sentence}</mark>'
                    )
                else:
                    shown_transcript = full_transcript
                st.markdown(f"""
                <div style="background:#fff7f7;border-left:3px solid #dc2626;border-radius:6px;padding:8px 14px;margin-bottom:6px;">
                  <div style="font-size:11px;color:#64748b;">Offensive speech detected between <b>{inst['timestamp']}</b></div>
                  <div style="font-size:13px;color:#111827;margin-top:6px;">"{shown_transcript}"</div>
                </div>
                """, unsafe_allow_html=True)

    with st.expander(f"Segment-by-Segment Analysis (30 sec, {len(all_windows)} segments)"):
        for w in all_windows:
            w_icon, w_label = window_style.get(w["verdict"], ("⚪", w["verdict"]))
            w_color = "#dc2626" if w["verdict"] == "HATE" else "#16a34a"
            w_bg = "#fff7f7" if w["verdict"] == "HATE" else "#f7fdf9"
            asr_tag = (
                f'<span title="{w["asr_model"]}" style="display:inline-block;width:16px;height:16px;'
                f'line-height:16px;text-align:center;border-radius:3px;background:#e2e8f0;'
                f'color:#334155;font-weight:700;font-size:10px;margin-left:6px;">'
                f'{asr_badge.get(w["asr_model"], "?")}</span>'
            )
            st.markdown(f"""
            <div style="background:{w_bg};border-left:3px solid {w_color};border-radius:6px;padding:8px 14px;margin-bottom:6px;">
              <div style="font-size:12px;color:#64748b;"><b>Timestamp:</b> {w['timestamp']} &nbsp;·&nbsp; <span style="color:{w_color};font-weight:700;">{w_icon} {w_label}</span> &nbsp;·&nbsp; Confidence: <b>{w['confidence']}%</b>{asr_tag}</div>
              <div style="font-size:13px;color:#111827;font-style:italic;margin-top:6px;">"{w['transcript']}"</div>
            </div>
            """, unsafe_allow_html=True)

    return {
        "overall_result": overall_result,
        "total_segments": total_segments,
        "flagged_segments": flagged_segments,
        "flagged_content": [
            {"timestamp": inst["timestamp"], "text": inst["sentence"]}
            for inst in hate_instances
        ],
        "all_windows": all_windows,
    }


def render_yt_comments_results(cmt_result):
    """Renders the comments stats/flagged list/expanders. Returns report_data['comments'] or None."""
    if not cmt_result:
        return None
    if "error" in cmt_result:
        st.error(f"Comments error: {cmt_result['error']}")
        return None
    if cmt_result.get("total", 0) == 0:
        st.info("No comments found on this video (or comments are disabled).")
        return None

    h = cmt_result["hate_count"]
    u = cmt_result["uncertain_count"]
    s = cmt_result["safe_count"]
    sk = cmt_result.get("skipped_count", 0)
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card"><div class="num num-blue">{cmt_result['checked_total']}</div><div class="lbl">Analyzed Comments</div></div>
      <div class="stat-card"><div class="num num-red">{h}</div><div class="lbl">Offensive</div></div>
      <div class="stat-card"><div class="num num-green">{s}</div><div class="lbl">Safe</div></div>
    </div>
    """, unsafe_allow_html=True)
    if sk > 0:
        st.caption(f"{sk} of {cmt_result['total']} fetched comments skipped (non-Tamil or empty/no content).")

    if h > 0:
        st.warning(f"{h} potentially offensive comment(s) were detected out of {cmt_result['checked_total']} analyzed comments.")
        for r in cmt_result["hate_comments"]:
            st.markdown(f"""
            <div style="background:#fff7f7;border-left:3px solid #dc2626;border-radius:6px;padding:8px 14px;margin-bottom:6px;">
              <div style="font-size:12px;color:#64748b;">Confidence: <b>{r['confidence']}%</b></div>
              <div style="font-size:13px;color:#111827;font-style:italic;margin-top:2px;">"{r['text']}"</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No potentially offensive comments were detected among the analyzed comments.")

    if u > 0:
        with st.expander(f"{u} uncertain comment(s) — needs review"):
            for r in cmt_result["results"]:
                if r["verdict"] == "UNCERTAIN":
                    st.markdown(f"- ({r['confidence']}%) \"{r['text']}\"")

    verdict_icon = {"HATE": "🔴", "UNCERTAIN": "🟡", "SAFE": "🟢", "SKIPPED": "⚪"}
    with st.expander(f"📋 View All Comments ({cmt_result['total']})"):
        for r in cmt_result["results"]:
            icon = verdict_icon.get(r["verdict"], "•")
            conf_txt = f" ({r['confidence']}%)" if r["verdict"] != "SKIPPED" else ""
            st.markdown(f"{icon} **{r['verdict']}**{conf_txt} — \"{r['text']}\"")

    return {
        "analyzed": cmt_result['checked_total'],
        "harmful": h,
        "safe": s,
        "flagged": [
            {"confidence": r["confidence"], "text": r["text"]}
            for r in cmt_result["hate_comments"]
        ],
    }


def get_wav_duration(file_bytes):
    """Total duration (seconds) of an uploaded WAV, read from its header. Returns
    None if it can't be read — used only to show the analyzed clip's time range."""
    try:
        with wave.open(io.BytesIO(file_bytes), 'rb') as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return None


def format_timestamp(seconds):
    if seconds is None:
        return "0:00"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def build_fusion_report(data):
    """Self-contained, styled HTML report for a single Audio Files Analysis run."""
    bar_color  = "#dc2626" if data["is_offensive"] else "#16a34a"
    badge_cls  = "result-harmful" if data["is_offensive"] else "result-safe"
    badge_icon = "🔴" if data["is_offensive"] else "🟢"

    info_rows = [
        ("File name", data["filename"]),
        ("Timestamp", data["timestamp"]),
        ("Report generated", data["generated"]),
    ]
    info_table = "".join(
        f'<tr><td class="label">{label}</td><td class="value">{value}</td></tr>'
        for label, value in info_rows
    )

    if data["is_offensive"]:
        assessment = (
            'This audio clip was flagged as <strong>offensive</strong> by the detection model. '
            'Review the transcript below alongside the original audio before making a final decision.'
        )
    else:
        assessment = 'No offensive content was detected in this audio clip by the detection model.'

    tone_html = ""
    if data.get("tone_label"):
        tone_html = (
            f'<div class="stat-card" style="border-left-color:{data["tone_color"]};margin-top:14px;">'
            f'<div class="lbl">Audio Tone</div><div class="num" style="color:{data["tone_color"]};font-size:1.1rem;">'
            f'{data["tone_icon"]} {data["tone_label"]}</div></div>'
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Audio Analysis Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:#f8fafc; margin:0; padding:32px; color:#1e293b; }}
  .container {{ max-width: 760px; margin: 0 auto; }}
  .header {{ text-align:center; padding: 10px 20px 24px; border-bottom:1px solid #e2e8f0; margin-bottom:24px; }}
  .header h1 {{ margin:0 0 6px; font-size:1.6rem; color:#0f172a; }}
  .header .subtitle {{ color:#64748b; font-size:0.9rem; }}
  .section {{ background:white; border-radius:14px; padding:24px 28px; margin-bottom:20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); }}
  .section h2 {{ margin:22px 0 10px; font-size:1rem; color:#0f172a; }}
  .section h2:first-child {{ margin-top:0; }}
  .info-table {{ width:100%; border-collapse:collapse; }}
  .info-table td {{ padding:10px 4px; font-size:0.88rem; border-bottom:1px solid #e2e8f0; }}
  .info-table td.label {{ color:#64748b; font-weight:600; width:170px; }}
  .info-table td.value {{ color:#111827; word-break:break-word; }}
  .result-badge {{ display:inline-block; padding:10px 22px; border-radius:999px; font-weight:700; font-size:1.05rem; }}
  .result-harmful {{ background:#fef2f2; color:#dc2626; border:2px solid #dc2626; }}
  .result-safe {{ background:#f0fdf4; color:#16a34a; border:2px solid #16a34a; }}
  .conf-label {{ font-size:0.9rem; color:#475569; margin-top:16px; }}
  .conf-bar-track {{ background:#e5e7eb; border-radius:999px; height:16px; width:100%; overflow:hidden; margin-top:8px; }}
  .conf-bar-fill {{ height:100%; border-radius:999px; }}
  .stat-card {{ background:#f8fafc; border-radius:10px; padding:14px 16px; border-left:4px solid #94a3b8; }}
  .stat-card .lbl {{ font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:.04em; }}
  .stat-card .num {{ font-size:1.5rem; font-weight:800; margin-top:4px; }}
  .assessment-box {{ background:#eff6ff; border-left:4px solid #1d4ed8; border-radius:8px; padding:12px 16px; font-size:0.85rem; color:#1e3a5f; margin:16px 0; }}
  .transcript-box {{ background:#f8fafc; border-left:4px solid #1d4ed8; border-radius:8px;
                      padding:16px 20px; font-style:italic; line-height:1.7; }}
  .footer {{ text-align:center; font-size:0.75rem; color:#94a3b8; margin-top:24px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🛡️ Audio Analysis Report</h1>
    <div class="subtitle">Tamil hate speech detection assessment</div>
  </div>
  <div class="section">
    <h2>File Information</h2>
    <table class="info-table">{info_table}</table>

    <h2>Result</h2>
    <span class="result-badge {badge_cls}">{badge_icon} {data['verdict_text']}</span>
    <div class="conf-label">Overall Confidence: <strong>{data['confidence']}%</strong></div>
    <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{data['confidence']}%;background:{bar_color};"></div></div>
    <div class="assessment-box">{assessment}</div>
    {tone_html}

    <h2>Transcript</h2>
    <div class="transcript-box">"{data['transcript_text']}"</div>
  </div>
  <div class="footer">Generated by Tamil Hate Speech Detector v1.0</div>
</div>
</body>
</html>"""


st.set_page_config(
    page_title="VigilAI — Hate Speech & Deepfake Detection",
    page_icon="🛡️",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "home"

# ── Global styles ─────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1d4ed8 100%);
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 6px; letter-spacing: -0.5px; }
.hero p  { font-size: 1rem; opacity: 0.9; font-weight: 500; margin: 0; }
.version-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.8rem; font-weight: 600;
    vertical-align: middle;
    margin-left: 10px;
}
.badge-row { display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 12px; font-weight: 600; color: white;
}

/* Home service cards */
.service-grid { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 24px; }
.service-card {
    flex: 1; min-width: 220px;
    background: white;
    border-radius: 16px;
    padding: 28px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    text-align: center;
    border: 2px solid transparent;
    transition: all 0.2s;
}
.service-card.active { border-color: #1d4ed8; }
.service-card.soon   { border-color: #e2e8f0; opacity: 0.65; }
.service-icon { font-size: 2.4rem; margin-bottom: 12px; }
.service-title { font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 6px; }
.service-desc  { font-size: 12px; color: #64748b; line-height: 1.5; margin-bottom: 14px; }
.soon-badge {
    display: inline-block;
    background: #f1f5f9; color: #94a3b8;
    border-radius: 999px; padding: 3px 12px;
    font-size: 11px; font-weight: 600;
}

/* ── Landing page (home page only — prefixed "lp-" so nothing here
   affects the Tamil / Sinhala / False Content pages, which keep using
   the .hero / .service-card rules above exactly as before) ────────── */

/* New palette for the home page: indigo → violet, replacing the flat
   corporate blue. Scoped entirely to lp- classes, so the Tamil page's
   own primary-blue buttons are untouched — see the extra <style> block
   injected only inside the home branch for the button-color override. */
:root {
    --lp-primary: #2563eb;
    --lp-primary-light: #3b82f6;
    --lp-primary-dark: #1d4ed8;
    --lp-accent2: #7c3aed;
    --lp-accent-cyan: #06b6d4;
    --lp-accent-green: #16a34a;
}

@keyframes lpFadeUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
@keyframes lpFadeIn  { from { opacity: 0; } to { opacity: 1; } }
@keyframes lpFloat   { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
@keyframes lpPulse   { 0%,100% { box-shadow: 0 0 0 0 rgba(79,70,229,0.35); } 50% { box-shadow: 0 0 0 10px rgba(79,70,229,0); } }

.lp-animate { animation: lpFadeUp 0.7s ease-out both; }
.lp-d1 { animation-delay: .05s; } .lp-d2 { animation-delay: .15s; }
.lp-d3 { animation-delay: .25s; } .lp-d4 { animation-delay: .35s; }

.lp-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 24px; margin-bottom: 20px;
    background: white; border-radius: 16px; box-shadow: 0 4px 18px rgba(15,23,42,0.07);
}
.lp-nav-brand { display: flex; align-items: center; gap: 12px; }
.lp-nav-logo {
    width: 44px; height: 44px; border-radius: 12px; flex-shrink: 0;
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 55%, #06b6d4 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; box-shadow: 0 6px 16px rgba(37,99,235,0.35);
}
.lp-nav-wordmark { font-weight: 800; font-size: 1.4rem; color: #0f172a; letter-spacing: -0.02em; }
.lp-nav-links { display: flex; gap: 32px; }
.lp-nav-links a { color: #475569; font-weight: 700; font-size: 0.95rem; text-decoration: none; transition: color .15s; }
.lp-nav-links a:hover { color: var(--lp-primary); }

.lp-hero-kicker {
    display: inline-block; background: #f2effd; color: var(--lp-primary-dark);
    border-radius: 999px; padding: 5px 14px; font-size: 0.75rem; font-weight: 700;
    letter-spacing: .03em; margin-bottom: 18px;
}
.lp-hero-title { font-size: 2.6rem; font-weight: 800; line-height: 1.15; color: #0f172a; margin: 0 0 18px; }
.lp-hero-title .accent { color: var(--lp-primary); }
.lp-hero-sub { font-size: 1.02rem; color: #64748b; line-height: 1.6; margin-bottom: 24px; max-width: 520px; }
.lp-hero-link-btn {
    display: inline-block; background: #ffffff; color: #1d4ed8 !important;
    border-radius: 10px; padding: 0.7rem 1.4rem; font-weight: 700; font-size: 0.95rem;
    text-decoration: none; border: 1.5px solid #93c5fd; transition: all .15s;
    text-align: center;
}
.lp-hero-link-btn:hover { background: #eff6ff; border-color: #2563eb; transform: translateY(-2px); }

.lp-btn-primary-link {
    display: inline-block; background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: #ffffff !important;
    border-radius: 10px; padding: 0.7rem 1.4rem; font-weight: 700; font-size: 0.95rem;
    text-decoration: none; border: none; transition: all .15s;
    box-shadow: 0 6px 16px rgba(37,99,235,0.25); text-align: center;
}
.lp-btn-primary-link:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(37,99,235,0.35); }
.lp-hero-actions { display: flex; gap: 12px; flex-wrap: wrap; }

.lp-trust-badges { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 16px; }
.lp-trust-badges span { font-size: 0.8rem; color: #475569; display: flex; align-items: center; gap: 6px; }
.lp-trust-badges .dot { width: 16px; height: 16px; border-radius: 50%; background: #dcfce7; color: #16a34a; display: inline-flex; align-items: center; justify-content: center; font-size: 0.6rem; }

.lp-preview-card {
    background: white; border-radius: 18px; padding: 0; overflow: hidden;
    box-shadow: 0 16px 45px rgba(37,99,235,0.16); border: 1px solid #eef2f7;
    animation: lpFadeUp 0.8s ease-out both, lpFloat 5s ease-in-out 1s infinite;
}
.lp-preview-chrome {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 16px; border-bottom: 1px solid #f1f5f9; background: #fafbfe;
}
.lp-preview-dots { display: flex; gap: 6px; }
.lp-preview-dots span { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.lp-preview-dots span:nth-child(1) { background: #f87171; }
.lp-preview-dots span:nth-child(2) { background: #fbbf24; }
.lp-preview-dots span:nth-child(3) { background: #34d399; }
.lp-preview-tag { font-size: 0.68rem; color: #94a3b8; font-family: monospace; }
.lp-preview-body { padding: 20px 22px; }
.lp-preview-row-meta { font-size: 0.68rem; color: #94a3b8; margin-top: 2px; }
.lp-preview-row.warn { background: #fef2f2; border: 1px solid #fecaca; }
.lp-preview-row.warn .lp-pill { background: #dc2626; }
.lp-wave-label { font-size: 0.68rem; color: #94a3b8; margin-top: 16px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .03em; }
.lp-preview-row {
    display: flex; align-items: center; justify-content: space-between;
    border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; font-size: 0.85rem;
}
.lp-preview-row.safe { background: #f0fdf4; border: 1px solid #bbf7d0; }
.lp-preview-row.safe .lp-pill { background: #16a34a; }
.lp-preview-label { font-weight: 600; color: #0f172a; }
.lp-pill { color: white; border-radius: 999px; padding: 3px 12px; font-size: 0.72rem; font-weight: 700; }
.lp-wave { display: flex; align-items: flex-end; gap: 3px; height: 44px; margin-top: 14px; }
.lp-wave span { flex: 1; background: linear-gradient(180deg, #93c5fd, var(--lp-primary)); border-radius: 3px; animation: lpFadeUp .6s ease-out both; }
.lp-wave span:nth-child(3n) { background: linear-gradient(180deg, #bfdbfe, var(--lp-primary-light)); }

.lp-stats { display: flex; gap: 16px; flex-wrap: wrap; margin: 40px 0 8px; }
.lp-stat { flex: 1; min-width: 150px; text-align: center; padding: 18px 12px; transition: transform .2s; }
.lp-stat:hover { transform: translateY(-4px); }
.lp-stat .num { font-size: 1.9rem; font-weight: 800; color: var(--lp-primary); }
.lp-stat .num.cyan { color: var(--lp-accent-cyan); }
.lp-stat .num.purple { color: var(--lp-accent2); }
.lp-stat .num.green { color: var(--lp-accent-green); }
.lp-stat .lbl { font-size: 0.78rem; color: #64748b; margin-top: 4px; }

.lp-section-kicker { text-align: center; color: var(--lp-primary); font-weight: 700; font-size: 0.78rem; letter-spacing: .06em; text-transform: uppercase; }
.lp-section-title { text-align: center; font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 6px 0 6px; }
.lp-section-sub { text-align: center; color: #64748b; font-size: 0.92rem; margin-bottom: 8px; }

.lp-module-card {
    background: white; border-radius: 16px; padding: 24px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06); border: 1px solid #eef2f7;
    height: 100%; transition: transform .25s, box-shadow .25s, border-color .25s;
}
.lp-module-card:hover { transform: translateY(-6px); box-shadow: 0 14px 30px rgba(79,70,229,0.14); border-color: #ddd6fe; }
.lp-module-card.soon { opacity: 0.6; }
.lp-module-card.soon:hover { transform: none; box-shadow: 0 2px 10px rgba(0,0,0,0.06); border-color: #eef2f7; }
.lp-module-icon {
    width: 42px; height: 42px; border-radius: 10px; background: #f2effd;
    display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
    margin-bottom: 14px;
}
.lp-module-title { font-weight: 700; font-size: 0.95rem; color: #0f172a; margin-bottom: 6px; }
.lp-module-desc { font-size: 0.8rem; color: #64748b; line-height: 1.5; margin-bottom: 14px; min-height: 55px; }

.lp-step { text-align: center; padding: 10px; }
.lp-step-num {
    width: 40px; height: 40px; border-radius: 50%;
    background: linear-gradient(135deg, var(--lp-primary), var(--lp-accent2)); color: white;
    display: flex; align-items: center; justify-content: center; font-weight: 800;
    margin: 0 auto 14px;
}
.lp-step-title { font-weight: 700; color: #0f172a; font-size: 0.95rem; margin-bottom: 6px; }
.lp-step-desc { font-size: 0.82rem; color: #64748b; line-height: 1.5; }

/* About panel with decorative visual instead of a stock photo */
.lp-panel {
    display: flex; align-items: center; gap: 40px; margin: 50px 0; flex-wrap: wrap;
}
.lp-panel-text { flex: 1.1; min-width: 280px; }
.lp-panel-kicker { color: var(--lp-primary); font-weight: 700; font-size: 0.78rem; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 10px; }
.lp-panel-text h3 { font-size: 1.5rem; font-weight: 800; color: #0f172a; margin: 0 0 12px; }
.lp-panel-text p { color: #64748b; font-size: 0.9rem; line-height: 1.7; margin-bottom: 16px; }
.lp-panel-check { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; font-size: 0.87rem; color: #334155; }
.lp-panel-check .dot {
    width: 20px; height: 20px; border-radius: 50%; background: #ecfdf5; color: #16a34a;
    display: flex; align-items: center; justify-content: center; font-size: 0.7rem; flex-shrink: 0; margin-top: 1px;
}
.lp-panel-visual {
    flex: 1; min-width: 260px; height: 260px; border-radius: 20px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #9333ea 100%);
    position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 20px 45px rgba(79,70,229,0.25);
}
.lp-panel-visual::before, .lp-panel-visual::after {
    content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,0.12);
}
.lp-panel-visual::before { width: 180px; height: 180px; top: -60px; right: -50px; }
.lp-panel-visual::after { width: 120px; height: 120px; bottom: -40px; left: -30px; }
.lp-panel-visual-glyph { font-size: 4.5rem; z-index: 1; animation: lpFloat 4s ease-in-out infinite; }

/* CTA band — one single block (header, text, and buttons all in the same
   HTML, since the buttons are plain anchor tags, not real Streamlit
   widgets) so it always renders as one seamless gradient card. */
.lp-cta-band {
    background: linear-gradient(120deg, #1d4ed8 0%, #2563eb 55%, #3b82f6 100%);
    border-radius: 22px; padding: 50px 40px; text-align: center; color: white;
    margin: 50px 0 30px; box-shadow: 0 20px 40px rgba(37,99,235,0.28);
}
.lp-cta-band h2 { font-size: 1.7rem; font-weight: 800; margin: 0 0 10px; }
.lp-cta-band p { opacity: 0.92; font-size: 0.92rem; margin-bottom: 26px; max-width: 480px; margin-left: auto; margin-right: auto; }
.lp-cta-band-buttons { display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; }

/* Expanded footer — real content only, no links to pages that don't exist */
.lp-footer { padding: 36px 0 16px; margin-top: 10px; border-top: 1px solid #e2e8f0; }
.lp-footer-grid { display: flex; gap: 40px; flex-wrap: wrap; margin-bottom: 24px; }
.lp-footer-brand-col { flex: 1.3; min-width: 220px; }
.lp-footer-brand-col .brand { font-weight: 800; color: #0f172a; font-size: 1.05rem; margin-bottom: 8px; }
.lp-footer-brand-col .tagline { color: #94a3b8; font-size: 0.8rem; line-height: 1.6; max-width: 260px; }
.lp-footer-col { flex: 1; min-width: 160px; }
.lp-footer-col h4 { font-size: 0.75rem; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 12px; }
.lp-footer-col a, .lp-footer-col .item { display: block; color: #64748b; font-size: 0.82rem; text-decoration: none; margin-bottom: 9px; transition: color .15s; }
.lp-footer-col a:hover { color: var(--lp-primary); }
.lp-footer-bottom { text-align: center; padding-top: 18px; border-top: 1px solid #f1f5f9; color: #94a3b8; font-size: 0.75rem; }

/* Stat cards */
.stat-row { display: flex; gap: 14px; margin: 16px 0; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 130px;
    background: white; border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    text-align: center;
}
.stat-card .num { font-size: 1.9rem; font-weight: 800; line-height: 1; }
.stat-card .lbl { font-size: 11px; color: #64748b; margin-top: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: .04em; }
.num-blue  { color: #1d4ed8; }
.num-red   { color: #dc2626; }
.num-green { color: #16a34a; }

/* Primary (Analyze) buttons — always blue */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"] {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #1e40af !important;
    border-color: #1e40af !important;
    color: #ffffff !important;
}

/* Bigger, nicer checkboxes app-wide */
[data-testid="stCheckbox"] {
    transform: scale(1.25);
    transform-origin: left center;
    margin: 6px 0;
}

/* Comment-analysis checkbox box (the one bordered container in this app) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #f0f7ff;
    border-radius: 12px;
    border-color: #93c5fd !important;
}

/* Clearer, more visible borders on URL / text input boxes */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    border: 2px solid #93c5fd !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #1d4ed8 !important;
    box-shadow: 0 0 0 2px rgba(29,78,216,0.15) !important;
}

/* Confidence bar (Audio Files Analysis result) */
.conf-bar-track {
    background: #e5e7eb; border-radius: 999px; height: 16px;
    width: 100%; overflow: hidden; margin-top: 10px;
}
.conf-bar-fill { height: 100%; border-radius: 999px; transition: width 0.4s; }
.tone-badge {
    display: inline-block; border-radius: 999px; font-weight: 700;
    font-size: 0.85rem; padding: 6px 16px; margin-top: 10px;
}

/* Result banners */
.result-hate {
    background: linear-gradient(90deg, #fef2f2, #fee2e2);
    border: 1.5px solid #fca5a5; border-left: 5px solid #dc2626;
    border-radius: 10px; padding: 18px 22px; margin: 16px 0;
}
.result-safe {
    background: linear-gradient(90deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #86efac; border-left: 5px solid #16a34a;
    border-radius: 10px; padding: 18px 22px; margin: 16px 0;
}
.result-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 4px; }
.result-hate .result-title { color: #dc2626; }
.result-safe .result-title { color: #16a34a; }
.result-sub { font-size: 13px; color: #475569; }

/* Info box */
.info-box {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 14px 18px;
    font-size: 13px; color: #1e40af;
    margin: 12px 0; line-height: 1.6;
}

/* Section card */
.section-card {
    background: white; border-radius: 14px;
    padding: 24px 26px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    margin-bottom: 16px;
}

/* Tab styling */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 14px; }
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600; font-size: 13px;
    margin-right: 12px;
    padding: 8px 10px;
    border-radius: 8px 8px 0 0;
    transition: background 0.15s;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    font-weight: 800;
    background: rgba(29,78,216,0.08);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #dc2626 !important;
    height: 3.5px !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    # Home-page-only button color override. This <style> block is only ever
    # rendered while page == "home" (Streamlit reruns the whole script per
    # page, so it simply never executes on the Tamil/Sinhala/False Content
    # branches) — the shared global button rule above stays exactly as-is
    # for those pages, and this later, more specific block wins only here.
    st.markdown("""
    <style>
    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.4rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 6px 16px rgba(37,99,235,0.25) !important;
        transition: transform .15s, box-shadow .15s !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #2563eb, #1e40af) !important;
        border-color: #1d4ed8 !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(37,99,235,0.35) !important;
    }
    .stButton > button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background: #ffffff !important;
        border: 1.5px solid #93c5fd !important;
        color: #1d4ed8 !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.4rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all .15s !important;
    }
    .stButton > button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background: #eff6ff !important;
        border-color: #2563eb !important;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Nav bar ─────────────────────────────────────────────
    st.markdown("""
    <div class="lp-nav lp-animate">
      <div class="lp-nav-brand">
        <span class="lp-nav-logo">🛡️</span>
        <span class="lp-nav-wordmark">TruthLenseAI</span>
      </div>
      <div class="lp-nav-links">
        <a href="#lp-modules">Modules</a>
        <a href="#lp-how">How it works</a>
        <a href="#lp-about">About</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero ────────────────────────────────────────────────
    hero_left, hero_right = st.columns([1.15, 1], gap="large")

    with hero_left:
        st.markdown("""
        <div class="lp-animate lp-d1">
        <span class="lp-hero-kicker">AI-Powered · Sinhala &amp; Tamil</span>
        <div class="lp-hero-title">Protecting Communities from<br><span class="accent">Hate Speech &amp; Deepfakes</span></div>
        <div class="lp-hero-sub">
          AI-powered detection for Sinhala &amp; Tamil language content — spanning text, audio,
          deepfake audio, and YouTube video and comment analysis, with results explained in
          plain language, not just a score.
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="lp-hero-actions">
          <a class="lp-btn-primary-link" href="#lp-modules">Try a Module →</a>
          <a class="lp-hero-link-btn" href="#lp-how">See How It Works</a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="lp-trust-badges">
          <span><span class="dot">✓</span> No signup required</span>
          <span><span class="dot">✓</span> Runs on real trained models</span>
          <span><span class="dot">✓</span> Sinhala &amp; Tamil, natively</span>
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        st.markdown("""
        <div class="lp-preview-card">
          <div class="lp-preview-chrome">
            <div class="lp-preview-dots"><span></span><span></span><span></span></div>
            <span class="lp-preview-tag">truthlenseai · example output</span>
          </div>
          <div class="lp-preview-body">
            <div class="lp-preview-row warn">
              <div>
                <div class="lp-preview-label">🗣️ Hate Speech Detected</div>
                <div class="lp-preview-row-meta">Language: Tamil (தமிழ்) · Text</div>
              </div>
              <span class="lp-pill">Confidence: 91.8%</span>
            </div>
            <div class="lp-preview-row safe">
              <div>
                <div class="lp-preview-label">🔐 Deepfake Check Passed</div>
                <div class="lp-preview-row-meta">Language: Sinhala (සිංහල) · Audio</div>
              </div>
              <span class="lp-pill">REAL</span>
            </div>
            <div class="lp-wave-label">Wav2Vec2 audio waveform (example)</div>
            <div class="lp-wave">
              <span style="height:35%"></span><span style="height:60%"></span><span style="height:40%"></span>
              <span style="height:85%"></span><span style="height:55%"></span><span style="height:70%"></span>
              <span style="height:30%"></span><span style="height:50%"></span><span style="height:90%"></span>
              <span style="height:45%"></span><span style="height:65%"></span><span style="height:38%"></span>
              <span style="height:75%"></span><span style="height:42%"></span><span style="height:58%"></span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Stats (sourced from the actual models — not placeholder numbers) ──
    st.markdown("""
    <div class="lp-stats lp-animate lp-d2">
      <div class="lp-stat"><div class="num">3</div><div class="lbl">Live Detection Modules</div></div>
      <div class="lp-stat"><div class="num cyan">2</div><div class="lbl">Languages — Tamil &amp; Sinhala</div></div>
      <div class="lp-stat"><div class="num purple">96.24%</div><div class="lbl">Tamil Text+Audio Fusion Accuracy</div></div>
      <div class="lp-stat"><div class="num green">5</div><div class="lbl">Content Types Supported</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Detection modules ───────────────────────────────────
    st.markdown('<div id="lp-modules"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="lp-section-kicker">Our Detection Modules</div>
    <div class="lp-section-title">Purpose-Built for Each Language &amp; Media Type</div>
    <div class="lp-section-sub">Separately trained models tuned for Sinhala and Tamil, across text, audio, and video.</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="lp-module-card">
          <div class="lp-module-icon">🇱🇰</div>
          <div class="lp-module-title">Tamil Hate Speech</div>
          <div class="lp-module-desc">Text, audio &amp; multimodal fusion detection using XLM-RoBERTa + SVM.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="go_tamil", use_container_width=True, type="primary"):
            st.session_state.page = "tamil"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="lp-module-card">
          <div class="lp-module-icon">🌐</div>
          <div class="lp-module-title">Sinhala Hate Speech</div>
          <div class="lp-module-desc">Text, audio, deepfake &amp; YouTube detection using XLM-RoBERTa + Wav2Vec2.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="go_sinhala", use_container_width=True, type="primary"):
            st.session_state.page = "sinhala"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="lp-module-card">
          <div class="lp-module-icon">🎬</div>
          <div class="lp-module-title">Video Deepfake Detection</div>
          <div class="lp-module-desc">Frame-level temporal analysis to detect manipulated video (face-swap style deepfakes).</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="go_video", use_container_width=True, type="primary"):
            st.session_state.page = "video_deepfake"
            st.rerun()

    with col4:
        st.markdown("""
        <div class="lp-module-card">
          <div class="lp-module-icon">📰</div>
          <div class="lp-module-title">Sinhala False Content</div>
          <div class="lp-module-desc">YouTube title/description/transcript consistency detection using XLM-RoBERTa.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="go_false_content", use_container_width=True, type="primary"):
            st.session_state.page = "false_content"
            st.rerun()

    # ── How it works ────────────────────────────────────────
    st.markdown('<div id="lp-how"></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="lp-section-kicker">Process Overview</div>
    <div class="lp-section-title">Three Simple Steps</div>
    <div class="lp-section-sub">The same flow across every module — text, audio, or a YouTube link.</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="lp-step">
          <div class="lp-step-num">01</div>
          <div class="lp-step-title">Choose a Module</div>
          <div class="lp-step-desc">Pick the language and content type — text, audio file, or a YouTube URL.</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="lp-step">
          <div class="lp-step-num">02</div>
          <div class="lp-step-title">Model Analysis</div>
          <div class="lp-step-desc">The relevant model(s) run locally — text classification, audio, deepfake, or transcription.</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="lp-step">
          <div class="lp-step-num">03</div>
          <div class="lp-step-title">Review Results</div>
          <div class="lp-step-desc">See confidence scores and explanations, then download a report where available.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── About panel ───────────────────────────────────────────
    st.markdown('<div id="lp-about"></div>', unsafe_allow_html=True)
    about_text, about_visual = st.columns([1.1, 1], gap="large")
    with about_text:
        st.markdown("""
        <div class="lp-panel-text">
          <div class="lp-panel-kicker">Why Two Separate Models</div>
          <h3>Built for Underserved Languages</h3>
          <p>
            Sinhala and Tamil are both underrepresented in mainstream content-moderation tooling.
            TruthLenseAI brings together separately fine-tuned models for each language instead of
            relying on a single general-purpose multilingual classifier.
          </p>
          <div class="lp-panel-check"><span class="dot">✓</span> XLM-RoBERTa fine-tuned separately for Sinhala and Tamil text</div>
          <div class="lp-panel-check"><span class="dot">✓</span> Wav2Vec2 models for audio hate speech and audio deepfake detection</div>
          <div class="lp-panel-check"><span class="dot">✓</span> Whisper / Wav2Vec2-XLS-R speech-to-text feeding text classification for YouTube audio</div>
          <div class="lp-panel-check"><span class="dot">✓</span> A separate consistency model checking Sinhala YouTube titles/descriptions against transcripts</div>
        </div>
        """, unsafe_allow_html=True)
    with about_visual:
        st.markdown("""
        <div class="lp-panel-visual"><span class="lp-panel-visual-glyph">🛡️</span></div>
        """, unsafe_allow_html=True)

    # ── CTA band ────────────────────────────────────────────
    st.markdown("""
    <div class="lp-cta-band">
      <h2>Ready to try it yourself?</h2>
      <p>Pick a module below and run text, audio, or a YouTube link through it — no signup required.</p>
      <div class="lp-cta-band-buttons">
        <a class="lp-hero-link-btn" style="background:#ffffff" href="#lp-modules">Explore the Modules ↑</a>
        <a class="lp-hero-link-btn" style="background:transparent;border-color:rgba(255,255,255,0.6);color:#ffffff !important" href="#lp-how">See How It Works</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ──────────────────────────────────────────────
    st.markdown("""
    <div class="lp-footer">
      <div class="lp-footer-grid">
        <div class="lp-footer-brand-col">
          <div class="brand">🛡️ TruthLenseAI</div>
          <div class="tagline">Multilingual hate speech and false-content detection for Sinhala &amp; Tamil.</div>
        </div>
        <div class="lp-footer-col">
          <h4>Modules</h4>
          <a href="#lp-modules">Tamil Hate Speech</a>
          <a href="#lp-modules">Sinhala Hate Speech</a>
          <a href="#lp-modules">Sinhala False Content</a>
          <a href="#lp-modules">Video Deepfake Detection</a>
        </div>
        <div class="lp-footer-col">
          <h4>Technology</h4>
          <span class="item">XLM-RoBERTa</span>
          <span class="item">Wav2Vec2</span>
          <span class="item">Whisper / Wav2Vec2-XLS-R</span>
          <span class="item">YouTube Data API</span>
        </div>
        <div class="lp-footer-col">
          <h4>About</h4>
          <span class="item">Research project covering Sinhala &amp; Tamil content moderation</span>
          <span class="item">Text, audio, deepfake &amp; video analysis</span>
        </div>
      </div>
      <div class="lp-footer-bottom">🛡️ TruthLenseAI &nbsp;·&nbsp; Sinhala &amp; Tamil hate speech and false-content detection</div>
    </div>
    """, unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════
# TAMIL HATE SPEECH PAGE
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "tamil":

    # Page-scoped dark theme, matching the Sinhala/Deepfake pages' look —
    # only injected while this page is being rendered, so Home/Sinhala/False
    # Content are completely unaffected. Only visual: no button's behavior,
    # key, or on_click logic below is touched, only how it's painted.
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"], .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #0d1538 30%, #111d4a 60%, #0a1230 100%) !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap:0; background:rgba(15,23,60,0.6); border-radius:16px; padding:6px; border:1px solid rgba(99,102,241,0.15); }
    .stTabs [data-baseweb="tab"] { height:50px; border-radius:12px; color:#94a3b8; font-weight:600; border:none; background:transparent; }
    .stTabs [aria-selected="true"] { background:linear-gradient(135deg,#4f46e5,#6366f1)!important; color:#fff!important; box-shadow:0 4px 15px rgba(99,102,241,0.3); }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
    .stButton > button {
        background:linear-gradient(135deg,#4f46e5,#6366f1,#7c3aed) !important;
        color:#fff !important;
        border:none !important;
        border-radius:12px !important;
        padding:0.7rem 2.2rem !important;
        font-weight:700 !important;
        box-shadow:0 4px 15px rgba(99,102,241,0.3) !important;
    }
    .stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 25px rgba(99,102,241,0.45) !important; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
        background: #f8fafc !important;
        border: 1px solid rgba(99,102,241,0.35) !important;
        color: #1e293b !important;
        border-radius: 12px !important;
    }
    [data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder { color: #64748b !important; }
    label { color: #cbd5e1 !important; }
    /* Section headers (e.g. "Video Summary", "Comments Section") default to
       a dark color meant for a light page — invisible against the new dark
       background. Only text color changes here, nothing structural. */
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
    /* .result-hate / .result-safe are their own light pink/green cards
       (not dark), so .result-sub's original dark text was already correct
       there — reverted back to dark after over-correcting for the dark
       page in the previous pass. */
    .result-sub { color: #475569 !important; }
    .result-hate .result-sub strong, .result-safe .result-sub strong, .result-sub strong {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    [data-testid="stExpander"] {
        background: rgba(15,23,60,0.5) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
        color: #e2e8f0 !important;
    }
    /* Plain text/captions (e.g. video title + "Channel · Duration" near the
       thumbnail) also default to dark, light-page text. */
    [data-testid="stCaptionContainer"], [data-testid="stCaption"] { color: #94a3b8 !important; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] strong { color: #e2e8f0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # Back button
    if st.button("← Back to Home", key="back_home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("""
    <div class="hero">
      <h1>🛡️ Tamil Hate Speech Detector <span class="version-badge">v1.0</span></h1>
      <p>Analyze Tamil and Tanglish content using text, audio, batch, and YouTube analysis.</p>

    </div>
    """, unsafe_allow_html=True)

    # ── Tabs shown in the UI. Flip a flag to True to bring that tab back
    # (e.g. for PP2) — the tab's code is untouched, just skipped while False.
    SHOW_AUDIO_TAB    = False
    SHOW_LIVE_TAB     = False
    SHOW_DOCUMENT_TAB = False
    SHOW_BATCH_TAB    = False

    tab_labels = ["📝  Text Analysis"]
    if SHOW_AUDIO_TAB:
        tab_labels.append("🎵  Audio Analysis")
    tab_labels.append("🎧  Audio Files Analysis")
    if SHOW_BATCH_TAB:
        tab_labels.append("📊  Batch Analysis")
    tab_labels.append("🎬  YouTube Analysis")
    if SHOW_LIVE_TAB:
        tab_labels.append("🎤  Live Detection")
    if SHOW_DOCUMENT_TAB:
        tab_labels.append("📄  Document Analysis")

    _tabs = iter(st.tabs(tab_labels))
    tab1 = next(_tabs)                              # Text Analysis
    tab2 = next(_tabs) if SHOW_AUDIO_TAB else None   # Audio Analysis
    tab3 = next(_tabs)                              # Audio Files Analysis
    tab4 = next(_tabs) if SHOW_BATCH_TAB else None   # Batch Analysis
    tab5 = next(_tabs)                              # YouTube Analysis
    tab6 = next(_tabs) if SHOW_LIVE_TAB else None    # Live Detection
    tab7 = next(_tabs) if SHOW_DOCUMENT_TAB else None  # Document Analysis

    # ── TAB 1: Text prediction ────────────────────────────────
    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Text Analysis")
        st.markdown('<p style="color:#64748b;margin-top:-8px">Analyze Tamil or Tanglish text for harmful or offensive language.</p>', unsafe_allow_html=True)

        text_input = st.text_area(
            "Enter Tamil or Tanglish text:",
            height=130,
            placeholder="Type or paste Tamil / Tanglish text here to analyze it...",
            label_visibility="collapsed"
        )

        if st.button("Analyze Content", type="primary", key="btn_text", use_container_width=True):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={"text": text_input, "use_lexicon": False},
                            timeout=30
                        )
                        result   = response.json()
                        verdict  = result.get("verdict", "SAFE")
                        is_hate  = verdict == "HATE"
                        source   = result.get("source")
                        conf_val = result.get("confidence", 0)

                        css_cls = "result-hate" if is_hate else "result-safe"
                        icon    = "🔴 Offensive Detected" if is_hate else "🟢 Not Offensive"

                        if is_hate:
                            if source == "lexicon":
                                reason = "Offensive language was detected."
                            else:
                                reason = ("Strong indicators of harmful language were detected." if conf_val >= 90 else
                                          "Potentially harmful language was detected.")
                        else:
                            reason = ("No harmful language was detected." if verdict == "SAFE" else
                                      "No harmful content was detected with sufficient confidence. Consider reviewing the content if needed.")

                        conf_level    = "High" if verdict in ("SAFE", "HATE") else "Medium"
                        detected_lang = "Tamil" if re.search(r'[஀-௿]', text_input) else "Tanglish / English"

                        st.markdown("#### Assessment")
                        st.markdown(f"""
                        <div class="{css_cls}">
                          <div class="result-title">{icon}</div>
                          <div class="result-sub" style="margin-top:10px;"><strong>Reason</strong><br>{reason}</div>
                          <div class="result-sub" style="margin-top:10px;"><strong>Detected language:</strong><br>{detected_lang}</div>
                          <div class="result-sub" style="margin-top:10px;"><strong>Confidence:</strong><br>{result['confidence']}% ({conf_level})</div>
                        </div>""", unsafe_allow_html=True)

                        with st.expander("Full API response"):
                            st.json(result)
                    except Exception as e:
                        st.error(f"API Error: {e} — Make sure Flask API is running!")
            else:
                st.warning("Please enter some Tamil text!")
        st.markdown('</div>', unsafe_allow_html=True)

    if SHOW_AUDIO_TAB:
        # ── TAB 2: Audio prediction ───────────────────────────────
        with tab2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Audio Analysis")
            st.markdown('<p style="color:#64748b;margin-top:-8px">104 acoustic features extracted with Librosa, classified by SVM</p>', unsafe_allow_html=True)

            audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])
            if audio_file is not None:
                st.audio(audio_file, format="audio/wav")

            if st.button("Analyze audio", type="primary", key="btn_audio", use_container_width=True):
                if audio_file is not None:
                    with st.spinner("Extracting 104 acoustic features..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/predict_audio",
                                files={"audio": (audio_file.name, audio_file.getvalue(), "audio/wav")},
                                timeout=300
                            )
                            result = response.json()
                            if "error" in result:
                                st.error(f"Processing error: {result['error']}")
                            else:
                                is_hate = result["label"] == "hate"
                                css_cls = "result-hate" if is_hate else "result-safe"
                                icon    = "⚠️ HATE SPEECH DETECTED" if is_hate else "✅ NOT HATE SPEECH"
                                st.markdown(f"""
                                <div class="{css_cls}">
                                  <div class="result-title">{icon}</div>
                                  <div class="result-sub">Confidence: <strong>{result['confidence']}%</strong> &nbsp;·&nbsp; Model: SVM (85.86%)</div>
                                </div>""", unsafe_allow_html=True)
                                with st.expander("Full API response"):
                                    st.json(result)
                        except requests.exceptions.Timeout:
                            st.error("Request timed out — try a shorter clip (under 30 seconds).")
                        except Exception as e:
                            st.error(f"API Error: {e}")
                else:
                    st.warning("Please upload a WAV file!")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 3: Audio Files Analysis (fusion) ───────────────────
    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Audio Files Analysis")
        st.markdown('<p style="color:#64748b;margin-top:-8px">Upload audio to check it for harmful or offensive language, using both what\'s said and how it\'s said.</p>', unsafe_allow_html=True)

        fusion_audio = st.file_uploader("Upload WAV audio file", type=["wav"], key="fusion_audio")
        if fusion_audio is not None:
            st.audio(fusion_audio, format="audio/wav")

        if st.button("Analyze Audio", type="primary", key="btn_fusion", use_container_width=True):
            if fusion_audio is not None:
                progress_bar     = st.progress(0)
                step_placeholder = st.empty()

                try:
                    step_placeholder.markdown("**Step 1 / 4** — Cleaning up audio...")
                    progress_bar.progress(10)

                    response = requests.post(
                        f"{API_URL}/predict_fusion_audio",
                        files={"audio": (fusion_audio.name, fusion_audio.getvalue(), "audio/wav")},
                        timeout=300
                    )
                    result = response.json()

                    if "error" in result:
                        progress_bar.empty()
                        step_placeholder.empty()
                        st.error(f"API error: {result['error']}")
                    else:
                        import time
                        steps = [
                            "**Step 1 / 4** — Cleaning up audio ✓",
                            "**Step 2 / 4** — Transcribing speech ✓",
                            "**Step 3 / 4** — Analyzing content ✓",
                            "**Step 4 / 4** — Finalizing result ✓",
                        ]
                        for i, step in enumerate(steps):
                            step_placeholder.markdown(step)
                            progress_bar.progress((i + 1) * 25)
                            time.sleep(0.25)

                        step_placeholder.empty()
                        progress_bar.empty()

                        is_hate    = result["label"] == "hate"
                        css_cls    = "result-hate" if is_hate else "result-safe"
                        icon       = "🔴 Offensive Speech Detected" if is_hate else "🟢 Not Offensive Speech Detected"
                        confidence = result["confidence"]
                        bar_color  = "#dc2626" if is_hate else "#16a34a"

                        # Tone tag — SVM (audio) signal only, and only shown when the
                        # overall result is offensive. Not shown for safe content at all.
                        tone_label = tone_icon = tone_color = None
                        if is_hate:
                            if result["audio_hate_prob"] >= 85:
                                tone_label, tone_icon, tone_color = "Aggressive Tone", "🔥", "#dc2626"
                            else:
                                tone_label, tone_icon, tone_color = "Calm Tone", "😐", "#2563eb"

                        # ── Main badge: verdict + confidence bar only ──
                        st.markdown(
                            f'<div class="{css_cls}">'
                            f'<div class="result-title">{icon}</div>'
                            f'<div class="result-sub">Overall Confidence: <strong>{confidence}%</strong></div>'
                            f'<div class="conf-bar-track"><div class="conf-bar-fill" style="width:{confidence}%;background:{bar_color};"></div></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        transcript_text = result["transcription"]
                        duration_sec    = get_wav_duration(fusion_audio.getvalue())
                        timestamp_str   = f"0:00 – {format_timestamp(duration_sec)}"

                        # ── Details: always visible, mirrors everything in the downloadable report ──
                        st.markdown("#### Details")
                        with st.container(border=True):
                            asr_badge = {"Sarvam AI": "S", "ElevenLabs": "E", "Google STT": "G", "Whisper": "W"}
                            asr_used = result.get("asr_model", "")
                            st.markdown(
                                f'**Transcript** '
                                f'<span title="{asr_used}" style="display:inline-block;width:16px;height:16px;'
                                f'line-height:16px;text-align:center;border-radius:3px;background:#e2e8f0;'
                                f'color:#334155;font-weight:700;font-size:10px;">'
                                f'{asr_badge.get(asr_used, "?")}</span>',
                                unsafe_allow_html=True
                            )
                            st.markdown(f'<div class="info-box">"{html.escape(transcript_text)}"</div>', unsafe_allow_html=True)

                            st.markdown(f"**Timestamp:** {timestamp_str}")

                            if tone_label:
                                st.markdown(
                                    f'<span class="tone-badge" style="background:{tone_color}1a;color:{tone_color};border:2px solid {tone_color};">'
                                    f'{tone_icon} Offensive content included, with {tone_label}</span>',
                                    unsafe_allow_html=True
                                )

                        st.markdown("---")
                        fusion_report_data = {
                            "filename":       fusion_audio.name,
                            "generated":      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "is_offensive":   is_hate,
                            "verdict_text":   "Offensive Speech Detected" if is_hate else "Not Offensive Speech Detected",
                            "confidence":     confidence,
                            "tone_label":     tone_label,
                            "tone_icon":      tone_icon,
                            "tone_color":     tone_color,
                            "transcript_text": html.escape(transcript_text),
                            "timestamp":      timestamp_str,
                        }
                        st.download_button(
                            "📄 Download Report",
                            data=build_fusion_report(fusion_report_data),
                            file_name="audio_analysis_report.html",
                            mime="text/html",
                            use_container_width=True,
                        )

                except Exception as e:
                    progress_bar.empty()
                    step_placeholder.empty()
                    st.error(f"API Error: {e} — Make sure Flask API is running!")
            else:
                st.warning("Please upload a WAV file!")
        st.markdown('</div>', unsafe_allow_html=True)

    if SHOW_BATCH_TAB:
        # ── TAB 4: Batch & API status ─────────────────────────────
        with tab4:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Batch Text Analysis")
            st.markdown('<p style="color:#64748b;margin-top:-8px">Analyze multiple comments at once for harmful or offensive language.</p>', unsafe_allow_html=True)

            default_batch = [
                "semma video da! super content pannra",
                "dei nee enna pannra idhu ellam wrong",
                "நல்ல பாடல் இது மிகவும் அழகாக இருக்கிறது",
                "இந்த ஜாதியினர் எல்லாம் இப்படித்தான்",
                "Romba nalla pesinanga today semma",
                "unga ellaarum oru mokka pasanga da",
                "super acting bro vera level",
                "dei nee pesra ellam poi da",
                "இந்த படம் மிகவும் சிறப்பாக உள்ளது",
                "nee enna nenachen unakku"
            ]

            if "batch_comments" not in st.session_state:
                st.session_state.batch_comments = list(default_batch)

            st.markdown('<p style="color:#64748b;font-size:13px">Click any comment to edit it, use the + row at the bottom to add new ones, select rows to delete them.</p>', unsafe_allow_html=True)

            edited_df = st.data_editor(
                pd.DataFrame({"Comment": st.session_state.batch_comments}),
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={"Comment": st.column_config.TextColumn("Comment", width="large")}
            )

            col_save, col_reset, _ = st.columns([1, 1, 2])
            with col_save:
                if st.button("Save changes", key="btn_save_batch", use_container_width=True):
                    saved = [r for r in edited_df["Comment"].tolist() if r and str(r).strip()]
                    st.session_state.batch_comments = saved
                    st.success(f"Saved {len(saved)} comments.")
                    st.rerun()
            with col_reset:
                if st.button("Reset to defaults", key="btn_reset_batch", use_container_width=True):
                    st.session_state.batch_comments = list(default_batch)
                    st.rerun()

            sample_batch = st.session_state.batch_comments
            table_placeholder = st.empty()

            if st.button("Analyze Comments", type="primary", key="btn_batch", use_container_width=True):
                try:
                    with st.spinner("Analyzing comments..."):
                        response = requests.post(
                            f"{API_URL}/predict_batch",
                            json={"texts": sample_batch},
                            timeout=300
                        )
                        batch_result = response.json()

                    results_data = [{
                        "Comment":    r["text"][:55] + "…" if len(r["text"]) > 55 else r["text"],
                        "Result":     "🔴 Harmful" if r["label"] == "hate" else "🟢 Safe",
                        "Confidence": f"{r['confidence']}%"
                    } for r in batch_result["results"]]

                    table_placeholder.dataframe(
                        pd.DataFrame(results_data), use_container_width=True, hide_index=True
                    )

                    h = batch_result["hate_count"]
                    s = batch_result["safe_count"]
                    t = batch_result["total"]
                    st.markdown(f"""
                    <div class="stat-row">
                      <div class="stat-card"><div class="num num-blue">{t}</div><div class="lbl">Analyzed Comments</div></div>
                      <div class="stat-card"><div class="num num-red">{h}</div><div class="lbl">Harmful</div></div>
                      <div class="stat-card"><div class="num num-green">{s}</div><div class="lbl">Safe</div></div>
                      <div class="stat-card"><div class="num num-red">{batch_result['hate_pct']}%</div><div class="lbl">Harmful Content Rate</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                    if h > 0:
                        st.warning(f"{h} potentially harmful comment(s) were detected out of {t} analyzed comments.")
                    else:
                        st.success("No potentially harmful comments were detected among the analyzed comments.")
                except Exception as e:
                    st.error(f"API Error: {e} — Make sure Flask API is running on port 5000")

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            with st.expander("Technical Details"):
                if st.button("Check API status", key="btn_health", use_container_width=True):
                    try:
                        response = requests.get(f"{API_URL}/health", timeout=5)
                        data = response.json()
                        st.success("Flask API is running.")
                        st.json(data)
                    except Exception:
                        st.error("Flask API is not running. Start it with: python api/app.py")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 5: YouTube Analysis ───────────────────────────────
    with tab5:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### YouTube Analysis")
        st.markdown('<p style="color:#64748b;margin-top:-8px">Paste a YouTube URL to analyze its audio and comments for harmful or offensive language.</p>', unsafe_allow_html=True)

        yt_url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")

        if st.button("Analyze Video and Comments", type="primary", key="btn_yt", use_container_width=True):
            if yt_url.strip():
                report_data = {
                    "url": yt_url.strip(),
                    "generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "video": None,
                    "comments": None,
                }

                with st.spinner("Fetching video details..."):
                    try:
                        info_response = requests.post(
                            f"{API_URL}/youtube_video_info",
                            json={"youtube_url": yt_url.strip()},
                            timeout=30
                        )
                        video_info = info_response.json()
                    except Exception:
                        video_info = {}
                preview = render_yt_video_preview(video_info)
                if preview:
                    report_data["video_title"], report_data["video_channel"], report_data["video_duration"] = preview

                yt_result = None
                with st.spinner("Downloading audio and analyzing chunks... (may take 1-2 minutes)"):
                    try:
                        response = requests.post(
                            f"{API_URL}/analyze_youtube",
                            json={"youtube_url": yt_url.strip(), "use_lexicon": False},
                            timeout=300
                        )
                        if not response.text.strip():
                            st.error(f"Backend returned empty response (HTTP {response.status_code}). Check Flask terminal for errors.")
                            st.stop()
                        try:
                            yt_result = response.json()
                        except Exception:
                            st.error(f"Backend error (HTTP {response.status_code}): {response.text[:300]}")
                            st.stop()
                    except requests.exceptions.Timeout:
                        yt_result = {"error": "Timed out — video may be too long. Try a shorter video."}
                    except Exception as e:
                        yt_result = {"error": str(e)}
                report_data["video"] = render_yt_video_results(yt_result)

                st.markdown("---")
                st.markdown("#### Comments Section")
                cmt_result = None
                with st.spinner("Fetching and analyzing comments..."):
                    try:
                        cmt_response = requests.post(
                            f"{API_URL}/analyze_youtube_comments",
                            json={"youtube_url": yt_url.strip(), "use_lexicon": False},
                            timeout=180
                        )
                        cmt_result = cmt_response.json()
                        if "error" in cmt_result:
                            cmt_result = {"error": cmt_result["error"]}
                    except requests.exceptions.Timeout:
                        cmt_result = {"error": "Timed out fetching comments — try again or a shorter comment list."}
                    except Exception as e:
                        cmt_result = {"error": str(e)}
                report_data["comments"] = render_yt_comments_results(cmt_result)

                st.session_state["yt_analysis"] = {
                    "url": yt_url.strip(),
                    "generated": report_data["generated"],
                    "video_info": video_info,
                    "yt_result": yt_result,
                    "cmt_result": cmt_result,
                }

                if report_data["video"] is not None:
                    st.markdown("---")
                    st.download_button(
                        "📄 Download Report",
                        data=build_html_report(report_data),
                        file_name="youtube_analysis_report.html",
                        mime="text/html",
                        use_container_width=True,
                        key="yt_download_btn",
                    )
            else:
                st.warning("Please paste a YouTube URL.")

        else:
            # Not a fresh "Analyze" click this run (e.g. the rerun triggered by
            # clicking Download Report below) — redraw the last results from
            # session_state instead of losing the whole page.
            analysis = st.session_state.get("yt_analysis")
            if analysis:
                video_info = analysis["video_info"]
                yt_result  = analysis["yt_result"]
                cmt_result = analysis["cmt_result"]

                report_data = {
                    "url": analysis["url"],
                    "generated": analysis["generated"],
                    "video": None,
                    "comments": None,
                }

                preview = render_yt_video_preview(video_info)
                if preview:
                    report_data["video_title"], report_data["video_channel"], report_data["video_duration"] = preview

                report_data["video"] = render_yt_video_results(yt_result)

                st.markdown("---")
                st.markdown("#### Comments Section")
                report_data["comments"] = render_yt_comments_results(cmt_result)

                if report_data["video"] is not None:
                    st.markdown("---")
                    st.download_button(
                        "📄 Download Report",
                        data=build_html_report(report_data),
                        file_name="youtube_analysis_report.html",
                        mime="text/html",
                        use_container_width=True,
                        key="yt_download_btn",
                    )
        st.markdown('</div>', unsafe_allow_html=True)

    if SHOW_LIVE_TAB:
        # ── TAB 6: Live Microphone Detection ─────────────────────
        with tab6:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Live Microphone Detection")
            st.markdown('<p style="color:#64748b;margin-top:-8px">Record your voice and classify it in real time using the SVM audio model</p>', unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
              Click the microphone button below to record. When done, click <strong>Analyze recording</strong> to classify the audio.
            </div>
            """, unsafe_allow_html=True)

            try:
                mic_audio = st.audio_input("🎤 Click to record", key="mic_input")
            except AttributeError:
                mic_audio = st.file_uploader("Upload recorded WAV audio", type=["wav"], key="mic_upload")

            if mic_audio is not None:
                st.audio(mic_audio, format="audio/wav")

            if st.button("Analyze recording", type="primary", key="btn_mic", use_container_width=True):
                if mic_audio is not None:
                    with st.spinner("Analyzing audio..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/predict_mic",
                                files={"audio": ("recording.wav", mic_audio.getvalue(), "audio/wav")},
                                timeout=60
                            )
                            result = response.json()
                            if "error" in result:
                                st.error(f"Error: {result['error']}")
                            else:
                                is_hate = result["label"] == "hate"
                                css_cls = "result-hate" if is_hate else "result-safe"
                                icon    = "⚠️ HATE SPEECH DETECTED" if is_hate else "✅ NOT HATE SPEECH"
                                st.markdown(f"""
                                <div class="{css_cls}">
                                  <div class="result-title">{icon}</div>
                                  <div class="result-sub">Confidence: <strong>{result['confidence']}%</strong> &nbsp;·&nbsp; Model: SVM Audio</div>
                                </div>""", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"API Error: {e}")
                else:
                    st.warning("Please record or upload audio first.")
            st.markdown('</div>', unsafe_allow_html=True)

    if SHOW_DOCUMENT_TAB:
        # ── TAB 7: Document Analysis ──────────────────────────────
        with tab7:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Document Analysis")
            st.markdown('<p style="color:#64748b;margin-top:-8px">Upload a .txt or .pdf file — each paragraph is analysed with XLM-RoBERTa</p>', unsafe_allow_html=True)

            doc_file = st.file_uploader("Upload document (.txt or .pdf)", type=["txt", "pdf"], key="doc_upload")

            if st.button("Analyze document", type="primary", key="btn_doc", use_container_width=True):
                if doc_file is not None:
                    with st.spinner("Extracting text and analyzing paragraphs..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/analyze_document",
                                files={"file": (doc_file.name, doc_file.getvalue(), "application/octet-stream")},
                                timeout=120
                            )
                            doc_result = response.json()

                            if "error" in doc_result:
                                st.error(f"Error: {doc_result['error']}")
                            else:
                                h = doc_result["hate_paragraphs"]
                                s = doc_result["safe_paragraphs"]
                                t = doc_result["total_paragraphs"]
                                st.markdown(f"""
                                <div class="stat-row">
                                  <div class="stat-card"><div class="num num-blue">{t}</div><div class="lbl">Paragraphs scanned</div></div>
                                  <div class="stat-card"><div class="num num-red">{h}</div><div class="lbl">Hate detected</div></div>
                                  <div class="stat-card"><div class="num num-green">{s}</div><div class="lbl">Safe</div></div>
                                </div>
                                """, unsafe_allow_html=True)

                                st.markdown("**Paragraph-by-paragraph results:**")
                                for i, r in enumerate(doc_result["results"]):
                                    is_hate  = r["label"] == "hate"
                                    bg       = "#fef2f2" if is_hate else "#f0fdf4"
                                    border   = "#dc2626" if is_hate else "#16a34a"
                                    lbl_col  = "#dc2626" if is_hate else "#16a34a"
                                    icon     = "🚨" if is_hate else "✅"
                                    lbl_txt  = "HATE SPEECH" if is_hate else "SAFE"
                                    text     = r["paragraph"][:300] + "..." if len(r["paragraph"]) > 300 else r["paragraph"]
                                    st.markdown(f"""
                                    <div style="background:{bg};border-left:5px solid {border};border-radius:8px;
                                                padding:12px 16px;margin:6px 0;">
                                      <div style="font-size:11px;font-weight:700;color:{lbl_col};margin-bottom:4px;">
                                        {icon} PARAGRAPH {i+1} — {lbl_txt} ({r['confidence']}%)
                                      </div>
                                      <div style="font-size:13px;color:#374151;line-height:1.5;">{text}</div>
                                    </div>""", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"API Error: {e}")
                else:
                    st.warning("Please upload a .txt or .pdf file.")
            st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SINHALA HATE SPEECH PAGE
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "sinhala":

    # Back button
    if st.button("← Back to Home", key="back_home_sinhala"):
        st.session_state.page = "home"
        st.rerun()

    render_sinhala_tabs()


# ════════════════════════════════════════════════════════════
# SINHALA FALSE CONTENT PAGE
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "false_content":

    # Back button
    if st.button("← Back to Home", key="back_home_false_content"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("""
    <div class="hero">
      <h1>📰 Sinhala False Content Detector</h1>
      <p>Checks whether a YouTube video's title and description are supported by its transcript.</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "This embeds the project's own existing page unchanged, running on its own backend "
        "(port 5002). Paste a YouTube URL or fetch by video id, then run the consistency check."
    )

    components.iframe(f"{FALSE_CONTENT_URL}/test", height=1400, scrolling=True)


# ════════════════════════════════════════════════════════════
# VIDEO DEEPFAKE DETECTION PAGE
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "video_deepfake":

    # Back button
    if st.button("← Back to Home", key="back_home_video_deepfake"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("""
    <div class="hero">
      <h1>🎬 Video Deepfake Detection</h1>
      <p>Frame-level temporal analysis — upload a video to check for manipulated (face-swapped) content.</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "This embeds the project's own existing page unchanged, running on its own backend "
        "(port 5003). Upload a video file directly — YouTube-URL analysis isn't wired up here yet."
    )

    components.iframe(VIDEO_DEEPFAKE_URL, height=1600, scrolling=True)

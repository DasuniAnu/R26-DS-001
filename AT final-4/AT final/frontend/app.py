import streamlit as st
import requests
import json
from io import BytesIO
from pathlib import Path
from html import escape
from datetime import datetime

API_BASE = "http://127.0.0.1:5000"

st.set_page_config(page_title="Sinhala Hate Speech Detector", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

# ── CSS Theme ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.stApp { background: linear-gradient(135deg, #0a0e27 0%, #0d1538 30%, #111d4a 60%, #0a1230 100%); font-family: 'Inter', sans-serif; }
#MainMenu, footer, header, .stDeployButton { display: none !important; visibility: hidden !important; }
.block-container { padding-top: 2rem; max-width: 1200px; }
.hero { text-align:center; padding:2rem 1rem 1.5rem; }
.hero-icon { font-size:3.5rem; display:block; animation:pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
.hero-title { font-size:2.4rem; font-weight:800; background:linear-gradient(135deg,#60a5fa,#a78bfa,#818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-sub { font-size:1rem; color:#94a3b8; }
.stTabs [data-baseweb="tab-list"] { gap:0; background:rgba(15,23,60,0.6); border-radius:16px; padding:6px; border:1px solid rgba(99,102,241,0.15); }
.stTabs [data-baseweb="tab"] { height:50px; border-radius:12px; color:#94a3b8; font-weight:600; border:none; background:transparent; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#4f46e5,#6366f1)!important; color:#fff!important; box-shadow:0 4px 15px rgba(99,102,241,0.3); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stTextArea textarea { background:rgba(10,14,39,0.7)!important; border:1px solid rgba(99,102,241,0.2)!important; border-radius:12px!important; color:#e2e8f0!important; font-size:1rem!important; }
.stTextArea textarea:focus { border-color:rgba(99,102,241,0.6)!important; box-shadow:0 0 20px rgba(99,102,241,0.15)!important; }
.stButton>button { background:linear-gradient(135deg,#4f46e5,#6366f1,#7c3aed)!important; color:#fff!important; border:none!important; border-radius:12px!important; padding:0.7rem 2.5rem!important; font-weight:700!important; font-size:1rem!important; box-shadow:0 4px 15px rgba(99,102,241,0.3)!important; width:100%; }
.stButton>button:hover { transform:translateY(-2px)!important; box-shadow:0 8px 25px rgba(99,102,241,0.45)!important; }
.rc { border-radius:20px; padding:1.8rem; margin-bottom:1rem; backdrop-filter:blur(15px); box-shadow:0 8px 32px rgba(0,0,0,0.25); animation:su 0.5s ease-out; }
@keyframes su { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
.rc-safe { background:linear-gradient(145deg,rgba(16,185,129,0.12),rgba(5,150,105,0.06)); border:1px solid rgba(16,185,129,0.25); }
.rc-danger { background:linear-gradient(145deg,rgba(239,68,68,0.12),rgba(220,38,38,0.06)); border:1px solid rgba(239,68,68,0.25); }
.rc-info { background:linear-gradient(145deg,rgba(59,130,246,0.12),rgba(37,99,235,0.06)); border:1px solid rgba(59,130,246,0.25); }
.rc-warn { background:linear-gradient(145deg,rgba(245,158,11,0.12),rgba(217,119,6,0.06)); border:1px solid rgba(245,158,11,0.25); }
.rl { font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.4rem; }
.rv { font-size:1.7rem; font-weight:800; margin-bottom:0.3rem; }
.rconf { font-size:0.9rem; color:#94a3b8; font-weight:500; }
.cbar-wrap { background:rgba(10,14,39,0.6); border-radius:10px; height:10px; margin-top:0.5rem; overflow:hidden; }
.cbar { height:100%; border-radius:10px; transition:width 1s ease-out; }
.cbar-g { background:linear-gradient(90deg,#10b981,#34d399); }
.cbar-r { background:linear-gradient(90deg,#ef4444,#f87171); }
.cbar-b { background:linear-gradient(90deg,#3b82f6,#60a5fa); }
.cbar-y { background:linear-gradient(90deg,#f59e0b,#fbbf24); }
.dc { background:linear-gradient(145deg,rgba(15,23,60,0.6),rgba(20,30,75,0.4)); border:1px solid rgba(99,102,241,0.1); border-radius:16px; padding:1.5rem; margin-top:1rem; }
.dt { font-size:0.85rem; font-weight:600; color:#818cf8; text-transform:uppercase; letter-spacing:1px; margin-bottom:1rem; }
.dr { display:flex; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid rgba(99,102,241,0.08); }
.dk { color:#94a3b8; font-size:0.85rem; }
.dv { color:#e2e8f0; font-size:0.85rem; font-weight:600; }
label { color:#94a3b8!important; font-weight:500!important; }
</style>
""", unsafe_allow_html=True)

def cbar(v, cls):
    return f'<div class="cbar-wrap"><div class="cbar {cls}" style="width:{int(v*100)}%"></div></div>'


def build_youtube_pdf_report(meta, summary, segments, comments, comments_summary, comments_source, source_url):
    """Create a downloadable, reader-friendly PDF for one YouTube analysis."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
        from xml.sax.saxutils import escape
    except ImportError as exc:
        raise RuntimeError("PDF reporting needs ReportLab. Install the frontend requirements first.") from exc

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=1.55 * cm, rightMargin=1.55 * cm,
        topMargin=1.65 * cm, bottomMargin=1.55 * cm,
        title="YouTube Analysis Report",
        author="Sinhala Hate Speech Detector",
    )

    # Nirmala UI is included with modern Windows versions and supports Sinhala.
    # If it is unavailable, the report still downloads with the built-in font.
    font_name, font_bold = "Helvetica", "Helvetica-Bold"
    try:
        regular = Path("C:/Windows/Fonts/Nirmala.ttf")
        bold = Path("C:/Windows/Fonts/NirmalaB.ttf")
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ReportNirmala", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportNirmalaBold", str(bold)))
            font_name, font_bold = "ReportNirmala", "ReportNirmalaBold"
    except Exception:
        pass

    navy, indigo, purple = colors.HexColor("#0D1538"), colors.HexColor("#4F46E5"), colors.HexColor("#7C3AED")
    slate, muted, green, red, amber = colors.HexColor("#172554"), colors.HexColor("#64748B"), colors.HexColor("#059669"), colors.HexColor("#DC2626"), colors.HexColor("#D97706")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("reportTitle", parent=styles["Title"], fontName=font_bold, fontSize=19, leading=24, textColor=navy, alignment=TA_CENTER, spaceAfter=4)
    subtitle = ParagraphStyle("reportSubtitle", parent=styles["Normal"], fontName=font_name, fontSize=9, leading=13, textColor=muted, alignment=TA_CENTER, spaceAfter=16)
    heading = ParagraphStyle("reportHeading", parent=styles["Heading2"], fontName=font_bold, fontSize=12, leading=16, textColor=navy, spaceBefore=13, spaceAfter=7)
    normal = ParagraphStyle("reportNormal", parent=styles["BodyText"], fontName=font_name, fontSize=8.5, leading=12, textColor=slate)
    small = ParagraphStyle("reportSmall", parent=normal, fontSize=7.4, leading=10)
    table_header = ParagraphStyle("reportTableHeader", parent=small, fontName=font_bold, textColor=colors.white)

    def text(value):
        return escape(str(value if value is not None else "Not available"))

    def section(name):
        return [Paragraph(name, heading), HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#C7D2FE"), spaceAfter=7)]

    def metric_card(label, value, accent):
        return Table([[Paragraph(f'<font color="#64748B" size="7">{text(label).upper()}</font><br/><font color="{accent}" size="15"><b>{text(value)}</b></font>', normal)]], colWidths=[4.15 * cm], rowHeights=[1.45 * cm], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#E2E8F0")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(accent)),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
        ]))

    def page_decor(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#C7D2FE"))
        canvas.line(document.leftMargin, 1.05 * cm, A4[0] - document.rightMargin, 1.05 * cm)
        canvas.setFont(font_name, 7)
        canvas.setFillColor(muted)
        canvas.drawString(document.leftMargin, 0.65 * cm, "Sinhala Hate Speech Detector - YouTube Analysis Report")
        canvas.drawRightString(A4[0] - document.rightMargin, 0.65 * cm, f"Page {document.page}")
        canvas.restoreState()

    duration_m, duration_s = divmod(int(meta.get("duration", 0) or 0), 60)
    report_url = meta.get("url", source_url)
    story = [
        Paragraph("YouTube Analysis Report", title),
        Paragraph("Sinhala hate speech and audio authenticity assessment", subtitle),
    ]
    story += section("Video information")
    info = [
        [Paragraph("Title", small), Paragraph(text(meta.get("title", "Untitled video")), normal)],
        [Paragraph("Channel", small), Paragraph(text(meta.get("channel", "Not available")), normal)],
        [Paragraph("Duration", small), Paragraph(f"{duration_m:02d}:{duration_s:02d}", normal)],
        [Paragraph("Source URL", small), Paragraph(text(report_url), small)],
        [Paragraph("Report generated", small), Paragraph(datetime.now().strftime("%d %B %Y, %H:%M"), normal)],
    ]
    story.append(Table(info, colWidths=[3.25 * cm, 14.0 * cm], style=TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2FF")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])))

    story += section("Analysis overview")
    cards = [[
        metric_card("Segments analysed", summary.get("total_segments", 0), "#4F46E5"),
        metric_card("Hate segments", f"{summary.get('hate_segments', 0)} ({summary.get('hate_percentage', 0)}%)", "#DC2626"),
        metric_card("Fake segments", f"{summary.get('fake_segments', 0)} ({summary.get('fake_percentage', 0)}%)", "#D97706"),
        metric_card("Clean segments", summary.get("clean_segments", 0), "#059669"),
    ]]
    story.append(Table(cards, colWidths=[4.18 * cm] * 4, style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")])))
    risk_parts = []
    if summary.get("hate_segments", 0):
        risk_parts.append(f"{summary['hate_segments']} audio segment(s) were flagged for hate speech")
    if summary.get("fake_segments", 0):
        risk_parts.append(f"{summary['fake_segments']} audio segment(s) were flagged as potentially synthetic")
    if not risk_parts:
        risk_parts.append("No audio segments were flagged by the selected models")
    story += [Spacer(1, 8), Paragraph("<b>Assessment:</b> " + text("; ".join(risk_parts)) + ". Review flagged timestamps alongside the original video before making a final decision.", normal)]

    story += section("Audio segment timeline")
    timeline_rows = [[Paragraph("#", table_header), Paragraph("Time", table_header), Paragraph("Hate speech", table_header), Paragraph("Confidence", table_header), Paragraph("Authenticity", table_header), Paragraph("Confidence", table_header)]]
    for index, segment in enumerate(segments, 1):
        hate_label = "OFF" if segment.get("hate_speech") else "NOT"
        fake_label = "FAKE" if not segment.get("audio_authentic", True) else "REAL"
        timeline_rows.append([
            Paragraph(str(index), small), Paragraph(f"{text(segment.get('start_fmt', ''))} - {text(segment.get('end_fmt', ''))}", small),
            Paragraph(f'<font color="{"#DC2626" if hate_label == "OFF" else "#059669"}"><b>{hate_label}</b></font>', small),
            Paragraph(f"{segment.get('hate_confidence', 0):.1%}", small),
            Paragraph(f'<font color="{"#D97706" if fake_label == "FAKE" else "#059669"}"><b>{fake_label}</b></font>', small),
            Paragraph(f"{segment.get('authenticity_confidence', 0):.1%}", small),
        ])
    story.append(Table(timeline_rows, repeatRows=1, colWidths=[0.75 * cm, 3.25 * cm, 3.05 * cm, 2.35 * cm, 3.2 * cm, 2.35 * cm], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])))

    if comments_summary is not None:
        story += section("Sinhala Unicode comment analysis")
        comment_rows = [[Paragraph("Comments analysed", small), Paragraph(str(comments_summary.get("total_fetched", 0)), normal)], [Paragraph("Offensive comments", small), Paragraph(f"{comments_summary.get('hate_comments', 0)} ({comments_summary.get('hate_percentage', 0)}%)", normal)], [Paragraph("Comment source", small), Paragraph(text(comments_source or "Not available"), normal)]]
        story.append(Table(comment_rows, colWidths=[5.4 * cm, 11.85 * cm], style=TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2FF")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E2E8F0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])))
        flagged = [comment for comment in comments if comment.get("is_offensive")]
        if flagged:
            story += [Spacer(1, 8), Paragraph("Flagged comments", ParagraphStyle("flagHeading", parent=normal, fontName=font_bold, textColor=red, spaceAfter=4))]
            flagged_rows = [[Paragraph("Author", table_header), Paragraph("Comment", table_header), Paragraph("Confidence", table_header)]]
            for comment in flagged:
                flagged_rows.append([Paragraph(text(comment.get("author", "Unknown")), small), Paragraph(text(comment.get("text", "")), small), Paragraph(f"{comment.get('confidence', 0):.1%}", small)])
            story.append(Table(flagged_rows, repeatRows=1, colWidths=[3.2 * cm, 11.0 * cm, 3.05 * cm], style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7F1D1D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#FECACA")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FEF2F2")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])))
        else:
            story += [Spacer(1, 8), Paragraph("No offensive Sinhala Unicode comments were detected.", normal)]

    story += section("Method notes")
    story.append(Paragraph("Audio is converted to 16 kHz mono and assessed in speech-aware segments. Audio hate speech uses Wav2Vec2; audio authenticity uses WavLM; Sinhala speech is transcribed with Wav2Vec2 XLS-R; and Sinhala text/comments are assessed with XLM-RoBERTa. Model outputs are decision-support signals and should be reviewed with the source media.", normal))
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return buffer.getvalue()


def build_youtube_html_report(meta, summary, segments, comments, comments_summary, comments_source, source_url):
    """Create a standalone, browser-friendly HTML report for download."""
    def value(item):
        return escape(str(item if item is not None else "Not available"))

    duration_m, duration_s = divmod(int(meta.get("duration", 0) or 0), 60)
    report_url = meta.get("url", source_url)
    timeline = []
    for index, segment in enumerate(segments, 1):
        hate_label = "OFF" if segment.get("hate_speech") else "NOT"
        fake_label = "FAKE" if not segment.get("audio_authentic", True) else "REAL"
        timeline.append(f"""
            <tr>
              <td>{index}</td><td>{value(segment.get('start_fmt', ''))} - {value(segment.get('end_fmt', ''))}</td>
              <td><span class="badge {'danger' if hate_label == 'OFF' else 'safe'}">{hate_label}</span></td><td>{segment.get('hate_confidence', 0):.1%}</td>
              <td><span class="badge {'warn' if fake_label == 'FAKE' else 'safe'}">{fake_label}</span></td><td>{segment.get('authenticity_confidence', 0):.1%}</td>
            </tr>""")

    comment_section = ""
    if comments_summary is not None:
        flagged = [comment for comment in comments if comment.get("is_offensive")]
        flagged_rows = "".join(
            f"<tr><td>{value(comment.get('author', 'Unknown'))}</td><td>{value(comment.get('text', ''))}</td><td>{comment.get('confidence', 0):.1%}</td></tr>"
            for comment in flagged
        ) or '<tr><td colspan="3" class="empty">No offensive Sinhala Unicode comments were detected.</td></tr>'
        comment_section = f"""
          <section>
            <h2>Sinhala Unicode comment analysis</h2>
            <div class="details"><div><b>Comments analysed</b><span>{comments_summary.get('total_fetched', 0)}</span></div><div><b>Offensive comments</b><span>{comments_summary.get('hate_comments', 0)} ({comments_summary.get('hate_percentage', 0)}%)</span></div><div><b>Comment source</b><span>{value(comments_source or 'Not available')}</span></div></div>
            <h3>Flagged comments</h3>
            <table><thead><tr><th>Author</th><th>Comment</th><th>Confidence</th></tr></thead><tbody>{flagged_rows}</tbody></table>
          </section>"""

    risk_notes = []
    if summary.get("hate_segments", 0):
        risk_notes.append(f"{summary['hate_segments']} audio segment(s) were flagged for hate speech")
    if summary.get("fake_segments", 0):
        risk_notes.append(f"{summary['fake_segments']} audio segment(s) were flagged as potentially synthetic")
    assessment = "; ".join(risk_notes) if risk_notes else "No audio segments were flagged by the selected models"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>YouTube Analysis Report - {value(meta.get('title', 'Video'))}</title>
<style>
  * {{ box-sizing:border-box }} body {{ margin:0; background:#f1f5f9; color:#172554; font:15px/1.5 Inter,Arial,'Nirmala UI',sans-serif }}
  main {{ max-width:1040px; margin:32px auto; background:#fff; padding:48px 54px; box-shadow:0 8px 30px #0f172a20 }}
  header {{ text-align:center; border-bottom:2px solid #c7d2fe; padding-bottom:24px }} h1 {{ margin:0; font-size:30px; color:#0d1538 }} .sub {{ color:#64748b; margin:6px 0 0 }}
  section {{ margin-top:32px }} h2 {{ font-size:20px; margin:0 0 12px; padding-bottom:8px; border-bottom:1px solid #c7d2fe }} h3 {{ font-size:15px; color:#b91c1c; margin:20px 0 8px }}
  .details {{ border:1px solid #e2e8f0; border-radius:8px; overflow:hidden }} .details div {{ display:grid; grid-template-columns:210px 1fr; border-bottom:1px solid #e2e8f0 }} .details div:last-child {{ border:0 }} .details b {{ padding:10px 12px; background:#eef2ff; font-size:13px }} .details span {{ padding:10px 12px; overflow-wrap:anywhere }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px }} .card {{ border:1px solid #e2e8f0; border-left:4px solid #4f46e5; border-radius:7px; padding:14px; background:#f8fafc }} .card.red {{ border-left-color:#dc2626 }} .card.orange {{ border-left-color:#d97706 }} .card.green {{ border-left-color:#059669 }} .label {{ display:block; color:#64748b; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.04em }} .number {{ display:block; font-size:22px; font-weight:800; margin-top:3px }}
  .assessment {{ background:#eef2ff; border-left:4px solid #4f46e5; padding:12px 15px; margin-top:16px; border-radius:4px }} table {{ width:100%; border-collapse:collapse; font-size:13px }} th {{ text-align:left; background:#0d1538; color:white; padding:10px }} td {{ padding:9px 10px; border:1px solid #e2e8f0; vertical-align:top }} tbody tr:nth-child(even) {{ background:#f8fafc }} .badge {{ display:inline-block; border-radius:99px; padding:2px 9px; font-size:11px; font-weight:800 }} .safe {{ color:#047857; background:#d1fae5 }} .danger {{ color:#b91c1c; background:#fee2e2 }} .warn {{ color:#b45309; background:#fef3c7 }} .empty {{ color:#64748b; text-align:center; padding:18px }} footer {{ border-top:1px solid #c7d2fe; color:#64748b; font-size:12px; margin-top:35px; padding-top:16px }}
  @media print {{ body {{ background:white }} main {{ max-width:none; margin:0; box-shadow:none; padding:20px }} }} @media(max-width:700px) {{ main {{ margin:0; padding:24px 16px }} .cards {{ grid-template-columns:repeat(2,1fr) }} .details div {{ grid-template-columns:1fr }} }}
</style></head><body><main>
  <header><h1>YouTube Analysis Report</h1><p class="sub">Sinhala hate speech and audio authenticity assessment</p></header>
  <section><h2>Video information</h2><div class="details"><div><b>Title</b><span>{value(meta.get('title', 'Untitled video'))}</span></div><div><b>Channel</b><span>{value(meta.get('channel', 'Not available'))}</span></div><div><b>Duration</b><span>{duration_m:02d}:{duration_s:02d}</span></div><div><b>Source URL</b><span>{value(report_url)}</span></div><div><b>Report generated</b><span>{datetime.now().strftime('%d %B %Y, %H:%M')}</span></div></div></section>
  <section><h2>Analysis overview</h2><div class="cards"><div class="card"><span class="label">Segments analysed</span><span class="number">{summary.get('total_segments', 0)}</span></div><div class="card red"><span class="label">Hate segments</span><span class="number">{summary.get('hate_segments', 0)} ({summary.get('hate_percentage', 0)}%)</span></div><div class="card orange"><span class="label">Fake segments</span><span class="number">{summary.get('fake_segments', 0)} ({summary.get('fake_percentage', 0)}%)</span></div><div class="card green"><span class="label">Clean segments</span><span class="number">{summary.get('clean_segments', 0)}</span></div></div><div class="assessment"><b>Assessment:</b> {value(assessment)}. Review flagged timestamps alongside the original video before making a final decision.</div></section>
  <section><h2>Audio segment timeline</h2><table><thead><tr><th>#</th><th>Time</th><th>Hate speech</th><th>Confidence</th><th>Authenticity</th><th>Confidence</th></tr></thead><tbody>{''.join(timeline)}</tbody></table></section>
  {comment_section}
  <footer><b>Method notes:</b> Audio is converted to 16 kHz mono and assessed in speech-aware segments. Audio hate speech uses Wav2Vec2; audio authenticity uses WavLM; Sinhala speech is transcribed with Wav2Vec2 XLS-R; and Sinhala text/comments are assessed with XLM-RoBERTa. Model outputs are decision-support signals and should be reviewed with the source media.</footer>
</main></body></html>"""

# ── Header ──
st.markdown('<div class="hero"><span class="hero-icon">🛡️</span><div class="hero-title">Sinhala Hate Speech Detector</div><div class="hero-sub">AI-powered detection for text & audio · Unicode & Romanized Sinhala</div></div>', unsafe_allow_html=True)

tab_text, tab_audio, tab_yt = st.tabs(["📝  Text Detection", "🎙️  Audio Detection", "🎬  YouTube Analysis"])

# ── TEXT TAB ──
with tab_text:
    st.markdown('<p style="color:#64748b;font-size:0.85rem;margin-bottom:0.5rem;">Supports <b style="color:#818cf8">Unicode Sinhala</b> (සිංහල) and <b style="color:#818cf8">Romanized Sinhala</b> (singhala)</p>', unsafe_allow_html=True)
    text_input = st.text_area("Text", height=150, placeholder="උදා: ඔයා නරකයි / oya naraka balla ...", label_visibility="collapsed")
    c1, c2 = st.columns([1, 2])
    with c1:
        text_go = st.button("🔍 Analyze Text", key="tb")
    if text_go and text_input.strip():
        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(f"{API_BASE}/api/text/predict", json={"text": text_input.strip()}, timeout=120).json()
                if resp.get("success"):
                    r = resp["result"]
                    off = r["is_offensive"]; conf = r["confidence"]; p = r["probabilities"]
                    c1, c2 = st.columns(2)
                    with c1:
                        if off:
                            st.markdown(f'<div class="rc rc-danger"><div class="rl" style="color:#fca5a5">Detection Result</div><div class="rv" style="color:#f87171">⚠️ Offensive</div><div class="rconf">Confidence: {conf:.1%}</div>{cbar(conf,"cbar-r")}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="rc rc-safe"><div class="rl" style="color:#6ee7b7">Detection Result</div><div class="rv" style="color:#34d399">✅ Not Offensive</div><div class="rconf">Confidence: {conf:.1%}</div>{cbar(conf,"cbar-g")}</div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="rc rc-info"><div class="rl" style="color:#93c5fd">Probability Breakdown</div><div style="margin-top:0.8rem"><div style="display:flex;justify-content:space-between;margin-bottom:0.3rem"><span style="color:#94a3b8;font-size:0.9rem">Not Offensive</span><span style="color:#e2e8f0;font-weight:700">{p["NOT"]:.1%}</span></div>{cbar(p["NOT"],"cbar-g")}<div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;margin-top:1rem"><span style="color:#94a3b8;font-size:0.9rem">Offensive</span><span style="color:#e2e8f0;font-weight:700">{p["OFF"]:.1%}</span></div>{cbar(p["OFF"],"cbar-r")}</div></div>', unsafe_allow_html=True)
                    pt = r["processed_text"][:100] + ("..." if len(r["processed_text"]) > 100 else "")
                    st.markdown(f'<div class="dc"><div class="dt">📋 Processing Details</div><div class="dr"><span class="dk">Script Detected</span><span class="dv">{r["script_type"].title()}</span></div><div class="dr"><span class="dk">Processed Text</span><span class="dv" style="max-width:60%;text-align:right;word-break:break-all">{pt}</span></div><div class="dr"><span class="dk">Model</span><span class="dv">XLM-RoBERTa (SOLD fine-tuned)</span></div></div>', unsafe_allow_html=True)
                else:
                    st.error(f"API Error: {resp.get('error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Start Flask server on port 5000.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    elif text_go:
        st.warning("Please enter text first.")

# ── AUDIO TAB ──
with tab_audio:
    st.markdown('<p style="color:#64748b;font-size:0.85rem;margin-bottom:0.5rem;">Supports <b style="color:#818cf8">WAV, MP3, MP4, OGG, FLAC, M4A</b> · Max 10s analyzed</p>', unsafe_allow_html=True)
    audio_file = st.file_uploader("Upload audio", type=["wav","mp3","mp4","ogg","flac","m4a","webm"], label_visibility="collapsed")
    if audio_file:
        st.audio(audio_file, format="audio/wav")
    c1, c2 = st.columns([1, 2])
    with c1:
        audio_go = st.button("🔍 Analyze Audio", key="ab")
    if audio_go and audio_file:
        with st.spinner("Analyzing audio (may take a moment on first run)..."):
            try:
                audio_file.seek(0)
                resp = requests.post(f"{API_BASE}/api/audio/predict", files={"audio": (audio_file.name, audio_file, "audio/wav")}, timeout=180).json()
                if resp.get("success"):
                    r = resp["result"]
                    c1, c2 = st.columns(2)
                    with c1:
                        ih = r["hate_speech"]; hc = r["hate_confidence"]; hp = r["hate_probabilities"]
                        if ih:
                            st.markdown(f'<div class="rc rc-danger"><div class="rl" style="color:#fca5a5">🗣️ Hate Speech</div><div class="rv" style="color:#f87171">⚠️ Offensive</div><div class="rconf">Confidence: {hc:.1%}</div>{cbar(hc,"cbar-r")}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="rc rc-safe"><div class="rl" style="color:#6ee7b7">🗣️ Hate Speech</div><div class="rv" style="color:#34d399">✅ Not Offensive</div><div class="rconf">Confidence: {hc:.1%}</div>{cbar(hc,"cbar-g")}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="dc"><div class="dt">Hate Probabilities</div><div class="dr"><span class="dk">NOT</span><span class="dv">{hp["NOT"]:.1%}</span></div><div class="dr"><span class="dk">OFF</span><span class="dv">{hp["OFF"]:.1%}</span></div></div>', unsafe_allow_html=True)
                    with c2:
                        ir = r["audio_authentic"]; ac = r["authenticity_confidence"]; ap = r["authenticity_probabilities"]
                        if ir:
                            st.markdown(f'<div class="rc rc-safe"><div class="rl" style="color:#6ee7b7">🔐 Deepfake Detection</div><div class="rv" style="color:#34d399">✅ Authentic</div><div class="rconf">Confidence: {ac:.1%}</div>{cbar(ac,"cbar-g")}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="rc rc-warn"><div class="rl" style="color:#fcd34d">🔐 Deepfake Detection</div><div class="rv" style="color:#fbbf24">⚠️ Fake Audio</div><div class="rconf">Confidence: {ac:.1%}</div>{cbar(ac,"cbar-y")}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="dc"><div class="dt">Authenticity Probabilities</div><div class="dr"><span class="dk">REAL</span><span class="dv">{ap["REAL"]:.1%}</span></div><div class="dr"><span class="dk">FAKE</span><span class="dv">{ap["FAKE"]:.1%}</span></div></div>', unsafe_allow_html=True)
                    st.markdown('<div class="dc" style="margin-top:1.5rem"><div class="dt">📋 Model Info</div><div class="dr"><span class="dk">Hate Model</span><span class="dv">Wav2Vec2 (fine-tuned)</span></div><div class="dr"><span class="dk">Deepfake Model</span><span class="dv">Wav2Vec2 (fine-tuned)</span></div><div class="dr"><span class="dk">Audio Processing</span><span class="dv">16kHz mono, 10s max</span></div></div>', unsafe_allow_html=True)
                else:
                    st.error(f"API Error: {resp.get('error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Start Flask server on port 5000.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    elif audio_go:
        st.warning("Please upload an audio file first.")


# ── YOUTUBE TAB ──
with tab_yt:
    st.markdown('<p style="color:#64748b;font-size:0.85rem;margin-bottom:0.5rem;">Analyzes both Audio (10s segments, max 10 min) and Comments</p>', unsafe_allow_html=True)
    
    with st.form(key="yt_form", clear_on_submit=False):
        yt_url = st.text_area("Paste YouTube URL Here:", height=68, placeholder="https://www.youtube.com/watch?v=...")
        analyze_comments = st.checkbox("Analyze Sinhala Unicode comments", value=False)
        c1, c2 = st.columns([1, 3])
        with c1:
            yt_go = st.form_submit_button("🎬 Analyze Video")

    if yt_go and yt_url.strip():
        comments = []
        summ_c = {"total_fetched": 0, "hate_comments": 0, "hate_percentage": 0}
        comments_source = "YouTube"
        comments_success = False
        comments_finished = False
        video_success = False
        metadata_ready = False
        metadata_preview = st.empty()
        progress = st.status("⏳ Fetching video information...", expanded=True)

        try:
            metadata_resp = requests.post(
                f"{API_BASE}/api/youtube/metadata",
                json={"url": yt_url.strip()},
                timeout=60,
            )
            if metadata_resp.status_code != 200:
                progress.update(label="❌ Could not fetch video information", state="error")
                st.error(f"❌ Video Information Error: {metadata_resp.json().get('error', 'Unknown Error')}")
            else:
                meta = metadata_resp.json()["metadata"]
                metadata_ready = True
                dur_m, dur_s = divmod(int(meta.get("duration", 0)), 60)
                with metadata_preview.container():
                    ci1, ci2 = st.columns([1, 2])
                    with ci1:
                        if meta.get("thumbnail"):
                            st.image(meta["thumbnail"], use_container_width=True)
                    with ci2:
                        st.markdown(f'''<div class="dc" style="margin-top:0">
                            <div class="dt">📹 Video Info</div>
                            <div class="dr"><span class="dk">Title</span><span class="dv" style="max-width:65%;text-align:right">{meta["title"]}</span></div>
                            <div class="dr"><span class="dk">Channel</span><span class="dv">{meta["channel"]}</span></div>
                            <div class="dr"><span class="dk">Duration</span><span class="dv">{dur_m:02d}:{dur_s:02d}</span></div>
                        </div>''', unsafe_allow_html=True)
                progress.write("✅ Video information loaded. Starting audio analysis...")
        except Exception as e:
            progress.update(label="❌ Could not fetch video information", state="error")
            st.error(f"❌ Video Information Failed: {e}")

        if not metadata_ready:
            st.stop()

        with st.spinner("⏳ Analyzing video audio... (this may take a few minutes)"):
            try:
                resp = requests.post(f"{API_BASE}/api/youtube/analyze", json={"url": yt_url.strip()}, timeout=600)
                if resp.status_code == 200:
                    data = resp.json()
                    meta = data["metadata"]
                    segs = data["segments"]
                    summ = data["summary"]
                    video_success = True
                    progress.update(label="✅ Audio analysis complete", state="complete")
                else:
                    progress.update(label="❌ Audio analysis failed", state="error")
                    st.error(f"❌ Video Analysis Error: {resp.json().get('error', 'Unknown Error')}")
            except Exception as e:
                progress.update(label="❌ Audio analysis failed", state="error")
                st.error(f"❌ Video Analysis Failed: {e}")
                
        if video_success:
            # ── 4. Overall Cards ──
            s1, s2, s3 = st.columns(3)
            
            # Audio Hate Speech Card
            s1.markdown(f'''
            <div class="rc {"rc-danger" if summ["hate_segments"] > 0 else "rc-safe"}" style="text-align:center;padding:1rem;height:100%">
                <div class="rl" style="color:#{"fca5a5" if summ["hate_segments"]>0 else "6ee7b7"}">🎙️ Fused Hate Analysis</div>
                <div class="rv" style="color:#{"f87171" if summ["hate_segments"]>0 else "34d399"}">
                    {summ["hate_segments"]} / {summ["total_segments"]}
                </div>
                <div class="rconf">audio + transcript ({summ["hate_percentage"]}%)</div>
            </div>''', unsafe_allow_html=True)
            
            # Audio Authenticity (Deepfake) Card
            s2.markdown(f'''
            <div class="rc {"rc-warn" if summ["fake_segments"] > 0 else "rc-safe"}" style="text-align:center;padding:1rem;height:100%">
                <div class="rl" style="color:#{"fcd34d" if summ["fake_segments"]>0 else "6ee7b7"}">🔐 Audio Authenticity</div>
                <div class="rv" style="color:#{"fbbf24" if summ["fake_segments"]>0 else "34d399"}">
                    {summ["fake_segments"]} / {summ["total_segments"]}
                </div>
                <div class="rconf">deepfake segments ({summ["fake_percentage"]}%)</div>
            </div>''', unsafe_allow_html=True)
            
            # Hate Comments Card
            comment_card = s3.empty()

            def render_comment_card():
                if comments_success:
                    comment_card.markdown(f'''
                    <div class="rc {"rc-danger" if summ_c["hate_comments"] > 0 else "rc-safe"}" style="text-align:center;padding:1rem;height:100%">
                        <div class="rl" style="color:#{"fca5a5" if summ_c["hate_comments"]>0 else "6ee7b7"}">💬 Comment Hate Speech</div>
                        <div class="rv" style="color:#{"f87171" if summ_c["hate_comments"]>0 else "34d399"}">
                            {summ_c["hate_comments"]} / {summ_c["total_fetched"]}
                        </div>
                        <div class="rconf">comments flagged ({summ_c["hate_percentage"]}%)</div>
                    </div>''', unsafe_allow_html=True)
                else:
                    comment_card_value = "Processing" if analyze_comments and not comments_finished else "N/A"
                    comment_card_detail = "Audio results are ready; comments are processing below" if analyze_comments and not comments_finished else "Comments disabled or failed"
                    comment_card.markdown(f'''
                    <div class="rc rc-info" style="text-align:center;padding:1rem;height:100%">
                        <div class="rl" style="color:#93c5fd">💬 Comment Hate Speech</div>
                        <div class="rv" style="color:#60a5fa">{comment_card_value}</div>
                        <div class="rconf">{comment_card_detail}</div>
                    </div>''', unsafe_allow_html=True)

            render_comment_card()

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 5. Visual Timeline ──
            total_dur = meta.get("duration", 1) or 1
            timeline_html = '<div style="margin-bottom:0.4rem;color:#818cf8;font-size:0.8rem;font-weight:600;letter-spacing:1px;text-transform:uppercase">⏱ Audio Analysis Timeline</div>'
            timeline_html += '<div style="display:flex;width:100%;height:28px;border-radius:8px;overflow:hidden;border:1px solid rgba(99,102,241,0.2)">'
            for seg in segs:
                seg_dur  = seg["end_time"] - seg["start_time"]
                width_pct = round(seg_dur / total_dur * 100, 2)
                is_no_speech = seg.get("hate_label") == "NO_SPEECH" or seg.get("authenticity_label") == "NO_SPEECH"
                is_uncertain = seg.get("hate_label") == "UNCERTAIN" or seg.get("authenticity_label") == "UNCERTAIN"
                is_hate  = seg["hate_speech"]
                is_fake  = not seg["audio_authentic"]

                if is_no_speech:
                    color = "#64748b"   # slate - no speech
                    label_text = "NO SPEECH"
                elif is_uncertain:
                    color = "#a78bfa"   # purple - uncertain
                    label_text = "UNCERTAIN"
                elif is_hate and is_fake:
                    color = "#b45309"   # orange — both
                    label_text = "HATE SPEECH & DEEPFAKE"
                elif is_hate:
                    color = "#ef4444"   # red — hate
                    label_text = "HATE SPEECH"
                elif is_fake:
                    color = "#f59e0b"   # yellow — deepfake
                    label_text = "AUDIO DEEPFAKE"
                else:
                    color = "#10b981"   # green — clean
                    label_text = "SAFE"
                
                label = f"{seg['start_fmt']}–{seg['end_fmt']} → {label_text}"
                
                timeline_html += f'<div title="{label}" style="width:{width_pct}%;background:{color};transition:opacity .2s" onmouseover="this.style.opacity=0.7" onmouseout="this.style.opacity=1"></div>'
            timeline_html += '</div>'
            timeline_html += '<div style="display:flex;justify-content:space-between;margin-top:4px;color:#475569;font-size:0.75rem"><span>0:00</span><span>{:02d}:{:02d}</span></div>'.format(int(total_dur)//60, int(total_dur)%60)
            timeline_html += '<div style="display:flex;gap:1rem;margin-top:0.6rem;flex-wrap:wrap">'
            for col, lbl in [("#10b981","✅ Safe Audio"),("#ef4444","🔴 Audio Hate Speech"),("#f59e0b","🟡 Audio Deepfake"),("#b45309","🟠 Both Issues"),("#a78bfa","❓ Uncertain"),("#64748b","🔇 No Speech")]:
                timeline_html += f'<span style="display:flex;align-items:center;gap:4px;font-size:0.78rem;color:#94a3b8"><span style="width:12px;height:12px;border-radius:3px;background:{col};display:inline-block"></span>{lbl}</span>'
            timeline_html += '</div>'
            st.markdown(f'<div class="dc">{timeline_html}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 6. Segment-by-Segment Analysis with Explanations ──
            st.markdown('<div style="color:#818cf8;font-size:0.85rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.5rem">🔍 Segment-by-Segment Analysis</div>', unsafe_allow_html=True)

            for seg_i, seg in enumerate(segs):
                is_flagged = seg["hate_speech"] or not seg["audio_authentic"] or seg.get("hate_label") == "UNCERTAIN" or seg.get("authenticity_label") == "UNCERTAIN"
                issues = []
                if seg.get("hate_label") == "NO_SPEECH" or seg.get("authenticity_label") == "NO_SPEECH":
                    issues.append("🔇 No Speech")
                elif seg.get("hate_label") == "UNCERTAIN" or seg.get("authenticity_label") == "UNCERTAIN":
                    issues.append("❓ Uncertain")
                else:
                    if seg["hate_speech"]: issues.append("🔴 Hate Speech")
                    if not seg["audio_authentic"]: issues.append("🟡 Deepfake")
                status_label = " · ".join(issues) if issues else "✅ Clean"

                with st.expander(f"Segment {seg_i+1}  ⏱ {seg['start_fmt']} – {seg['end_fmt']}  │  {status_label}", expanded=is_flagged):
                    expl = seg.get("explanation", {})
                    h_expl = expl.get("hate_explanation", {})
                    f_expl = expl.get("fake_explanation", {})
                    a_feat = expl.get("audio_features", {})

                    transcript = seg.get("transcript", "")
                    transcript_result = seg.get("transcript_analysis") or {}
                    fusion_label = seg.get("hate_label", "NOT_ASSESSED")
                    fusion_color = "#f87171" if fusion_label == "OFF" else "#fbbf24" if fusion_label == "UNCERTAIN" else "#60a5fa" if fusion_label == "NOT_ASSESSED" else "#34d399"
                    if transcript:
                        transcript_text = transcript
                    elif seg.get("asr_status") == "LOW_QUALITY_OUTPUT":
                        transcript_text = "A reliable Sinhala transcript could not be produced for this segment, so it was excluded from text hate classification."
                    else:
                        transcript_text = "No transcript produced (no speech detected or no usable speech content)."
                    transcript_status = transcript_result.get("prediction", "NOT ASSESSED")
                    st.markdown(f'''
                    <div class="dc" style="margin:0 0 1rem 0;padding:1rem 1.2rem">
                        <div class="dt">📝 Sinhala Wav2Vec2-XLS-R ASR + XLM-RoBERTa Hate Fusion</div>
                        <div style="color:#e2e8f0;font-size:0.9rem;line-height:1.5;margin-bottom:0.7rem">{transcript_text}</div>
                        <div style="display:flex;gap:1.5rem;flex-wrap:wrap;font-size:0.8rem">
                            <span style="color:#94a3b8">Audio: <b style="color:#e2e8f0">{seg.get("audio_hate_label", "N/A")}</b></span>
                            <span style="color:#94a3b8">Transcript: <b style="color:#e2e8f0">{transcript_status}</b></span>
                            <span style="color:#94a3b8">Fused Result: <b style="color:{fusion_color}">{fusion_label}</b></span>
                        </div>
                        <div style="color:#94a3b8;font-size:0.78rem;margin-top:0.6rem">{seg.get("fusion_reason", "")}</div>
                    </div>''', unsafe_allow_html=True)

                    hp = seg.get("hate_probabilities", {"NOT": 0, "OFF": 0})
                    ap = seg.get("authenticity_probabilities", {"REAL": 0, "FAKE": 0})

                    ec1, ec2 = st.columns(2)

                    # ── Hate Speech Explanation Column ──
                    with ec1:
                        h_label = seg.get("hate_label")
                        if h_label == "NO_SPEECH":
                            h_cls = "rc-info"; h_icon = "🔇 No Speech"; h_col = "#60a5fa"; h_lbl_col = "#93c5fd"; h_bar_cls = "cbar-b"
                        elif h_label == "UNCERTAIN":
                            h_cls = "rc-warn"; h_icon = "❓ Uncertain"; h_col = "#fbbf24"; h_lbl_col = "#fcd34d"; h_bar_cls = "cbar-y"
                        else:
                            h_cls = "rc-danger" if seg["hate_speech"] else "rc-safe"
                            h_icon = "⚠️ Offensive" if seg["hate_speech"] else "✅ Not Offensive"
                            h_col = "#f87171" if seg["hate_speech"] else "#34d399"
                            h_lbl_col = "#fca5a5" if seg["hate_speech"] else "#6ee7b7"
                            h_bar_cls = "cbar-r" if seg["hate_speech"] else "cbar-g"

                        card = f'<div class="rc {h_cls}" style="padding:1.2rem">'
                        card += f'<div class="rl" style="color:{h_lbl_col}">🗣️ Hate Speech Analysis</div>'
                        card += f'<div class="rv" style="color:{h_col};font-size:1.3rem">{h_icon}</div>'
                        card += f'<div class="rconf">Confidence: {seg["hate_confidence"]:.1%} ({h_expl.get("confidence_level", "N/A")})</div>'
                        card += cbar(seg["hate_confidence"], h_bar_cls)

                        # Probability bars
                        card += '<div style="margin-top:1rem">'
                        card += f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px"><span style="color:#94a3b8">NOT</span><span style="color:#e2e8f0;font-weight:600">{hp["NOT"]:.1%}</span></div>'
                        card += cbar(hp["NOT"], "cbar-g")
                        card += f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-top:0.5rem;margin-bottom:2px"><span style="color:#94a3b8">OFF</span><span style="color:#e2e8f0;font-weight:600">{hp["OFF"]:.1%}</span></div>'
                        card += cbar(hp["OFF"], "cbar-r")
                        card += '</div>'

                        # Key factors
                        factors = h_expl.get("factors", [])
                        if factors:
                            card += '<div style="margin-top:1rem;border-top:1px solid rgba(99,102,241,0.1);padding-top:0.8rem">'
                            card += '<div style="color:#818cf8;font-size:0.75rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.5rem">Key Factors</div>'
                            for f in factors:
                                card += f'<div style="color:#cbd5e1;font-size:0.82rem;padding:3px 0;display:flex;gap:6px"><span style="color:#818cf8">•</span> {f}</div>'
                            card += '</div>'

                        # Summary
                        summary = h_expl.get("summary", "")
                        if summary:
                            card += f'<div style="margin-top:0.8rem;padding:0.7rem;background:rgba(10,14,39,0.5);border-radius:8px;border-left:3px solid {h_col}">'
                            card += f'<div style="color:#94a3b8;font-size:0.78rem;font-style:italic">{summary}</div></div>'

                        card += '</div>'
                        st.markdown(card, unsafe_allow_html=True)

                    # ── Deepfake Explanation Column ──
                    with ec2:
                        f_label = seg.get("authenticity_label")
                        if f_label == "NO_SPEECH":
                            f_cls = "rc-info"; f_icon = "🔇 No Speech"; f_col = "#60a5fa"; f_lbl_col = "#93c5fd"; f_bar_cls = "cbar-b"
                        elif f_label == "UNCERTAIN":
                            f_cls = "rc-warn"; f_icon = "❓ Uncertain"; f_col = "#fbbf24"; f_lbl_col = "#fcd34d"; f_bar_cls = "cbar-y"
                        else:
                            f_cls = "rc-warn" if not seg["audio_authentic"] else "rc-safe"
                            f_icon = "⚠️ Fake Audio" if not seg["audio_authentic"] else "✅ Authentic"
                            f_col = "#fbbf24" if not seg["audio_authentic"] else "#34d399"
                            f_lbl_col = "#fcd34d" if not seg["audio_authentic"] else "#6ee7b7"
                            f_bar_cls = "cbar-y" if not seg["audio_authentic"] else "cbar-g"

                        card = f'<div class="rc {f_cls}" style="padding:1.2rem">'
                        card += f'<div class="rl" style="color:{f_lbl_col}">🔐 Deepfake Analysis</div>'
                        card += f'<div class="rv" style="color:{f_col};font-size:1.3rem">{f_icon}</div>'
                        card += f'<div class="rconf">Confidence: {seg["authenticity_confidence"]:.1%} ({f_expl.get("confidence_level", "N/A")})</div>'
                        card += cbar(seg["authenticity_confidence"], f_bar_cls)

                        # Probability bars
                        card += '<div style="margin-top:1rem">'
                        card += f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px"><span style="color:#94a3b8">REAL</span><span style="color:#e2e8f0;font-weight:600">{ap["REAL"]:.1%}</span></div>'
                        card += cbar(ap["REAL"], "cbar-g")
                        card += f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-top:0.5rem;margin-bottom:2px"><span style="color:#94a3b8">FAKE</span><span style="color:#e2e8f0;font-weight:600">{ap["FAKE"]:.1%}</span></div>'
                        card += cbar(ap["FAKE"], "cbar-y")
                        card += '</div>'

                        # Key factors
                        factors = f_expl.get("factors", [])
                        if factors:
                            card += '<div style="margin-top:1rem;border-top:1px solid rgba(99,102,241,0.1);padding-top:0.8rem">'
                            card += '<div style="color:#818cf8;font-size:0.75rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.5rem">Key Factors</div>'
                            for f in factors:
                                card += f'<div style="color:#cbd5e1;font-size:0.82rem;padding:3px 0;display:flex;gap:6px"><span style="color:#818cf8">•</span> {f}</div>'
                            card += '</div>'

                        # Summary
                        summary = f_expl.get("summary", "")
                        if summary:
                            card += f'<div style="margin-top:0.8rem;padding:0.7rem;background:rgba(10,14,39,0.5);border-radius:8px;border-left:3px solid {f_col}">'
                            card += f'<div style="color:#94a3b8;font-size:0.78rem;font-style:italic">{summary}</div></div>'

                        card += '</div>'
                        st.markdown(card, unsafe_allow_html=True)

                    # ── Audio Signal Characteristics (full width) ──
                    if a_feat:
                        energy_col = "#f87171" if a_feat.get("energy_level") == "High" else "#fbbf24" if a_feat.get("energy_level") == "Medium" else "#34d399"
                        speech_icon = "✅ Yes" if a_feat.get("has_speech") else "⚠️ Minimal"
                        st.markdown(f'''
                        <div style="background:rgba(15,23,60,0.4);border:1px solid rgba(99,102,241,0.08);border-radius:10px;padding:0.8rem 1.2rem;margin-top:0.5rem;display:flex;gap:2rem;flex-wrap:wrap;font-size:0.8rem">
                            <div><span style="color:#64748b">Energy Level:</span> <span style="color:{energy_col};font-weight:600">{a_feat.get("energy_level", "N/A")}</span></div>
                            <div><span style="color:#64748b">RMS Energy:</span> <span style="color:#e2e8f0;font-weight:600">{a_feat.get("rms_energy", 0):.4f}</span></div>
                            <div><span style="color:#64748b">Silence Ratio:</span> <span style="color:#e2e8f0;font-weight:600">{a_feat.get("silence_ratio", 0):.1%}</span></div>
                            <div><span style="color:#64748b">Speech Detected:</span> <span style="font-weight:600">{speech_icon}</span></div>
                            <div><span style="color:#64748b">Zero-Crossing Rate:</span> <span style="color:#e2e8f0;font-weight:600">{a_feat.get("zero_crossing_rate", 0):.4f}</span></div>
                        </div>''', unsafe_allow_html=True)

            # Audio results above have already been sent to the page.  Run the
            # optional, potentially slower comment request only after that.
            if analyze_comments:
                with st.spinner("⏳ Audio analysis is ready. Analyzing Sinhala comments..."):
                    try:
                        resp_c = requests.post(
                            f"{API_BASE}/api/youtube/comments",
                            json={"url": yt_url.strip()},
                            timeout=1800,
                        )
                        if resp_c.status_code == 200:
                            data_c = resp_c.json()
                            comments = data_c["comments"]
                            summ_c = data_c["summary"]
                            comments_source = data_c.get("source", "YouTube")
                            comments_success = True
                        else:
                            st.warning(f"⚠️ Comment Analysis Error: {resp_c.json().get('error', 'Unknown Error')}")
                    except Exception as e:
                        st.warning(f"⚠️ Comment Analysis Failed: {e}")
                comments_finished = True
                render_comment_card()

            # ── 7. Comment Analysis Section ──
            if comments_success:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'''<div class="dc">
                    <div class="dt">💬 Comment Hate Speech Analysis</div>
                    <div class="dr"><span class="dk">Total Analyzed</span><span class="dv">{summ_c["total_fetched"]}</span></div>
                    <div class="dr"><span class="dk">Retrieved Via</span><span class="dv">{comments_source}</span></div>
                    <div class="dr"><span class="dk">Offensive Count</span><span class="dv" style="color:{"#f87171" if summ_c["hate_comments"]>0 else "#34d399"}">{summ_c["hate_comments"]}</span></div>
                    <div class="dr" style="border-bottom:none"><span class="dk">Offensive Percentage</span><span class="dv">{summ_c["hate_percentage"]}%</span></div>
                ''', unsafe_allow_html=True)
                
                hate_comments = [c for c in comments if c["is_offensive"]]
                if hate_comments:
                    tbl = '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;margin-top:1rem">'
                    tbl += '<tr style="border-bottom:1px solid rgba(99,102,241,0.2)"><th style="color:#818cf8;text-align:left;padding:0.4rem;width:20%">Author</th><th style="color:#818cf8;text-align:left;padding:0.4rem;width:60%">Detected Offensive Comment</th><th style="color:#818cf8;text-align:center;padding:0.4rem;width:20%">Confidence</th></tr>'
                    for c in hate_comments:
                        tbl += f'<tr style="border-bottom:1px solid rgba(99,102,241,0.06)"><td style="color:#e2e8f0;padding:0.4rem;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{c["author"]}</td><td style="color:#fca5a5;padding:0.4rem">{c["text"]}</td><td style="color:#f87171;text-align:center;padding:0.4rem">{c["confidence"]:.1%}</td></tr>'
                    tbl += '</table></div>'
                    st.markdown(tbl, unsafe_allow_html=True)
                else:
                    st.markdown('<div style="margin-top:1rem;color:#34d399;font-weight:600">✅ No offensive comments detected.</div></div>', unsafe_allow_html=True)

            # ── Download Report Button ──
            st.markdown("<br>", unsafe_allow_html=True)
            safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in meta.get('title', 'video')).strip().replace(' ', '_')[:50]
            try:
                report_html = build_youtube_html_report(
                    meta, summ, segs, comments if comments_success else [],
                    summ_c if comments_success else None,
                    comments_source if comments_success else None,
                    yt_url.strip(),
                )
                st.download_button(
                    label="📥 Download Full HTML Report",
                    data=report_html,
                    file_name=f"report_{safe_title or 'video'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    key="yt_report_dl",
                )
            except Exception as error:
                st.error(f"HTML report could not be generated: {error}")
    elif yt_go:
        st.warning("Please enter a YouTube URL first.")

# ── Footer ──
st.markdown('<div style="text-align:center;padding:2rem 0 1rem;margin-top:3rem;border-top:1px solid rgba(99,102,241,0.1)"><p style="color:#475569;font-size:0.8rem">🛡️ Sinhala Hate Speech Detection ·</p><p style="color:#334155;font-size:0.7rem">XLM-RoBERTa · Wav2Vec2 · SOLD Dataset · CPU Inference</p></div>', unsafe_allow_html=True)

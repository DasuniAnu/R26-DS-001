import streamlit as st
import requests

API_BASE = "http://localhost:5000"

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

# ── Header ──
st.markdown('<div class="hero"><span class="hero-icon">🛡️</span><div class="hero-title">Sinhala Hate Speech Detector</div><div class="hero-sub">AI-powered detection for text & audio · Unicode & Romanized Sinhala</div></div>', unsafe_allow_html=True)

tab_text, tab_audio = st.tabs(["📝  Text Detection", "🎙️  Audio Detection"])

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

# ── Footer ──
st.markdown('<div style="text-align:center;padding:2rem 0 1rem;margin-top:3rem;border-top:1px solid rgba(99,102,241,0.1)"><p style="color:#475569;font-size:0.8rem">🛡️ Sinhala Hate Speech Detection ·</p><p style="color:#334155;font-size:0.7rem">XLM-RoBERTa · Wav2Vec2 · SOLD Dataset · CPU Inference</p></div>', unsafe_allow_html=True)

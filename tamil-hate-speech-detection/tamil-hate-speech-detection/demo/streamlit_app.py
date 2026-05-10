import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:5000"

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
.hero p  { font-size: 1rem; opacity: 0.75; margin: 0; }
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
[data-testid="stTabs"] [role="tab"] { font-weight: 600; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    st.markdown("""
    <div class="hero">
      <h1>🛡️ VigilAI</h1>
      <p>Multilingual Hate Speech &amp; Deepfake Detection Platform — Tamil &amp; Sinhala</p>
      <div class="badge-row">
        <span class="badge">Tamil NLP</span>
        <span class="badge">Sinhala NLP</span>
        <span class="badge">Multimodal Fusion · 96.24%</span>
        <span class="badge">Deepfake Detection</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Select a Service")
    st.markdown('<p style="color:#64748b;margin-top:-10px">Choose a module below to begin analysis</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="service-card active">
          <div class="service-icon">🇱🇰</div>
          <div class="service-title">Tamil Hate Speech</div>
          <div class="service-desc">Text, audio & multimodal fusion detection using XLM-RoBERTa + SVM</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="go_tamil", use_container_width=True, type="primary"):
            st.session_state.page = "tamil"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="service-card soon">
          <div class="service-icon">🌐</div>
          <div class="service-title">Sinhala Hate Speech</div>
          <div class="service-desc">Hate speech detection for Sinhala language content</div>
          <span class="soon-badge">Coming Soon</span>
        </div>
        """, unsafe_allow_html=True)
        st.button("Coming Soon", key="go_sinhala", use_container_width=True, disabled=True)

    with col3:
        st.markdown("""
        <div class="service-card soon">
          <div class="service-icon">🎬</div>
          <div class="service-title">Video Hate Detection</div>
          <div class="service-desc">Real-time video content analysis for hate speech identification</div>
          <span class="soon-badge">Coming Soon</span>
        </div>
        """, unsafe_allow_html=True)
        st.button("Coming Soon", key="go_video", use_container_width=True, disabled=True)

    with col4:
        st.markdown("""
        <div class="service-card soon">
          <div class="service-icon">📈</div>
          <div class="service-title">Trend Analysis</div>
          <div class="service-desc">Visualise hate speech trends across time and platforms</div>
          <span class="soon-badge">Coming Soon</span>
        </div>
        """, unsafe_allow_html=True)
        st.button("Coming Soon", key="go_trends", use_container_width=True, disabled=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
      <strong>VigilAI</strong> is a research platform for detecting hate speech and deepfake content in Tamil and Sinhala.
      The Tamil module uses a multimodal late-fusion approach combining XLM-RoBERTa (text) and SVM (audio)
      achieving <strong>96.24% accuracy</strong>.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAMIL HATE SPEECH PAGE
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "tamil":

    # Back button
    if st.button("← Back to Home", key="back_home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("""
    <div class="hero">
      <h1>🛡️ Tamil Hate Speech Detector</h1>
      <p>Multimodal detection using Text, Audio, and Fusion analysis</p>
      <div class="badge-row">
        <span class="badge">XLM-RoBERTa · 83.18%</span>
        <span class="badge">SVM Audio · 85.86%</span>
        <span class="badge">Fusion DNN · 96.24%</span>
        <span class="badge">Whisper ASR</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝  Text Analysis",
        "🎵  Audio Analysis",
        "🔀  Fusion Analysis",
        "📊  Batch & Status"
    ])

    # ── TAB 1: Text prediction ────────────────────────────────
    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Text Analysis — XLM-RoBERTa")
        st.markdown('<p style="color:#64748b;margin-top:-8px">Fine-tuned multilingual model for Tamil hate speech classification</p>', unsafe_allow_html=True)

        sample_texts = [
            "dei nee enna pannra idhu romba wrong da",
            "semma movie da super acting",
            "உங்கள் எல்லாரும் ஒரு kelvi kekka maateenga",
            "nalla video bro keep it up",
            "இந்த ஜாதி பாருங்க எப்படி பேசுறாங்க"
        ]

        col1, col2 = st.columns([3, 1])
        with col1:
            text_input = st.text_area(
                "Enter Tamil or Tanglish text:",
                height=130,
                placeholder="Type Tamil text here...",
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("**Try a sample:**")
            for sample in sample_texts[:3]:
                if st.button(sample[:22] + "…", key=f"sample_{sample[:10]}"):
                    text_input = sample

        if st.button("Analyze text", type="primary", key="btn_text", use_container_width=True):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={"text": text_input},
                            timeout=30
                        )
                        result = response.json()
                        is_hate = result["label"] == "hate"
                        css_cls = "result-hate" if is_hate else "result-safe"
                        icon    = "⚠️ HATE SPEECH DETECTED" if is_hate else "✅ NOT HATE SPEECH"
                        st.markdown(f"""
                        <div class="{css_cls}">
                          <div class="result-title">{icon}</div>
                          <div class="result-sub">Confidence: <strong>{result['confidence']}%</strong> &nbsp;·&nbsp; Model: XLM-RoBERTa (83.18%)</div>
                        </div>""", unsafe_allow_html=True)
                        with st.expander("Full API response"):
                            st.json(result)
                    except Exception as e:
                        st.error(f"API Error: {e} — Make sure Flask API is running!")
            else:
                st.warning("Please enter some Tamil text!")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 2: Audio prediction ───────────────────────────────
    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Audio Analysis — SVM (85.86%)")
        st.markdown('<p style="color:#64748b;margin-top:-8px">104 acoustic features extracted with Librosa, classified by SVM</p>', unsafe_allow_html=True)

        audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

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

    # ── TAB 3: Fusion prediction ──────────────────────────────
    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Multimodal Fusion — 96.24%")
        st.markdown('<p style="color:#64748b;margin-top:-8px">Whisper ASR → XLM-RoBERTa (30%) + SVM Audio (70%) weighted fusion</p>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
          Upload one WAV file — the system denoises it, transcribes with <strong>Whisper</strong>,
          runs <strong>XLM-RoBERTa</strong> on the text and <strong>SVM</strong> on 104 acoustic features,
          then combines both with a <strong>30 / 70 weighted fusion</strong>.
        </div>
        """, unsafe_allow_html=True)

        fusion_audio = st.file_uploader("Upload WAV audio file", type=["wav"], key="fusion_audio")

        if st.button("Run fusion analysis", type="primary", key="btn_fusion", use_container_width=True):
            if fusion_audio is not None:
                progress_bar     = st.progress(0)
                step_placeholder = st.empty()

                try:
                    step_placeholder.markdown("**Step 1 / 5** — Removing background noise...")
                    progress_bar.progress(5)

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
                            "**Step 1 / 5** — Noise removal (noisereduce) ✓",
                            "**Step 2 / 5** — Tamil transcription (Whisper) ✓",
                            "**Step 3 / 5** — Text analysis (XLM-RoBERTa) ✓",
                            "**Step 4 / 5** — Audio features (Librosa + SVM) ✓",
                            "**Step 5 / 5** — Weighted fusion (30% text + 70% audio) ✓",
                        ]
                        for i, step in enumerate(steps):
                            step_placeholder.markdown(step)
                            progress_bar.progress((i + 1) * 20)
                            time.sleep(0.25)

                        step_placeholder.empty()
                        progress_bar.empty()

                        st.markdown("**Whisper transcription:**")
                        st.markdown(f"""
                        <div class="info-box">"{result['transcription']}"</div>
                        """, unsafe_allow_html=True)

                        st.divider()

                        is_hate = result["label"] == "hate"
                        css_cls = "result-hate" if is_hate else "result-safe"
                        icon    = "⚠️ HATE SPEECH DETECTED" if is_hate else "✅ NOT HATE SPEECH"
                        st.markdown(f"""
                        <div class="{css_cls}">
                          <div class="result-title">{icon}</div>
                          <div class="result-sub">Fusion confidence: <strong>{result['confidence']}%</strong></div>
                        </div>""", unsafe_allow_html=True)

                        st.markdown("**Model breakdown — P(hate):**")
                        st.markdown(f"""
                        <div class="stat-row">
                          <div class="stat-card">
                            <div class="num num-blue">{result['text_hate_prob']}%</div>
                            <div class="lbl">XLM-RoBERTa · 30%</div>
                          </div>
                          <div class="stat-card">
                            <div class="num num-blue">{result['audio_hate_prob']}%</div>
                            <div class="lbl">SVM Audio · 70%</div>
                          </div>
                          <div class="stat-card">
                            <div class="num {'num-red' if is_hate else 'num-green'}">{result['fusion_hate_prob']}%</div>
                            <div class="lbl">Fusion P(hate)</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.expander("Full API response"):
                            st.json(result)

                except Exception as e:
                    progress_bar.empty()
                    step_placeholder.empty()
                    st.error(f"API Error: {e} — Make sure Flask API is running!")
            else:
                st.warning("Please upload a WAV file!")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 4: Batch & API status ─────────────────────────────
    with tab4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Batch Text Analysis")
        st.markdown('<p style="color:#64748b;margin-top:-8px">Run prediction on multiple comments at once</p>', unsafe_allow_html=True)

        sample_batch = [
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

        comments_df = pd.DataFrame({"Comment": sample_batch, "Status": ["Pending"] * len(sample_batch)})
        table_placeholder = st.empty()
        table_placeholder.dataframe(comments_df, use_container_width=True, hide_index=True)

        if st.button("Run batch analysis", type="primary", key="btn_batch", use_container_width=True):
            try:
                with st.spinner("Analyzing batch..."):
                    response = requests.post(
                        f"{API_URL}/predict_batch",
                        json={"texts": sample_batch},
                        timeout=300
                    )
                    batch_result = response.json()

                results_data = [{
                    "Comment":    r["text"][:55] + "…" if len(r["text"]) > 55 else r["text"],
                    "Result":     "🚨 HATE" if r["label"] == "hate" else "✅ Safe",
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
                  <div class="stat-card"><div class="num num-blue">{t}</div><div class="lbl">Total scanned</div></div>
                  <div class="stat-card"><div class="num num-red">{h}</div><div class="lbl">Hate detected</div></div>
                  <div class="stat-card"><div class="num num-green">{s}</div><div class="lbl">Safe comments</div></div>
                  <div class="stat-card"><div class="num num-red">{batch_result['hate_pct']}%</div><div class="lbl">Hate rate</div></div>
                </div>
                """, unsafe_allow_html=True)

                if h > 0:
                    st.warning(f"{h} hate speech comment(s) detected — would be hidden by the Chrome extension.")
                else:
                    st.success("No hate speech detected in this batch.")
            except Exception as e:
                st.error(f"API Error: {e} — Make sure Flask API is running on port 5000")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### API Health Check")
        if st.button("Check API status", key="btn_health", use_container_width=True):
            try:
                response = requests.get(f"{API_URL}/health", timeout=5)
                data = response.json()
                st.success(f"Flask API is running — model: {data['model']}")
                st.json(data)
            except Exception:
                st.error("Flask API is not running. Start it with: python api/app.py")
        st.markdown('</div>', unsafe_allow_html=True)

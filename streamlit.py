import streamlit as st
import requests
import tempfile
import os
import time

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="DeepShield",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================
# CSS
# ======================================================

st.markdown("""
<style>

/* =======================================================
GLOBAL
======================================================= */

html,
body,
[class*="css"] {

    font-family:
    Inter,
    sans-serif;
}

.stApp {

    background:
    radial-gradient(
        circle at top,
        #0f1d3a 0%,
        #08101f 45%,
        #050816 100%
    );

    color: white;
}

/* HIDE STREAMLIT */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* MAIN WIDTH */

.block-container {

    max-width: 1250px;

    padding-top: 2rem;

    padding-bottom: 4rem;
}

/* =======================================================
HERO
======================================================= */

.hero {

    background:
    linear-gradient(
        135deg,
        rgba(11,23,48,0.96),
        rgba(19,54,110,0.94)
    );

    border-radius: 32px;

    padding: 55px;

    border:
    1px solid rgba(255,255,255,0.06);

    box-shadow:
    0 0 40px rgba(0,0,0,0.35);

    margin-bottom: 38px;
}

.hero-flex {

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 40px;
}

.hero-left {

    flex: 1;
}

.hero-title {

    font-size: 68px;

    font-weight: 800;

    line-height: 1.05;

    letter-spacing: -2px;

    margin-bottom: 18px;

    color: white;
}

.green {

    color: #00f5b0;
}

.hero-sub {

    color: #a9b5d1;

    font-size: 20px;

    line-height: 1.7;

    max-width: 760px;

    margin-bottom: 28px;
}

/* TAGS */

.tag-row {

    display: flex;

    flex-wrap: wrap;

    gap: 14px;
}

.tag {

    background:
    rgba(0,255,180,0.08);

    border:
    1px solid rgba(0,255,180,0.20);

    color: #00f5b0;

    padding: 12px 22px;

    border-radius: 999px;

    font-size: 14px;

    font-weight: 700;

    backdrop-filter: blur(6px);
}

/* RIGHT CARD */

.hero-right {

    width: 300px;
}

.hero-card {

    background:
    rgba(255,255,255,0.04);

    border:
    1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 24px;

    min-height: 260px;

    text-align: center;

    backdrop-filter: blur(12px);
}

.hero-stat {

    font-size: 42px;

    margin-bottom: 6px;
}

.hero-label {

    color: #a3afc8;

    font-size: 15px;

    line-height: 1.6;

    margin-bottom: 16px;
}

/* =======================================================
UPLOAD
======================================================= */

section[data-testid="stFileUploader"] {

    background:
    rgba(255,255,255,0.03);

    border:
    1px solid rgba(255,255,255,0.06);

    border-radius: 22px;

    padding: 16px;

    margin-top: 12px;
}

section[data-testid="stFileUploader"] label {

    color: white !important;

    font-size: 18px !important;

    font-weight: 700 !important;
}

/* =======================================================
VIDEO
======================================================= */

video {

    width: 100% !important;

    max-height: 420px !important;

    border-radius: 22px;

    object-fit: contain;

    background: black;

    margin-top: 10px;

    border:
    1px solid rgba(255,255,255,0.08);
}

/* =======================================================
BUTTON
======================================================= */

.stButton > button {

    width: 100%;

    height: 64px;

    border-radius: 18px;

    border: none;

    background:
    linear-gradient(
        90deg,
        #00f5b0,
        #00c8ff
    );

    color: #04111f;

    font-size: 20px;

    font-weight: 800;

    transition: 0.25s;

    margin-top: 14px;

    box-shadow:
    0 8px 30px rgba(0,255,180,0.18);
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
    0 12px 40px rgba(0,255,180,0.28);
}

/* =======================================================
RESULT
======================================================= */

.result-real {

    background:
    rgba(0,255,120,0.08);

    border:
    2px solid rgba(0,255,120,0.35);

    border-radius: 24px;

    padding: 32px;

    text-align: center;

    font-size: 46px;

    font-weight: 800;

    color: #00ff99;
}

.result-fake {

    background:
    rgba(255,0,80,0.08);

    border:
    2px solid rgba(255,0,80,0.30);

    border-radius: 24px;

    padding: 32px;

    text-align: center;

    font-size: 46px;

    font-weight: 800;

    color: #ff4d79;
}

/* =======================================================
METRICS
======================================================= */

.metric-card {

    background:
    rgba(13,19,37,0.92);

    border:
    1px solid rgba(255,255,255,0.06);

    border-radius: 24px;

    padding: 28px;

    text-align: center;

    backdrop-filter: blur(8px);

    margin-top: 10px;
}

.metric-value {

    font-size: 44px;

    font-weight: 800;

    color: #00f5b0;
}

.metric-label {

    color: #95a1ba;

    margin-top: 10px;

    font-size: 15px;
}

/* =======================================================
TIMELINE
======================================================= */

.segment {

    background:
    rgba(12,18,34,0.96);

    border:
    1px solid rgba(255,255,255,0.06);

    border-left:
    5px solid #ff4d79;

    border-radius: 20px;

    padding: 24px;

    margin-bottom: 18px;
}

.segment-title {

    color: white;

    font-size: 24px;

    font-weight: 700;

    line-height: 1.6;
}

.segment-sub {

    color: #a4aec5;

    margin-top: 10px;

    font-size: 16px;

    line-height: 1.7;
}

/* =======================================================
SUCCESS / ERROR
======================================================= */

.stSuccess {

    border-radius: 18px !important;
}

.stError {

    border-radius: 18px !important;
}

</style>

""", unsafe_allow_html=True)

# ======================================================
# HERO
# ======================================================

st.markdown("""

<div class="hero">

<div class="hero-flex">

<div class="hero-left">

<div class="hero-title">

Check if a Video is
<span class="green">Real or Fake</span>

</div>

<div class="hero-sub">

Upload a Sinhala or Tamil video and instantly detect fake content,
including the exact time where it appears in the video.

</div>

<div class="tag-row">

<span class="tag">Sinhala Support</span>

<span class="tag">Tamil Support</span>

<span class="tag">Fast Detection</span>

<span class="tag">AI Analysis</span>

<span class="tag">Timeline Detection</span>

</div>

</div>

<div class="hero-right">

<div class="hero-card">

<div class="hero-stat">

🛡️

</div>

<div class="hero-label">

Secure Local Video Analysis

</div>

<br>

<div class="hero-stat">

⏱️

</div>

<div class="hero-label">

Detects Fake Time Segments

</div>

<br>

<div class="hero-stat">

🌏

</div>

<div class="hero-label">

Sinhala & Tamil Video Support

</div>

</div>

</div>

""", unsafe_allow_html=True)

# ======================================================
# FILE UPLOAD
# ======================================================

uploaded = st.file_uploader(
    "Upload Video",
    type=['mp4', 'avi', 'mov', 'mkv']
)

# ======================================================
# VIDEO
# ======================================================

if uploaded is not None:

    st.video(uploaded)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Analyze Video"):

        with st.spinner("Scanning video..."):

            suffix = os.path.splitext(uploaded.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:

                tmp.write(uploaded.read())

                tmp_path = tmp.name

            start = time.time()

            try:

                with open(tmp_path, 'rb') as f:

                    response = requests.post(
                        'http://127.0.0.1:5000/analyze',
                        files={'video': f},
                        timeout=300
                    )

                process_time = round(
                    time.time() - start,
                    2
                )

                if response.status_code != 200:

                    st.error(response.text)

                else:

                    data = response.json()

                    verdict = data.get(
                        'verdict',
                        'UNKNOWN'
                    )

                    fake_percent = data.get(
                        'fake_percentage',
                        0
                    )

                    segments = data.get(
                        'fake_segments',
                        []
                    )

                    # ==================================
                    # FINAL VERDICT
                    # ==================================

                    if verdict == "REAL":

                        st.markdown("""

                        <div class="result-real">

                        ✅ REAL VIDEO

                        </div>

                        """, unsafe_allow_html=True)

                    else:

                        st.markdown("""

                        <div class="result-fake">

                        ❌ FAKE VIDEO

                        </div>

                        """, unsafe_allow_html=True)

                    st.markdown("<br>",
                                unsafe_allow_html=True)

                    # ==================================
                    # METRICS
                    # ==================================

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.markdown(f"""

                        <div class="metric-card">

                        <div class="metric-value">

                        {round(fake_percent)}%

                        </div>

                        <div class="metric-label">

                        Fake Score

                        </div>

                        </div>

                        """, unsafe_allow_html=True)

                    with c2:

                        st.markdown(f"""

                        <div class="metric-card">

                        <div class="metric-value">

                        {len(segments)}

                        </div>

                        <div class="metric-label">

                        Fake Sections

                        </div>

                        </div>

                        """, unsafe_allow_html=True)

                    with c3:

                        st.markdown(f"""

                        <div class="metric-card">

                        <div class="metric-value">

                        {process_time}s

                        </div>

                        <div class="metric-label">

                        Scan Time

                        </div>

                        </div>

                        """, unsafe_allow_html=True)

                    st.markdown("<br><br>",
                                unsafe_allow_html=True)

                    # ==================================
                    # TIMELINE
                    # ==================================

                    st.subheader(
                        "Detected Fake Sections"
                    )

                    if len(segments) == 0:

                        st.success(
                            "No fake sections detected."
                        )

                    else:

                        for seg in segments:

                            start_t = round(
                                seg.get(
                                    'start_time',
                                    0
                                ),
                                2
                            )

                            end_t = round(
                                seg.get(
                                    'end_time',
                                    0
                                ),
                                2
                            )

                            duration = round(
                                seg.get(
                                    'duration',
                                    0
                                ),
                                2
                            )

                            confidence = round(
                                seg.get(
                                    'confidence',
                                    0
                                ) * 100,
                                1
                            )

                            st.markdown(f"""

                            <div class="segment">

                            <div class="segment-title">

                            ❌ Fake Detected

                            <br><br>

                            {start_t}s → {end_t}s

                            </div>

                            <div class="segment-sub">

                            Duration:
                            {duration}s

                            <br>

                            Confidence:
                            {confidence}%

                            </div>

                            </div>

                            """, unsafe_allow_html=True)

            except Exception as e:

                st.error(str(e))

            finally:

                if os.path.exists(tmp_path):

                    os.remove(tmp_path)
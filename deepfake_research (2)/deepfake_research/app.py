# =========================================================
# frontend/app.py
# DeepShield - Streamlit Frontend
# =========================================================

import streamlit as st
import requests
import tempfile
import os
import time
import re
import subprocess
import shutil
import sys
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="DeepShield",

    page_icon="🛡️",

    layout="wide",

    initial_sidebar_state="collapsed"

)


# =========================================================
# CONSTANTS
# =========================================================

# Defaults to the original 127.0.0.1:5001, unchanged, so running this file
# the normal standalone way is unaffected. Only the integration launcher
# sets DEEPFAKE_BACKEND_URL, since port 5001 is already used by the Sinhala
# backend in the combined setup.
API_URL = os.environ.get(
    "DEEPFAKE_BACKEND_URL",
    "http://127.0.0.1:5001"
)

ANALYZE_URL = (
    f"{API_URL}/analyze"
)

HEALTH_URL = (
    f"{API_URL}/health"
)

MAX_VIDEO_SIZE_MB = 500

# Explicit Deno path used by yt-dlp for YouTube JS challenges. The original
# path (C:\Users\LENOVO\...) was the original developer's machine and never
# existed here, which is why the YouTube-URL analysis path always failed
# with "Deno was not found" while direct video upload (a separate code
# path that never touches this) kept working fine. Deno itself was already
# installed on this machine via winget — just pointing at the real location.
DENO_PATH = r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe\deno.exe"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
r"""
<style>

/* =====================================================
   GLOBAL
===================================================== */

html, body, [class*="css"] {

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

}

.stApp {

    background:
        radial-gradient(
            circle at 15% 0%,
            #162a50 0%,
            #0b1428 32%,
            #050912 72%,
            #03050a 100%
        );

    color: #f7f9fc;

}


/* =====================================================
   HIDE STREAMLIT DEFAULTS
===================================================== */

#MainMenu {

    visibility: hidden;

}

header {

    visibility: hidden;

}

footer {

    visibility: hidden;

}


/* =====================================================
   MAIN CONTAINER
===================================================== */

.block-container {

    max-width: 1320px;

    padding-top: 2rem;

    padding-bottom: 5rem;

}


/* =====================================================
   HERO
===================================================== */

.hero {

    position: relative;

    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            rgba(14, 31, 63, 0.98),
            rgba(8, 20, 43, 0.96)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 32px;

    padding:
        52px 55px;

    margin-bottom: 28px;

    box-shadow:
        0 25px 80px rgba(0,0,0,0.35);

}

.hero::before {

    content: "";

    position: absolute;

    width: 350px;

    height: 350px;

    border-radius: 50%;

    background:
        rgba(0,245,176,0.08);

    filter: blur(80px);

    right: -100px;

    top: -120px;

}

.hero-grid {

    display: grid;

    grid-template-columns:
        1fr 320px;

    gap: 45px;

    align-items: center;

}

.hero-title {

    font-size: 64px;

    font-weight: 850;

    line-height: 1.04;

    letter-spacing: -2.8px;

    margin: 0 0 20px 0;

}

.green {

    color: #00f5b0;

}

.hero-sub {

    max-width: 780px;

    color: #aab8d2;

    font-size: 18px;

    line-height: 1.75;

    margin-bottom: 28px;

}

.hero-tags {

    display: flex;

    flex-wrap: wrap;

    gap: 10px;

}

.hero-tag {

    padding:
        9px 16px;

    border-radius: 999px;

    background:
        rgba(0,245,176,0.07);

    border:
        1px solid rgba(0,245,176,0.18);

    color: #66ffd4;

    font-size: 13px;

    font-weight: 700;

}

.hero-side {

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 24px;

    backdrop-filter: blur(14px);

}

.side-row {

    display: flex;

    align-items: center;

    gap: 14px;

    padding: 15px 0;

    border-bottom:
        1px solid rgba(255,255,255,0.06);

}

.side-row:last-child {

    border-bottom: none;

}

.side-icon {

    width: 45px;

    height: 45px;

    border-radius: 14px;

    display: flex;

    align-items: center;

    justify-content: center;

    background:
        rgba(0,245,176,0.08);

    font-size: 22px;

}

.side-title {

    font-weight: 750;

    color: #f7f9fc;

}

.side-sub {

    font-size: 12px;

    color: #8f9bb5;

    margin-top: 3px;

}


/* =====================================================
   SECTION CARD
===================================================== */

.section-card {

    background:
        rgba(9,16,31,0.86);

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius: 26px;

    padding: 28px;

    margin-top: 20px;

    box-shadow:
        0 16px 50px rgba(0,0,0,0.18);

}

.section-title {

    font-size: 23px;

    font-weight: 800;

    margin-bottom: 5px;

}

.section-subtitle {

    color: #8794ae;

    font-size: 14px;

    margin-bottom: 22px;

}


/* =====================================================
   INPUTS
===================================================== */

div[data-baseweb="input"] {

    background:
        rgba(255,255,255,0.035);

    border-radius: 14px;

}

div[data-baseweb="input"] input {

    color: white !important;

}

div[data-baseweb="input"] input::placeholder {

    color: #71809d !important;

}


/* =====================================================
   FILE UPLOADER
===================================================== */

section[data-testid="stFileUploader"] {

    background:
        rgba(255,255,255,0.025);

    border:
        1px dashed rgba(255,255,255,0.15);

    border-radius: 18px;

    padding: 14px;

}

section[data-testid="stFileUploader"] label {

    color: #eaf0fb !important;

    font-weight: 700 !important;

}


/* =====================================================
   VIDEO
===================================================== */

video {

    width: 100% !important;

    max-height: 500px !important;

    border-radius: 20px;

    background: #000;

    border:
        1px solid rgba(255,255,255,0.08);

}


/* =====================================================
   PRIMARY BUTTON
===================================================== */

.stButton > button {

    width: 100%;

    min-height: 58px;

    border-radius: 16px;

    border: none;

    background:
        linear-gradient(
            90deg,
            #00f5b0,
            #00c8ff
        );

    color: #04101d;

    font-size: 17px;

    font-weight: 850;

    box-shadow:
        0 10px 35px rgba(0,245,176,0.15);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;

}

.stButton > button:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 15px 45px rgba(0,245,176,0.24);

}


/* =====================================================
   RESULT
===================================================== */

.result {

    border-radius: 26px;

    padding: 34px;

    text-align: center;

    margin-top: 25px;

}

.result-real {

    background:
        linear-gradient(
            135deg,
            rgba(0,255,150,0.08),
            rgba(0,200,130,0.025)
        );

    border:
        1px solid rgba(0,255,150,0.25);

}

.result-fake {

    background:
        linear-gradient(
            135deg,
            rgba(255,50,100,0.10),
            rgba(180,20,60,0.025)
        );

    border:
        1px solid rgba(255,60,100,0.28);

}

.result-icon {

    font-size: 42px;

    margin-bottom: 8px;

}

.result-title {

    font-size: 40px;

    font-weight: 850;

    letter-spacing: -1px;

}

.result-real .result-title {

    color: #00ff9d;

}

.result-fake .result-title {

    color: #ff527d;

}


.result-partial {

    background:
        linear-gradient(
            135deg,
            rgba(255,190,40,0.10),
            rgba(180,110,10,0.025)
        );

    border:
        1px solid rgba(255,190,40,0.30);

}

.result-partial .result-title {

    color: #ffc857;

}

.result-description {

    color: #9aa8c2;

    margin-top: 8px;

    font-size: 14px;

}


/* =====================================================
   METRICS
===================================================== */

.metric-card {

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius: 20px;

    padding: 24px;

    text-align: center;

    min-height: 130px;

}

.metric-value {

    color: #00f5b0;

    font-size: 36px;

    font-weight: 850;

}

.metric-label {

    color: #8592ac;

    font-size: 13px;

    margin-top: 8px;

}


/* =====================================================
   TIMELINE
===================================================== */

.timeline-header {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 18px;

}

.timeline-title {

    font-size: 23px;

    font-weight: 800;

}

.timeline-count {

    padding:
        7px 13px;

    border-radius: 999px;

    background:
        rgba(255,70,110,0.09);

    border:
        1px solid rgba(255,70,110,0.18);

    color: #ff668b;

    font-size: 12px;

    font-weight: 750;

}

.segment {

    background:
        linear-gradient(
            135deg,
            rgba(20,27,45,0.94),
            rgba(10,16,29,0.94)
        );

    border:
        1px solid rgba(255,255,255,0.065);

    border-left:
        4px solid #ff4d79;

    border-radius: 18px;

    padding: 20px 22px;

    margin-bottom: 14px;

}

.segment-main {

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 20px;

}

.segment-time {

    font-size: 21px;

    font-weight: 800;

    color: white;

}

.segment-time span {

    color: #ff5c84;

}

.segment-meta {

    color: #8d99b2;

    font-size: 13px;

    margin-top: 7px;

}

.confidence {

    text-align: right;

}

.confidence-value {

    color: #ff668b;

    font-size: 22px;

    font-weight: 800;

}

.confidence-label {

    color: #7e8aa4;

    font-size: 11px;

}


/* =====================================================
   FOOTER
===================================================== */

.deep-footer {

    text-align: center;

    color: #596680;

    font-size: 12px;

    margin-top: 45px;

    padding-top: 20px;

    border-top:
        1px solid rgba(255,255,255,0.05);

}


/* =====================================================
   MOBILE
===================================================== */

@media (max-width: 900px) {

    .hero-grid {

        grid-template-columns: 1fr;

    }

    .hero-title {

        font-size: 45px;

    }

    .hero {

        padding: 35px 28px;

    }

}

</style>
""",
unsafe_allow_html=True
)


# =========================================================
# HELPERS
# =========================================================

def format_time(seconds):

    seconds = max(
        0,
        float(seconds)
    )

    minutes = int(
        seconds // 60
    )

    remaining = seconds % 60

    return (
        f"{minutes:02d}:"
        f"{remaining:05.2f}"
    )


def youtube_url_valid(url):

    if not url:
        return False

    patterns = [

        r"(youtube\.com/watch\?v=)",

        r"(youtu\.be/)",

        r"(youtube\.com/shorts/)",

        r"(youtube\.com/embed/)"

    ]

    return any(
        re.search(
            pattern,
            url,
            re.IGNORECASE
        )
        for pattern in patterns
    )


def download_youtube_video(
    url,
    output_directory
):
    """
    Download a YouTube video with yt-dlp.

    This version explicitly supplies the Deno JavaScript runtime,
    because modern YouTube extraction may fail with HTTP 403 when
    yt-dlp cannot solve YouTube's JavaScript challenges.
    """

    # -----------------------------------------------------
    # CHECK DENO
    # -----------------------------------------------------

    if not os.path.exists(DENO_PATH):

        raise RuntimeError(
            "Deno was not found at:\n"
            f"{DENO_PATH}\n\n"
            "Install Deno or update DENO_PATH in frontend/app.py."
        )

    # -----------------------------------------------------
    # OUTPUT TEMPLATE
    # -----------------------------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_template = os.path.join(
        output_directory,
        "youtube_video.%(ext)s"
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Run yt-dlp through the same Python interpreter that
    # is running Streamlit. This prevents the frontend from
    # accidentally using a different yt-dlp installation.
    # -----------------------------------------------------

    command = [

        sys.executable,
        "-m",
        "yt_dlp",

        "--no-playlist",

        "--restrict-filenames",

        "--js-runtimes",
        f"deno:{DENO_PATH}",

        "--remote-components",
        "ejs:npm",

        # Prefer a reasonable YouTube quality for analysis.
        # If separate video/audio streams are selected,
        # yt-dlp/ffmpeg will merge them.
        "-f",
        "bv*[height<=720]+ba/b[height<=720]/best",

        "--merge-output-format",
        "mp4",

        "-o",
        output_template,

        url
    ]

    print()
    print("=" * 70)
    print("YOUTUBE DOWNLOAD")
    print("=" * 70)
    print("URL:", url)
    print("Python:", sys.executable)
    print("Deno:", DENO_PATH)
    print("Deno exists:", os.path.exists(DENO_PATH))
    print("Output directory:", output_directory)

    try:

        process = subprocess.run(

            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            text=True,

            encoding="utf-8",

            errors="replace",

            timeout=300
        )

    except subprocess.TimeoutExpired:

        raise RuntimeError(
            "YouTube download timed out after 5 minutes."
        )

    print()
    print("YT-DLP STDOUT:")
    print(process.stdout)

    print()
    print("YT-DLP STDERR:")
    print(process.stderr)

    # -----------------------------------------------------
    # DOWNLOAD FAILED
    # -----------------------------------------------------

    if process.returncode != 0:

        details = (
            process.stderr
            or process.stdout
            or "Unknown yt-dlp error."
        )

        if "HTTP Error 403" in details:

            raise RuntimeError(
                "Could not download the YouTube video.\n\n"
                "YouTube returned HTTP 403. "
                "DeepShield attempted the download using the "
                "configured Deno JavaScript runtime.\n\n"
                + details[-2500:]
            )

        raise RuntimeError(
            "Could not download the YouTube video.\n\n"
            + details[-2500:]
        )

    # -----------------------------------------------------
    # LOCATE FINAL VIDEO
    # -----------------------------------------------------

    downloaded_files = list(
        Path(output_directory).glob(
            "youtube_video.*"
        )
    )

    downloaded_files = [

        p

        for p in downloaded_files

        if (
            p.is_file()
            and
            p.suffix.lower()
            not in {
                ".part",
                ".ytdl"
            }
        )
    ]

    if not downloaded_files:

        raise RuntimeError(
            "YouTube download completed, but the final "
            "video file could not be located."
        )

    # Prefer MP4 if yt-dlp produced one.
    mp4_files = [
        p
        for p in downloaded_files
        if p.suffix.lower() == ".mp4"
    ]

    if mp4_files:

        downloaded_file = max(
            mp4_files,
            key=lambda p: p.stat().st_mtime
        )

    else:

        downloaded_file = max(
            downloaded_files,
            key=lambda p: p.stat().st_mtime
        )

    print(
        "Downloaded video:",
        str(downloaded_file)
    )

    return str(
        downloaded_file
    )


def check_backend():

    try:

        response = requests.get(

            HEALTH_URL,

            timeout=5

        )

        if response.status_code == 200:

            return True, response.json()

        return False, None

    except Exception:

        return False, None


# =========================================================
# HERO
# =========================================================

st.markdown(
"""
<div class="hero">

<div class="hero-grid">

<div>

<div class="hero-title">

Verify a Video.
<br>

<span class="green">Find the Fake.</span>

</div>

<div class="hero-sub">

DeepShield analyzes video frames over time to determine
whether the content appears real or manipulated and identifies
the time intervals where suspicious content is detected.

</div>

<div class="hero-tags">

<span class="hero-tag">
🛡️ Deepfake Detection
</span>

<span class="hero-tag">
🎬 Temporal Analysis
</span>

<span class="hero-tag">
⏱️ Fake Timestamp Detection
</span>

<span class="hero-tag">
🌏 Sinhala & Tamil
</span>

</div>

</div>


<div class="hero-side">

<div class="side-row">

<div class="side-icon">
🎥
</div>

<div>

<div class="side-title">
Video Analysis
</div>

<div class="side-sub">
Local frame-level processing
</div>

</div>

</div>


<div class="side-row">

<div class="side-icon">
🧠
</div>

<div>

<div class="side-title">
Temporal Scoring
</div>

<div class="side-sub">
Sliding-window analysis
</div>

</div>

</div>


<div class="side-row">

<div class="side-icon">
⏱️
</div>

<div>

<div class="side-title">
Timeline Results
</div>

<div class="side-sub">
Locate suspicious intervals
</div>

</div>

</div>

</div>

</div>

</div>
""",
unsafe_allow_html=True
)


# =========================================================
# BACKEND STATUS
# =========================================================

backend_ok, backend_info = check_backend()

if backend_ok:

    st.success(
        "DeepShield detection engine is online.",
        
    )

else:

    st.error(
        "Detection engine is offline. "
        "Start backend/app.py before analyzing a video."
    )


# =========================================================
# SOURCE SELECTION
# =========================================================

st.markdown(
"""
<div class="section-card">

<div class="section-title">
Choose Video Source
</div>

<div class="section-subtitle">
Analyze a YouTube video or upload a local video file.
</div>

</div>
""",
unsafe_allow_html=True
)

source = st.radio(

    "Video source",

    [
        "YouTube URL",
        "Upload Video"
    ],

    horizontal=True,

    label_visibility="collapsed"

)


# =========================================================
# SESSION STATE
# =========================================================

if "video_path" not in st.session_state:

    st.session_state.video_path = None


if "video_source_name" not in st.session_state:

    st.session_state.video_source_name = None


if "video_source_type" not in st.session_state:

    st.session_state.video_source_type = None


if "youtube_url" not in st.session_state:

    st.session_state.youtube_url = ""


# =========================================================
# YOUTUBE SOURCE
# =========================================================

if source == "YouTube URL":

    st.markdown(
        """
        <div class="section-card">
        """,
        unsafe_allow_html=True
    )

    youtube_url = st.text_input(

        "YouTube URL",

        placeholder=
        "https://www.youtube.com/watch?v=...",

        help=
        "Paste the URL of the YouTube video you want to analyze."

    )

    if youtube_url:

        if youtube_url_valid(
            youtube_url
        ):

            # ---------------------------------------------
            # PREVIEW
            # ---------------------------------------------

            st.markdown(
                "### Video Preview"
            )

            try:

                st.video(
                    youtube_url
                )

            except Exception:

                st.info(
                    "YouTube preview could not be embedded, "
                    "but the video can still be downloaded for analysis."
                )

            # ---------------------------------------------
            # DOWNLOAD
            # ---------------------------------------------

            if st.button(
                "Prepare YouTube Video",
                key="prepare_youtube"
            ):

                with st.spinner(
                    "Preparing YouTube video for analysis..."
                ):

                    try:

                        youtube_temp_dir = tempfile.mkdtemp(
                            prefix="deepshield_youtube_"
                        )

                        video_path = download_youtube_video(

                            youtube_url,

                            youtube_temp_dir

                        )

                        st.session_state.video_path = (
                            video_path
                        )

                        st.session_state.video_source_name = (
                            "YouTube video"
                        )

                        st.session_state.video_source_type = (
                            "youtube"
                        )

                        st.session_state.youtube_url = (
                            youtube_url
                        )

                        st.success(
                            "YouTube video is ready for analysis.",
                            
                        )

                    except Exception as error:

                        st.error(
                            str(error)
                        )

        else:

            st.warning(
                "Please enter a valid YouTube URL."
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# UPLOAD SOURCE
# =========================================================

else:

    st.markdown(
        """
        <div class="section-card">
        """,
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(

        "Upload Video",

        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm"
        ],

        help=
        "Supported formats: MP4, AVI, MOV, MKV and WebM."

    )

    if uploaded is not None:

        file_size_mb = (
            uploaded.size
            /
            (1024 * 1024)
        )

        if file_size_mb > MAX_VIDEO_SIZE_MB:

            st.error(
                f"Video is too large "
                f"({file_size_mb:.1f} MB). "
                f"Maximum is {MAX_VIDEO_SIZE_MB} MB."
            )

        else:

            # ---------------------------------------------
            # CREATE TEMP FILE
            # ---------------------------------------------

            suffix = os.path.splitext(
                uploaded.name
            )[1].lower()

            upload_directory = tempfile.mkdtemp(
                prefix="deepshield_upload_"
            )

            upload_path = os.path.join(

                upload_directory,

                f"uploaded_video{suffix}"

            )

            with open(
                upload_path,
                "wb"
            ) as file:

                file.write(
                    uploaded.getbuffer()
                )

            st.session_state.video_path = (
                upload_path
            )

            st.session_state.video_source_name = (
                uploaded.name
            )

            st.session_state.video_source_type = (
                "upload"
            )

            st.markdown(
                "### Video Preview"
            )

            st.video(
                uploaded
            )

            st.caption(
                f"{uploaded.name} · "
                f"{file_size_mb:.1f} MB"
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# ANALYSIS SECTION
# =========================================================

if st.session_state.video_path:

    st.markdown(
        """
        <div class="section-card">
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-title">
        Ready for Analysis
        </div>

        <div class="section-subtitle">
        The video will be processed by the DeepShield temporal
        deepfake detection pipeline.
        </div>
        """,
        unsafe_allow_html=True
    )

    source_name = (
        st.session_state.video_source_name
        or
        "Selected video"
    )

    st.info(
        f"Source: {source_name}",
        icon="🎬"
    )

    analyze_button = st.button(
        "🔍  Analyze Video",
        key="analyze_video"
    )

    if analyze_button:

        if not backend_ok:

            st.error(
                "The Flask backend is not running."
            )

        else:

            video_path = (
                st.session_state.video_path
            )

            if not os.path.exists(
                video_path
            ):

                st.error(
                    "The selected video file is no longer available. "
                    "Please select it again."
                )

            else:

                progress = st.progress(
                    0
                )

                status = st.empty()

                start_time = time.time()

                try:

                    status.info(
                        "Uploading video to the detection engine..."
                    )

                    progress.progress(
                        10
                    )

                    with open(
                        video_path,
                        "rb"
                    ) as video_file:

                        response = requests.post(

                            ANALYZE_URL,

                            files={
                                "video": (
                                    os.path.basename(
                                        video_path
                                    ),
                                    video_file,
                                    (
                                        "video/webm"
                                        if video_path.lower().endswith(".webm")
                                        else "video/mp4"
                                    )
                                )
                            },

                            timeout=1800

                        )

                    progress.progress(
                        95
                    )

                    processing_time = round(

                        time.time()
                        -
                        start_time,

                        2

                    )

                    if response.status_code != 200:

                        try:

                            error_data = (
                                response.json()
                            )

                            error_message = (
                                error_data.get(
                                    "error",
                                    "Unknown backend error."
                                )
                            )

                        except Exception:

                            error_message = (
                                response.text
                            )

                        progress.empty()

                        status.empty()

                        st.error(
                            f"Analysis failed: "
                            f"{error_message}"
                        )

                    else:

                        data = response.json()

                        progress.progress(
                            100
                        )

                        status.success(
                            "Analysis completed.",
                            
                        )

                        # =================================
                        # RESULT DATA
                        # =================================

                        verdict = str(
                            data.get(
                                "verdict",
                                "UNKNOWN"
                            )
                        ).upper()

                        fake_percentage = float(
                            data.get(
                                "fake_percentage",
                                0
                            )
                            or 0
                        )

                        segments = data.get(
                            "fake_segments",
                            []
                        ) or []

                        api_processing_time = float(
                            data.get(
                                "api_processing_time",
                                processing_time
                            )
                            or processing_time
                        )

                        video_duration = float(
                            data.get(
                                "video_duration",
                                0
                            )
                            or 0
                        )

                        windows_scored = int(
                            data.get(
                                "windows_scored",
                                0
                            )
                            or 0
                        )

                        candidate_windows = int(
                            data.get(
                                "candidate_windows",
                                0
                            )
                            or 0
                        )

                        rejected_windows = int(
                            data.get(
                                "rejected_windows",
                                0
                            )
                            or 0
                        )

                        analysable_percentage = float(
                            data.get(
                                "analysable_percentage",
                                0
                            )
                            or 0
                        )

                        fake_windows = int(
                            data.get(
                                "fake_windows",
                                0
                            )
                            or 0
                        )

                        max_fake_probability = float(
                            data.get(
                                "max_fake_probability",
                                0
                            )
                            or 0
                        )

                        # =================================
                        # FINAL v3 VERDICT
                        # =================================

                        st.markdown(
                            "<br>",
                            unsafe_allow_html=True
                        )

                        if verdict == "REAL":

                            st.markdown(
                                """
                                <div class="result result-real">
                                <div class="result-title">
                                REAL VIDEO
                                </div>
                                <div class="result-description">
                                No sustained manipulated facial-content
                                interval passed the final Condition C v3
                                decision rules.
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        elif verdict == "PARTIALLY_MANIPULATED":

                            st.markdown(
                                """
                                <div class="result result-partial">
                                <div class="result-title">
                                PARTIALLY MANIPULATED
                                </div>
                                <div class="result-description">
                                Sustained manipulated content was detected
                                in localized sections of the analysed video.
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        elif verdict == "FAKE":

                            st.markdown(
                                """
                                <div class="result result-fake">
                                <div class="result-title">
                                FAKE / MANIPULATED VIDEO
                                </div>
                                <div class="result-description">
                                Sustained manipulated content was detected
                                across enough analysable windows to satisfy
                                the final video-level fake rule.
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        elif verdict == "INSUFFICIENT_FACE_DATA":

                            st.warning(
                                "Not enough detectable facial content "
                                "was available for a reliable analysis."
                            )

                        elif verdict in {
                            "VIDEO_TOO_SHORT",
                            "VIDEO_LOAD_FAILED",
                            "VIDEO_NOT_FOUND"
                        }:

                            st.warning(
                                f"Analysis status: {verdict}"
                            )

                        elif verdict == "ERROR":

                            st.error(
                                data.get(
                                    "error",
                                    "The analysis failed."
                                )
                            )

                        else:

                            st.warning(
                                f"Analysis status: {verdict}"
                            )

                        # =================================
                        # FINAL v3 METRICS
                        # =================================

                        st.markdown(
                            "<br>",
                            unsafe_allow_html=True
                        )

                        c1, c2, c3, c4 = st.columns(4)

                        with c1:

                            st.markdown(
                                f"""
                                <div class="metric-card">
                                <div class="metric-value">
                                {fake_percentage:.1f}%
                                </div>
                                <div class="metric-label">
                                Sustained Fake Window Coverage
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with c2:

                            st.markdown(
                                f"""
                                <div class="metric-card">
                                <div class="metric-value">
                                {len(segments)}
                                </div>
                                <div class="metric-label">
                                Detected Fake Sections
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with c3:

                            st.markdown(
                                f"""
                                <div class="metric-card">
                                <div class="metric-value">
                                {analysable_percentage:.1f}%
                                </div>
                                <div class="metric-label">
                                Analysable Face Coverage
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with c4:

                            st.markdown(
                                f"""
                                <div class="metric-card">
                                <div class="metric-value">
                                {api_processing_time:.1f}s
                                </div>
                                <div class="metric-label">
                                Processing Time
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        if (
                            analysable_percentage > 0
                            and
                            analysable_percentage < 60
                        ):

                            st.info(
                                "Only part of this video contained enough "
                                "detectable facial evidence for analysis. "
                                "Interpret the verdict together with the "
                                "Analysable Face Coverage value."
                            )

                        # =================================
                        # TEMPORAL TIMELINE
                        # =================================

                        st.markdown(
                            "<br>",
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f"""
                            <div class="timeline-header">
                            <div class="timeline-title">
                            Detected Manipulated Timeline
                            </div>
                            <div class="timeline-count">
                            {len(segments)} SECTION(S)
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        if not segments:

                            st.success(
                                "No sustained manipulated time segments "
                                "were detected."
                            )

                        else:

                            for index, segment in enumerate(
                                segments,
                                start=1
                            ):

                                start_sec = float(
                                    segment.get(
                                        "start_time",
                                        0
                                    )
                                    or 0
                                )

                                end_sec = float(
                                    segment.get(
                                        "end_time",
                                        0
                                    )
                                    or 0
                                )

                                duration = float(
                                    segment.get(
                                        "duration",
                                        max(
                                            0,
                                            end_sec - start_sec
                                        )
                                    )
                                    or 0
                                )

                                confidence = float(
                                    segment.get(
                                        "confidence",
                                        0
                                    )
                                    or 0
                                )

                                peak_confidence = float(
                                    segment.get(
                                        "peak_confidence",
                                        confidence
                                    )
                                    or confidence
                                )

                                confidence_percent = (
                                    confidence * 100
                                )

                                peak_percent = (
                                    peak_confidence * 100
                                )

                                st.markdown(
                                    f"""
                                    <div class="segment">
                                    <div class="segment-main">
                                    <div>
                                    <div class="segment-time">
                                    Section {index}
                                    &nbsp;&nbsp;
                                    <span>
                                    {format_time(start_sec)}
                                    →
                                    {format_time(end_sec)}
                                    </span>
                                    </div>
                                    <div class="segment-meta">
                                    Duration:
                                    {duration:.2f} seconds
                                    &nbsp; • &nbsp;
                                    Mean fake probability:
                                    {confidence_percent:.1f}%
                                    &nbsp; • &nbsp;
                                    Peak:
                                    {peak_percent:.1f}%
                                    </div>
                                    </div>
                                    <div class="confidence">
                                    <div class="confidence-value">
                                    {confidence_percent:.0f}%
                                    </div>
                                    <div class="confidence-label">
                                    MEAN PROBABILITY
                                    </div>
                                    </div>
                                    </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        # =================================
                        # TECHNICAL DETAILS
                        # =================================

                        with st.expander(
                            "View technical analysis details"
                        ):

                            col_a, col_b = st.columns(2)

                            with col_a:

                                st.write(
                                    "**Model:**",
                                    data.get(
                                        "model",
                                        "N/A"
                                    )
                                )

                                st.write(
                                    "**Windows scored:**",
                                    windows_scored
                                )

                                st.write(
                                    "**Candidate windows:**",
                                    candidate_windows
                                )

                                st.write(
                                    "**Rejected windows:**",
                                    rejected_windows
                                )

                                st.write(
                                    "**Analysable coverage:**",
                                    f"{analysable_percentage:.1f}%"
                                )

                            with col_b:

                                st.write(
                                    "**Window threshold:**",
                                    data.get(
                                        "window_threshold",
                                        "N/A"
                                    )
                                )

                                st.write(
                                    "**Minimum consecutive fake windows:**",
                                    data.get(
                                        "minimum_consecutive_fake_windows",
                                        "N/A"
                                    )
                                )

                                st.write(
                                    "**Full fake coverage threshold:**",
                                    data.get(
                                        "full_fake_coverage_threshold",
                                        "N/A"
                                    )
                                )

                                st.write(
                                    "**Sustained fake windows:**",
                                    fake_windows
                                )

                                st.write(
                                    "**Peak fake probability:**",
                                    f"{max_fake_probability * 100:.1f}%"
                                )

                                st.write(
                                    "**Video duration:**",
                                    f"{video_duration:.2f}s"
                                )

                        # =================================
                        # IMPORTANT RESEARCH DISCLAIMER
                        # =================================

                        st.caption(
                            "DeepShield's result is an automated model "
                            "prediction. A detected interval represents "
                            "content identified as suspicious by the "
                            "current temporal detection pipeline."
                        )

                except requests.exceptions.Timeout:

                    progress.empty()

                    status.empty()

                    st.error(
                        "The analysis timed out. "
                        "Long videos may require additional processing time."
                    )

                except requests.exceptions.ConnectionError:

                    progress.empty()

                    status.empty()

                    st.error(
                        "Could not connect to the DeepShield backend. "
                        "Make sure backend/app.py is running."
                    )

                except Exception as error:

                    progress.empty()

                    status.empty()

                    st.error(
                        f"Unexpected error: {error}"
                    )

                finally:

                    # Don't delete the uploaded file here because
                    # Streamlit reruns can still need the path.
                    pass

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
"""
<div class="deep-footer">

DeepShield · Multimodal Deepfake Detection Research Prototype
<br>
Temporal video analysis for Sinhala and Tamil digital content

</div>
""",
unsafe_allow_html=True
)
# R26-DS-001 — Multimodal Hate Speech & Deepfake Detection

Unified system combining four independently-built detection modules into one product:

| Module | Detects | Tech |
|---|---|---|
| **Tamil Hate Speech** | Offensive Tamil speech & text (text, audio, YouTube) | Flask + Streamlit |
| **Sinhala Hate Speech + Audio Deepfake** | Offensive Sinhala speech, AI-faked audio | Flask + Streamlit |
| **Sinhala False Content** | Misleading/false video titles & descriptions | Flask |
| **Video Deepfake (DeepShield)** | AI-generated/manipulated video | Flask + Streamlit |

Plus a **Chrome Extension** for real-time YouTube checking, and a **unified web app (Shell)** that brings all four modules into one interface via a lightweight **Gateway**.

---

## Architecture

```
Chrome Extension ──┬── Gateway (:8000) ── Tamil backend (:5000)
                    │                └── Sinhala backend (:5001)
                    ├── False Content backend (:5002)
                    └── Deepfake backend (:5003)

Shell / unified web app (:8501) ── embeds Deepfake frontend (:8502) via iframe
```

Every original project's own code is untouched — the Gateway only routes requests, and the Shell only adds a shared UI layer on top.

---

## ⚠️ What's NOT in this repo (and why)

To keep this repo pushable to GitHub (which hard-rejects any file over 100MB), the following are **excluded via `.gitignore`** and must be set up locally after cloning:

| Excluded | Why | What to do |
|---|---|---|
| `venv/`, `.venv/`, `venv_py311/` (all sub-projects) | Reinstallable from `requirements.txt` | Create your own venv per sub-project (see Setup below) |
| `models/` folders, `*.safetensors`, `*.onnx`, `*.pt`, `*.pkl` | Trained model files, several 300MB–1.5GB each | Contact the team for the model download link (Google Drive) |
| `dataset/`, `outputs/`, `*.npy` | Training data / large intermediate outputs | Not needed to *run* the system, only to retrain |
| `.env` files | **Live API keys** (Sarvam, ElevenLabs, Google Speech, YouTube Data API) | Create your own `.env` from the `.env.example` in each sub-project, with your own keys |
| `roop/` | Third-party face-swap tool, cloned with its own separate git history | Not required to run detection — only used as a reference during development |
| `logs/`, `*.log` | Runtime-generated, not source | Created automatically when you run the app |

**In short: this repo has all the code, none of the trained weights or secrets.** You need your own model files and your own API keys to actually run it.

---

## Setup (per sub-project)

Each of the 4 detection projects has its **own separate Python virtual environment** — they are not meant to share one.

```bash
# Example for one sub-project (repeat per sub-project folder)
cd "<sub-project folder>"
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

Then:
1. Place the trained model files into the correct `models/` (or equivalent) subfolder for that project.
2. Copy `.env.example` → `.env` in the Tamil project (and any other project that has one), and fill in your own API keys.

---

## Running the full system

Each service runs in its own terminal:

```powershell
# Tamil backend (port 5000)
cd "tamil-hate-speech-detection inte 23rd\tamil-hate-speech-detection"
.\venv\Scripts\python.exe api\app.py

# Sinhala backend (port 5001)
cd "AT final-4\AT final\backend"
$env:SINHALA_BACKEND_PORT = "5001"
& "..\.venv\Scripts\python.exe" app.py

# False Content backend (port 5002)
cd "false 2\false 2"
$env:FALSE_CONTENT_PORT = "5002"
.\venv\Scripts\python.exe app.py

# Deepfake backend (port 5003)
cd "deepfake_research (2)\deepfake_research\webapp"
$env:DEEPFAKE_BACKEND_PORT = "5003"
& "..\venv_py311\Scripts\python.exe" app.py

# Gateway (port 8000)
cd gateway
python app.py

# Shell — unified web app (port 8501)
cd shell
& "<tamil venv>\Scripts\python.exe" -m streamlit run app.py --server.port 8501

# Deepfake frontend (port 8502)
cd "deepfake_research (2)\deepfake_research"
$env:DEEPFAKE_BACKEND_URL = "http://127.0.0.1:5003"
.\venv_py311\Scripts\python.exe -m streamlit run app.py --server.port 8502 --server.enableXsrfProtection false
```

Start the 4 backends first, then the Gateway, then the Shell. Once all show "Running on..." (Streamlit: "You can now view your app..."), open **http://localhost:8501**.

The **Chrome Extension** is loaded separately: `chrome://extensions` → Developer mode → Load unpacked → select the `chrome_extension/` folder.

---

## Team

* Tamil Hate Speech
* Sinhala Hate Speech + Audio Deepfake
* Sinhala False/Misleading Content
* Video Deepfake Detection

*Hate Speech and Deep Fake Identification for Sinhala and Tamil Low-Resource Languages.*

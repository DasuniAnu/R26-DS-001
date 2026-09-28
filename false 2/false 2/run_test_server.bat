@echo off
set "TRANSFORMERS_NO_TORCHAUDIO=1"
set "TRANSFORMERS_NO_TORCHVISION=1"
if not defined WHISPER_RUNTIME set "WHISPER_RUNTIME=xlsr_sinhala"
if not defined WHISPER_MODEL set "WHISPER_MODEL=janiduchamika/wav2vec2-xls-r-300m-sinhala-general-185k"
if not defined XLSR_PYTHON set "XLSR_PYTHON=C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe"
if not defined MAX_AUDIO_TRANSCRIPT_SECONDS set "MAX_AUDIO_TRANSCRIPT_SECONDS=900"
cd /d "C:\Users\User\Desktop\false 2"
"C:\Users\User\torch-env\Scripts\python.exe" -c "from app import app; app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)"

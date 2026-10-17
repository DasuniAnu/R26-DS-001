$ErrorActionPreference = "Stop"
$env:TRANSFORMERS_NO_TORCHAUDIO = "1"
$env:TRANSFORMERS_NO_TORCHVISION = "1"
if (-not $env:WHISPER_RUNTIME) { $env:WHISPER_RUNTIME = "xlsr_sinhala" }
if (-not $env:WHISPER_MODEL) { $env:WHISPER_MODEL = "janiduchamika/wav2vec2-xls-r-300m-sinhala-general-185k" }
if (-not $env:XLSR_PYTHON) { $env:XLSR_PYTHON = "C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe" }
if (-not $env:MAX_AUDIO_TRANSCRIPT_SECONDS) { $env:MAX_AUDIO_TRANSCRIPT_SECONDS = "900" }
Set-Location "C:\Users\User\Desktop\false 2"
& "C:\Users\User\torch-env\Scripts\python.exe" -c "from app import app; app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)"

# =============================================================
# AUDIO PREPROCESSING MODULE
# Extracted from: notebooks/final audio module.ipynb
#
# DO NOT MODIFY — must match training preprocessing exactly.
# =============================================================

import numpy as np
import librosa


# Constants — must match training exactly
SAMPLE_RATE  = 16000
MAX_DURATION = 10
MAX_LENGTH   = SAMPLE_RATE * MAX_DURATION   # 160,000 samples


def load_audio(filepath, max_length=MAX_LENGTH):
    """
    Load audio file and prepare for Wav2Vec2.

    Handles:
      - Any sample rate → resampled to 16kHz
      - Stereo → mono
      - 5s clips → zero-padded to 10s (160,000 samples)
      - 10s clips → used as-is
      - >10s clips → truncated to first 10s
      - Silent/empty files → returns zeros with warning
    """
    try:
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return np.zeros(max_length, dtype=np.float32)

    # Normalise amplitude to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    # Pad (5s → 10s) or truncate (>10s → 10s)
    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
    else:
        audio = audio[:max_length]

    return audio.astype(np.float32)

# YouTube transcription

Each 10-second analysis segment is transcribed with up to 2 seconds of original
audio on either side. The final ASR window is not padded with artificial silence.
CTC token runs are assigned to the central segment using their frame midpoints
and the model's convolutional stride/receptive field. Extra context text is
excluded. These are approximate character alignments, not word-level forced
alignment; words may still span displayed segment boundaries.

Audio hate detection and deepfake detection retain their existing inputs.

The recognition score is the mean of the maximum posterior in each emitted,
non-blank CTC run. Blank frames are excluded. An initial heuristic flags a
transcript if its mean is below 0.65 or more than 25% of its tokens score below
0.65. These constants are in `asr_manager.py`. They are NOT calibrated accuracy
probabilities. High-scoring transcripts can still be wrong.

Uncertain or unusable Sinhala transcripts are retained for review but are not
sent to the text hate classifier. An independently offensive audio result
remains offensive; otherwise an uncertain transcript yields an uncertain fused
result and is excluded from the clean count. ASR errors affect only that segment.
The existing audio model's NO_SPEECH decision still skips transcription.

Run offline regression checks from the project root:

    .\.venv\Scripts\python.exe -X utf8 -m unittest discover -s backend -p test_asr.py

Before claiming improved transcription accuracy, collect manually corrected
Sinhala transcripts for clear, slow, quiet, noisy, and music-backed examples.
Compare word/character error rates, review rate, and resulting hate predictions
against the previous pipeline. Tune thresholds on a development split and
report results on a separate held-out split. Noise reduction, time stretching,
and model replacement are not enabled by this change.

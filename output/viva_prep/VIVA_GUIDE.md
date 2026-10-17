# Final viva preparation guide

## 1. How to use this guide

**Project:** R26-DS-001. **Student:** IT22155870, Suwandarathne N.S. **Component:** Multimodal Sinhala Hate Speech and Deepfake Detection in Text and Audio.

This is a code-based study guide for the project in `D:\Studies\Y4S1\RP\AT final`, reviewed on 14 October 2026. It explains the current implementation, not an idealized version of it. File and line references are for this snapshot. The companion `SOURCE_ATLAS.html` contains the complete, numbered application source, a function index, and local call sites. Search it for a function name when practising a code demonstration.

Read chapters 2–5 first to understand the system. Then follow the backend chapters in execution order, study the frontend, and practise the question bank aloud. For any highlighted code, answer in this order: **purpose → input → operation → output → caller → limitation**. Do not memorize a paragraph without being able to trace its variables.

The guide explains all application functions, nested report helpers, test methods, configuration files, important statements and UI blocks. Third-party packages in virtual environments are dependencies rather than code you wrote. Model files and training evidence are explained separately. No application behavior has been changed while preparing this guide.

Suggested revision sessions: 45 minutes architecture and request traces; 60 minutes preprocessing and model inference; 60 minutes YouTube and ASR; 45 minutes frontend, reports and tests; 30 minutes mock questions and demonstration. If time is short, prioritize chapters 2, 4, 10, 15 and 19.

## 2. Your opening explanation

### A 60-second explanation

“My component provides Sinhala text and audio analysis through a Streamlit interface and a Flask API. Text is cleaned and classified with a fine-tuned XLM-RoBERTa checkpoint. Audio hate speech uses a Wav2Vec2 classifier, and the deployed audio-authenticity checkpoint is WavLM. For YouTube, I download the audio, resample it to 16 kHz mono and divide it into 10-second analysis segments. The hate-audio path uses speech-masked segments, while the deepfake path preserves the original resampled waveform. A separate Sinhala XLS-R speech recognizer creates transcripts using overlapping context. Reliable-enough transcripts are classified by the text model, and a rule-based decision combines audio and transcript results. Uncertain transcripts are retained for review. Optional Sinhala Unicode comments are analysed separately. The interface shows timestamps, model scores and an HTML report.”

### What the research component is—and is not

The integration covers two modalities, text and audio, with several tasks. It is not one end-to-end multimodal neural network: the classifiers have separate weights and are combined at the decision level. A transcript is derived from the audio, so audio and transcript evidence are related, not statistically independent. Comment predictions are separate from segment-level fusion. YouTube video frames are not analysed; this is audio-deepfake detection, not visual face-swap detection.

The implemented target labels are `OFF` and `NOT`, meaning offensive and non-offensive under the training task. Explain your research title, but do not imply that every offensive word is legally or semantically hate speech against a protected group. The code does not separately classify insult, threat, protected-group targeting and quoted news. Contextual mistakes remain possible.

Your engineering contribution includes the preprocessing and inference pipeline, endpoint integration, YouTube segmentation, independent handling of audio authenticity, transcript screening and fusion, UI and tests. Pretrained architectures and the external Sinhala ASR checkpoint are reused components. Claim your actual training and evaluation work with its records; do not claim that you invented Wav2Vec2, WavLM or XLM-RoBERTa.

## 3. File map and responsibilities

| File | Responsibility | Called by / entry point |
| --- | --- | --- |
| `backend/app.py` | HTTP validation, orchestration, fusion, JSON responses, temporary upload cleanup | Flask routes invoked by frontend requests |
| `backend/preprocessing.py` | Text cleaning, script heuristic, targeted romanized-word mapping | `model_manager.predict_text` and `predict_text_batch` |
| `backend/audio_utils.py` | Load, resample, normalize, pad/truncate an uploaded file | `model_manager.predict_audio` |
| `backend/model_manager.py` | Lazy model loading, prediction, audio features, template explanations | Backend routes |
| `backend/youtube_utils.py` | Metadata, audio download, VAD, segmentation, comment retrieval | YouTube routes |
| `backend/asr_manager.py` | Cached Sinhala CTC transcription, temporal selection, review score | `youtube_analyze` |
| `backend/test_asr.py` | Offline mocked regression tests | `unittest` discovery, not the running UI |
| `backend/ASR_NOTES.md` | Overlap, score and validation documentation | Developer reference; not executable |
| `backend/requirements.txt` | Backend dependency lower bounds | pip installation |
| `frontend/app.py` | Streamlit widgets, HTTP calls, result presentation, report builders | Streamlit script runner |
| `frontend/.streamlit/config.toml` | Theme and server settings | Streamlit started from frontend directory |
| `frontend/requirements.txt` | Streamlit, Requests and ReportLab | pip installation |
| `models/` | Saved model configurations, tokenizers/processors and weights | Model loaders |
| `notebooks/` | Training-related material; not imported by the application | Separate experimentation |
| `.venv/`, `frontend/venv/` | Installed Python environments | Python executables used to start services |
| `tmp/`, `output/` | Local test/export artifacts | Development and report preparation |

**Important distinction:** the app's temporary audio files go to the operating-system temporary directory, not necessarily the repository's `tmp` folder. `tempfile.gettempdir()` decides the actual location.

## 4. Architecture and exact call chains

### Two Python services

```text
Browser
  ↕ Streamlit UI protocol
Streamlit Python process, port 8501 — frontend/app.py
  ↕ requests.post: JSON or multipart file upload
Flask Python process, port 5000 — backend/app.py
  ↕ Python function calls
Preprocessing / downloads / model loaders / PyTorch inference
  → Python dictionaries → JSON → Streamlit display
```

The `requests.post` calls run in the Streamlit server process. They are not JavaScript fetch requests directly from the browser. `127.0.0.1` means the machine hosting the Streamlit process. If the frontend is moved to another server, its API address must point to the actual backend. `CORS(app)` enables cross-origin browser access, but CORS is not needed for these server-to-server Requests calls and is not authentication.

### Manual text

```text
Text tab / Analyze Text
→ POST /api/text/predict {text: ...}
→ text_predict()
→ predict_text()
→ preprocess() → clean_text() → is_romanized()
   → if romanized: clean_noise() → convert_hate_words()
→ _load_text_model() → tokenizer → XLM-R classification
→ softmax → argmax → result JSON
→ label, scores and processed text displayed
```

### Uploaded audio

```text
Audio tab / file upload / Analyze Audio
→ POST /api/audio/predict, multipart field audio
→ audio_predict() → validate extension → save UUID temporary file
→ predict_audio(filepath) → load_audio(filepath)
→ 16 kHz mono, peak normalization, first 10 seconds or zero padding
→ Wav2Vec2 hate classifier AND WavLM authenticity classifier
→ JSON → two result cards
→ finally: remove uploaded temporary file
```

This path does not use ASR, YouTube VAD, transcript fusion or the 0.60 hate-confidence abstention rule. Both classifiers receive the same prepared upload waveform.

### YouTube audio and transcript

```text
YouTube form submitted
→ POST /api/youtube/metadata → get_video_metadata()
→ preview title/channel/thumbnail
→ POST /api/youtube/analyze → youtube_analyze()
→ analyze_youtube_video()
   → metadata and duration validation → download_youtube_audio()
   → librosa.load → preserve original copy
   → _load_vad() → speech timestamps → mask non-speech samples
   → fixed 10-second segments + original ASR context windows
→ for each segment:
   predict_audio_array(masked, deepfake_audio=original)
   → audio features, hate classifier, authenticity classifier
   if not NO_SPEECH:
     transcribe_sinhala_audio(context, core offsets)
     → _load_model() → CTC logits → decode_segment()
     → recognition screen + Sinhala usability screen
     → accepted text only: predict_text()
   _fuse_youtube_hate_predictions()
→ segment dictionaries and summary
→ timeline, transcript warnings, explanations, HTML report
```

### Optional comments

After audio results have been drawn, the frontend makes a separate comments request if the checkbox is enabled. `youtube_comments()` calls `fetch_youtube_comments()`, then filters for at least one Sinhala Unicode character, then calls `predict_text_batch()`. The results affect the comment summary only. They do not change the audio timestamps or fused segment labels. These operations are sequential, not background jobs.

## 5. Models, tensors and terminology

### What is actually deployed

| Task | Runtime class / evidence | Output |
| --- | --- | --- |
| Text classification | `AutoModelForSequenceClassification`; local config says `XLMRobertaForSequenceClassification`, `model_type=xlm-roberta` | Two class logits: NOT, OFF |
| Audio hate classification | `Wav2Vec2ForSequenceClassification`; local config agrees | Two class logits: NOT, OFF |
| Audio authenticity | `AutoModelForAudioClassification`; local config says `WavLMForSequenceClassification`, `model_type=wavlm` | REAL, FAKE, with indices read from config |
| Sinhala ASR | `AutoModelForCTC`; default external Sinhala XLS-R checkpoint | A vocabulary distribution at every audio frame |
| Speech activity | Silero VAD from Torch Hub | Speech start/end sample positions |

The health endpoint and manual audio tab still contain a hard-coded Wav2Vec2 deepfake label. **The saved deepfake checkpoint is WavLM.** Point to `models/model_audio_deepfake/config.json`, not stale display text, to establish the architecture.

All three local classifier configs show hidden size 768 and 12 Transformer layers/12 attention heads. Audio classifier projection size is 256. These are saved architecture settings, not proof of which layers were trained. The text config vocabulary size is 250002; that is a tokenizer vocabulary size, not the number of training sentences. Text max position embeddings 514 does not mean this app uses 514 tokens: inference explicitly limits input to 128.

XLM-RoBERTa is a multilingual Transformer encoder; the app adds/loads a sequence-classification head rather than generating a free-form response. [Official architecture reference](https://huggingface.co/docs/transformers/model_doc/xlm-roberta).

Wav2Vec2 takes waveform samples and learns speech representations; its classification and CTC heads solve different tasks. The classifier outputs one class decision for a clip, while CTC outputs a token sequence. [Official Wav2Vec2 reference](https://huggingface.co/docs/transformers/model_doc/wav2vec2).

WavLM is another pretrained speech representation architecture. This app uses its sequence-classification checkpoint for authenticity; it does not run its pretraining objective during inference. [Official WavLM reference](https://huggingface.co/docs/transformers/model_doc/wavlm).

### Shapes you should be able to explain

| Object | Typical shape | Meaning |
| --- | --- | --- |
| Mono 10-second waveform | `[160000]` | 16000 samples per second × 10 seconds |
| Audio processor input values | `[1, 160000]` | Batch dimension added |
| Text `input_ids` and mask | `[1, 128]`, or `[B, 128]` | Subword IDs and padding visibility |
| Classifier logits | `[1, 2]`, or `[B, 2]` | Unnormalized class scores |
| Classifier softmax | `[1, 2]` | Scores normalized over the two classes |
| ASR logits | `[1, T, V]` | Frames × token vocabulary, with a batch dimension |
| ASR logits after `[0]` | `[T, V]` | Single example selected |
| CTC argmax IDs | `[T]` | One highest-scoring token per frame |

`softmax(z_i) = exp(z_i) / sum_j exp(z_j)`. It preserves the ranking of logits but makes values sum to one over the selected axis. `argmax` gives the winning index; it does not give the winning probability. With `[0.2, 0.8]`, argmax is 1 and confidence is 0.8. No sigmoid is used because these are mutually exclusive two-class outputs.

`dim=1` selects classes in `[batch, classes]`; `dim=-1` selects vocabulary in `[frames, vocabulary]`. A tensor is a multidimensional numeric array with device/dtype information; NumPy arrays are used for waveform handling, and PyTorch tensors for model execution. Conversion back to Python `float`, `int`, `bool` and lists makes responses JSON-serializable.

`model.eval()` changes training-sensitive behavior such as dropout. `torch.no_grad()` disables gradient recording; `torch.inference_mode()` additionally removes some inference overhead. Neither context manager replaces `eval()`, and none of them trains or updates weights. [PyTorch inference-mode reference](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html).

## 6. backend/preprocessing.py — every function

### Dictionaries: HATE_WORDS and HATE_VARIATIONS

`HATE_WORDS` maps selected romanized tokens to Sinhala Unicode; `HATE_VARIATIONS` maps spelling variants to canonical keys first. These dictionaries are input transformations, not the final classifier. A matched token does not immediately return OFF. Some entries are ordinary or context-dependent words, so dictionary membership is not evidence that a sentence is hateful.

The intended rationale is to reduce selected spelling variation while keeping the rest of a mixed sentence available to the learned model. This is targeted normalization, not complete Sinhala transliteration or translation. Unlisted romanized forms remain unconverted; correctness and bias of the mappings need evaluation.

### clean_text(text), lines 134–147

**Called by:** `preprocess`. **Returns:** a string. It converts the input to a string, strips outer whitespace, removes URL patterns, the standalone literal `URL`, mentions, hashtags and standalone digits, then collapses whitespace.

`re.sub(pattern, replacement, text)` replaces regex matches. Raw strings such as `r'\s+'` make regex backslashes readable. `\S+` means non-whitespace sequence; `\s+` means whitespace sequence; `\b` marks word boundaries. The code removes hashtag content rather than merely deleting `#`; this can remove meaningful evidence. Regex word classes do not amount to Sinhala linguistic tokenization. The literal `URL` removal is case-sensitive at this stage.

**Viva answer:** “I reuse the inference preprocessing expected by the text model. Cleaning reduces some social-media noise, but it can also discard useful content; its benefit needs an ablation rather than assumption.”

### clean_noise(text), lines 154–159

**Called by:** `preprocess`, only on the romanized branch. It lowercases and uses `(.)\1{2,}` with replacement `\1\1` to reduce runs of three or more identical characters to two. Therefore `hondaaaa` becomes `hondaa`, not `honda` as one comment suggests. It is a character rule, not a spelling-correction model. Lowercasing also means the full pipeline does not preserve original casing even though the dictionary converter alone preserves unmatched token spelling.

### is_romanized(text), lines 166–176

**Called by:** `preprocess`. It counts characters in U+0D80–U+0DFF and divides by length after removing ordinary spaces. Below 10% Sinhala is considered romanized; an empty denominator returns False. This is script detection, not language identification: an English sentence can be labelled romanized. The ratio includes punctuation and does not explicitly remove every kind of whitespace.

### convert_hate_words(text), lines 183–215

**Called by:** the romanized branch of `preprocess`. Split on whitespace; detach one trailing punctuation character from `.,!?;:"'`; lowercase the lookup token; normalize via `HATE_VARIATIONS.get(w, w)`; look up `HATE_WORDS`; reattach punctuation; join with spaces.

`dict.get(key, default)` avoids a KeyError for unknown words. Only one trailing punctuation character is removed, so repeated punctuation can prevent a match. There is no stemming, morphological analyser, fuzzy edit-distance matching or language model in this function. Variation normalization runs before the direct dictionary lookup and can override a direct key's intended mapping.

### preprocess(text), lines 223–251

**Called by:** both single and batch text prediction. It always calls `clean_text`. It then chooses the script branch. Romanized input gets noise cleaning and targeted conversion; Unicode input gets no extra conversion. Returns `(processed_text, script_type)`, a tuple unpacked by the caller.

The script label describes the selected branch, not the final converted output. The final text may be mixed script. A mixed sentence with sufficient Sinhala characters goes down the Unicode branch, so its remaining romanized terms will not be mapped. This is a limitation you can demonstrate honestly.

## 7. backend/audio_utils.py — uploaded-file preparation

### Constants and load_audio(filepath, max_length=MAX_LENGTH), lines 18–47

**Called by:** `model_manager.predict_audio`, not the YouTube segmentation function. `SAMPLE_RATE=16000`, `MAX_DURATION=10`, `MAX_LENGTH=160000`.

`librosa.load(filepath, sr=16000, mono=True)` decodes supported media, resamples to the requested rate and produces a mono waveform. Resampling changes sample count to preserve duration; it is not speeding up playback. Mono reduces channel handling and matches expected model inputs. The code then divides by the maximum absolute amplitude when the peak is positive, making peak magnitude one. This preserves relative shape within the recording but changes amplitude and can boost noise.

Short clips are padded on the right with zeros. Longer clips are truncated to the first 160000 samples. For a 30-second upload, only the first ten seconds are classified; the original full-file peak is calculated before truncation. A five-second clip contains 80000 real samples and 80000 padding samples. Float32 uses four bytes per sample: a 10-second mono array alone is about 640000 bytes, excluding model memory and temporary copies.

**Failure behavior:** a decoding exception returns zeros, which the upload path still classifies. That can produce a misleading result rather than an explicit decode failure. Also, `np.max` on a successfully decoded empty array can raise an error; the docstring promises more than the implementation guarantees. Do not say this function reliably rejects all corrupt or empty files.

**Why fixed duration?** It aligns input size with the project's training/inference convention and bounds CPU work. It is a design tradeoff, not proof that ten seconds is optimal for Sinhala meaning.

## 8. backend/model_manager.py — all loaders and inference

### Module setup

`Path(__file__).resolve().parent.parent` locates the project root relative to this source file, rather than relying on whichever directory the terminal happens to use. Model paths then resolve under `models`. `DEVICE=torch.device("cpu")` makes runtime inference CPU-only. `_cache={}` holds loaded processors and models for the lifetime of this backend process. It does not cache predictions, persist after restart, or share memory between separate worker processes.

### _load_text_model(), lines 42–62

**Called by:** `predict_text` and `predict_text_batch`. Returns cached tokenizer/model if both are present; otherwise loads the local tokenizer and sequence classifier, supplies two labels `{0: NOT, 1: OFF}`, uses `ignore_mismatched_sizes=False`, sets evaluation mode/CPU, and stores both in `_cache`.

Failing on a mismatched classification head is preferable to silently using incompatible trained weights. `from_pretrained` loads configuration and weights; it does not mean the request trains a model. “Auto” selects a concrete architecture from configuration. The label mapping in this loader is explicit, so training label semantics must agree.

### _load_audio_hate_model(), lines 65–81

**Called by:** both audio prediction functions. Loads `Wav2Vec2Processor` and `Wav2Vec2ForSequenceClassification` from the hate checkpoint. Caches them and sets eval/CPU. The processor bundles preparation logic; its presence does not mean this classification call produces a transcript.

The saved processor specifies `do_normalize=true`. This feature-extractor normalization is distinct from the earlier manual peak normalization. Explain both stages if asked why amplitudes change twice.

### _load_audio_fake_model(), lines 84–106

**Called by:** both audio prediction functions. Uses `AutoFeatureExtractor` and `AutoModelForAudioClassification` so the configured audio-classifier architecture can be selected. Checks that config labels include REAL and FAKE before caching. The current checkpoint is WavLM, despite a stale comment referring to a future WavLM checkpoint.

The deepfake preprocessor has `do_normalize=false`, making the choice to preserve the original YouTube waveform especially relevant. It specifies `return_attention_mask=true`, but these inference calls pass only input_values to the classifier. Padding therefore is not explicitly masked at model call time; do not claim otherwise.

### _deepfake_label_indices(model), lines 109–112

**Called by:** `predict_audio` and `predict_audio_array`. Reverses `id2label` into uppercase label-to-integer positions. Returns `(real_idx, fake_idx)`. This avoids assuming that index 1 always means FAKE across arbitrary checkpoints. A mislabeled checkpoint is still a problem: reading configuration cannot prove its semantic correctness.

### _compute_audio_features(audio), lines 119–143

**Called by:** `predict_audio_array` only. Calculates peak amplitude, root-mean-square energy, an approximate zero-crossing rate and a sample-amplitude “silence ratio.” RMS is `sqrt(mean(audio**2))`. Zero-crossing rate here is `mean(abs(diff(sign(audio)))) / 2`; zero samples can contribute half-transitions.

A sample is called silent if `abs(sample)<0.02`. `has_speech` is True if the fraction of such samples is below 0.7. Energy is High if RMS > 0.15, Medium if > 0.05, otherwise Low. All these thresholds are heuristics.

These features are computed on the VAD-masked, peak-normalized, padded hate waveform in the YouTube path. They are not independent measurements of raw microphone loudness. Padding, slow speech with pauses, and weak voiced samples can drive the silence ratio upward. This is not the same as Silero's speech detector and is not a rigorous SNR estimate. RMS/ZCR are not concatenated into the neural model input; they support explanation text and speech-presence decisions.

### _generate_explanations(...), lines 146–236

**Called by:** `predict_audio_array`. Uses label, winning probability, probability margin, energy and has_speech to select textual templates. Returns hate_explanation, fake_explanation and audio_features. Margin is the absolute difference between the two class probabilities. For binary probabilities summing to one, margin > 0.5 corresponds to winning score > 0.75.

Confidence categories are High above 0.85, Moderate from 0.60 to 0.85 inclusive, and Low below 0.60. A separate text chooses “decisive/moderate/narrow,” but the `decisive` variable is currently unused. There is a FAKE UNCERTAIN explanation branch, but the current authenticity inference always returns REAL or FAKE for speech, so that branch is not normally reached.

**Crucial viva answer:** “These are rule-based descriptions of outputs and simple signal statistics, not SHAP, LIME, attention explanations or proof that the model detected a particular causal artifact.” Claims about aggressive tone or manipulation artifacts in the displayed templates should not be treated as measured features. The template says classification was skipped for no speech, but both model forward passes actually occur before those labels are overridden.

### predict_text(raw_text), lines 243–297

**Called by:** manual text route and accepted YouTube transcripts. Preprocess → load tokenizer/model → tokenize to exactly 128 positions using padding and truncation → pass input_ids and attention_mask → obtain logits → softmax over classes → NumPy → argmax → construct result.

`attention_mask` tells the text model which positions are real input versus padding. 128 means subword tokens including tokenizer-added special tokens, not 128 words or characters. Truncation can lose decisive information near the end of a long comment or transcript. A nonempty raw input can become empty after cleaning; this function still feeds the tokenizer because there is no second empty-text rejection.

Returned keys: input_text, processed_text, script_type, prediction, is_offensive, confidence, probabilities. The confidence is the winning class softmax score, not evaluated accuracy or a calibrated probability of correctness. The function has no low-confidence abstention branch.

### predict_text_batch(raw_texts, batch_size=16), lines 300–353

**Called by:** comments route. Empty list returns an empty list; batch_size below one raises ValueError. It preprocesses each comment once, loads the shared model once, and loops over chunks of at most 16. Each batch is tokenized to `[B,128]` and classified in one forward pass. It reconstructs results in original order using `index=start+offset`.

Batching improves throughput by reducing repeated model-call overhead, but it is not parallel network fetching or asynchronous processing. Padding all texts to 128 simplifies shapes but costs more than dynamic padding for short comments. Result schema matches single-text inference. `zip(comments,predictions)` later relies on this ordering and matching length.

### predict_audio(filepath), lines 356–432

**Called by:** upload route. Prepares the first ten seconds via `load_audio`. Runs hate processor/model, gets class probabilities and returns OFF/NOT. Runs the deepfake processor/model on the same waveform and maps REAL/FAKE indices from config. Returns both independent predictions.

There is no VAD, no speech heuristic, no ASR and no 0.60 abstention here. This explains why an uploaded clip and the same clip analysed through YouTube can produce different outputs. Audio authenticity and offensiveness are different questions: genuine speech can be offensive; synthetic speech can be non-offensive.

### predict_audio_array(audio, deepfake_audio=None), lines 435–522

**Called by:** YouTube analysis route. Accepts a prepared NumPy array directly, avoiding a temporary file for every segment. Computes features, runs hate inference, then overrides its label to NO_SPEECH if has_speech is false or UNCERTAIN if winning confidence < 0.60. Otherwise returns the winning hate label. Model probabilities remain present even when the final label abstains.

For authenticity, use deepfake_audio when supplied, otherwise audio. It always runs inference; if has_speech is false it sets authenticity_label=NO_SPEECH and audio_authentic=True. That Boolean is a bookkeeping choice to avoid a fake flag, **not proof of authenticity**. For speech it always returns REAL or FAKE, with no 0.60 abstention threshold. It attaches template explanations based on the original audio-model results.

**Caching limitation:** these three classifier loaders do not use locks, unlike the ASR loader. Concurrent first requests can race and load duplicate model instances. CPU-only and lazy loading help deployment but do not guarantee low memory: all used models stay cached, and loading all of them can consume substantial RAM.

## 9. backend/youtube_utils.py — video preparation and comments

### Module constants and imports

`SAMPLE_RATE=16000`, `SEGMENT_DURATION=10`, `SEGMENT_SAMPLES=160000`, `ASR_CONTEXT_SAMPLES=32000`, `MAX_VIDEO_DURATION=600`. UUID produces collision-resistant filenames. `tempfile` locates operating-system scratch storage. `yt_dlp` retrieves metadata/media/comments; FFmpeg converts extracted audio to WAV; librosa converts decoded samples to 16 kHz mono. Requests is used for the official comments API. These are distinct responsibilities: FFmpeg is not a classifier or a speech recognizer.

### format_time(seconds), lines 25–28

**Called by:** duration error messages and segment creation. Integer division `//60` gives whole minutes; `%60` gives remaining seconds; `:02d` prints two digits. Fractions are truncated for display, while numeric start_time/end_time retain two decimals. A formatted timestamp is not a word alignment.

### get_video_id(url), lines 30–49

**Called by:** `fetch_youtube_comments`. Uses `urlparse` and `parse_qs` to support youtu.be paths, watch?v=, embed, v, shorts and live forms. It checks the ID against `[A-Za-z0-9_-]{11}`. Returns None for unsupported or malformed inputs.

The host check uses `endswith("youtube.com")`, which is broader than exact host/subdomain validation. It should not be presented as a strong domain allowlist. Also, metadata/download functions pass their URL directly to yt-dlp and do not use this validator. Validation coverage differs across endpoints.

### get_video_metadata(url), lines 51–64

**Called by:** metadata endpoint and `analyze_youtube_video`. Opens YoutubeDL as a context manager and uses `extract_info(download=False)`. Returns title, duration, channel/uploader fallback, thumbnail and URL. It does not download the media file, although metadata extraction still involves network requests.

Every extraction exception is wrapped as ValueError with “Invalid URL or video unavailable.” Therefore this message can also represent network, DNS or permission failures. The WinError 10013 seen earlier was blocked network access, not proof that the video was invalid. The frontend preview and analysis each retrieve metadata, so there are repeated network lookups.

### download_youtube_audio(url), lines 66–98

**Called by:** `analyze_youtube_video`. Builds a UUID output template and requests `bestaudio/best`. The preferred extraction clients are android/web. A postprocessor named FFmpegExtractAudio converts the selected source to WAV. Returns `(filepath, metadata)`; normally the path is UUID.wav, with requested_downloads as a fallback if that path is absent.

The download stage does not explicitly force 16 kHz mono; that occurs in librosa afterward. Converting a compressed YouTube source to WAV cannot restore information lost by original compression. The exception wrapper mentions FFmpeg, but a download failure may instead involve connectivity, platform restrictions or extraction changes. Failed/intermediate downloads may leave artifacts because cleanup focuses on the successfully selected path.

### _load_vad(), lines 105–114

**Called by:** `analyze_youtube_video`. Loads Silero VAD through torch.hub once, retaining model and utility objects in module globals. `force_reload=False` allows cached reuse; `trust_repo=True` permits repository code. The first uncached use needs network access. No repository revision is pinned, which weakens reproducibility. No initialization lock is present here.

VAD means voice activity detection: locating speech intervals, not identifying a speaker, transcribing words or detecting hateful content. The local `get_speech_timestamps` name is a callable retrieved from the utility tuple, not a function defined in this repository.

### analyze_youtube_video(url), lines 116–201

**Called by:** Flask `youtube_analyze`. It prepares data; it does not itself run the hate, authenticity or ASR classifiers.

1. Retrieve metadata. Reject duration <= 0 or duration > 600 seconds.
2. Download audio, verify a path exists, decode as 16 kHz mono and reject an empty waveform.
3. Copy the decoded original into asr_audio before modifying the hate path.
4. Obtain Silero speech timestamps. Make a Boolean mask initially false, set speech intervals true, multiply audio by the mask. Outside speech, samples become zero; no samples are removed, preserving time alignment.
5. If VAD raises an exception, warn and continue with raw audio. This fallback means behavior can differ from a successful-VAD run.
6. Iterate by 160000 samples. Make a masked hate segment and original same-interval segment. Pad both final classification segments to ten seconds.
7. Peak-normalize only the masked hate segment. The original YouTube deepfake waveform is not manually peak-normalized here.
8. Build an unpadded original ASR context window extending two seconds to each available side. Store core sample offsets into it.
9. Return metadata and segment dictionaries. The finally block removes the downloaded file after arrays have been prepared, even before neural classification starts in the caller.

Multiplying by a speech mask is not denoising inside speech or separating background music from overlapping speech. It only silences intervals judged non-speech. The split remains fixed-time, not sentence-aware or silence-boundary segmentation. The complete waveform and prepared segment copies are held in memory; it is not streaming inference.

Example for 19 seconds: segment 0 core is 0–10 seconds with context 0–12; segment 1 core is 10–19 with context 8–19. The second classifier input has one second of padding, but its ASR context does not. Numeric end time can differ from metadata duration because downloaded sample duration and metadata rounding can differ.

### _normalise_comments(raw_comments, max_comments=None), lines 203–217

**Called by:** official API retrieval and yt-dlp fallback. Strips text, skips empty comments, supplies Unknown author, normalizes likes/like_count, and optionally caps accepted comments. Returns a consistent list of dictionaries. This is schema normalization, not language cleaning or hate classification.

### _fetch_comments_from_api(video_id, api_key, max_comments=None), lines 220–263

**Called by:** `fetch_youtube_comments` when YOUTUBE_API_KEY is present. Requests commentThreads with part=snippet, plain text and at most 100 results per page, following nextPageToken. Each request has a 20-second timeout. It extracts the topLevelComment snippet, not every reply in a thread. A non-200 response becomes RuntimeError. No retries, backoff or quota-management policy is implemented.

If a cap is supplied it applies before the later Sinhala filter, so requesting 100 comments does not guarantee 100 Sinhala comments. With no cap, it follows available pages until completion; this does not guarantee every historical/deleted/private/moderated comment is accessible.

### fetch_youtube_comments(url, max_comments=None), lines 266–312

**Called by:** Flask comments route. Validate/extract video ID. If an environment API key exists, try the official API. If successful with comments, return them and source='YouTube Data API'. If the API succeeds but returns none, raise ValueError and do not try fallback. Other exceptions are saved as api_error and allow fallback.

Fallback uses yt-dlp with skip_download=True and getcomments=True, optionally an extractor comment cap. Normalize returned comments and identify source='yt-dlp'. If both paths fail, construct an informative error. The yt-dlp path does not independently filter top-level versus reply comments, so its scope can differ from the official API path. A source field makes that distinction visible.

## 10. backend/asr_manager.py — speech recognition in detail

### What CTC means

Connectionist Temporal Classification supports mapping many audio frames to a shorter token sequence without requiring a manual frame-to-character alignment during training. At inference, this code takes the highest-scoring token at each frame, collapses repeated runs and removes blanks. It is greedy decoding: no beam search, external language model or grammar correction is added.

Example using B for blank: `[B, ක, ක, B, ක, ම, ම, B]` becomes `කකම`. The two ක tokens separated by a blank must remain two tokens. That is why the final tokenizer decode uses group_tokens=False: our own run loop has already performed collapsing.

### Globals and thresholds

`_processor` and `_model` initially None. `_model_lock=threading.Lock()` protects initialization. `MODEL_ID` is read from SINHALA_ASR_MODEL or defaults to the Sinhala XLS-R checkpoint. An environment override must remain compatible with the CTC model and convolution-config assumptions; it is not a plug-in switch for any arbitrary recognizer.

`MIN_TOKEN_SCORE=0.65`, `MAX_WEAK_TOKEN_FRACTION=0.25`. These are initial engineering heuristics. They were not calibrated against a labelled Sinhala transcription set in the work shown here. A high score does not prove correct spelling, meaning or even faithful transcription.

### _load_model(), lines 24–42

**Called by:** `transcribe_sinhala_audio` after cheap input checks. Uses a double check around a lock: test whether loading is needed, acquire the lock, check again in case another thread already loaded. It first loads cached processor/model using local_files_only=True; on OSError it tries normal from_pretrained, which can retrieve missing files. Sets eval/CPU and returns both.

This lock protects initialization, not every inference operation and not other model loaders. The ASR weights are external pretrained/fine-tuned weights. No ASR training occurs in this file. Disk cache and in-process cache are different: disk cache survives a restart; the Python globals do not.

### transcribe_sinhala_audio(audio, core_start=0, core_end=None), lines 82–116

**Called by:** `youtube_analyze`. Empty input returns review without loading; invalid core interval raises ValueError; NaN/infinity samples return review; essentially zero peak in the core (<1e-7) returns silent review. These checks are not a robust detector of low-volume speech or noise.

Processor input is the full original context window as float32 at 16 kHz. `model(**inputs)` unpacks all processor fields as keyword arguments, including any attention mask it provides. `logits[0]` removes the batch dimension. Inference is CPU and gradients are disabled.

To map sample offsets to convolutional frame coordinates, initialize stride S=1 and receptive field R=1. For each convolution kernel k and stride s: `R = R + (k-1)*S`, then `S = S*s`. Offset is `(R-1)/2`. A frame's approximate centre is `offset + frame_index*S`. Therefore a sample boundary maps to `(sample_boundary-offset)/S`.

For kernels [10,3,3,3,3,2,2] and strides [5,2,2,2,2,2,2], S=320 samples (20 ms at 16 kHz), R=400 samples (25 ms), and offset=199.5 samples. These figures illustrate that configuration; the function calculates from the loaded model rather than hard-coding them. This is the convolutional frame grid, not the full Transformer context receptive field: attention can use the wider window.

It calls decode_segment with the core frame boundaries. A caught model/decoding exception becomes RuntimeError; the route then converts that segment into review status. An exception in the loader is outside this inner try but is still caught by the route.

### decode_segment(logits, processor, blank_id, frame_start=0, frame_end=None), lines 45–79

**Called by:** transcribe_sinhala_audio and tests. Input is `[frames,vocabulary]` logits. Softmax produces a token distribution per frame. `.max(dim=-1)` gives each frame's winning score and token ID. The loop finds maximal consecutive runs of identical winning IDs.

For a run covering `[start,end)`, midpoint is `(start+end-1)/2`. Keep it only if it is not blank and its midpoint is in `[frame_start,frame_end)`. Append one token and the maximum probability in its run. Decode with group_tokens=False and skip_special_tokens=True, then strip whitespace.

The recognition score is the mean of retained run maxima, not the product of token probabilities, not mean confidence across all frames, and not WER. Blank runs are excluded so confident silence cannot inflate the score. A long token run contributes one value just like a short run. Nonblank special/delimiter tokens can contribute to scoring even if tokenizer decoding later suppresses or transforms them.

`weak_fraction` counts retained run scores strictly below 0.65. Review is required if text is empty, mean score is strictly below 0.65, or weak_fraction is strictly greater than 0.25. Equality matters: exactly 0.65 is not weak; exactly 25% weak does not trigger the fraction rule. No tokens means score=0 and weak_fraction=1. Return text, four-decimal score, review Boolean and reason.

Worked example: run scores [0.90, 0.80, 0.40, 0.40] average 0.625 and have weak_fraction=0.5, so review is required. [0.90,0.80,0.70,0.40] average 0.70 with weak_fraction=0.25 passes this numerical screen, but the route's Sinhala check may still reject it.

### Why overlap, and what it cannot guarantee

Two seconds either side provide acoustic context near a hard boundary and preserve the core's intended timestamp. Interior windows can reach 14 seconds. This costs more inference than non-overlapping ten-second windows. It may help boundary recognition, but improvement has not been measured here.

Each window is inferred separately. Midpoint selection prevents duplicate assignment when decoding intervals from the same logits; it does not prove perfect deduplication across independent window predictions, whose token timing can differ. Words can split across display segments. This is not global forced alignment or sentence-level decoding. Slow speech may still be excluded earlier by the sample-amplitude speech heuristic. There is no automatic speed correction or denoising.

## 11. backend/app.py — server and every endpoint

### Import-time setup and executable entry

On Windows, search the user's WinGet Packages tree for FFmpeg bin folders and prepend the first existing match to PATH. This helps yt-dlp locate the executable; it does not install FFmpeg or verify which build is best. Flask(__name__) creates the app, CORS(app) enables cross-origin headers, and os.makedirs(..., exist_ok=True) creates the upload directory if needed.

`if __name__ == "__main__":` starts the server only when run as the main script. Importing this module in tests registers routes and performs setup but does not start app.run. Host 0.0.0.0 binds all network interfaces; localhost is one way to reach it, not the only exposed interface. Port is 5000 and debug=False. This is the Flask development server, with no application authentication, database or worker queue here.

### Route contracts

| Method and path | Input | Success output | Frontend timeout |
| --- | --- | --- | --- |
| GET /api/health | None | status, service, model description strings | Not routinely polled by UI |
| POST /api/text/predict | JSON text | success and result dictionary | 120 seconds |
| POST /api/audio/predict | multipart audio file | success and both predictions | 180 seconds |
| POST /api/youtube/metadata | JSON url | success and metadata | 60 seconds |
| POST /api/youtube/analyze | JSON url | success, metadata, segments, summary | 600 seconds |
| POST /api/youtube/comments | JSON url; optional max_comments | success, comments, summary, source | 1800 seconds |

Timeouts stop the client waiting; they do not guarantee cancellation of an already running backend operation. Returned percentages are segment/comment fractions, not test-set accuracy and not necessarily duration-weighted exposure.

### health(), lines 93–104

Returns an “ok” dictionary through jsonify. It does not load or test models, check FFmpeg, contact YouTube or run a prediction. Thus a healthy endpoint proves the Flask process responds, not that all inference dependencies work. Its deepfake description is stale relative to the WavLM config.

### text_predict(), lines 108–127

Decorator registers the path and POST method. `request.get_json(silent=True)` returns parsed JSON or None instead of raising for parsing failure. Missing text/blank text gives 400. Valid stripped text goes to predict_text. Exceptions inside inference return 500 with an error message.

It checks presence and emptiness, not strict JSON schema/type: a number or null as text can fail at `.strip()` outside the inference try. Length limits, authentication and rate limiting are absent. The endpoint should not be described as production-hardened.

### audio_predict(), lines 131–168

Requires multipart field audio, a nonempty filename and a lowercased extension in the allowed set. It generates a new UUID filename rather than trusting the client's path. Saves under UPLOAD_DIR, calls predict_audio and returns JSON. Finally removes the file, swallowing OSError on deletion.

Extension validation does not prove content type or safe decoding. There is no explicit upload-size limit in this code. UUID reduces name collisions and prevents use of a supplied directory path, but does not replace content checks. Cleanup occurs after success or inference failure; a process crash can still leave files.

### _is_usable_sinhala_transcript(text), lines 83–89

**Called by:** youtube_analyze after ASR. Builds a list of non-whitespace alphanumeric characters. Requires at least three Sinhala-block characters and a Sinhala fraction >= 0.25 within that list. `sum(Boolean expressions)` counts True values because True behaves as 1.

This rejects some empty/non-Sinhala outputs, not all hallucinations. Sinhala vowel signs/combining marks may not pass isalnum, so this is not grapheme counting or language validation. Valid very short utterances can be rejected; syntactically wrong Sinhala can pass. The docstring's reference to rejecting hallucinations overstates the guarantee.

### _fuse_youtube_hate_predictions(...), lines 46–80

**Called by:** youtube_analyze, after audio and optional text predictions. This is decision-level rule fusion, not learned fusion, probability averaging or majority vote.

| Audio label | Accepted transcript | ASR needs review | Fused label |
| --- | --- | --- | --- |
| NO_SPEECH | Normally absent | Any | NOT_ASSESSED |
| OFF | Any | Any | OFF |
| NOT or UNCERTAIN | OFF | False | OFF |
| NOT | Absent | True | UNCERTAIN |
| UNCERTAIN | Absent | Any | UNCERTAIN |
| UNCERTAIN | NOT | False | NOT |
| NOT | NOT | False | NOT |

Branch order matters: no-speech wins first, then any offensive evidence, then review/uncertainty, then non-offensive. Setting hate_speech=False for UNCERTAIN means “not positively flagged,” not “known safe.” The returned dictionary preserves audio_hate_label, transcript_hate_label and fusion_reason separately from the fused hate_label.

Why OR for OFF? It lets either usable path surface offensive content that the other misses. Tradeoff: a false positive in either branch can flag the segment, and ASR errors can propagate into text decisions. There is no measured proof here that this fusion improves precision/recall over every single model.

### youtube_analyze(), lines 173–283

Validate URL presence/blankness. Ask youtube_utils for arrays and metadata. For each segment, pop array fields and ASR offsets so they do not enter jsonify. `.pop(key, default)` both retrieves and removes a field; this also reduces references retaining large arrays.

Run predict_audio_array with masked audio and original deepfake audio. Initialize transcript fields. If the audio label is not NO_SPEECH, call ASR. Catch a transcription exception locally, log its traceback and produce a review result for this segment. Keep even an uncertain candidate transcript for display; call predict_text only when both ASR score screening and Sinhala usability pass. A later text-classifier exception is not handled by this inner ASR catch and can still fail the whole request.

Build the result using `{**seg, **pred, **fusion, ...}`. In a dictionary merge, later values overwrite earlier duplicate keys. Thus fusion replaces pred's hate_label and hate_speech, while hate_confidence, hate_probabilities and explanations still refer to the audio classifier. There is **no computed fused probability**. This distinction is essential when a transcript causes OFF while the audio classifier says NOT.

`del audio` etc. removes local references, not every copy or cached model. `gc.collect()` asks Python to collect unreachable cyclic objects; it does not unload cached model weights or guarantee that OS memory immediately drops.

Summary counts are based on segment decisions. Hate/fake/both overlap and must not be added as mutually exclusive categories. Clean requires fused NOT and authenticity REAL. review_segments counts ASR NEEDS_REVIEW, not every source of uncertainty. Percentages divide flagged count by all segments, so a one-second last segment counts equally to a ten-second segment and unassessed segments remain in the denominator.

ValueError and RuntimeError in the outer block return 400; other exceptions return 500. This is a simple error convention, not precise HTTP classification for all upstream/network/model failures. The endpoint processes segments sequentially and returns the final JSON only when all are processed; it does not stream per-segment progress.

### youtube_metadata(), lines 287–298

Requires a nonempty URL, calls get_video_metadata and returns success/metadata. ValueError becomes 400, other errors 500. It gives the frontend a quick preview before a longer analysis call. It does not enforce the ten-minute limit itself; analysis does that later.

### youtube_comments(), lines 302–363

Requires URL. Optional max_comments is converted with int then clamped to at least one. This is not a strict whole-number validator: some numeric inputs can be truncated, and booleans are integer-like. Fetch comments, keep only those containing any Sinhala Unicode-block character, batch-classify text, and zip original comments with ordered predictions.

Returns author, text, likes, offensive Boolean and confidence per retained comment. total_fetched is actually the number retained and analysed after filtering, not every raw retrieved comment. One Sinhala character is enough, so mixed-language comments can pass. Romanized-only Sinhala comments are excluded here even though manual text inference supports a romanized branch. Comment likes are displayed data, not a model input or weighting factor.

## 12. frontend/app.py — complete UI walkthrough

### Imports and startup, lines 1–54

Streamlit draws widgets and results. Requests calls Flask. BytesIO holds PDF bytes in memory. Path locates fonts. html.escape protects inserted text from being interpreted as markup. datetime timestamps exported reports. The json import is presently unused.

API_BASE is fixed to http://127.0.0.1:5000. set_page_config establishes title, shield icon, wide layout and collapsed sidebar. The CSS defines the dark gradient, typography, tabs, text boxes, buttons, cards, probability bars, detail rows and animations. Class names such as rc-safe/rc-danger only style the UI; colors do not change predictions. The Inter font import needs an external font request, with a fallback font if unavailable. Selectors based on Streamlit's rendered DOM may be fragile across upgrades.

`st.markdown(..., unsafe_allow_html=True)` enables custom markup rather than displaying it as text. Some external values are escaped, but title/channel, manual processed text and displayed comment author/text are inserted without consistent explicit escaping. Do not claim complete sanitization because the transcript alone is escaped. Inline event attributes in custom HTML should not be assumed to work identically across Streamlit versions/sanitization behavior.

### Streamlit execution model

Streamlit ordinarily reruns the Python script top to bottom after widget interactions. A form batches its widgets until submission. The app creates three tab containers and uses `with tab_text`, `with tab_audio`, `with tab_yt` to place elements. Those blocks are presentation containers, not separate Flask routes or separate Python processes. [Streamlit execution model](https://docs.streamlit.io/get-started/fundamentals/main-concepts).

This file does not use st.session_state to preserve analysis results across reruns and does not cache prediction requests. Results live in local variables under button/form-submit conditions. A later widget interaction may therefore cause results to disappear, rather than retrieving stored results. Backend model caching is independent of frontend reruns. The UI is not a persistent case-management system.

### cbar(v, cls), lines 57–58

**Called by:** manual text/audio cards and YouTube segment cards. Returns an HTML outer bar plus inner bar whose width is int(v*100) percent. It truncates the displayed width to an integer; text elsewhere may show one decimal place. The function assumes v is a valid 0–1 score and has no clamp. cls chooses a color class such as cbar-r or cbar-g.

### Header and text tab, lines 279–315

Hero markup and `st.tabs` create the shell. The text area supplies text_input; Analyze Text produces text_go. Only a clicked button with nonblank input performs the POST. `json={...}` sends structured JSON, not query parameters. A spinner shows the request is running, not a quantified progress estimate.

The response is decoded with .json(), then success determines the branch. Display uses is_offensive, confidence and probabilities. Processing details show script_type and the first 100 processed characters; the actual model input is not truncated at 100 characters by this UI operation. Model input truncation is 128 tokenizer positions in the backend. ConnectionError gets a specific startup hint; other exceptions use a generic error. The text tab does not call raise_for_status before JSON parsing.

### Audio tab, lines 317–356

file_uploader permits the listed extensions; st.audio provides playback. The upload object is a file-like buffer. `audio_file.seek(0)` rewinds it so earlier playback/read operations do not leave the POST starting at the end. `files={"audio": (...)}` creates multipart upload matching request.files in Flask.

The MIME value is always audio/wav even for other accepted formats; it is a declared label rather than a conversion operation. The backend decoder actually reads the content. Two columns show independent hate and authenticity results, plus separate probabilities. The “Max 10s analysed” note is important. Model info hard-codes Wav2Vec2 for deepfake, while the deployed checkpoint is WavLM.

### YouTube form and preview, lines 359–411

st.form keeps URL typing and the optional comment checkbox together. clear_on_submit=False preserves widget input. On submit, initialize result/status variables and placeholders. st.empty makes a replaceable UI slot. st.status supplies a progress container. The metadata request uses 60-second timeout; non-200 or exceptions show errors. Successful metadata displays thumbnail/title/channel/duration. `divmod(duration,60)` returns minutes and seconds together.

If metadata_ready is false, st.stop terminates the current script execution, not the Flask process. It prevents later analysis from running with missing metadata. Success starts a second request to /api/youtube/analyze with a 600-second timeout. This is a synchronous request; progress messages are phase-level, not live backend segment updates.

### Summary and render_comment_card(), lines 433–481

If video_success, display a review warning where necessary and three cards: fused hate, authenticity, comments. The nested render_comment_card function reads comment_card, comments_success, comments_finished, analyze_comments and summ_c from the surrounding module execution context. It is called once before optional comment retrieval and again after completion, updating the same placeholder.

Its success branch shows offensive count/total and percentage. Otherwise it shows Processing or N/A. “Processing” is a UI state before the later synchronous comments request, not evidence that another thread is fetching comments simultaneously. The hate summary card can be green when there are zero flagged segments even if some segments need review; read it with the warning and detailed statuses.

### Visual timeline, lines 485–530

Each segment's width is its actual numeric duration divided by video metadata duration. The timeline widths are duration-weighted, unlike summary percentages based on segment counts. Color priority: no speech, uncertainty, both offensive+fake, offensive, fake, otherwise green. A segment with uncertain hate and fake authenticity can therefore appear purple; both facts should be read in the detail panel.

The “SAFE”/“Clean” wording is a model-result display label, not a guarantee of safety or truth. There is a tooltip per interval, but no timestamp-linked playback/navigation is implemented by these divs. Rounding durations can produce tiny visual gaps or totals slightly different from 100%.

### Segment expanders and transcript panel, lines 535–579

Each segment builds an issue list and expands automatically if flagged/uncertain. It extracts stored explanation dictionaries. NEEDS_REVIEW shows a warning and reason; a non-None ASR score is shown to two decimals with an explicit statement that it is not transcription accuracy. Candidate text is escaped before HTML insertion. The panel separates audio_hate_label, transcript label and fused label and includes fusion_reason.

If the transcript is empty, it distinguishes unavailable/rejected text from the no-speech case. LOW_QUALITY_OUTPUT is supported as a legacy display value; new code returns NEEDS_REVIEW. Labels can legitimately differ because different paths assess different evidence.

### Hate and authenticity detail cards, lines 581–681

Cards show confidence, probability bars, rule-based factors and summaries. The hate card selects its headline using the fused hate_label but shows hate_confidence/probabilities/explanation from the audio classifier. Therefore it can display a fused OFF headline alongside an audio NOT explanation. The API already exposes the separate labels; the presentation should be described as an area needing clearer separation, not as a calibrated fused-confidence display.

Another mismatch: fused no-speech is NOT_ASSESSED, but the hate-card branch checks NO_SPEECH. It can therefore display “Not Offensive” inside a no-speech segment even though the transcript/fusion panel correctly says NOT_ASSESSED. The timeline still notices authenticity_label=NO_SPEECH. The deepfake UNCERTAIN display branch is currently defensive/legacy because speech authenticity classification is binary.

### Signal features, comments and downloads, lines 683–775

The feature strip shows energy, RMS, silence ratio, speech heuristic and ZCR supplied by the backend. It does not calculate new acoustic explanations in the browser. Optional comments are requested after audio rendering, with timeout=1800. Comment failure warns without discarding already rendered audio results. Only offensive comments are listed, though summary counts include all analysed Sinhala-filtered comments.

The report filename retains alphanumeric characters, spaces, dashes and underscores, replaces spaces with underscores, truncates the title to 50 characters, and appends a timestamp. This sanitizes a download filename; it does not sanitize every HTML value. The active download button calls build_youtube_html_report and serves text/html. No current button calls the PDF builder. Footer/model branding is static text rather than a live reflection of loaded checkpoint metadata.

## 13. Report functions and their nested helpers

### build_youtube_pdf_report(...), frontend lines 61–215

**Current caller:** none in the active frontend download flow. It remains defined and could be called programmatically. It returns PDF bytes from BytesIO, using ReportLab's SimpleDocTemplate, Paragraph, Table, Spacer and HRFlowable. A story is a list of flowable layout objects; doc.build lays them out into pages. ParagraphStyle sets font/spacing, TableStyle sets borders/fills/padding, repeatRows repeats table headers.

It imports ReportLab inside the function so failure becomes a specific RuntimeError only if called. It attempts to register Nirmala UI fonts from Windows, otherwise falls back to Helvetica. A fallback font does not guarantee Sinhala glyph coverage or correct shaping; real Sinhala output requires visual verification. `datetime.now()` uses host local time without an explicit project timezone.

Inputs: video metadata; summary; segment dictionaries; comments; optional comments summary; source string; original URL. It builds video information, metric cards, timeline rows, optional flagged-comment tables, method notes and page footers. It uses the existing prediction result; it does not rerun a model.

Nested functions:

| Function | Called by | Purpose and detail |
| --- | --- | --- |
| text(value), 109–110 | PDF metadata, cards, comments and explanatory text | Converts None to Not available, converts to string, XML-escapes for ReportLab Paragraph markup |
| section(name), 112–113 | Repeated PDF section starts | Returns a heading Paragraph and horizontal-rule flowable |
| metric_card(label,value,accent), 115–121 | Four overview cards | Returns a one-cell styled table; fixed height can be sensitive to long content |
| page_decor(canvas,document), 123–131 | doc.build onFirstPage/onLaterPages callbacks | Draws footer line, title and page number, preserving canvas state with saveState/restoreState |

Inside this function, importing xml.sax.saxutils.escape shadows the global html.escape name locally. Both are escaping utilities but serve different markup contexts. KeepTogether is imported but not used. Some color variables are also unused; unused imports/variables are not active functionality.

### build_youtube_html_report(...), lines 218–277

**Called by:** the active download block near the end of the YouTube tab. Returns a full standalone HTML string with UTF-8 declaration, responsive CSS and print rules. It builds timeline rows, metadata, summary cards and optionally flagged-comment rows. Nested `value(item)` converts None and applies html.escape. `''.join(timeline)` concatenates row strings without separators. Doubled braces in f-strings represent literal CSS braces.

A standalone HTML download is portable and can be opened offline; fonts use local fallbacks. It is a static snapshot, not an interactive API-connected report. The page contains predictions already computed, not saved model weights or a database record.

### Report limitation examiners may notice

Both builders reconstruct hate_label as OFF if hate_speech else NOT, and authenticity as FAKE if not audio_authentic else REAL. This collapses UNCERTAIN/NOT_ASSESSED/NO_SPEECH into apparently decisive labels. They do not include transcript text, ASR review scores or the new review count in the timeline. Their hate confidence is the audio classifier score even if the label resulted from fusion.

**Defensible answer:** “The detailed UI/API preserve uncertainty, but the older export format still reduces it to Booleans. I would update export columns to use the explicit label fields and include transcript review information before treating the report as a complete audit record.” This is a verified limitation, not a feature added in this review.

## 14. Tests, documentation and configuration

### backend/test_asr.py — class and helper map

Tokenizer is a tiny fake tokenizer, not the real Sinhala vocabulary. Tokenizer.decode asserts group_tokens=False and maps IDs 1,2,3 to a few Sinhala characters. RecognitionTests inherits unittest.TestCase. Its decode helper creates synthetic four-class logits and invokes the actual decode_segment. Its request helper patches heavy dependencies, creates a Flask test client, posts a fake URL and returns result JSON plus mock objects so tests can inspect calls.

patch.object replaces an attribute temporarily inside a context manager and restores it afterward. Patch where the caller looks up the name: routes use imported functions on the app module, so patching api.predict_text is appropriate. return_value simulates success; side_effect simulates an exception. SimpleNamespace creates an object with attributes such as tokenizer. A lambda is a small anonymous function; here it returns no speech timestamps.

| Test method | What it verifies | What it does not prove |
| --- | --- | --- |
| test_context_is_excluded_and_repeated_letters_survive | CTC context crop and repeated letters across blanks | Natural Sinhala transcription accuracy |
| test_boundary_run_is_assigned_once | One run belongs to one adjacent interval in the same logits | Perfect consistency across separately inferred windows |
| test_low_scores_and_blank_only_need_review | Weak predictions/empty decoding require review | Empirically optimal score thresholds |
| test_blank_frames_do_not_inflate_score | Strong blank frames do not hide a weak emitted token | Probability calibration |
| test_silence_and_nonfinite_input_do_not_load_model | Empty, zero and NaN input avoid a model load | All corrupt media handling |
| test_context_and_original_deepfake_audio_are_separate | 19-second context boundaries, classification padding, original waveform preservation | VAD accuracy; VAD is mocked |
| test_uncertain_transcript_does_not_classify_or_count_clean | No text classification for review; uncertain fusion and count | Real-world false-positive rate |
| test_accepted_transcript_still_contributes | Accepted transcript can produce fused OFF | Correctness of the mocked classifier |
| test_asr_failure_keeps_audio_result | ASR failure does not erase independent audio OFF | Recovery from every backend failure |
| test_no_speech_skips_asr | No-speech skips transcription and text prediction | That neural audio inference itself was skipped |

`unittest.main()` enables direct execution; discovery finds methods beginning test_. assertEqual checks values, assert_not_called verifies a dependency was not invoked, and np.testing.assert_array_equal compares waveform samples exactly. A deliberately simulated ASR failure may print a traceback while the test still passes—that is expected logging of the injected exception.

The earlier work recorded ten tests passing and one real-video end-to-end request succeeding. A fresh test result for this guide is recorded in the companion validation notes. Functional tests establish control-flow correctness within the cases tested, not model precision, fairness, or robustness.

### ASR_NOTES.md

This file documents overlap, frame assignment, heuristic thresholds, review behavior, the test command and the need for held-out manual reference transcripts. It is not loaded by Python and cannot change runtime behavior. If you edit a threshold in documentation without changing asr_manager, the program will not change.

### Requirements and environments

Backend requirements list Flask, flask-cors, Transformers, torch, safetensors, NumPy, librosa, soundfile and yt-dlp. Frontend requirements list Streamlit, Requests and ReportLab. Requests is also imported by youtube_utils but is not explicitly listed in the backend requirements: the current environment may obtain it transitively, but it should be declared for reproducibility. FFmpeg is an external executable, not installed by those pip requirements.

The version specifications are lower bounds (`>=`), not a lockfile. A fresh install can resolve different versions. The local saved model configs record Transformers 5.0.0 as serialization metadata, not necessarily the currently installed version. Exact runtime versions should be recorded separately before reproducing experiments.

### frontend/.streamlit/config.toml

Dark theme, primary/background/text colors and sans-serif font establish defaults. headless=true suppresses automatic browser launching; port=8501 selects the frontend port. CSS in app.py can override or supplement theme settings. Start Streamlit from the frontend directory so this project-local .streamlit folder is discovered as intended.

## 15. Most important limitations to know before the viva

| Observation in current code | Accurate explanation / improvement direction |
| --- | --- |
| Health/manual UI says Wav2Vec2 deepfake; checkpoint says WavLM | Trust configuration and actual loaded class; synchronize display labels |
| Upload and YouTube use different preprocessing and decision rules | Explain each path; do not promise identical predictions |
| Masking non-speech is described like noise removal | It does not remove overlapping noise during speech |
| Silence heuristic examines padded normalized samples | Pauses, padding and short speech can lead to NO_SPEECH; evaluate separately |
| Model forwards run before no-speech override | It skips usable output/ASR, not all neural computation |
| Explanation templates mention tone/artifacts | They are not direct feature attribution or causal explanations |
| Fused label overwrites audio label but not confidence | No fused probability exists; retain/display separate evidence |
| Export collapses review/no-speech statuses into NOT/REAL | Older report schema needs explicit multi-state labels |
| ASR thresholds are initial heuristics | Calibrate on development references; evaluate held-out data |
| Overlap uses independently inferred windows | Boundary mistakes and split words remain possible |
| Script heuristic calls low-Sinhala input romanized | It cannot distinguish English from romanized Sinhala reliably |
| Comments filter accepts any Sinhala-block character | It is not full language identification; Romanized-only comments are excluded |
| Temporary uploads use extension checks only | Content/size validation and explicit decode errors are future hardening |
| Classifier/VAD lazy loaders have no locks | Concurrent first requests can duplicate loading |
| CORS is permissive; server binds all interfaces | Prototype access configuration, not access control |
| No session-state result persistence | UI reruns may lose displayed results |
| No background job queue or backend progress stream | Long CPU requests are synchronous and can time out |
| Dependencies/Torch Hub revision are not pinned | Reproducibility needs versions, hashes and model lineage |
| Text cleaning removes hashtag content and numbers | It may remove meaningful context; evaluate cleaning ablations |
| Boolean false is used for unassessed hate | Always interpret alongside the explicit label |

This table is preparation for honest technical discussion. It is not a claim that every limitation has already caused a measured failure, and no fixes were applied as part of writing this guide.

## 16. “Point at this line and explain it” practice

### A. Decorators and JSON

```python
@app.route("/api/text/predict", methods=["POST"])
def text_predict():
    data = request.get_json(silent=True)
```

The decorator associates a URL and method with a Python handler. Flask invokes it after a matching HTTP request; the frontend does not import and call text_predict directly. silent=True concerns JSON parsing errors, not logging or model confidence. HTTP POST carries user content for computation; GET health requests simply inspect service status.

```python
return jsonify({"success": True, "result": result})
```

jsonify serializes supported Python values and produces an HTTP JSON response. Arrays must be removed or converted beforehand. The frontend's resp.json() reverses this transport step into a Python dictionary. HTTP success, application success and model classification correctness are three different ideas.

### B. Dictionaries, mutation and precedence

```python
asr_context = seg.pop("asr_context", asr_audio)
results.append({**seg, **pred, **fusion, "transcript": transcript})
```

pop removes a field from seg and returns it, falling back if absent. The fallback supports older segment dictionaries. Double-star unpacks mappings. If pred and fusion both contain hate_label, fusion wins because it comes later. append adds one result to the list; it does not send an incremental HTTP response.

```python
transcript_result = seg.get("transcript_analysis") or {}
```

get may return None; `or {}` replaces a falsey value with an empty mapping so later .get calls are safe. This does not fabricate a prediction. In contrast, `if value is not None` preserves valid zero values, which matters for showing ASR score 0.0.

### C. Arrays, slicing and copies

```python
end = min(start + SEGMENT_SAMPLES, total_samples)
asr_seg = asr_audio[start:end].copy()
seg = np.pad(seg, (0, SEGMENT_SAMPLES - len(seg)), mode="constant")
```

Python slices include start and exclude end. min avoids stepping beyond available samples. NumPy slicing can produce a view; copy makes an independent array so subsequent modifications do not alter the original. The pad tuple adds zero samples to the left and the required number to the right. The length guard prevents negative padding.

```python
mask[ts['start']:ts['end']] = True
audio = audio * mask
```

A Boolean mask multiplies speech by one and non-speech by zero. Array positions stay in place, preserving timestamp correspondence. Removing samples with audio[mask] would instead concatenate speech and shift timing; that is not what this implementation does.

### D. Model execution and numeric conversions

```python
with torch.no_grad():
    logits = h_model(input_values=inp["input_values"].to(DEVICE)).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
```

with scopes gradient disabling. `.to(DEVICE)` selects CPU here. Calling the model invokes its forward pass. `.logits` selects raw classifier scores from the returned object. Softmax normalizes the class dimension; `.cpu()` ensures host memory before NumPy; `[0]` selects the first batch item. int removes a NumPy scalar type for ordinary Python/JSON use. There is no backward pass or optimizer step.

```python
model.eval()
```

This does not evaluate test-set accuracy. It selects inference behavior for the model's layers. Accuracy evaluation requires labelled examples and metric computation elsewhere.

### E. CTC runs

```python
while end < len(ids) and ids[end] == ids[start]:
    end += 1
midpoint = (start + end - 1) / 2
```

The inner loop extends one identical-token run. The subtraction of one reflects that end is exclusive, so the last frame is end-1. Midpoints can be fractional. Then frame_start <= midpoint < frame_end determines membership in the central interval.

```python
token_scores.append(max(scores[start:end]))
```

It uses the most confident frame of each run. This makes the score less dominated by uncertain edges of a token but can make it optimistic. It is a heuristic design choice to validate, not a mathematically calibrated transcript correctness estimate.

### F. Locks and global state

```python
with _model_lock:
    if _processor is None or _model is None:
        ...
```

Only one thread enters the protected initialization at a time. Checking again after acquiring the lock avoids redundant loading if another thread completed it while this one waited. global declarations let the function assign module-level model references. This lock is in-process, not a distributed lock across multiple servers.

### G. Exception handling and resource cleanup

```python
try:
    audio_file.save(temp_path)
    result = predict_audio(temp_path)
finally:
    ...
```

The finally block runs when the try block exits normally or through an exception, including when a return was prepared. It is for cleanup rather than deciding whether the prediction was correct. `except OSError: pass` tolerates failed deletion, at the cost of hiding a cleanup problem.

`raise RuntimeError(...) from exc` keeps the original exception as the cause. `app.logger.exception` records a traceback from the active exception. Returning raw exception text to clients is convenient in a prototype but can reveal implementation details.

### H. File uploads and file buffers

```python
audio_file.seek(0)
files={"audio": (audio_file.name, audio_file, "audio/wav")}
```

seek rewinds a stream. The tuple supplies the multipart filename, readable content object and MIME declaration. The field name must match the backend's request.files lookup. MIME text does not transcode the audio.

### I. Templates and escaping

```python
transcript_text = escape(transcript)
```

Escaping represents markup characters as text so user-derived content cannot become unintended HTML structure. This is unrelated to cleaning text for ML; escape belongs at the output boundary, while preprocess belongs before classification. A single escaped field does not secure other unescaped interpolations.

```python
f"{confidence:.1%}"
f"{minutes:02d}:{seconds:02d}"
```

The first formats 0.853 as 85.3%; the second prints two-digit integers. Formatting changes presentation, not underlying predictions. Raw CSS braces inside an f-string are doubled as {{ and }}.

### J. Python shorthand you will see throughout

| Syntax | Meaning in this project |
| --- | --- |
| `def name(...):` | Define a reusable function; its body runs when called |
| `-> dict`, `audio: np.ndarray` | Type hints; they do not enforce validation automatically |
| `dict \| None` | Type annotation allowing a dictionary or None; requires compatible Python syntax |
| `_helper` | Convention for internal function; not access enforcement |
| `**inputs` in a call | Expand dictionary entries into keyword arguments |
| `**mapping` in a dict | Merge key/value pairs, with later duplicates overriding earlier ones |
| `list[str]` | Type hint for a list of strings |
| `[r for r in results if ...]` | List comprehension: filter and build a new list |
| `sum(condition for r in results)` | Count True conditions |
| `enumerate(segs)` | Obtain zero-based index and value together |
| `zip(comments,predictions)` | Pair corresponding items; stops at the shorter input |
| `value if condition else other` | Conditional expression, often for labels/colors |
| `None` | Explicit absence; different from empty string, zero or False |
| `with ...` | Enter/exit a context for a resource, lock, inference mode or UI container |
| `return a,b` | Return a tuple that can be unpacked by the caller |
| `os.environ.get(...)` | Read configuration from the process environment |
| `Path(__file__)` | Locate this source file, not the current command directory |
| `del variable` | Remove a reference; not guaranteed complete deallocation |
| `if __name__ == '__main__'` | Execute startup only when run as a script |

## 17. Training versus inference: evidence you can defend

The application files contain inference, not training loops. There are no optimizer updates, loss.backward calls or Trainer.train calls in the backend request paths. Config/weight loading reuses learned parameters. Training history, dataset provenance and reported metrics must be linked to the exact exported checkpoint.

### Text notebook evidence inspected

`notebooks/final text module.ipynb` is a JSON notebook. Inspected cells load `sinhala-nlp/SOLD`, map NOT to 0 and OFF to 1, use xlm-roberta-base, and define a SinhalaDataset with max_len=128 and both input IDs and attention masks. The shown run_experiment uses three epochs, training batch size 16, evaluation batch size 32, warmup_steps=100, weight_decay=0.01, epoch evaluation, save_strategy='no', load_best_model_at_end=False, and fp16=True. Therefore its shown final evaluation uses last-epoch weights, not automatically selected best-epoch weights.

A metrics function computes accuracy, macro-F1 and ROC-AUC. Some romanized comparisons reuse the same split indices for raw and transformed inputs, an important fairness measure. This guide does not certify the entire notebook's experiment lineage or copy historical output scores as the deployed model's performance. The snippet does not explicitly set learning_rate in TrainingArguments; do not invent a custom learning rate for it without verifying the exact runtime/default or other experiment records.

### Audio training artifact inspected

`notebooks/final audio module.ipynb` is actually plain Python text despite its .ipynb filename. Its opening comments describe an older 60-file-per-task setup (30 per class). It uses stratified 80/20 splits with seed 42, Wav2Vec2ForSequenceClassification, frozen wav2vec2 parameters, 15 epochs, batch size 4, and learning rates 1e-4 for hate and 5e-5 for the older deepfake experiment. It uses epoch evaluation and no checkpoint-saving selection.

These are facts about that script, not proof that the deployed WavLM model was trained with those data/settings. The current WavLM checkpoint config proves architecture, not dataset size, data splits, augmentation, measured metrics or training provenance. Obtain the actual WavLM training/comparison notebook and experiment logs if your viva includes those claims. Do not quote old comments as the final dataset count.

### Questions to prepare evidence for

Where did each dataset come from? How were labels defined and verified? What licence/consent applies? How many independent speakers, recordings and source videos are in each split? Were duplicates or near-duplicates removed before splitting? Did the same speaker/source/generator appear in train and test? Were augmentation variants split after the originals were separated? Was there a validation set independent of the final test set? Did repeated test inspection influence model selection? What checkpoint hash corresponds to the results table?

Stratification preserves label proportions but does not prevent speaker/source leakage. A fixed random seed supports repeatability but is not proof of fully deterministic hardware execution. A small test set produces uncertain estimates. Frozen pretrained encoders reduce trainable parameters but do not remove the need for representative data.

## 18. Metrics and experimental design in plain language

### Classification metrics

Treat OFF (or FAKE for authenticity) as the positive class. True positive means correctly flagged; false positive means flagged but actually negative; false negative means missed positive; true negative means correctly unflagged. State which class is positive before giving formulas.

| Metric | Formula / idea | What it helps explain |
| --- | --- | --- |
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness; can hide class imbalance |
| Precision | TP/(TP+FP) | Of flagged cases, how many are truly positive? |
| Recall | TP/(TP+FN) | Of real positives, how many were detected? |
| F1 | 2×precision×recall/(precision+recall) | Balance between precision and recall |
| Macro-F1 | Average of per-class F1 values | Gives both classes equal weight |
| Confusion matrix | Counts by true and predicted class | Reveals types and direction of errors |
| ROC-AUC | Ranking performance across score thresholds | Requires continuous positive-class scores, not only argmax labels |
| Review rate | Reviewed / all eligible samples | Cost and coverage of abstention |

Zero denominators need explicit handling. A high softmax confidence on one example is not dataset accuracy. The video's hate_percentage is a flagged-segment rate, not model precision. If a system abstains, report coverage/review rate along with performance on accepted samples; otherwise rejecting difficult cases can create misleading accuracy improvements.

### Transcription metrics

WER = (substitutions + deletions + insertions) / reference word count. CER uses the same edit operations at character level. Both need human reference transcripts and a declared normalization/tokenization policy. Sinhala spacing and combining-character handling can affect counts. ASR score 0.93 is not “93% WER accuracy.” A transcript can sound plausible and still replace a critical word that changes the hate decision.

### An evaluation plan for the new ASR changes

Build a fixed set of clear, slow, quiet, noisy and music-backed Sinhala clips with independently checked reference text. Run old non-overlap and new overlap pipelines on the same audio/model. Compare WER/CER, latency, review rate and downstream text/fused errors. Evaluate the speech-presence gate separately so missed speech is not hidden by evaluating only accepted clips. Tune thresholds on a development set, then report untouched test results. Include cases around segment boundaries and rare/offensive words rather than only easy news clips.

Ablations can compare audio-only, transcript-only, OR fusion, overlap versus none, screening versus none, and VAD mask versus original audio. Keep splits and compute conditions consistent. A successful smoke test proves the workflow executes, not that the proposed enhancement improves these metrics.

## 19. Mock viva: concise answers plus follow-up probes

### Architecture and scope

**Q1. What exactly makes the project multimodal?** Text and audio are processed through different learned encoders. For YouTube, audio and derived-text decisions are combined. It is a modular decision-fusion system, not one jointly trained cross-modal network. **Follow-up:** show _fuse_youtube_hate_predictions.

**Q2. Why Flask and Streamlit?** Flask provides clear reusable HTTP inference endpoints; Streamlit lets the prototype expose inputs and results quickly in Python. The separation makes backend logic testable without the browser. **Tradeoff:** the current UI is synchronous and not designed for high-concurrency deployment.

**Q3. Where is the database?** There is no database in this component. Results are returned to the UI and can be exported; temporary audio files are cleaned up. **Follow-up:** explain why no persistent history is available.

**Q4. Is the YouTube thumbnail analysed?** No. It is displayed as metadata; no vision model processes video frames. Authenticity refers to audio.

**Q5. Why can the first request be slower?** Model files are read and model objects initialized lazily on first use; later requests reuse process caches. A restart clears those caches. Network and CPU time also contribute.

**Q6. Does health=ok prove all models work?** No. It only proves that the health handler responds. Use representative inference and dependency checks for deeper readiness.

### Models and preprocessing

**Q7. Why XLM-RoBERTa?** It provides multilingual pretrained text representations and can be fine-tuned for the Sinhala offensive-language label task. This is a suitability rationale; superiority over alternatives requires comparison results. **Follow-up:** show the local config and 128-token inference call.

**Q8. Which deepfake model is running?** The local config identifies WavLMForSequenceClassification. AutoModelForAudioClassification reads that configuration. Some display strings still say Wav2Vec2 and should be corrected separately.

**Q9. Does the dictionary make the final decision?** No. It normalizes selected romanized words before the learned text classifier. The model still decides OFF/NOT from tokenized context.

**Q10. Does is_romanized recognise Sinhala language?** No. It measures the fraction of Sinhala Unicode-block characters. It can call English romanized, so it is a script heuristic.

**Q11. Why normalize text the same way as training?** To reduce train–inference mismatch. Changing preprocessing can change token distributions and invalidate comparisons. The exact checkpoint's training pipeline must be verified, not assumed solely from a comment.

**Q12. Why 16 kHz?** It matches the saved processors and the project's inference convention, providing consistent timing and input representation. Setting a rate parameter does not make any arbitrary waveform correctly resampled; librosa performs that conversion here.

**Q13. Why ten seconds?** It bounds CPU/memory work and matches the application's fixed-length classifier preparation. It may truncate meaning in uploads; YouTube is segmented to cover its audio. Optimality requires duration experiments.

**Q14. Can this classify a silent uploaded file?** The upload route lacks the YouTube no-speech gate and still produces binary model predictions. That is a limitation; a positive model score does not establish meaningful speech.

**Q15. Why preserve original YouTube audio for deepfake detection?** Masking and peak normalization can alter waveform cues relevant to authenticity. The deepfake branch therefore receives the original resampled interval, while the hate branch receives the speech mask. This rationale needs evaluation on representative deepfakes; it is not an automatic accuracy guarantee.

### Transcription and uncertainty

**Q16. How is ASR different from audio hate classification?** ASR outputs a character/token sequence over time. Audio hate classification returns two clip-level logits. They use different trained heads/checkpoints despite both being in the Wav2Vec2 family.

**Q17. Why overlap?** The model sees some neighbouring sound instead of encountering every boundary with no context. Only core token runs are retained for that segment. It increases computation and does not completely solve boundary words.

**Q18. How do you map a token to a segment?** Use the model's convolution stride and receptive-field centre to map sample offsets to frames; assign each nonblank identical-token run by its midpoint.

**Q19. Why remove blanks from scoring?** Many confident blanks can dominate a mean and hide uncertain actual characters. We score emitted token runs instead.

**Q20. Why 0.65 and 25%?** They are initial heuristic screening settings, not empirically optimal thresholds. The next evaluation should tune them on a labelled development set and assess a held-out set.

**Q21. Can a high-score transcript be wrong?** Yes. Model confidence and correctness differ, especially with noise, domain shift and unfamiliar vocabulary. The Sinhala-character check cannot validate meaning.

**Q22. Are you using Whisper?** No. The default runtime ASR is a Sinhala fine-tuned XLS-R CTC checkpoint. Historical comments or cached alternative models do not change what MODEL_ID loads.

**Q23. What if transcription fails?** The current route logs the exception, marks the segment for review, skips text classification and retains independent audio results. It does not silently label an absent transcript safe.

**Q24. Can slow speech still fail?** Yes. Long pauses or a short final utterance can trigger the upstream silence-ratio gate, preventing ASR entirely. Overlap helps a different problem—context near boundaries.

### Fusion and outputs

**Q25. How do you combine audio and text?** A rule sets OFF if either the audio or an accepted transcript is OFF. It handles no-speech and uncertainty explicitly. This is decision-level OR fusion, not averaging calibrated probabilities.

**Q26. If audio says OFF and transcript is uncertain?** Keep OFF from independent audio evidence, while marking the transcript for review.

**Q27. If audio is UNCERTAIN and accepted text says NOT?** The implemented rule returns NOT. This allows accepted text to resolve that audio uncertainty; its tradeoff should be evaluated.

**Q28. Is displayed hate confidence a fused score?** No. After fusion the API keeps the original audio score. The separate transcript result has its own text score. The UI needs careful reading and clearer labeling.

**Q29. Why distinguish NOT, UNCERTAIN and NOT_ASSESSED?** NOT is a usable negative decision; UNCERTAIN means insufficient reliability; NOT_ASSESSED means no suitable speech assessment occurred. Collapsing these into false loses meaning.

**Q30. Does REAL mean the recording is certainly genuine?** No. It is the classifier's label within its learned distribution. It is not forensic proof or a cryptographic authenticity guarantee.

**Q31. Are explanations based on attention or SHAP?** No. They are templates selected from labels, score margins and simple audio statistics. They should not be described as causally identifying model evidence.

**Q32. Does 50% hate mean half the video duration?** Not necessarily. It means half the segments were flagged; the last short segment has equal count weight. Timeline widths, in contrast, use duration.

### Comments, UI and reports

**Q33. Are all comments analysed?** Only successfully retrieved comments passing the Sinhala Unicode filter. The official API retrieves top-level comments; unavailable/private/deleted content is not accessible. No cap is sent by the current UI, but completeness is not guaranteed.

**Q34. Why exclude romanized comments when the text tab accepts them?** That is the chosen comments scope in this implementation, enforced by the Unicode-character filter. It is a product scope difference, not a limitation of the manual endpoint's input type.

**Q35. What happens without an API key?** Comment retrieval uses yt-dlp fallback. A key enables the official Data API attempt; it is read from the environment, not embedded in code.

**Q36. Are comments processed in parallel with audio?** No. The frontend displays completed audio results, then makes the optional comments request. Batch inference processes several texts per model call but is not asynchronous fetching.

**Q37. Where is the PDF button?** The PDF builder is defined but not called by the current UI. The active download is HTML. A defined function is not evidence of an exposed feature.

**Q38. Why can the report disagree with the detailed panel?** Its older timeline derives labels from Booleans and loses uncertainty/no-speech states. Explicit labels and review fields should be included in the export.

### Testing, evaluation and research

**Q39. What do the ten tests establish?** Specific CTC decoding, context separation, review gating and API failure-handling behavior using synthetic data and mocks. They do not measure model accuracy on real Sinhala speech.

**Q40. How will you prove overlap improved accuracy?** Compare baseline and overlap on the same manually transcribed held-out clips, reporting WER/CER, downstream classification results and latency. A successful demo alone is insufficient.

**Q41. Why macro-F1 instead of accuracy alone?** It gives equal weight to per-class F1 and makes minority-class failures less easy to hide. Report the confusion matrix and class supports as well.

**Q42. How do you prevent leakage?** Split by relevant sources/speakers/recordings before augmentation and duplicate handling, and keep final evaluation separate from tuning. The shown stratified split alone does not establish those guarantees.

**Q43. Did you train the deployed WavLM with the shown older audio script?** The inspected older script uses Wav2Vec2. I must use the actual WavLM training records to substantiate that checkpoint's data, training and metrics. Do not fill missing evidence with guesses.

**Q44. What would you improve next?** Validate/calibrate transcript screening, improve speech-presence handling, align report labels with the API, document checkpoint provenance, pin dependencies, and add robust input validation and a job queue for longer analyses. Separate proposed work from completed work.

## 20. A live demonstration and debugging rehearsal

### Starting the application

From the project root, use two terminals. Backend:

```powershell
cd backend
..\.venv\Scripts\python.exe -X utf8 app.py
```

Frontend, from the project root in the other terminal:

```powershell
cd frontend
.\venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. UTF-8 mode helps terminal output contain Sinhala without the earlier Windows code-page encoding error. It does not change the recognizer's linguistic accuracy.

### Minimal health and regression checks

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
Invoke-RestMethod http://127.0.0.1:8501/_stcore/health
```

From project root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s backend -p test_asr.py
```

Do not stop an existing server simply because you open another terminal. If restarting, first identify and stop your existing instance so two processes do not compete for the same port. Network access is necessary for YouTube; cached model weights alone do not make YouTube extraction offline.

### A five-minute demonstration

1. Show the three tabs and the architecture in one sentence.
2. Submit a prepared Unicode sentence and a romanized sentence. Point to processed_text and clarify that the dictionary is not the classifier.
3. Upload a short known clip and explain the separate OFF/NOT and REAL/FAKE questions. State the first-ten-seconds limitation.
4. Analyse a short accessible YouTube clip. Show preview, time segments, separate audio/transcript labels, recognition score and fusion reason.
5. If comments are enabled, explain the Unicode filter and retrieval source. Open the HTML export and acknowledge its current uncertainty limitation if relevant.
6. Show test output and clearly distinguish functional verification from evaluation metrics. Keep a previously saved result for discussion if network availability changes, and label it as a saved result.

### Diagnose by layer

| Symptom | First layer to inspect | Useful explanation |
| --- | --- | --- |
| Frontend cannot connect | Backend process and port 5000 | ConnectionError occurs before model inference |
| Address already in use | Existing process on the port | Do not keep launching duplicate servers |
| YouTube WinError 10013 | Process network restrictions | Socket permission failure, not necessarily invalid URL |
| DNS resolution error | Internet/DNS | The hostname could not be resolved |
| Metadata succeeds, download fails | yt-dlp extraction/FFmpeg/network | Separate metadata and media operations |
| Model folder error | Path, config, processor and weights | All compatible checkpoint files are needed |
| First request is slow | Lazy loading, disk, RAM, CPU | Subsequent requests can reuse model objects |
| Transcript missing | Audio NO_SPEECH gate, ASR status/reason | Do not assume the ASR model was called |
| Transcript seems wrong but score high | ASR calibration/domain mismatch | Confidence does not certify correctness |
| Fused label conflicts with explanation | Audio fields versus fused fields | Explain dictionary overwrite and retained audio confidence |
| Report shows NOT for uncertain segment | Boolean-based export rendering | Known presentation limitation |
| Result vanishes after interaction | Streamlit rerun/state | Results are not persisted in session_state |

## 21. Revision checklist and glossary

Before the viva, be able to locate every endpoint, trace one text request and one YouTube segment, name the deployed checkpoint architectures, explain why uploaded and YouTube audio differ, distinguish all labels and scores, and reproduce the CTC example without reading it. Bring the actual final training results and model provenance rather than relying on UI labels.

| Term | Meaning here |
| --- | --- |
| API / endpoint | A network-accessible operation such as POST /api/text/predict |
| Backend / frontend | Inference/orchestration service versus user-interface service |
| JSON | Text serialization format for dictionaries, lists and scalar values |
| Multipart | HTTP body format used to transmit an uploaded file |
| Checkpoint | Saved model weights and configuration from a training state |
| Tokenizer | Converts text into model vocabulary IDs and back where supported |
| Feature extractor / processor | Prepares audio arrays and/or packages tokenizer functionality |
| Logit | Unnormalized model score before softmax |
| Softmax | Normalization across mutually exclusive output choices |
| Argmax | Index of the largest value |
| Embedding | Learned numeric representation of a token or signal feature |
| Attention | Model mechanism relating sequence positions; not used as an explanation here |
| Fine-tuning | Updating pretrained weights/head on task data; not done by a prediction request |
| Frozen parameters | Parameters not updated by gradient optimization during training |
| Dropout | Training-time regularization disabled in evaluation mode |
| Padding / truncation | Filling to a target length / discarding excess input |
| Sampling rate | Number of audio samples per second |
| Mono | One channel of audio |
| Waveform | Audio amplitude samples over time |
| VAD | Speech-activity interval detection |
| RMS | Root-mean-square amplitude measure |
| ZCR | Approximate sign-change rate in the waveform |
| ASR | Automatic speech recognition: audio to text |
| CTC blank | Alignment symbol for emitting no token at a frame |
| Greedy decoding | Taking the highest-scoring token at each frame |
| Receptive field | Input region associated with a local feature; distinguish conv grid from attention context |
| Stride | Input displacement between successive convolution outputs |
| Fusion | Combining multiple model decisions using the stated rules |
| Abstention / review | Declining a definitive decision when evidence is insufficient |
| Calibration | Agreement between stated confidence and empirical correctness |
| Domain shift | Test inputs differ from training conditions |
| Regression test | Test that established software behavior remains correct after a change |
| Mock | Controlled replacement of a dependency for a test |
| Smoke test | Small execution check that the workflow runs |
| Ablation | Controlled comparison with a component removed or changed |
| Data leakage | Evaluation information entering training or tuning improperly |
| Provenance | Records linking data, code, training run and exported checkpoint |

## 22. References and evidence boundaries

The primary evidence for file/function behavior is the numbered source snapshot in SOURCE_ATLAS.html and the model configurations recorded there. Line numbers refer to this review's snapshot. Third-party concepts were checked against official references; project-specific limitations above come from the local code, not claims about the libraries in general.

- [Hugging Face Wav2Vec2 reference](https://huggingface.co/docs/transformers/model_doc/wav2vec2): waveform models and task heads.
- [Hugging Face WavLM reference](https://huggingface.co/docs/transformers/model_doc/wavlm): configured speech-model family.
- [Hugging Face XLM-RoBERTa reference](https://huggingface.co/docs/transformers/model_doc/xlm-roberta): multilingual text architecture.
- [PyTorch inference_mode reference](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html): gradient mode versus eval mode.
- [Streamlit execution fundamentals](https://docs.streamlit.io/get-started/fundamentals/main-concepts): reruns and Python UI execution.
- [CTC audio chunking explanation](https://huggingface.co/blog/asr-chunking): background on using overlapping context; this app uses its own token-run selection rather than that pipeline verbatim.

No new dataset accuracy, WER improvement, latency benchmark or human-reviewed transcript correctness has been established by preparing this document. The study guide helps you explain what exists, identify what requires evidence, and answer limitations precisely.

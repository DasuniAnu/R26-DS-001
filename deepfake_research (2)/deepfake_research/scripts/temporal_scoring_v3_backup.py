# ============================================================
# scripts/temporal_scoring.py
# CONDITION C v3 — FINAL TEMPORAL DEEPFAKE SCORER
#
# Model:
#   MobileNetV2 + CBAM + BiGRU
#
# Classes:
#   0 = REAL
#   1 = FAKE
#
# Training / inference parity:
#   MTCNN face detector
#   2-second temporal windows
#   1-second stride
#   15 frames per window
#   224x224 RGB faces
#   minimum 8 detected faces
#   nearest valid face filling
#   input normalization /255.0
#
# Final video outputs:
#   REAL
#   PARTIALLY_MANIPULATED
#   FAKE
# ============================================================

import os
import json
import cv2
import numpy as np
import tensorflow as tf

from mtcnn import MTCNN
from keras.saving import register_keras_serializable


# ============================================================
# CLASS MAPPING
# ============================================================

REAL_CLASS = 0
FAKE_CLASS = 1


# ============================================================
# CUSTOM CBAM LAYERS
# ============================================================

@register_keras_serializable()
class ChannelAttention(tf.keras.layers.Layer):

    def __init__(
        self,
        ratio=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):

        channels = int(
            input_shape[-1]
        )

        hidden = max(
            channels // self.ratio,
            1
        )

        self.fc1 = tf.keras.layers.Dense(
            hidden,
            activation="relu"
        )

        self.fc2 = tf.keras.layers.Dense(
            channels
        )

        super().build(
            input_shape
        )

    def call(self, x):

        avg = tf.reduce_mean(
            x,
            axis=[1, 2],
            keepdims=True
        )

        mx = tf.reduce_max(
            x,
            axis=[1, 2],
            keepdims=True
        )

        avg_out = self.fc2(
            self.fc1(avg)
        )

        max_out = self.fc2(
            self.fc1(mx)
        )

        attention = tf.nn.sigmoid(
            avg_out + max_out
        )

        return x * attention

    def get_config(self):

        config = super().get_config()

        config.update({
            "ratio": self.ratio
        })

        return config


@register_keras_serializable()
class SpatialAttention(tf.keras.layers.Layer):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.conv = tf.keras.layers.Conv2D(
            1,
            kernel_size=7,
            padding="same",
            activation="sigmoid"
        )

        super().build(
            input_shape
        )

    def call(self, x):

        avg = tf.reduce_mean(
            x,
            axis=-1,
            keepdims=True
        )

        mx = tf.reduce_max(
            x,
            axis=-1,
            keepdims=True
        )

        concat = tf.concat(
            [avg, mx],
            axis=-1
        )

        attention = self.conv(
            concat
        )

        return x * attention


# ============================================================
# MODEL CACHE
# ============================================================

MODEL_CACHE = {}


def load_model_cached(model_path):

    model_path = os.path.abspath(
        model_path
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    if model_path not in MODEL_CACHE:

        print()
        print("=" * 70)
        print("LOADING CONDITION C v3 MODEL")
        print("=" * 70)

        print(
            "Model:",
            model_path
        )

        model = tf.keras.models.load_model(
            model_path,

            custom_objects={
                "ChannelAttention":
                    ChannelAttention,

                "SpatialAttention":
                    SpatialAttention
            },

            compile=False,

            safe_mode=False
        )

        print(
            "Input shape:",
            model.input_shape
        )

        print(
            "Output shape:",
            model.output_shape
        )

        expected_input = (
            15,
            224,
            224,
            3
        )

        if (
            model.input_shape[1:]
            !=
            expected_input
        ):

            raise ValueError(
                "Unexpected model input shape.\n"
                f"Expected: (None, {expected_input})\n"
                f"Actual:   {model.input_shape}"
            )

        if (
            not model.output_shape
            or
            model.output_shape[-1] != 2
        ):

            raise ValueError(
                "Condition C v3 expects a "
                "2-class output:\n"
                "0 = REAL\n"
                "1 = FAKE\n"
                f"Actual output shape: "
                f"{model.output_shape}"
            )

        MODEL_CACHE[
            model_path
        ] = model

        print(
            "MODEL LOADED SUCCESSFULLY"
        )

    return MODEL_CACHE[
        model_path
    ]


# ============================================================
# LOAD DEPLOYMENT CONFIG
# ============================================================

def load_deployment_config(
    config_path
):

    config_path = os.path.abspath(
        config_path
    )

    if not os.path.exists(
        config_path
    ):

        raise FileNotFoundError(
            f"Deployment config not found:\n"
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    required = [
        "window_seconds",
        "window_stride_seconds",
        "frames_per_window",
        "image_size",
        "minimum_detected_faces",
        "mtcnn_confidence",
        "window_fake_threshold",
        "minimum_consecutive_fake_windows",
        "full_fake_coverage_threshold"
    ]

    missing = [
        key
        for key in required
        if key not in config
    ]

    if missing:

        raise ValueError(
            "Missing deployment config values: "
            +
            ", ".join(missing)
        )

    return config


# ============================================================
# SAFE MTCNN
# ============================================================

def safe_mtcnn_detect(
    detector,
    rgb
):

    try:

        detections = detector.detect_faces(
            rgb
        )

        if detections is None:
            return []

        return detections

    except ValueError as e:

        text = str(e).lower()

        if (
            "empty output" in text
            or
            "shape=(0" in text
        ):

            return []

        raise

    except Exception as e:

        print(
            "MTCNN warning:",
            str(e)
        )

        return []


# ============================================================
# EXTRACT LARGEST FACE
# ============================================================

def extract_largest_face(
    rgb,
    detector,
    image_size=224,
    confidence_threshold=0.80
):

    detections = safe_mtcnn_detect(
        detector,
        rgb
    )

    if not detections:

        return None

    valid = []

    for detection in detections:

        box = detection.get(
            "box"
        )

        confidence = float(
            detection.get(
                "confidence",
                0.0
            )
        )

        if box is None:
            continue

        if len(box) != 4:
            continue

        x, y, w, h = box

        if (
            w <= 0
            or
            h <= 0
        ):
            continue

        if (
            confidence
            <
            confidence_threshold
        ):
            continue

        valid.append(
            detection
        )

    if not valid:

        return None

    # --------------------------------------------------------
    # Select largest detected face
    # --------------------------------------------------------

    best = max(
        valid,
        key=lambda d:
            d["box"][2]
            *
            d["box"][3]
    )

    x, y, w, h = (
        best["box"]
    )

    frame_h, frame_w = (
        rgb.shape[:2]
    )

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    # MTCNN can return negative x/y
    x = max(
        0,
        x
    )

    y = max(
        0,
        y
    )

    x2 = min(
        frame_w,
        x + w
    )

    y2 = min(
        frame_h,
        y + h
    )

    if (
        x2 <= x
        or
        y2 <= y
    ):

        return None

    face = rgb[
        y:y2,
        x:x2
    ]

    if face.size == 0:

        return None

    face = cv2.resize(
        face,
        (
            image_size,
            image_size
        ),
        interpolation=cv2.INTER_AREA
    )

    return face.astype(
        np.uint8
    )


# ============================================================
# READ VIDEO INFORMATION
# ============================================================

def get_video_info(
    cap
):

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if (
        not fps
        or
        fps <= 0
    ):

        fps = 25.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames > 0:

        duration = (
            total_frames
            /
            fps
        )

    else:

        duration_ms = float(
            cap.get(
                cv2.CAP_PROP_POS_MSEC
            )
        )

        duration = (
            duration_ms
            /
            1000.0
            if duration_ms > 0
            else 0.0
        )

    return (
        fps,
        total_frames,
        float(duration)
    )


# ============================================================
# EXTRACT ONE 2-SECOND TEMPORAL WINDOW
# ============================================================

def extract_temporal_window(
    cap,
    detector,
    frame_cache,
    start_sec,
    end_sec,
    frames_per_window,
    image_size,
    minimum_detected_faces,
    mtcnn_confidence
):

    timestamps = np.linspace(
        float(start_sec),
        float(end_sec),
        int(frames_per_window),
        endpoint=False
    )

    detected_faces = {}

    for position, timestamp in enumerate(
        timestamps
    ):

        # Round because neighboring overlapping
        # windows reuse many identical timestamps.
        cache_key = round(
            float(timestamp),
            4
        )

        if cache_key in frame_cache:

            face = frame_cache[
                cache_key
            ]

        else:

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                float(
                    timestamp
                    *
                    1000.0
                )
            )

            ok, frame = cap.read()

            if not ok:

                face = None

            else:

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                face = extract_largest_face(
                    rgb,
                    detector,
                    image_size=
                        image_size,
                    confidence_threshold=
                        mtcnn_confidence
                )

            frame_cache[
                cache_key
            ] = face

        if face is not None:

            detected_faces[
                position
            ] = face

    detected_count = len(
        detected_faces
    )

    # --------------------------------------------------------
    # Reject if insufficient true face detections
    # --------------------------------------------------------

    if (
        detected_count
        <
        minimum_detected_faces
    ):

        return (
            None,
            detected_count
        )

    # --------------------------------------------------------
    # Fill missed positions with nearest valid face
    # This exactly follows the final training preprocessing.
    # --------------------------------------------------------

    valid_positions = sorted(
        detected_faces.keys()
    )

    sequence = []

    for position in range(
        frames_per_window
    ):

        if position in detected_faces:

            sequence.append(
                detected_faces[
                    position
                ]
            )

        else:

            nearest_position = min(
                valid_positions,
                key=lambda p:
                    abs(
                        p
                        -
                        position
                    )
            )

            sequence.append(
                detected_faces[
                    nearest_position
                ]
            )

    sequence = np.stack(
        sequence,
        axis=0
    )

    expected_shape = (
        frames_per_window,
        image_size,
        image_size,
        3
    )

    if (
        sequence.shape
        !=
        expected_shape
    ):

        return (
            None,
            detected_count
        )

    return (
        sequence.astype(
            np.uint8
        ),
        detected_count
    )


# ============================================================
# FILTER ISOLATED FALSE POSITIVES
# ============================================================

def apply_minimum_fake_run(
    window_results,
    minimum_run,
    stride_seconds
):

    if not window_results:

        return []

    # --------------------------------------------------------
    # IMPORTANT:
    # Only analysable windows are present in window_results.
    # A temporal gap therefore breaks a fake run.
    # --------------------------------------------------------

    keep = np.zeros(
        len(window_results),
        dtype=bool
    )

    current_run = []

    for i, window in enumerate(
        window_results
    ):

        is_fake = bool(
            window[
                "is_fake_raw"
            ]
        )

        if not is_fake:

            if (
                len(current_run)
                >=
                minimum_run
            ):

                keep[
                    current_run
                ] = True

            current_run = []

            continue

        if not current_run:

            current_run = [i]

            continue

        previous_index = (
            current_run[-1]
        )

        previous_start = float(
            window_results[
                previous_index
            ][
                "start_time"
            ]
        )

        current_start = float(
            window[
                "start_time"
            ]
        )

        # Consecutive windows should be
        # approximately one stride apart.
        if (
            current_start
            -
            previous_start
            <=
            stride_seconds
            +
            0.10
        ):

            current_run.append(
                i
            )

        else:

            if (
                len(current_run)
                >=
                minimum_run
            ):

                keep[
                    current_run
                ] = True

            current_run = [i]

    if (
        len(current_run)
        >=
        minimum_run
    ):

        keep[
            current_run
        ] = True

    filtered = []

    for i, window in enumerate(
        window_results
    ):

        item = dict(
            window
        )

        item[
            "is_fake"
        ] = bool(
            keep[i]
        )

        filtered.append(
            item
        )

    return filtered


# ============================================================
# MERGE FAKE WINDOWS INTO SEGMENTS
# ============================================================

def merge_fake_segments(
    window_results,
    video_duration
):

    fake_windows = [
        window
        for window in window_results
        if window.get(
            "is_fake",
            False
        )
    ]

    if not fake_windows:

        return []

    fake_windows = sorted(
        fake_windows,
        key=lambda x:
            x["start_time"]
    )

    merged = []

    current = {

        "start_time":
            float(
                fake_windows[0][
                    "start_time"
                ]
            ),

        "end_time":
            float(
                fake_windows[0][
                    "end_time"
                ]
            ),

        "probabilities": [
            float(
                fake_windows[0][
                    "fake_probability"
                ]
            )
        ]
    }

    for window in fake_windows[1:]:

        start = float(
            window[
                "start_time"
            ]
        )

        end = float(
            window[
                "end_time"
            ]
        )

        # 2-sec windows / 1-sec stride overlap.
        # If the next fake window touches/overlaps,
        # treat it as the same fake segment.
        if (
            start
            <=
            current[
                "end_time"
            ]
            +
            0.10
        ):

            current[
                "end_time"
            ] = max(
                current[
                    "end_time"
                ],
                end
            )

            current[
                "probabilities"
            ].append(
                float(
                    window[
                        "fake_probability"
                    ]
                )
            )

        else:

            merged.append(
                current
            )

            current = {

                "start_time":
                    start,

                "end_time":
                    end,

                "probabilities": [
                    float(
                        window[
                            "fake_probability"
                        ]
                    )
                ]
            }

    merged.append(
        current
    )

    final_segments = []

    for segment in merged:

        start = max(
            0.0,
            float(
                segment[
                    "start_time"
                ]
            )
        )

        end = min(
            float(
                video_duration
            ),
            float(
                segment[
                    "end_time"
                ]
            )
        )

        if end <= start:

            continue

        probabilities = (
            segment[
                "probabilities"
            ]
        )

        final_segments.append({

            "start_time":
                round(
                    start,
                    2
                ),

            "end_time":
                round(
                    end,
                    2
                ),

            "duration":
                round(
                    end
                    -
                    start,
                    2
                ),

            "confidence":
                round(
                    float(
                        np.mean(
                            probabilities
                        )
                    ),
                    4
                ),

            "peak_confidence":
                round(
                    float(
                        np.max(
                            probabilities
                        )
                    ),
                    4
                )
        })

    return final_segments


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(seconds):

    seconds = max(
        0,
        int(
            round(
                float(seconds)
            )
        )
    )

    minutes = (
        seconds // 60
    )

    secs = (
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# MAIN TEMPORAL SCORER
# ============================================================

def sliding_window_temporal_scorer(
    video_path,
    model_path,
    config_path
):

    """
    Final Condition C v3 inference.

    Returns:
        REAL
        PARTIALLY_MANIPULATED
        FAKE
        INS UFFICIENT_FACE_DATA
        ERROR

    Along with:
        fake percentage
        fake segments
        window probabilities
        analysable coverage
    """

    cap = None

    try:

        # ====================================================
        # VALIDATION
        # ====================================================

        video_path = os.path.abspath(
            video_path
        )

        if not os.path.exists(
            video_path
        ):

            return {

                "verdict":
                    "VIDEO_NOT_FOUND",

                "fake_percentage":
                    0.0,

                "fake_segments":
                    [],

                "error":
                    f"Video not found: "
                    f"{video_path}"
            }

        # ====================================================
        # CONFIG
        # ====================================================

        config = (
            load_deployment_config(
                config_path
            )
        )

        window_seconds = float(
            config[
                "window_seconds"
            ]
        )

        stride_seconds = float(
            config[
                "window_stride_seconds"
            ]
        )

        frames_per_window = int(
            config[
                "frames_per_window"
            ]
        )

        image_size = int(
            config[
                "image_size"
            ]
        )

        minimum_detected_faces = int(
            config[
                "minimum_detected_faces"
            ]
        )

        mtcnn_confidence = float(
            config[
                "mtcnn_confidence"
            ]
        )

        window_fake_threshold = float(
            config[
                "window_fake_threshold"
            ]
        )

        minimum_fake_run = int(
            config[
                "minimum_consecutive_fake_windows"
            ]
        )

        full_fake_coverage_threshold = float(
            config[
                "full_fake_coverage_threshold"
            ]
        )

        # ====================================================
        # MODEL
        # ====================================================

        model = load_model_cached(
            model_path
        )

        # ====================================================
        # MTCNN
        # ====================================================

        print()
        print("=" * 70)
        print("INITIALIZING MTCNN")
        print("=" * 70)

        detector = MTCNN()

        # ====================================================
        # OPEN VIDEO
        # ====================================================

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            return {

                "verdict":
                    "VIDEO_LOAD_FAILED",

                "fake_percentage":
                    0.0,

                "fake_segments":
                    [],

                "error":
                    "OpenCV could not open video."
            }

        (
            fps,
            total_frames,
            duration
        ) = get_video_info(
            cap
        )

        print()
        print("=" * 70)
        print("CONDITION C v3 TEMPORAL ANALYSIS")
        print("=" * 70)

        print(
            "Video:",
            video_path
        )

        print(
            f"FPS: {fps:.2f}"
        )

        print(
            "Total frames:",
            total_frames
        )

        print(
            f"Duration: {duration:.2f}s"
        )

        print(
            f"Window: {window_seconds:.1f}s"
        )

        print(
            f"Stride: {stride_seconds:.1f}s"
        )

        print(
            "Frames/window:",
            frames_per_window
        )

        print(
            "Minimum faces:",
            minimum_detected_faces
        )

        print(
            "Window fake threshold:",
            window_fake_threshold
        )

        print(
            "Minimum consecutive fake windows:",
            minimum_fake_run
        )

        print(
            "Full fake coverage threshold:",
            full_fake_coverage_threshold
        )

        # ====================================================
        # VERY SHORT VIDEO
        # ====================================================

        if (
            duration
            <
            window_seconds
        ):

            return {

                "verdict":
                    "VIDEO_TOO_SHORT",

                "fake_percentage":
                    0.0,

                "fake_segments":
                    [],

                "video_duration":
                    round(
                        duration,
                        2
                    )
            }

        # ====================================================
        # BUILD TIME-BASED WINDOWS
        # ====================================================

        window_starts = np.arange(
            0.0,
            duration
            -
            window_seconds
            +
            1e-6,
            stride_seconds
        )

        total_candidate_windows = len(
            window_starts
        )

        frame_cache = {}

        window_results = []

        rejected_windows = 0

        # ====================================================
        # SCORE WINDOWS
        # ====================================================

        for window_index, start in enumerate(
            window_starts,
            start=1
        ):

            end = (
                float(start)
                +
                window_seconds
            )

            sequence, face_count = (
                extract_temporal_window(
                    cap,
                    detector,
                    frame_cache,
                    float(start),
                    float(end),
                    frames_per_window,
                    image_size,
                    minimum_detected_faces,
                    mtcnn_confidence
                )
            )

            if sequence is None:

                rejected_windows += 1

                continue

            # ------------------------------------------------
            # EXACT TRAINING NORMALIZATION
            # ------------------------------------------------

            batch = (
                sequence.astype(
                    np.float32
                )
                /
                255.0
            )

            batch = np.expand_dims(
                batch,
                axis=0
            )

            prediction = model.predict(
                batch,
                verbose=0
            )[0]

            prediction = np.asarray(
                prediction,
                dtype=np.float32
            )

            prediction = np.nan_to_num(
                prediction,
                nan=0.0,
                posinf=1.0,
                neginf=0.0
            )

            if (
                prediction.shape[-1]
                !=
                2
            ):

                raise ValueError(
                    "Unexpected model prediction "
                    f"shape: {prediction.shape}"
                )

            real_probability = float(
                prediction[
                    REAL_CLASS
                ]
            )

            fake_probability = float(
                prediction[
                    FAKE_CLASS
                ]
            )

            raw_fake = bool(
                fake_probability
                >=
                window_fake_threshold
            )

            window_results.append({

                "start_time":
                    round(
                        float(start),
                        2
                    ),

                "end_time":
                    round(
                        float(end),
                        2
                    ),

                "real_probability":
                    round(
                        real_probability,
                        6
                    ),

                "fake_probability":
                    round(
                        fake_probability,
                        6
                    ),

                "face_count":
                    int(
                        face_count
                    ),

                "is_fake_raw":
                    raw_fake
            })

            print(
                f"[{window_index}/"
                f"{total_candidate_windows}] "
                f"{start:.1f}-"
                f"{end:.1f}s | "
                f"faces={face_count}/"
                f"{frames_per_window} | "
                f"fake={fake_probability:.3f}"
            )

        # ====================================================
        # NO USABLE WINDOWS
        # ====================================================

        analysable_windows = len(
            window_results
        )

        if analysable_windows == 0:

            return {

                "verdict":
                    "INSUFFICIENT_FACE_DATA",

                "fake_percentage":
                    0.0,

                "fake_segments":
                    [],

                "windows_scored":
                    0,

                "candidate_windows":
                    total_candidate_windows,

                "rejected_windows":
                    rejected_windows,

                "analysable_percentage":
                    0.0,

                "video_duration":
                    round(
                        duration,
                        2
                    )
            }

        # ====================================================
        # REQUIRE SUSTAINED FAKE WINDOWS
        # ====================================================

        window_results = (
            apply_minimum_fake_run(
                window_results,
                minimum_fake_run,
                stride_seconds
            )
        )

        fake_windows = int(
            sum(
                bool(
                    window[
                        "is_fake"
                    ]
                )
                for window
                in window_results
            )
        )

        # ====================================================
        # FAKE COVERAGE
        # ====================================================

        fake_coverage = (
            fake_windows
            /
            analysable_windows
        )

        fake_percentage = (
            fake_coverage
            *
            100.0
        )

        # ====================================================
        # MERGED TIMESTAMP SEGMENTS
        # ====================================================

        fake_segments = (
            merge_fake_segments(
                window_results,
                duration
            )
        )

        # ====================================================
        # FINAL VIDEO VERDICT
        # ====================================================

        if fake_windows == 0:

            verdict = "REAL"

        elif (
            fake_coverage
            >=
            full_fake_coverage_threshold
        ):

            verdict = "FAKE"

        else:

            verdict = (
                "PARTIALLY_MANIPULATED"
            )

        # ====================================================
        # ANALYSABLE COVERAGE
        # ====================================================

        analysable_percentage = (

            analysable_windows
            /
            max(
                total_candidate_windows,
                1
            )
            *
            100.0
        )

        # ====================================================
        # RAW FAKE WINDOWS
        # ====================================================

        raw_fake_windows = int(
            sum(
                bool(
                    window[
                        "is_fake_raw"
                    ]
                )
                for window
                in window_results
            )
        )

        # ====================================================
        # MEAN / MAX FAKE PROBABILITY
        # ====================================================

        fake_probabilities = [
            float(
                window[
                    "fake_probability"
                ]
            )
            for window
            in window_results
        ]

        mean_fake_probability = float(
            np.mean(
                fake_probabilities
            )
        )

        max_fake_probability = float(
            np.max(
                fake_probabilities
            )
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        result = {

            "verdict":
                verdict,

            "fake_percentage":
                round(
                    fake_percentage,
                    1
                ),

            "fake_segments":
                fake_segments,

            "segments_found":
                len(
                    fake_segments
                ),

            "video_duration":
                round(
                    duration,
                    2
                ),

            "candidate_windows":
                int(
                    total_candidate_windows
                ),

            "windows_scored":
                int(
                    analysable_windows
                ),

            "rejected_windows":
                int(
                    rejected_windows
                ),

            "analysable_percentage":
                round(
                    analysable_percentage,
                    1
                ),

            "raw_fake_windows":
                raw_fake_windows,

            "fake_windows":
                fake_windows,

            "mean_fake_probability":
                round(
                    mean_fake_probability,
                    4
                ),

            "max_fake_probability":
                round(
                    max_fake_probability,
                    4
                ),

            "window_threshold":
                window_fake_threshold,

            "minimum_consecutive_fake_windows":
                minimum_fake_run,

            "full_fake_coverage_threshold":
                full_fake_coverage_threshold,

            "model":
                os.path.basename(
                    model_path
                ),

            "class_mapping": {
                "0":
                    "REAL",

                "1":
                    "FAKE"
            },

            "window_results":
                window_results
        }

        # ====================================================
        # PRINT SUMMARY
        # ====================================================

        print()
        print("=" * 70)
        print("FINAL CONDITION C v3 RESULT")
        print("=" * 70)

        print(
            "Verdict:",
            verdict
        )

        print(
            f"Fake coverage: "
            f"{fake_percentage:.1f}%"
        )

        print(
            "Candidate windows:",
            total_candidate_windows
        )

        print(
            "Analysable windows:",
            analysable_windows
        )

        print(
            "Rejected windows:",
            rejected_windows
        )

        print(
            "Raw fake windows:",
            raw_fake_windows
        )

        print(
            "Sustained fake windows:",
            fake_windows
        )

        print(
            "Segments:",
            len(
                fake_segments
            )
        )

        if fake_segments:

            print()
            print(
                "Detected manipulated segments:"
            )

            for segment in fake_segments:

                print(
                    "  "
                    f"{format_time(segment['start_time'])}"
                    " -> "
                    f"{format_time(segment['end_time'])}"
                    " | "
                    f"mean confidence="
                    f"{segment['confidence']:.4f}"
                    " | "
                    f"peak="
                    f"{segment['peak_confidence']:.4f}"
                )

        print("=" * 70)

        return result

    except Exception as e:

        print()
        print("=" * 70)
        print("CONDITION C v3 SCORER ERROR")
        print("=" * 70)

        print(
            str(e)
        )

        return {

            "verdict":
                "ERROR",

            "fake_percentage":
                0.0,

            "fake_segments":
                [],

            "error":
                str(e)
        }

    finally:

        if cap is not None:

            try:
                cap.release()

            except Exception:
                pass


# ============================================================
# TEMPORAL IoU
# ============================================================

def temporal_iou(
    predicted_segments,
    gt_start,
    gt_end
):

    """
    Calculates IoU between the UNION of predicted
    fake segments and one ground-truth fake interval.

    This is better than taking only the best individual
    predicted segment.
    """

    if not predicted_segments:

        return 0.0

    gt_start = float(
        gt_start
    )

    gt_end = float(
        gt_end
    )

    if (
        gt_end
        <=
        gt_start
    ):

        return 0.0

    # --------------------------------------------------------
    # First merge overlapping predicted segments
    # defensively
    # --------------------------------------------------------

    intervals = []

    for segment in predicted_segments:

        start = float(
            segment.get(
                "start_time",
                0.0
            )
        )

        end = float(
            segment.get(
                "end_time",
                0.0
            )
        )

        if end > start:

            intervals.append(
                (
                    start,
                    end
                )
            )

    if not intervals:

        return 0.0

    intervals.sort()

    merged = [
        list(
            intervals[0]
        )
    ]

    for start, end in intervals[1:]:

        previous = merged[-1]

        if (
            start
            <=
            previous[1]
        ):

            previous[1] = max(
                previous[1],
                end
            )

        else:

            merged.append(
                [
                    start,
                    end
                ]
            )

    # --------------------------------------------------------
    # Predicted duration
    # --------------------------------------------------------

    predicted_duration = sum(
        end - start
        for start, end
        in merged
    )

    # --------------------------------------------------------
    # Intersection with GT
    # --------------------------------------------------------

    intersection = 0.0

    for start, end in merged:

        overlap_start = max(
            start,
            gt_start
        )

        overlap_end = min(
            end,
            gt_end
        )

        intersection += max(
            0.0,
            overlap_end
            -
            overlap_start
        )

    gt_duration = (
        gt_end
        -
        gt_start
    )

    union = (
        predicted_duration
        +
        gt_duration
        -
        intersection
    )

    if union <= 0:

        return 0.0

    return float(
        intersection
        /
        union
    )
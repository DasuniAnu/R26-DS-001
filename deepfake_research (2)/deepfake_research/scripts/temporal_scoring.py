# ============================================================
# scripts/temporal_scoring.py
# CONDITION C v3 — SEQUENTIAL-DECODE TEMPORAL SCORER
# ============================================================

import os
import json
import time

import cv2
import numpy as np
import tensorflow as tf

from mtcnn import MTCNN
from keras.saving import register_keras_serializable


# ============================================================
# CONSTANTS
# ============================================================

REAL_CLASS = 0
FAKE_CLASS = 1

INFERENCE_BATCH_SIZE = 4

# ============================================================
# FAST FACE EXTRACTION SETTINGS
# ============================================================

# Run full-resolution MTCNN only once every N required frames.
# Intermediate required frames reuse the latest reliable face box.
MTCNN_DETECTION_INTERVAL = 4

# Smallest usable detected face box.
MIN_FACE_BOX_SIZE = 24

MODEL_CACHE = {}
DETECTOR_CACHE = {}


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

    def build(
        self,
        input_shape
    ):

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

    def call(
        self,
        x
    ):

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

    def get_config(
        self
    ):

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

    def build(
        self,
        input_shape
    ):

        self.conv = tf.keras.layers.Conv2D(
            1,
            kernel_size=7,
            padding="same",
            activation="sigmoid"
        )

        super().build(
            input_shape
        )

    def call(
        self,
        x
    ):

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
# MODEL
# ============================================================

def load_model_cached(
    model_path
):

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

        expected = (
            15,
            224,
            224,
            3
        )

        if (
            model.input_shape[1:]
            !=
            expected
        ):

            raise ValueError(
                "Wrong model input shape.\n"
                f"Expected: (None, {expected})\n"
                f"Actual: {model.input_shape}"
            )

        if (
            model.output_shape[-1]
            !=
            2
        ):

            raise ValueError(
                "Expected 2 output classes."
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
# MTCNN
# ============================================================

def get_mtcnn():

    key = "default"

    if key not in DETECTOR_CACHE:

        print()
        print("=" * 70)
        print("INITIALIZING MTCNN")
        print("=" * 70)

        DETECTOR_CACHE[
            key
        ] = MTCNN()

    return DETECTOR_CACHE[
        key
    ]


# ============================================================
# CONFIG
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
            f"Config not found:\n"
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(
            file
        )

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
            "Missing config values: "
            +
            ", ".join(
                missing
            )
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

    except ValueError as error:

        text = str(
            error
        ).lower()

        if (
            "empty output" in text
            or
            "shape=(0" in text
            or
            "incompatible shapes" in text
        ):
            return []

        raise

    except Exception as error:

        print(
            "MTCNN warning:",
            str(error)
        )

        return []

def detect_largest_face_box(
    rgb,
    detector,
    confidence_threshold
):
    """
    Run MTCNN and return the largest reliable face box.

    Returns:
        (x, y, w, h)

    or:
        None
    """

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

        if not box:
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

    best = max(
        valid,
        key=lambda detection:
            detection["box"][2]
            *
            detection["box"][3]
    )

    x, y, w, h = (
        best["box"]
    )

    return (
        int(x),
        int(y),
        int(w),
        int(h)
    )

def crop_face_from_box(
    rgb,
    box,
    image_size
):
    """
    Crop face from ORIGINAL RGB frame.
    """

    if box is None:
        return None

    x, y, w, h = box

    frame_height, frame_width = (
        rgb.shape[:2]
    )

    x = max(
        0,
        int(x)
    )

    y = max(
        0,
        int(y)
    )

    x2 = min(
        frame_width,
        x + int(w)
    )

    y2 = min(
        frame_height,
        y + int(h)
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

    if (
        face.shape
        !=
        (
            image_size,
            image_size,
            3
        )
    ):
        return None

    return face.astype(
        np.uint8
    )
# ============================================================
# FACE EXTRACTION
# ============================================================
def extract_largest_face(
    rgb,
    detector,
    image_size,
    confidence_threshold
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

        if not box:
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

    best = max(
        valid,
        key=lambda detection:
            detection["box"][2]
            *
            detection["box"][3]
    )

    x, y, w, h = (
        best["box"]
    )

    frame_height, frame_width = (
        rgb.shape[:2]
    )

    x = max(
        0,
        int(x)
    )

    y = max(
        0,
        int(y)
    )

    x2 = min(
        frame_width,
        x + int(w)
    )

    y2 = min(
        frame_height,
        y + int(h)
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
# VIDEO INFO
# ============================================================

def get_video_info(
    video_path
):
    """
    Return:
        fps,
        total_frames,
        duration_seconds
    """

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video:\n{video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if (
        not fps
        or fps <= 0
        or np.isnan(fps)
    ):
        fps = 25.0

    if total_frames > 0:
        duration = (
            total_frames
            /
            fps
        )
    else:
        duration = 0.0

    cap.release()

    return (
        fps,
        total_frames,
        float(duration)
    )


# ============================================================
# BUILD WINDOW FRAME REQUESTS
# ============================================================

def build_window_frame_requests(
    window_starts,
    window_seconds,
    frames_per_window,
    fps,
    total_frames
):

    window_requests = []

    required_frame_indices = set()

    for start in window_starts:

        end = (
            float(start)
            +
            window_seconds
        )

        timestamps = np.linspace(
            float(start),
            float(end),
            int(
                frames_per_window
            ),
            endpoint=False
        )

        frame_indices = []

        for timestamp in timestamps:

            frame_index = int(
                round(
                    float(timestamp)
                    *
                    float(fps)
                )
            )

            if (
                total_frames
                >
                0
            ):

                frame_index = min(
                    max(
                        frame_index,
                        0
                    ),
                    total_frames - 1
                )

            frame_indices.append(
                frame_index
            )

            required_frame_indices.add(
                frame_index
            )

        window_requests.append({

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

            "frame_indices":
                frame_indices
        })

    return (
        window_requests,
        required_frame_indices
    )


# ============================================================
# SEQUENTIAL FACE EXTRACTION
# ============================================================

def extract_faces_sequentially(
    video_path,
    required_frame_indices,
    detector,
    image_size,
    mtcnn_confidence
):
    """
    FAST SEQUENTIAL FACE EXTRACTION

    Strategy:
    ------------------------------------------------------------
    1. Decode video sequentially exactly once.
    2. Process only frames required by temporal windows.
    3. Run full-resolution MTCNN only every
       MTCNN_DETECTION_INTERVAL required frames.
    4. Reuse the last reliable bounding box between detections.
    5. Crop every face from the ORIGINAL frame.
    ------------------------------------------------------------

    This preserves model input behaviour while reducing the
    number of expensive MTCNN calls substantially.
    """

    required = sorted(
        set(
            required_frame_indices
        )
    )

    required_set = set(
        required
    )

    face_by_frame = {}

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video "
            "for sequential preprocessing."
        )

    preprocess_start = (
        time.perf_counter()
    )

    total_mtcnn_time = 0.0

    mtcnn_calls = 0
    reused_box_frames = 0

    frame_index = 0
    required_index = 0

    last_face_box = None

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        # --------------------------------------------------------
        # Skip frames that are not required by any temporal window
        # --------------------------------------------------------

        if (
            frame_index
            not in
            required_set
        ):

            frame_index += 1
            continue

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------------------
        # Decide whether MTCNN must run
        # --------------------------------------------------------

        should_detect = (
            last_face_box is None
            or
            required_index
            %
            MTCNN_DETECTION_INTERVAL
            ==
            0
        )

        # --------------------------------------------------------
        # PERIODIC FULL-RESOLUTION MTCNN
        # --------------------------------------------------------

        if should_detect:

            detect_start = (
                time.perf_counter()
            )

            detected_box = (
                detect_largest_face_box(
                    rgb,
                    detector,
                    mtcnn_confidence
                )
            )

            total_mtcnn_time += (
                time.perf_counter()
                -
                detect_start
            )

            mtcnn_calls += 1

            if (
                detected_box
                is not None
            ):

                last_face_box = (
                    detected_box
                )

        else:

            reused_box_frames += 1

        # --------------------------------------------------------
        # Crop using latest reliable original-frame box
        # --------------------------------------------------------

        face = crop_face_from_box(
            rgb,
            last_face_box,
            image_size
        )

        # --------------------------------------------------------
        # If reused box became invalid, do immediate MTCNN recovery
        # --------------------------------------------------------

        if (
            face is None
            and
            not should_detect
        ):

            detect_start = (
                time.perf_counter()
            )

            detected_box = (
                detect_largest_face_box(
                    rgb,
                    detector,
                    mtcnn_confidence
                )
            )

            total_mtcnn_time += (
                time.perf_counter()
                -
                detect_start
            )

            mtcnn_calls += 1

            if (
                detected_box
                is not None
            ):

                last_face_box = (
                    detected_box
                )

                face = crop_face_from_box(
                    rgb,
                    last_face_box,
                    image_size
                )

            else:

                last_face_box = None

        face_by_frame[
            frame_index
        ] = face

        required_index += 1

        if (
            len(
                face_by_frame
            )
            >=
            len(
                required_set
            )
        ):
            break

        frame_index += 1

    cap.release()

    total_preprocess_time = (
        time.perf_counter()
        -
        preprocess_start
    )

    # ============================================================
    # SPEED PROFILE
    # ============================================================

    print()
    print("=" * 70)
    print("FAST FACE EXTRACTION PROFILE")
    print("=" * 70)

    print(
        "Required frames      :",
        len(
            required_set
        )
    )

    print(
        "MTCNN interval       :",
        MTCNN_DETECTION_INTERVAL
    )

    print(
        "Actual MTCNN calls   :",
        mtcnn_calls
    )

    print(
        "Reused-box frames    :",
        reused_box_frames
    )

    print(
        "MTCNN total time     :",
        f"{total_mtcnn_time:.2f}s"
    )

    print("=" * 70)

    return (
        face_by_frame,
        total_preprocess_time,
        total_mtcnn_time
    )
# ============================================================
# BUILD WINDOW SEQUENCE
# ============================================================

def build_sequence_from_faces(
    frame_indices,
    face_by_frame,
    frames_per_window,
    image_size,
    minimum_detected_faces
):

    detected_faces = {}

    for position, frame_index in enumerate(
        frame_indices
    ):

        face = face_by_frame.get(
            frame_index
        )

        if face is not None:

            detected_faces[
                position
            ] = face

    detected_count = len(
        detected_faces
    )

    if (
        detected_count
        <
        minimum_detected_faces
    ):

        return (
            None,
            detected_count
        )

    valid_positions = sorted(
        detected_faces.keys()
    )

    sequence = []

    for position in range(
        frames_per_window
    ):

        if (
            position
            in
            detected_faces
        ):

            sequence.append(
                detected_faces[
                    position
                ]
            )

        else:

            nearest = min(
                valid_positions,
                key=lambda valid_position:
                    abs(
                        valid_position
                        -
                        position
                    )
            )

            sequence.append(
                detected_faces[
                    nearest
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
# BATCH MODEL INFERENCE
# ============================================================

def predict_window_batch(
    model,
    sequences
):

    if not sequences:

        return np.empty(
            (
                0,
                2
            ),
            dtype=np.float32
        )

    batch = np.stack(
        sequences,
        axis=0
    ).astype(
        np.float32
    )

    batch = (
        batch
        /
        255.0
    )

    predictions = model(
        batch,
        training=False
    )

    if hasattr(
        predictions,
        "numpy"
    ):

        predictions = (
            predictions.numpy()
        )

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    if (
        predictions.ndim
        !=
        2
        or
        predictions.shape[1]
        !=
        2
    ):

        raise ValueError(
            "Unexpected prediction shape: "
            f"{predictions.shape}"
        )

    return predictions


# ============================================================
# MINIMUM FAKE RUN
# ============================================================

def apply_minimum_fake_run(
    window_results,
    minimum_run,
    stride_seconds
):

    if not window_results:
        return []

    keep = np.zeros(
        len(
            window_results
        ),
        dtype=bool
    )

    current_run = []

    for index, window in enumerate(
        window_results
    ):

        is_fake = bool(
            window[
                "is_fake_raw"
            ]
        )

        if not is_fake:

            if (
                len(
                    current_run
                )
                >=
                minimum_run
            ):

                keep[
                    current_run
                ] = True

            current_run = []

            continue

        if not current_run:

            current_run = [
                index
            ]

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
                index
            )

        else:

            if (
                len(
                    current_run
                )
                >=
                minimum_run
            ):

                keep[
                    current_run
                ] = True

            current_run = [
                index
            ]

    if (
        len(
            current_run
        )
        >=
        minimum_run
    ):

        keep[
            current_run
        ] = True

    output = []

    for index, window in enumerate(
        window_results
    ):

        item = dict(
            window
        )

        item[
            "is_fake"
        ] = bool(
            keep[
                index
            ]
        )

        output.append(
            item
        )

    return output


# ============================================================
# MERGE FAKE SEGMENTS
# ============================================================

def merge_fake_segments(
    window_results,
    video_duration
):

    fake_windows = [
        window
        for window
        in window_results
        if window.get(
            "is_fake",
            False
        )
    ]

    if not fake_windows:
        return []

    fake_windows = sorted(
        fake_windows,
        key=lambda item:
            item[
                "start_time"
            ]
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

    segments = []

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

        if (
            end
            <=
            start
        ):
            continue

        probabilities = (
            segment[
                "probabilities"
            ]
        )

        segments.append({

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
                    end - start,
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

    return segments


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(
    seconds
):

    seconds = max(
        0,
        int(
            round(
                float(
                    seconds
                )
            )
        )
    )

    minutes = (
        seconds // 60
    )

    seconds = (
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# MAIN SCORER
# ============================================================

def sliding_window_temporal_scorer(
    video_path,
    model_path,
    config_path
):

    total_start = (
        time.perf_counter()
    )

    try:

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

        config = load_deployment_config(
            config_path
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

        model = load_model_cached(
            model_path
        )

        detector = get_mtcnn()

        (
            fps,
            total_frames,
            duration
        ) = get_video_info(
            video_path
        )

        print()
        print("=" * 70)

        print(
            "CONDITION C v3 — "
            "SEQUENTIAL-DECODE ANALYSIS"
        )

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
            f"Duration: "
            f"{duration:.2f}s"
        )

        print(
            f"Window: "
            f"{window_seconds:.1f}s"
        )

        print(
            f"Stride: "
            f"{stride_seconds:.1f}s"
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
            "Window threshold:",
            window_fake_threshold
        )

        print(
            "Minimum fake run:",
            minimum_fake_run
        )

        print(
            "Full fake coverage:",
            full_fake_coverage_threshold
        )

        print(
            "Inference batch size:",

            INFERENCE_BATCH_SIZE
        )

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
        # WINDOW REQUESTS
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

        (
            window_requests,
            required_frame_indices
        ) = build_window_frame_requests(
            window_starts,
            window_seconds,
            frames_per_window,
            fps,
            total_frames
        )

        print(
            "Candidate windows:",
            total_candidate_windows
        )

        print(
            "Unique required source frames:",
            len(
                required_frame_indices
            )
        )

        # ====================================================
        # SEQUENTIAL VIDEO PROCESSING
        # ====================================================

        (
            face_by_frame,
            total_preprocess_time,
            total_mtcnn_time
        ) = extract_faces_sequentially(
            video_path,
            required_frame_indices,
            detector,
            image_size,
            mtcnn_confidence
        )

        print(
            "Sequential preprocessing complete."
        )

        # ====================================================
        # BUILD TEMPORAL WINDOWS
        # ====================================================

        valid_sequences = []
        valid_metadata = []

        rejected_windows = 0

        for window_index, request in enumerate(
            window_requests,
            start=1
        ):

            (
                sequence,
                face_count
            ) = build_sequence_from_faces(
                request[
                    "frame_indices"
                ],
                face_by_frame,
                frames_per_window,
                image_size,
                minimum_detected_faces
            )

            if sequence is None:

                rejected_windows += 1

                print(
                    f"[{window_index}/"
                    f"{total_candidate_windows}] "
                    f"{request['start_time']:.1f}-"
                    f"{request['end_time']:.1f}s | "
                    f"faces={face_count}/"
                    f"{frames_per_window} | "
                    f"REJECTED"
                )

                continue

            valid_sequences.append(
                sequence
            )

            valid_metadata.append({

                "start_time":
                    request[
                        "start_time"
                    ],

                "end_time":
                    request[
                        "end_time"
                    ],

                "face_count":
                    int(
                        face_count
                    )
            })

        analysable_windows = len(
            valid_sequences
        )

        if (
            analysable_windows
            ==
            0
        ):

            total_time = (
                time.perf_counter()
                -
                total_start
            )

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
                    ),

                "processing_time_seconds":
                    round(
                        total_time,
                        2
                    )
            }

        # ====================================================
        # BATCH CLASSIFICATION
        # ====================================================

        model_start = (
            time.perf_counter()
        )

        prediction_chunks = []

        for batch_start in range(
            0,
            analysable_windows,
            INFERENCE_BATCH_SIZE
        ):

            batch_sequences = (
                valid_sequences[
                    batch_start:
                    batch_start
                    +
                    INFERENCE_BATCH_SIZE
                ]
            )

            prediction_chunks.append(
                predict_window_batch(
                    model,
                    batch_sequences
                )
            )

        predictions = np.concatenate(
            prediction_chunks,
            axis=0
        )

        total_model_time = (
            time.perf_counter()
            -
            model_start
        )

        window_results = []

        for metadata, prediction in zip(
            valid_metadata,
            predictions
        ):

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

            item = {

                "start_time":
                    metadata[
                        "start_time"
                    ],

                "end_time":
                    metadata[
                        "end_time"
                    ],

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
                    metadata[
                        "face_count"
                    ],

                "is_fake_raw":
                    raw_fake
            }

            window_results.append(
                item
            )

            print(
                "      "
                f"{item['start_time']:.1f}-"
                f"{item['end_time']:.1f}s | "
                f"faces="
                f"{item['face_count']}/"
                f"{frames_per_window} | "
                f"fake="
                f"{item['fake_probability']:.3f}"
            )

        # ====================================================
        # SAME TEMPORAL POSTPROCESSING
        # ====================================================

        postprocess_start = (
            time.perf_counter()
        )

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

        fake_segments = (
            merge_fake_segments(
                window_results,
                duration
            )
        )

        if (
            fake_windows
            ==
            0
        ):

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

        postprocess_time = (
            time.perf_counter()
            -
            postprocess_start
        )

        total_time = (
            time.perf_counter()
            -
            total_start
        )

        realtime_factor = (
            total_time
            /
            duration
            if duration > 0
            else 0.0
        )

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
                "0": "REAL",
                "1": "FAKE"
            },

            "timing": {

                "preprocessing_seconds":
                    round(
                        total_preprocess_time,
                        2
                    ),

                "mtcnn_seconds":
                    round(
                        total_mtcnn_time,
                        2
                    ),

                "model_inference_seconds":
                    round(
                        total_model_time,
                        2
                    ),

                "postprocessing_seconds":
                    round(
                        postprocess_time,
                        4
                    ),

                "total_seconds":
                    round(
                        total_time,
                        2
                    ),

                "realtime_factor":
                    round(
                        realtime_factor,
                        2
                    )
            },

            "window_results":
                window_results
        }

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print()
        print("=" * 70)
        print(
            "FINAL CONDITION C v3 RESULT"
        )
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
                    f"mean="
                    f"{segment['confidence']:.4f}"
                    " | "
                    f"peak="
                    f"{segment['peak_confidence']:.4f}"
                )

        print()
        print("=" * 70)
        print(
            "PERFORMANCE PROFILE"
        )
        print("=" * 70)

        print(
            "Unique required frames :",
            len(
                required_frame_indices
            )
        )

        print(
            "Preprocessing total    :",
            f"{total_preprocess_time:.2f}s"
        )

        print(
            "MTCNN face detect      :",
            f"{total_mtcnn_time:.2f}s"
        )

        print(
            "Model inference        :",
            f"{total_model_time:.2f}s"
        )

        print(
            "Postprocessing         :",
            f"{postprocess_time:.4f}s"
        )

        print(
            "TOTAL                  :",
            f"{total_time:.2f}s"
        )

        print(
            "Real-time factor       :",
            f"{realtime_factor:.2f}x"
        )

        print("=" * 70)

        return result

    except Exception as error:

        print()
        print("=" * 70)
        print(
            "CONDITION C v3 SCORER ERROR"
        )
        print("=" * 70)

        print(
            str(error)
        )

        return {

            "verdict":
                "ERROR",

            "fake_percentage":
                0.0,

            "fake_segments":
                [],

            "error":
                str(error)
        }


# ============================================================
# TEMPORAL IoU
# ============================================================

def temporal_iou(
    predicted_segments,
    gt_start,
    gt_end
):

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

        if (
            end
            >
            start
        ):

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

        previous = (
            merged[-1]
        )

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

    predicted_duration = sum(
        end - start
        for start, end
        in merged
    )

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
import os
import sys
import json
import time

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(
    0,
    os.path.join(PROJECT_DIR, "scripts")
)

from temporal_scoring import sliding_window_temporal_scorer


MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "condition_c_v3_best.keras"
)

CONFIG_PATH = os.path.join(
    PROJECT_DIR,
    "config",
    "condition_c_v3_deployment_config.json"
)


if len(sys.argv) < 2:
    print(
        'Usage: python scripts\\test_v3_video.py '
        '"C:\\path\\to\\video.mp4"'
    )
    raise SystemExit(1)


VIDEO_PATH = os.path.abspath(sys.argv[1])


print("=" * 70)
print("CONDITION C v3 — LOCAL VIDEO TEST")
print("=" * 70)
print("Video :", VIDEO_PATH)
print("Model :", MODEL_PATH)
print("Config:", CONFIG_PATH)


if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Video not found:\n{VIDEO_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"Config not found:\n{CONFIG_PATH}"
    )


start = time.time()

result = sliding_window_temporal_scorer(
    VIDEO_PATH,
    MODEL_PATH,
    CONFIG_PATH
)

elapsed = time.time() - start


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    json.dumps(
        result,
        indent=2
    )
)

print()
print(
    "Processing time:",
    round(elapsed, 2),
    "seconds"
)
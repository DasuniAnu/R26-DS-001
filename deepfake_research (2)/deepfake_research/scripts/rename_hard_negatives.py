from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

FOLDER = (
    PROJECT_DIR
    / "dataset"
    / "dataset"
    / "hard_negative_real"
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}


if not FOLDER.exists():
    raise FileNotFoundError(
        f"Folder not found:\n{FOLDER}"
    )


videos = sorted(
    [
        p for p in FOLDER.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
    ],
    key=lambda p: p.name.lower()
)


print("=" * 70)
print("HARD-NEGATIVE REAL VIDEO RENAMER")
print("=" * 70)
print("Folder:", FOLDER)
print("Videos found:", len(videos))
print()


# ---------------------------------------------------------
# Stage 1
# Rename to temporary names first.
#
# This prevents collisions if a file already happens to
# have a name such as hardreal_001.mp4.
# ---------------------------------------------------------

temporary_files = []

for i, old_path in enumerate(videos, start=1):

    temp_path = FOLDER / (
        f"__temp_hardreal_{i:03d}{old_path.suffix.lower()}"
    )

    print(
        f"TEMP: {old_path.name}"
        f"\n   -> {temp_path.name}"
    )

    old_path.rename(temp_path)

    temporary_files.append(temp_path)


# ---------------------------------------------------------
# Stage 2
# Give every video its final standardized name.
# ---------------------------------------------------------

print()
print("-" * 70)
print("FINAL NAMES")
print("-" * 70)

for i, temp_path in enumerate(
    temporary_files,
    start=1
):

    final_path = FOLDER / (
        f"hardreal_{i:03d}{temp_path.suffix.lower()}"
    )

    temp_path.rename(final_path)

    print(final_path.name)


print()
print("=" * 70)
print(
    f"DONE — {len(temporary_files)} "
    "hard-negative REAL videos renamed."
)
print("=" * 70)
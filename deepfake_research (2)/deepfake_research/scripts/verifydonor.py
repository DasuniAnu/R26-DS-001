import cv2
import glob
import os
import sys

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

donors = sorted(glob.glob("dataset/donor_faces/donor*.JPG"))

if not donors:
    print(" NO DONOR FILES FOUND in dataset/donor_faces/")
    print(" Save your images as donor1.JPG, donor2.JPG etc.")
    sys.exit(1)

print(f"\nFound {len(donors)} donor face files\n")

all_ok = True

for path in donors:
    img = cv2.imread(path)

    if img is None:
        print(f" FAIL — cannot read: {path}")
        all_ok = False
        continue

    h, w = img.shape[:2]

    # Check resolution
    if h < 150 or w < 150:
        print(f" FAIL — image too small ({w}x{h}): {path}")
        all_ok = False
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print(f" FAIL — no face detected: {path}")
        all_ok = False
    elif len(faces) > 1:
        print(f" WARNING — multiple faces detected: {path}")
    else:
        print(f" OK — {path}")

print("\n==============================")

if all_ok:
    print("✅ All donor faces verified — ready for fake generation 🚀")
else:
    print(" Some donor images failed.")
    print(" Replace failed images with clearer front-facing faces.")
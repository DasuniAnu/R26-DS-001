"""
Check if faces are being detected in a specific video time range
This helps diagnose why partial fakes aren't working
"""
import subprocess
import cv2
import json
import sys
import os

def get_duration(path):
    r = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            path
        ],
        capture_output=True,
        text=True
    )
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except:
        return 0.0

def check_faces_in_range(video_path, start_sec, end_sec):
    """Extract frames from the specified range and check for faces"""
    
    print(f"\n{'='*60}")
    print(f"Checking video: {video_path}")
    print(f"Time range: {start_sec}s to {end_sec}s")
    print(f"{'='*60}")
    
    # Create temp extraction
    tmp_segment = "dataset/tmp/check_segment.mp4"
    tmp_frames = "dataset/tmp/check_frames_%04d.jpg"
    
    # Extract segment
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-c:v", "libx264",
        tmp_segment
    ], capture_output=True)
    
    if result.returncode != 0:
        print("✗ Failed to extract segment")
        return
    
    print(f"✓ Extracted segment to {tmp_segment}")
    
    # Extract frames from segment
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_segment,
        "-vf", "fps=1",  # 1 frame per second
        tmp_frames
    ], capture_output=True)
    
    if result.returncode != 0:
        print("✗ Failed to extract frames")
        return
    
    print(f"✓ Extracted frames")
    
    # Check frames for faces using opencv DNN
    print("\nChecking frames for faces...")
    
    net = cv2.dnn.readNetFromCaffe(
        "c:\\Users\\LENOVO\\.opencv\\opencv-face-detection-model\\deploy.prototxt",
        "c:\\Users\\LENOVO\\.opencv\\opencv-face-detection-model\\res10_300x300_ssd_iter_140000.caffemodel"
    )
    
    import glob
    frames = sorted(glob.glob("dataset/tmp/check_frames_*.jpg"))
    
    faces_found = 0
    for i, frame_path in enumerate(frames):
        img = cv2.imread(frame_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        
        num_faces = 0
        for j in range(detections.shape[2]):
            confidence = detections[0, 0, j, 2]
            if confidence > 0.5:
                num_faces += 1
        
        if num_faces > 0:
            faces_found += 1
            print(f"  Frame {i+1}: {num_faces} face(s) detected ✓")
        else:
            print(f"  Frame {i+1}: No faces detected ✗")
    
    print(f"\n{'='*60}")
    print(f"Summary: {faces_found}/{len(frames)} frames have detectable faces")
    if faces_found == 0:
        print("⚠ WARNING: No faces detected in this time range!")
        print("This is why the face swap is failing.")
    print(f"{'='*60}\n")
    
    # Cleanup
    for f in [tmp_segment] + frames:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python check_face_detection.py <video_path> <start_sec> <end_sec>")
        print("Example: python check_face_detection.py dataset/roop_input_v2/tamil_015.mp4 56.54 64.54")
        sys.exit(1)
    
    video_path = sys.argv[1]
    start_sec = float(sys.argv[2])
    end_sec = float(sys.argv[3])
    
    check_faces_in_range(video_path, start_sec, end_sec)

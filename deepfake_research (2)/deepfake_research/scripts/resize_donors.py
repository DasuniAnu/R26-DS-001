from PIL import Image
import glob
import os

donors = glob.glob("dataset/donor_faces/donor*.jpg") + \
         glob.glob("dataset/donor_faces/donor*.JPG")

for path in donors:
    img = Image.open(path)
    original_size = img.size
    img = img.resize((256, 256), Image.LANCZOS)
    img.save(path)
    print(f"Resized: {os.path.basename(path)} "
          f"{original_size} → (256, 256)")

print(f"\nAll {len(donors)} donor faces resized to 256x256")
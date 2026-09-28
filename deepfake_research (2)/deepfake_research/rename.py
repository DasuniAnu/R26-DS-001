import os

folder = "dataset/trimmed"

for i, filename in enumerate(os.listdir(folder)):
    if filename.endswith(".mp4"):
        new_name = f"video_{i+1}.mp4"
        
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        
        os.rename(old_path, new_path)

print("Renaming done ✅")
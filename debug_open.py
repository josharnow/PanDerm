from PIL import Image
import os
import time

# --- IMPORTANT ---
# Manually set these two variables to match your training setup.
# Use the same paths that your main script would use.
root_path = "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/pad-ufes/images"
image_filename = "PAT_67_104_730.png"
# -----------------

full_path = os.path.join(root_path, image_filename)

print(f"Attempting to open image: {full_path}")
print("Timestamp before opening:", time.time())

try:
    # This is the line we are testing
    img = Image.open(full_path)
    print("Timestamp after opening:", time.time())
    print("\nSuccess! The image was opened without hanging.")
    print("Image details:", img.format, img.size, img.mode)

except Exception as e:
    print("\nAn error occurred while trying to open the image:")
    print(e)
"""
This is a utility script to convert images to raw binary format.
You'll need to run this once using PIL or OpenCV to create the raw files,
then the main model will only use NumPy to read these files.
"""
from PIL import Image
import os
import numpy as np

def convert_to_raw(input_dir):
    """Convert images to raw binary format (32x32 grayscale)"""
    # Convert in the same directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Open and convert to grayscale
            img_path = os.path.join(input_dir, filename)
            with Image.open(img_path) as img:
                # Convert to grayscale and resize
                img_gray = img.convert('L').resize((32, 32))
                # Convert to numpy array
                img_array = np.array(img_gray)
                # Save as raw binary in the same directory
                raw_path = os.path.join(input_dir, f"{os.path.splitext(filename)[0]}.raw")
                img_array.tofile(raw_path)
                print(f"Converted {filename} to raw format")

if __name__ == "__main__":
    # Get the absolute path to the img directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(current_dir, "data", "img")
    print(f"Converting images in: {img_dir}")
    convert_to_raw(img_dir)

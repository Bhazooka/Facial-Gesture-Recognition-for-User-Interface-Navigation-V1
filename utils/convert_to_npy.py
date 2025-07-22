import numpy as np
import os

def convert_images_to_npy(input_dir):
    """Convert all images in the input directory to .npy format"""
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(input_dir, filename.rsplit('.', 1)[0] + '.npy')
            
            # Read raw image bytes
            with open(input_path, 'rb') as f:
                img_bytes = f.read()
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                
                # Reshape to 32x32
                img = img_array[-1024:].reshape(32, 32)
                
                # Save as .npy
                np.save(output_path, img)
                print(f"Converted {filename} to {os.path.basename(output_path)}")

if __name__ == "__main__":
    # Convert images in the data/img directory
    img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "img")
    convert_images_to_npy(img_dir)
    print("Conversion complete!")

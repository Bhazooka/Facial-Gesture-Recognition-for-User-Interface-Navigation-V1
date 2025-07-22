import sys
import os
import numpy as np
import cv2

# Add the parent directory to Python path to import model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.model import EyeNN, preprocess_eye_image

def process_image(image_path, target_size=(32, 32)):
    """Process an image file into the format needed by the model"""
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to 32x32
    img_resized = cv2.resize(img, target_size)
    
    # Normalize to [0,1] range and flatten
    img_normalized = img_resized.astype(float) / 255.0
    img_flat = img_normalized.flatten().reshape(1, -1)
    
    return img_flat

def test_single_image(image_path, model_path='trained_model.npz'):
    """Test a single image with the trained model"""
    # Initialize model
    model = EyeNN(input_size=1024, hidden_size=64, output_size=1)
    
    try:
        # Load trained weights
        model.load_weights(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'agent', model_path))
        print("Loaded pre-trained weights")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process and test image
    try:
        # For .raw files use preprocess_eye_image, for others use process_image
        if image_path.endswith('.raw'):
            img_input = preprocess_eye_image(image_path)
        else:
            img_input = process_image(image_path)
        
        prediction = model.forward(img_input)
        
        print(f"\nTesting image: {os.path.basename(image_path)}")
        print(f"Probability of eye: {prediction[0,0]:.4f}")
        print("Prediction:", "Eye detected!" if prediction[0,0] > 0.5 else "No eye detected!")
        
        return prediction[0,0]
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def test_all_images(img_dir):
    """Test all images in the given directory"""
    # Get all image files (both .raw and other image formats)
    image_files = []
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.raw')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No images found in {img_dir}")
        return
    
    print(f"Found {len(image_files)} images to test")
    print("-" * 50)
    
    results = []
    for img_path in image_files:
        prob = test_single_image(img_path)
        if prob is not None:
            results.append((img_path, prob))
    
    print("\nSummary:")
    print("-" * 50)
    print(f"Total images tested: {len(results)}")
    
    # Count detections
    detections = sum(1 for _, prob in results if prob > 0.5)
    print(f"Eyes detected: {detections}")
    print(f"No eyes detected: {len(results) - detections}")
    
    # Show highest and lowest probabilities
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 5 highest probabilities:")
        for path, prob in results[:5]:
            print(f"{os.path.basename(path)}: {prob:.4f}")
        
        print("\nTop 5 lowest probabilities:")
        for path, prob in results[-5:]:
            print(f"{os.path.basename(path)}: {prob:.4f}")

if __name__ == "__main__":
    # Get the path to the img directory
    img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "img")
    test_all_images(img_dir)

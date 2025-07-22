import os
import numpy as np

def create_labels_from_folders():
    """
    Create labels file based on image folder structure.
    Assumes images are organized in folders like:
    data/
        img/
            eye_images/
            mouth_images/
            eyebrow_images/
    """
    # Get the path to the data directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_base_dir = os.path.join(current_dir, "data", "img")
    labels_dir = os.path.join(current_dir, "data", "labels")
    
    # Create labels directory if it doesn't exist
    os.makedirs(labels_dir, exist_ok=True)

    # Dictionary to store all labels
    all_labels = {}
    
    # Process each feature folder
    feature_folders = [d for d in os.listdir(img_base_dir) 
                      if os.path.isdir(os.path.join(img_base_dir, d))]
    
    if not feature_folders:
        print("No feature folders found! Expected folders like 'eyes', 'mouth', etc.")
        return
        
    print(f"Found feature folders: {feature_folders}")
    
    # Process each folder
    for feature in feature_folders:
        feature_dir = os.path.join(img_base_dir, feature)
        feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.raw')]
        
        print(f"\nProcessing {feature} folder: {len(feature_files)} images found")
        
        # Add labels for this feature
        for filename in feature_files:
            full_path = os.path.join(feature, filename)  # Store relative path
            all_labels[full_path] = feature  # Store the folder name as the label
    
    # Save labels
    labels_file = os.path.join(labels_dir, "feature_labels.txt")
    with open(labels_file, 'w') as f:
        for path, label in all_labels.items():
            f.write(f"{path},{label}\n")
    
    print(f"\nLabels saved to {labels_file}")
    print(f"Total images labeled: {len(all_labels)}")
    
    # Print summary
    label_counts = {}
    for label in all_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nDataset summary:")
    for label, count in label_counts.items():
        print(f"{label}: {count} images")

if __name__ == "__main__":
    create_labels_from_folders()

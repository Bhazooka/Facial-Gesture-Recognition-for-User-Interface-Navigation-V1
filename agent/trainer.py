import numpy as np
import os
from model import EyeNN, preprocess_eye_image

class ModelTrainer:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.training_losses = []

    def load_dataset(self, data_dir):
        """Load all .raw images and their labels from feature folders"""
        images = []
        labels = []
        
        # Load labels if they exist
        labels_file = os.path.join(os.path.dirname(data_dir), "labels", "feature_labels.txt")
        if not os.path.exists(labels_file):
            raise ValueError("No labels file found. Run utils/create_labels.py first!")
            
        # Read all labels and build label mapping
        unique_labels = set()
        label_dict = {}
        with open(labels_file, 'r') as f:
            for line in f:
                rel_path, feature = line.strip().split(',')
                unique_labels.add(feature)
                label_dict[rel_path] = feature
        
        # Create binary mapping for eye detection
        label_to_idx = {
            'eye_images': 1,     # Eyes are positive class (1)
            'non_eye_images': 0  # Non-eyes are negative class (0)
        }
        print(f"\nFeature mapping: {label_to_idx}")
        
        # Load images and their corresponding labels
        for feature_folder in os.listdir(data_dir):
            feature_path = os.path.join(data_dir, feature_folder)
            if not os.path.isdir(feature_path):
                continue
                
            for filename in os.listdir(feature_path):
                if filename.endswith('.raw'):
                    rel_path = os.path.join(feature_folder, filename)
                    if rel_path not in label_dict:
                        print(f"Warning: No label for {rel_path}, skipping")
                        continue
                        
                    image_path = os.path.join(data_dir, rel_path)
                    img = preprocess_eye_image(image_path)
                    images.append(img)
                    # Convert feature name to binary label (1 for eyes, 0 for others)
                    label_idx = label_to_idx.get(label_dict[rel_path], 0)
                    labels.append(label_idx)
        
        if not images:
            raise ValueError("No labeled images found!")
        
        # Stack all images into a single array
        X = np.vstack(images)
        y = np.array(labels).reshape(-1, 1)
        
        self.label_mapping = label_to_idx  # Save for later reference
        return X, y

    def train(self, X, y, epochs=100, learning_rate=0.01):
        """Train the model for multiple epochs"""
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.model.forward(X_batch)
                
                # Compute loss
                loss = self.model.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                
                # Backward pass
                self.model.backward(X_batch, y_batch, learning_rate=learning_rate)
            
            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def evaluate(self, X, y):
        """Evaluate the model on test data"""
        y_pred = self.model.forward(X)
        loss = self.model.compute_loss(y_pred, y)
        accuracy = np.mean((y_pred > 0.5) == y)
        return loss, accuracy

def main():
    # Initialize model for eye detection only
    model = EyeNN(input_size=1024, hidden_size=64, output_size=1)
    trainer = ModelTrainer(model, batch_size=16)
    
    # Load dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "img")
    print(f"Loading data from {data_dir}")
    X, y = trainer.load_dataset(data_dir)
    
    # Split into train and validation sets
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    
    # Train the model
    trainer.train(X_train, y_train, epochs=100, learning_rate=0.01)
    
    # Evaluate
    val_loss, val_acc = trainer.evaluate(X_val, y_val)
    print(f"\nValidation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # Save the trained model
    np.savez('trained_model.npz',
             W1=model.W1,
             b1=model.b1,
             W2=model.W2,
             b2=model.b2)
    print("\nModel saved to trained_model.npz")

if __name__ == "__main__":
    main()

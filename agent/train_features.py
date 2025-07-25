import numpy as np
import os
from model import EyeNN, EyebrowNN, MouthNN, preprocess_eye_image, preprocess_eyebrow_image, preprocess_mouth_image

class FeatureTrainer:
    def __init__(self, feature_type, batch_size=32, hidden_size=64):
        """Initialize trainer for a specific feature
        
        Args:
            feature_type: str, one of ['eye', 'eyebrow', 'mouth']
            batch_size: int, batch size for training
            hidden_size: int, size of hidden layer
        """
        self.feature_type = feature_type
        self.batch_size = batch_size
        
        # Initialize appropriate model and processing the images
        if feature_type == 'eye':
            self.model = EyeNN(input_size=1024, hidden_size=hidden_size, output_size=1)
            self.preprocess_fn = preprocess_eye_image
            self.model_file = 'eye_model.npz'
        elif feature_type == 'eyebrow':
            self.model = EyebrowNN(input_size=1024, hidden_size=hidden_size, output_size=1)
            self.preprocess_fn = preprocess_eyebrow_image
            self.model_file = 'eyebrow_model.npz'
        elif feature_type == 'mouth':
            self.model = MouthNN(input_size=1024, hidden_size=hidden_size, output_size=1)
            self.preprocess_fn = preprocess_mouth_image  
            self.model_file = 'mouth_model.npz'
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
        self.training_losses = []
        
    def load_dataset(self, data_dir):
        """Load dataset for the specific feature"""
        images = []
        labels = []
        
        # Define positive and negative folders
        positive_folder = f'{self.feature_type}_images'
        negative_folder = f'non_{self.feature_type}_images'
        
        # Load positive examples (label 1)
        pos_path = os.path.join(data_dir, positive_folder)
        if os.path.exists(pos_path):
            for filename in os.listdir(pos_path):
                if filename.endswith(('.raw', '.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(pos_path, filename)
                    img = self.preprocess_fn(image_path)
                    images.append(img)
                    labels.append(1)
        
        # Load negative examples (label 0)
        neg_path = os.path.join(data_dir, negative_folder)
        if os.path.exists(neg_path):
            for filename in os.listdir(neg_path):
                if filename.endswith(('.raw', '.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(neg_path, filename)
                    img = self.preprocess_fn(image_path)
                    images.append(img)
                    labels.append(0)
        
        if not images:
            raise ValueError(f"No images found for {self.feature_type}")
            
        # Convert to numpy arrays
        X = np.vstack(images)
        y = np.array(labels).reshape(-1, 1)
        
        print(f"\nLoaded dataset for {self.feature_type}:")
        print(f"Total images: {len(images)}")
        print(f"Positive examples: {sum(labels)}")
        print(f"Negative examples: {len(labels) - sum(labels)}")
        
        return X, y
    
    def train(self, X, y, epochs=100, learning_rate=0.01, validation_split=0.2):
        """Train the model"""
        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_train = X[indices[:n_train]]
        y_train = y[indices[:n_train]]
        X_val = X[indices[n_train:]]
        y_val = y[indices[n_train:]]
        
        print(f"\nTraining on {n_train} samples, validating on {n_val} samples")
        
        n_batches = (n_train + self.batch_size - 1) // self.batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            # Shuffle training data
            train_indices = np.random.permutation(n_train)
            X_shuffled = X_train[train_indices]
            y_shuffled = y_train[train_indices]
            
            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, n_train)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward and backward pass
                y_pred = self.model.forward(X_batch)
                loss = self.model.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                self.model.backward(X_batch, y_batch, learning_rate=learning_rate)
            
            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)
            
            # Validate
            if (epoch + 1) % 10 == 0:
                val_pred = self.model.forward(X_val)
                val_loss = self.model.compute_loss(val_pred, y_val)
                val_acc = np.mean((val_pred > 0.5) == y_val)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Final validation
        val_pred = self.model.forward(X_val)
        val_loss = self.model.compute_loss(val_pred, y_val)
        val_acc = np.mean((val_pred > 0.5) == y_val)
        print(f"\nFinal Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # Save model
        weights = {
            'W1': self.model.W1,
            'b1': self.model.b1,
            'W2': self.model.W2,
            'b2': self.model.b2
        }
        np.savez(self.model_file, **weights)
        print(f"\nModel saved to {self.model_file}")

def train_feature(feature_type, data_dir, **kwargs):
    """Helper function to train a specific feature"""
    trainer = FeatureTrainer(feature_type, **kwargs)
    X, y = trainer.load_dataset(data_dir)
    trainer.train(X, y, **kwargs)

if __name__ == "__main__":
    # Get the path to the data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "img")
    
    # Train each feature separately
    features = ['eye', 'eyebrow', 'mouth']
    
    for feature in features:
        print(f"\n{'='*50}")
        print(f"Training {feature} detector")
        print('='*50)
        
        try:
            train_feature(
                feature_type=feature,
                data_dir=data_dir,
                batch_size=32,
                hidden_size=64,
                epochs=100,
                learning_rate=0.01
            )
        except Exception as e:
            print(f"Error training {feature} detector: {e}")
            continue

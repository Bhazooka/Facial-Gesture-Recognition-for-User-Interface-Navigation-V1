import numpy as np
import os

class MLModel:
    pass

class EyeNN:
    def __init__(self, input_size=1024, hidden_size=32, output_size=1):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def load_weights(self, weights_file):
        """Load pre-trained weights from a .npz file"""
        weights = np.load(weights_file)
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        # Binary cross-entropy
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)) / m
        return loss

    def backward(self, x, y_true, learning_rate=0.01):
        # Forward pass
        y_pred = self.forward(x)
        m = y_true.shape[0]

        # Output layer gradients
        dz2 = y_pred - y_true  # (batch, output)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = x.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

def preprocess_eye_image(image_path):
    """
    Process an image file into the format needed for the neural network.
    Expects either:
    1. A .raw file containing a 32x32 grayscale image (1024 bytes)
    2. A .npy file containing the preprocessed image
    """
    try:
        if image_path.endswith('.npy'):
            # Load preprocessed numpy array
            img = np.load(image_path)
        elif image_path.endswith('.raw'):
            # Load raw binary image data
            # Each pixel is 1 byte (uint8), image is 32x32 = 1024 bytes total
            img = np.fromfile(image_path, dtype=np.uint8).reshape(32, 32)
        else:
            raise ValueError("Unsupported image format. Please convert to .raw format first.")
            
        # Normalize to [0,1] range and flatten
        img_normalized = img.astype(float) / 255.0
        img_flat = img_normalized.flatten().reshape(1, -1)
        return img_flat
    except Exception as e:
        print("Error loading image:", e)
        # Fallback to random image
        img = np.random.rand(32, 32)
        img_normalized = img / np.max(img)
        img_flat = img_normalized.flatten().reshape(1, -1)
        return img_flat

if __name__ == "__main__":
    # Initialize model with same architecture as training
    model = EyeNN(input_size=1024, hidden_size=64, output_size=1)  # Start with eyes only
    
    # Load trained weights if they exist
    try:
        model.load_weights('trained_model.npz')
        print("Loaded pre-trained weights")
    except Exception as e:
        print("No pre-trained weights found. Please run trainer.py first")
        exit(1)

    # Get the path to the img directory
    img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "img")
    
    # Test with eye images
    eye_folder = 'eye_images'
    
    print("\nTesting eye images:")
    eye_folder_path = os.path.join(img_dir, eye_folder)
    if os.path.exists(eye_folder_path):
        # Get up to 5 images from the folder
        image_files = [f for f in os.listdir(eye_folder_path) if f.endswith('.raw')][:5]
        
        for filename in image_files:
            image_path = os.path.join(eye_folder_path, filename)
            img_input = preprocess_eye_image(image_path)
            prediction = model.forward(img_input)
            
            print(f"\nImage: {filename}")
            print(f"Probability of eye: {prediction[0,0]:.4f}")
            print("Prediction: " + ("Eye detected!" if prediction[0,0] > 0.5 else "No eye detected!"))


class EyebrowNN:
    def __init__(self, input_size=1024, hidden_size=32, output_size=1):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def load_weights(self, weights_file):
        """Load pre-trained weights from a .npz file"""
        weights = np.load(weights_file)
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        # Binary cross-entropy
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)) / m
        return loss

    def backward(self, x, y_true, learning_rate=0.01):
        # Forward pass
        y_pred = self.forward(x)
        m = y_true.shape[0]

        # Output layer gradients
        dz2 = y_pred - y_true
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = x.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

def preprocess_eyebrow_image(image_path):
    """
    Process an eyebrow image file into the format needed for the neural network.
    Similar to preprocess_eye_image but for eyebrow ROIs
    """
    try:
        if image_path.endswith('.npy'):
            img = np.load(image_path)
        elif image_path.endswith('.raw'):
            img = np.fromfile(image_path, dtype=np.uint8).reshape(32, 32)
        else:
            raise ValueError("Unsupported image format. Please convert to .raw format first.")
            
        img_normalized = img.astype(float) / 255.0
        img_flat = img_normalized.flatten().reshape(1, -1)
        return img_flat
    except Exception as e:
        print("Error loading image:", e)
        img = np.random.rand(32, 32)
        img_normalized = img / np.max(img)
        img_flat = img_normalized.flatten().reshape(1, -1)
        return img_flat

if __name__ == "__main__":
    # Test both networks
    eye_model = EyeNN(input_size=1024, hidden_size=64, output_size=1)
    eyebrow_model = EyebrowNN(input_size=1024, hidden_size=32, output_size=1)
    
    # Load trained weights if they exist
    try:
        eye_model.load_weights('eye_model.npz')
        print("Loaded pre-trained eye weights")
        eyebrow_model.load_weights('eyebrow_model.npz')
        print("Loaded pre-trained eyebrow weights")
    except Exception as e:
        print("No pre-trained weights found. Please run trainer.py first")
        exit(1)

    # Get the path to the img directory
    img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "img")
    
    # Test with eye images
    print("\nTesting eye images:")
    eye_folder_path = os.path.join(img_dir, 'eye_images')
    if os.path.exists(eye_folder_path):
        image_files = [f for f in os.listdir(eye_folder_path) if f.endswith('.raw')][:5]
        for filename in image_files:
            image_path = os.path.join(eye_folder_path, filename)
            img_input = preprocess_eye_image(image_path)
            prediction = eye_model.forward(img_input)
            print(f"\nImage: {filename}")
            print(f"Probability of eye: {prediction[0,0]:.4f}")
    
    # Test with eyebrow images
    print("\nTesting eyebrow images:")
    eyebrow_folder_path = os.path.join(img_dir, 'eyebrow_images')
    if os.path.exists(eyebrow_folder_path):
        image_files = [f for f in os.listdir(eyebrow_folder_path) if f.endswith('.raw')][:5]
        for filename in image_files:
            image_path = os.path.join(eyebrow_folder_path, filename)
            img_input = preprocess_eyebrow_image(image_path)
            prediction = eyebrow_model.forward(img_input)
            print(f"\nImage: {filename}")
            print(f"Probability of raised eyebrow: {prediction[0,0]:.4f}")
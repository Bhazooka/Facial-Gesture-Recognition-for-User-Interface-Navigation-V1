import numpy as np

def preprocess_frame(frame, resize_dim=(640, 480), flip=True):
    # frame: numpy array of shape (H, W, 3) with RGB values
    
    # Convert to grayscale using standard weights
    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])

    # Resize (basic nearest neighbor)
    orig_h, orig_w = gray.shape
    new_h, new_w = resize_dim
    row_idx = (np.linspace(0, orig_h - 1, new_h)).astype(int)
    col_idx = (np.linspace(0, orig_w - 1, new_w)).astype(int)
    resized = gray[row_idx][:, col_idx]

    # Flip horizontally if needed
    if flip:
        resized = np.fliplr(resized)
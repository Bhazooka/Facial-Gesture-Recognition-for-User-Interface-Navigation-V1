import cv2

def preprocess_frame(frame, resize_dim=(640, 480), flip=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, resize_dim)
    if flip:
        resized = cv2.flip(resized, 1)
    return resized
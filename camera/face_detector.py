import cv2
import numpy as np

def detect_face(frame, cascade_path="haarcascade_frontalface_default.xml"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    return faces

def get_heuristic_face_roi(frame):
    h, w = frame.shape
    # Heuristic: center-upper portion (e.g., 30% from top, 40% height, centered horizontally)
    x = int(w * 0.3)
    y = int(h * 0.15)
    w_roi = int(w * 0.4)
    h_roi = int(h * 0.4)
    return (x, y, w_roi, h_roi)

def extract_eye_mouth_rois(face_rect):
    x, y, w, h = face_rect
    # Eyes: upper 40% of face, split horizontally
    eye_y = y + int(h * 0.15)
    eye_h = int(h * 0.25)
    left_eye = (x + int(w * 0.13), eye_y, int(w * 0.32), eye_h)
    right_eye = (x + int(w * 0.55), eye_y, int(w * 0.32), eye_h)
    # Mouth: lower 30% of face
    mouth_y = y + int(h * 0.65)
    mouth_h = int(h * 0.20)
    mouth = (x + int(w * 0.22), mouth_y, int(w * 0.56), mouth_h)
    return left_eye, right_eye, mouth

def detect_eye_mouth_contours(roi):
    # Threshold to binary image
    _, thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Return largest contour (assume it's the feature)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        return largest
    return None

def predict_landmarks_with_ml(roi, model):
    # roi: grayscale image patch
    # model: pre-trained ML model (loaded elsewhere)
    roi_flat = roi.flatten().reshape(1, -1)
    landmarks = model.predict(roi_flat)
    return landmarks



from camera.camera import get_camera_stream
from camera.processing import preprocess_frame
from camera.face_detector import detect_face, get_heuristic_face_roi, extract_eye_mouth_rois, detect_eye_mouth_contours, predict_landmarks_with_ml
import cv2

def main():
    cap = get_camera_stream(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        processed = preprocess_frame(frame, resize_dim=(640, 480), flip=True)
        faces = detect_face(processed)

        if len(faces) > 0:
            face_rect = faces[0]
        else:
            face_rect = get_heuristic_face_roi(processed)

        # Draw face ROI
        x, y, w, h = face_rect
        cv2.rectangle(processed, (x, y), (x + w, y + h), 128, 2)

        # Extract and draw eye and mouth ROIs
        left_eye, right_eye, mouth = extract_eye_mouth_rois(face_rect)
        for ex, ey, ew, eh in [left_eye, right_eye, mouth]:
            cv2.rectangle(processed, (ex, ey), (ex + ew, ey + eh), 200, 1)

            # For each ROI (left_eye, right_eye, mouth):
            roi = processed[ey:ey+eh, ex:ex+ew]
            contour = detect_eye_mouth_contours(roi)
            if contour is not None:
                # Draw contour or extract landmarks
                cv2.drawContours(roi, [contour], -1, 255, 1)
            elif ml_model is not None:
                landmarks = predict_landmarks_with_ml(roi, ml_model)
                # Draw landmarks
            else:
                # Use hard-coded heuristics as fallback
                pass

        cv2.imshow("ROI Detection", processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
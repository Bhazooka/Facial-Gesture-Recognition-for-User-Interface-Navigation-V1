
# controllers/system_controller.py

import cv2
import pyautogui
from agent.model_adapter import predict_gaze_from_frame

def run_gaze_mouse_control():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            continue

        x, y = predict_gaze_from_frame(frame)
        pyautogui.moveTo(x, y)

        cv2.imshow("Live Gaze Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



class SystemController:
    def __init__(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def restart(self):
        pass

    def handle_event(self, event):
        pass
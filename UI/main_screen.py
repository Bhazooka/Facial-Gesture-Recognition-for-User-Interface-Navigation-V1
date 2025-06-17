import tkinter as tk
from tkinter import messagebox
import cv2
import subprocess
import sys
import os

def open_config_screen_and_exit():
    from config_screen import ConfigScreen
    app = ConfigScreen()
    app.mainloop()

class MainUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Gesture UI Control System")
        self.geometry("800x600")
        self.resizable(False, False)
        self.configure(bg="white")

        self.title_label = tk.Label(self, text="Face Gesture UI Control System", font=("Arial", 20), bg="white")
        self.title_label.place(x=50, y=30)

        self.exit_btn = tk.Button(self, text="Exit", font=("Arial", 14), width=10, height=1, command=self.quit_application)
        self.exit_btn.place(x=650, y=30)

        self.config_btn = tk.Button(self, text="Configuration", font=("Arial", 14), width=15, height=1, command=self.open_config)
        self.config_btn.place(x=100, y=500)

        self.start_btn = tk.Button(self, text="Start Program", font=("Arial", 14), width=15, height=1, command=self.start_program)
        self.start_btn.place(x=550, y=500)

        self.camera_frame = tk.Frame(self, bg="black", bd=2, relief="solid")
        self.camera_frame.place(x=200, y=100, width=500, height=400)
        self.camera_label = tk.Label(self.camera_frame, bg="black")
        self.camera_label.pack(expand=True, fill="both")

        self.cap = None
        self.is_camera_running = False

        self.protocol("WM_DELETE_WINDOW", self.quit_application)

    def start_program(self):
        # Run main.py in a new process and close this window
        main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main.py'))
        python_exe = sys.executable
        try:
            self.quit_application()  # Close the main_screen window first
            subprocess.Popen([python_exe, main_py_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not start main program:\n{e}")

    def open_config(self):
        self.quit_application()
        open_config_screen_and_exit()

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera.")
                self.cap = None
                return

        self.is_camera_running = True
        self.update_camera_feed()

    def update_camera_feed(self):
        if self.is_camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                from PIL import Image, ImageTk
                img = Image.fromarray(cv2image)
                img = img.resize((self.camera_frame.winfo_width() - 4, self.camera_frame.winfo_height() - 4), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            self.after(10, self.update_camera_feed)

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
            self.camera_label.configure(image='')

    def quit_application(self):
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = MainUI()
    app.mainloop()
import tkinter as tk
from tkinter import messagebox

class ConfigScreen(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Configuration")
        self.geometry("800x600")
        self.resizable(False, False)
        self.configure(bg="white")

        # --- Top Left: Configuration Title ---
        self.config_title_label = tk.Label(self, text="Configuration", font=("Arial", 20), bg="white")
        self.config_title_label.place(x=50, y=30)

        # --- Right Side: Map Gesture Controls Section ---
        self.gesture_controls_frame = tk.Frame(self, bg="white", bd=2, relief="solid")
        self.gesture_controls_frame.place(x=450, y=150, width=300, height=300)

        self.map_title_label = tk.Label(self.gesture_controls_frame, text="Map Gesture Controls", font=("Arial", 16), bg="white")
        self.map_title_label.pack(pady=10)

        self.gestures = [
            "Open Mouth",
            "Right Blink",
            "Left Blink",
            "Left Eyebrow Raise",
            "Right Eyebrow Raise"
        ]
        self.entry_fields = {}

        self.grid_frame = tk.Frame(self.gesture_controls_frame, bg="white")
        self.grid_frame.pack(pady=10)

        for i, gesture in enumerate(self.gestures):
            label = tk.Label(self.grid_frame, text=gesture, font=("Arial", 12), bg="white")
            label.grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry = tk.Entry(self.grid_frame, font=("Arial", 12), width=15)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entry_fields[gesture] = entry

        self.cancel_btn = tk.Button(self, text="Cancel", font=("Arial", 14), width=10, height=1, command=self.cancel_changes)
        self.cancel_btn.place(x=500, y=500)

        self.save_btn = tk.Button(self, text="Save", font=("Arial", 14), width=10, height=1, command=self.save_changes)
        self.save_btn.place(x=650, y=500)

    def cancel_changes(self):
        messagebox.showinfo("Cancelled", "Changes have been cancelled.")
        self.destroy()

    def save_changes(self):
        saved_data = {
            gesture: entry.get() for gesture, entry in self.entry_fields.items()
        }
        messagebox.showinfo("Saved", f"Changes have been saved: {saved_data}")
        self.destroy()

if __name__ == "__main__":
    app = ConfigScreen()
    app.mainloop()


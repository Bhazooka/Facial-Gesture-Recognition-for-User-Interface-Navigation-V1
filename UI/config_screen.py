import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk # Required for image handling

class ConfigScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Configuration")
        self.geometry("800x600") # Adjust size based on the image's proportions
        self.resizable(False, False)
        self.configure(bg="white") # Set background to white

        # --- Top Left: Configuration Title ---
        self.config_title_label = tk.Label(self, text="Configuration", font=("Arial", 20), bg="white")
        self.config_title_label.place(x=50, y=30)

        # --- Center Left: Face Icon Placeholder ---
        try:
            # Load the face icon, resize it, and place it.
            # Make sure 'image_3c0cdf.png' (or a suitable face icon image)
            # is in the same directory as this script.
            self.face_img_original = Image.open("image_3c0cdf.png").resize((250, 250), Image.LANCZOS)
            self.face_img = ImageTk.PhotoImage(self.face_img_original)
            self.face_icon_label = tk.Label(self, image=self.face_img, bg="white")
            self.face_icon_label.place(x=100, y=200) # Position to the left
        except FileNotFoundError:
            self.face_icon_label = tk.Label(self, text="[Face Icon Placeholder]", font=("Arial", 16), bg="white", fg="gray")
            self.face_icon_label.place(x=100, y=200)
            messagebox.showwarning("Image Not Found", "image_3c0cdf.png not found. Please provide the image file.")

        # --- Right Side: Map Gesture Controls Section ---
        self.gesture_controls_frame = tk.Frame(self, bg="white", bd=2, relief="solid")
        self.gesture_controls_frame.place(x=450, y=150, width=300, height=300) # Position and size the frame

        self.map_title_label = tk.Label(self.gesture_controls_frame, text="Map Gesture Controls", font=("Arial", 16), bg="white")
        self.map_title_label.pack(pady=10)

        # Labels and Entry fields for gestures using grid
        self.gestures = [
            "Open Mouth",
            "Right Blink",
            "Left Blink",
            "Left Eyebrow Raise",
            "Right Eyebrow Raise"
        ]
        self.entry_fields = {} # Dictionary to store references to Entry widgets

        # Create a sub-frame for the grid to center it within gesture_controls_frame
        self.grid_frame = tk.Frame(self.gesture_controls_frame, bg="white")
        self.grid_frame.pack(pady=10)

        for i, gesture in enumerate(self.gestures):
            label = tk.Label(self.grid_frame, text=gesture, font=("Arial", 12), bg="white")
            label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

            entry = tk.Entry(self.grid_frame, width=20, font=("Arial", 12), relief="solid", bd=1)
            entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
            self.entry_fields[gesture] = entry

        # --- Bottom Buttons: Cancel and Save ---
        self.cancel_btn = tk.Button(self, text="Cancel", font=("Arial", 14), width=10, height=1, command=self.cancel_changes)
        self.cancel_btn.place(x=500, y=500)

        self.save_btn = tk.Button(self, text="Save", font=("Arial", 14), width=10, height=1, command=self.save_changes)
        self.save_btn.place(x=650, y=500)

    def cancel_changes(self):
        """Called when the Cancel button is pressed."""
        messagebox.showinfo("Cancelled", "Changes have been cancelled.")
        self.destroy() # Close the configuration window

    def save_changes(self):
        """Called when the Save button is pressed."""
        # Here you would retrieve the values from the entry fields
        # and save them (e.g., to a configuration file or database).
        saved_data = {
            gesture: entry.get() for gesture, entry in self.entry_fields.items()
        }
        messagebox.showinfo("Saved", f"Changes have been saved: {saved_data}")
        # Example of how to access a specific value:
        # open_mouth_control = self.entry_fields["Open Mouth"].get()
        self.destroy() # Close the configuration window

# Example of how to run this screen independently for testing:
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() # Hide the main Tkinter window for this example
    config_app = ConfigScreen(root)
    root.mainloop()


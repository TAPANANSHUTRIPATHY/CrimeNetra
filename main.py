import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
import os
from ultralytics import YOLO
from utils.helper import get_tk_image

# Constants
MODEL_PATH = 'models'
SUPPORTED_MODELS = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pt')]

class CrimeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crime Detection App")
        self.root.geometry("1000x600")
        self.root.configure(bg="#1e1e1e")

        self.model = None
        self.cap = None
        self.running = False

        self.setup_ui()

    def setup_ui(self):
        ttk.Label(self.root, text="Select Model:", background="#1e1e1e", foreground="white").place(x=20, y=20)
        self.model_combo = ttk.Combobox(self.root, values=SUPPORTED_MODELS, width=40)
        self.model_combo.place(x=120, y=20)
        if SUPPORTED_MODELS:
            self.model_combo.current(0)

        ttk.Button(self.root, text="Load Model", command=self.load_model).place(x=400, y=18)
        ttk.Button(self.root, text="Open Video File", command=self.open_video).place(x=500, y=18)
        ttk.Button(self.root, text="Use Webcam", command=self.use_webcam).place(x=640, y=18)
        ttk.Button(self.root, text="Stop", command=self.stop_detection).place(x=760, y=18)

        self.canvas = tk.Label(self.root, bg="black")
        self.canvas.place(x=20, y=70, width=960, height=500)

    def load_model(self):
        selected = self.model_combo.get()
        model_path = os.path.join(MODEL_PATH, selected)
        try:
            self.model = YOLO(model_path)
            messagebox.showinfo("Success", f"Model '{selected}' loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if path:
            self.start_detection(path)

    def use_webcam(self):
        self.start_detection(0)

    def start_detection(self, source):
        if not self.model:
            messagebox.showwarning("No Model", "Please load a model first!")
            return

        self.running = True
        self.cap = cv2.VideoCapture(source)
        threading.Thread(target=self.detect_loop, daemon=True).start()

    def detect_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)
            annotated_frame = results[0].plot()

            tk_img = get_tk_image(annotated_frame)
            self.canvas.configure(image=tk_img)
            self.canvas.image = tk_img

        if self.cap:
            self.cap.release()

    def stop_detection(self):
        self.running = False
        self.canvas.configure(image=None)
        self.canvas.image = None

if __name__ == "__main__":
    if not os.path.exists("models") or not SUPPORTED_MODELS:
        print("[ERROR] No models found in the 'models/' folder.")
        exit()

    root = tk.Tk()
    app = CrimeDetectionApp(root)
    root.mainloop()

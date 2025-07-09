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
        self.root.title("Crime Detection — Modern Crime Detector")
        self.root.geometry("1024x720")
        self.root.minsize(900, 600)
        self.root.configure(bg="#121212")

        self.model = None
        self.cap = None
        self.running = False
        self.conf_threshold = 0.5  # default sensitivity

        self.style = ttk.Style()
        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        self.style.theme_use('clam')

        self.style.configure('TLabel',
                             background='#121212',
                             foreground='#E0E0E0',
                             font=('Segoe UI', 11))

        self.style.configure('TButton',
                             background='#1F1F1F',
                             foreground='#FFFFFF',
                             font=('Segoe UI Semibold', 11),
                             borderwidth=0,
                             focusthickness=3,
                             focuscolor='none')
        self.style.map('TButton',
                       background=[('active', '#333333'), ('pressed', '#555555')])

        self.style.configure('TCombobox',
                             fieldbackground='#1F1F1F',
                             background='#1F1F1F',
                             foreground='#E0E0E0',
                             font=('Segoe UI', 11))

        self.style.configure('Horizontal.TScale',
                             troughcolor='#2A2A2A',
                             background='#4CAF50',
                             thickness=15)

        self.style.configure('TEntry',
                             fieldbackground='#1F1F1F',
                             foreground='#E0E0E0',
                             font=('Segoe UI', 11))

    def setup_ui(self):
        # Top control frame
        control_frame = ttk.Frame(self.root, style='TFrame')
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=15)

        # Model label + combobox
        ttk.Label(control_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.model_combo = ttk.Combobox(control_frame, values=SUPPORTED_MODELS, width=38, state='readonly')
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        if SUPPORTED_MODELS:
            self.model_combo.current(0)

        # Load model button
        self.load_btn = ttk.Button(control_frame, text="Load Model", command=self.load_model)
        self.load_btn.grid(row=0, column=2, padx=8)

        # Open video button
        self.open_btn = ttk.Button(control_frame, text="Open Video File", command=self.open_video)
        self.open_btn.grid(row=0, column=3, padx=8)

        # Webcam button
        self.webcam_btn = ttk.Button(control_frame, text="Use Webcam", command=self.use_webcam)
        self.webcam_btn.grid(row=0, column=4, padx=8)

        # Stop button
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_detection)
        self.stop_btn.grid(row=0, column=5, padx=8)

        # RTSP URL label + entry + button
        ttk.Label(control_frame, text="RTSP URL:").grid(row=1, column=0, padx=5, pady=10, sticky='w')
        self.rtsp_entry = ttk.Entry(control_frame, width=50)
        self.rtsp_entry.grid(row=1, column=1, columnspan=3, padx=5, pady=10, sticky='w')

        self.rtsp_btn = ttk.Button(control_frame, text="Start RTSP Stream", command=self.start_rtsp_stream)
        self.rtsp_btn.grid(row=1, column=4, padx=8)

        # Sensitivity slider frame
        sensitivity_frame = ttk.Frame(self.root)
        sensitivity_frame.pack(side=tk.TOP, fill=tk.X, padx=20)

        ttk.Label(sensitivity_frame, text="Sensitivity (Confidence Threshold):").pack(side=tk.LEFT, padx=5, pady=5)

        self.sensitivity_slider = ttk.Scale(sensitivity_frame, from_=0.1, to=1.0,
                                            value=self.conf_threshold,
                                            command=self.update_sensitivity,
                                            style='Horizontal.TScale',
                                            length=280)
        self.sensitivity_slider.pack(side=tk.LEFT, padx=10)

        self.sensitivity_value_label = ttk.Label(sensitivity_frame, text=f"{self.conf_threshold:.2f}")
        self.sensitivity_value_label.pack(side=tk.LEFT)

        # Video display canvas frame
        video_frame = ttk.Frame(self.root)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        self.canvas = tk.Label(video_frame, bg="#000000", bd=2, relief=tk.SUNKEN)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind('<Configure>', self.resize_canvas)

    def resize_canvas(self, event):
        # Placeholder for future responsive resizing
        pass

    def update_sensitivity(self, val):
        self.conf_threshold = round(float(val), 2)
        self.sensitivity_value_label.config(text=f"{self.conf_threshold:.2f}")

    def load_model(self):
        selected = self.model_combo.get()
        model_path = os.path.join(MODEL_PATH, selected)
        try:
            self.model = YOLO(model_path)
            messagebox.showinfo("Success", f"Model '{selected}' loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if path:
            self.start_detection(path)

    def use_webcam(self):
        self.start_detection(0)

    def start_rtsp_stream(self):
        url = self.rtsp_entry.get().strip()
        if not url:
            messagebox.showwarning("RTSP URL Missing", "Please enter a valid RTSP URL.")
            return
        self.start_detection(url)

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

            results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
            annotated_frame = results[0].plot()

            tk_img = get_tk_image(annotated_frame, size=(self.canvas.winfo_width(), self.canvas.winfo_height()))
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

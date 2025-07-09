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
        self.root.geometry("1100x720")
        self.root.minsize(1000, 650)
        self.root.configure(bg="#FFFFFF")  # white background

        self.model = None
        self.cap = None
        self.running = False
        self.conf_threshold = 0.5  # default sensitivity

        self.setup_ui()

    def setup_ui(self):
        # Custom fonts
        self.font_heading = ("Segoe UI Semibold", 14)
        self.font_normal = ("Segoe UI", 11)
        self.font_small = ("Segoe UI", 9)

        # Top control frame with padding & white bg
        control_frame = tk.Frame(self.root, bg="#FFFFFF", bd=0)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=25, pady=20)

        top_row = tk.Frame(control_frame, bg="#FFFFFF")
        top_row.pack(fill=tk.X, pady=(5,10))
        bottom_row = tk.Frame(control_frame, bg="#FFFFFF")
        bottom_row.pack(fill=tk.X, pady=(0,10))

        # Model label + combobox
        lbl_model = tk.Label(top_row, text="Select Model:", fg="#000000", bg="#FFFFFF", font=self.font_normal)
        lbl_model.grid(row=0, column=0, padx=(0,8), pady=2, sticky='w')

        self.model_combo = ttk.Combobox(top_row, values=SUPPORTED_MODELS, width=38, state='readonly',
                                        font=self.font_normal)
        self.model_combo.grid(row=0, column=1, padx=8, pady=2, sticky='w')
        if SUPPORTED_MODELS:
            self.model_combo.current(0)

        # Buttons styling
        btn_style = {
            "bg": "#F0F0F0",  # light gray background
            "fg": "#000000",  # black text
            "activebackground": "#D9D9D9",  # slightly darker on active
            "activeforeground": "#000000",
            "font": ("Segoe UI Semibold", 11),
            "bd": 0,
            "relief": tk.FLAT,
            "cursor": "hand2",
            "width": 14,
            "height": 1
        }

        self.load_btn = tk.Button(top_row, text="Load Model", command=self.load_model, **btn_style)
        self.load_btn.grid(row=0, column=2, padx=8, pady=2)

        self.open_btn = tk.Button(top_row, text="Open Video File", command=self.open_video, **btn_style)
        self.open_btn.grid(row=0, column=3, padx=8, pady=2)

        self.webcam_btn = tk.Button(top_row, text="Use Webcam", command=self.use_webcam, **btn_style)
        self.webcam_btn.grid(row=0, column=4, padx=8, pady=2)

        # For Stop button, use red background separately - define a new style dict without duplication
        stop_btn_style = btn_style.copy()
        stop_btn_style.update({
            "bg": "#F44336",            # Red
            "activebackground": "#E53935",
            "fg": "#FFFFFF",
            "activeforeground": "#FFFFFF",
        })

        self.stop_btn = tk.Button(top_row, text="Stop", command=self.stop_detection, **stop_btn_style)
        self.stop_btn.grid(row=0, column=5, padx=8, pady=2)

        # RTSP label + entry + button (bottom row)
        lbl_rtsp = tk.Label(bottom_row, text="RTSP URL:", fg="#000000", bg="#FFFFFF", font=self.font_normal)
        lbl_rtsp.grid(row=0, column=0, padx=(0,8), pady=6, sticky='w')

        self.rtsp_entry = tk.Entry(bottom_row, bg="#F7F7F7", fg="#000000", insertbackground="#000000", font=self.font_normal,
                                   relief=tk.FLAT, width=60)
        self.rtsp_entry.grid(row=0, column=1, padx=8, pady=6, sticky='w')

        self.rtsp_btn = tk.Button(bottom_row, text="Start RTSP Stream", command=self.start_rtsp_stream, **btn_style)
        self.rtsp_btn.grid(row=0, column=2, padx=8, pady=6)

        # Sensitivity slider frame with label
        sensitivity_frame = tk.Frame(self.root, bg="#FFFFFF")
        sensitivity_frame.pack(side=tk.TOP, fill=tk.X, padx=25)

        lbl_sensitivity = tk.Label(sensitivity_frame, text="Sensitivity (Confidence Threshold):", fg="#000000",
                                   bg="#FFFFFF", font=self.font_normal)
        lbl_sensitivity.pack(side=tk.LEFT, padx=(0,10), pady=10)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TScale',
                        troughcolor='#DDDDDD',
                        background='#4CAF50',
                        thickness=14,
                        sliderlength=20,
                        sliderthickness=20)
        style.configure('TLabel', background='#FFFFFF', foreground='#000000', font=self.font_normal)

        self.sensitivity_slider = ttk.Scale(sensitivity_frame, from_=0.1, to=1.0,
                                            value=self.conf_threshold,
                                            command=self.update_sensitivity,
                                            style='TScale',
                                            length=280)
        self.sensitivity_slider.pack(side=tk.LEFT)

        self.sensitivity_value_label = tk.Label(sensitivity_frame, text=f"{self.conf_threshold:.2f}", fg="#4CAF50",
                                                bg="#FFFFFF", font=("Segoe UI Semibold", 12))
        self.sensitivity_value_label.pack(side=tk.LEFT, padx=10)

        # Video display frame with white bg and black border
        video_frame = tk.Frame(self.root, bg="#FFFFFF", bd=2, relief=tk.SOLID)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)

        self.canvas = tk.Label(video_frame, bg="#000000", bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind('<Configure>', self.resize_canvas)

    def resize_canvas(self, event):
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

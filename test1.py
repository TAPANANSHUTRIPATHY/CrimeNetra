import tkinter as tk
from tkinter import ttk, filedialog
import threading
import cv2
import os
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO
from utils.helper import get_tk_image

# Constants
MODEL_PATH = 'models'
SUPPORTED_MODELS = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pt')]

class CrimeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crime Detection — Modern Crime Detector")
        self.root.geometry("1100x780")  # Extra height for message log
        self.root.minsize(1000, 700)
        self.root.configure(bg="#FFFFFF")  # White background

        self.model = None
        self.cap = None
        self.running = False
        self.conf_threshold = 0.5  # default confidence threshold

        self.setup_ui()

    def setup_ui(self):
        # Fonts
        self.font_normal = ("Segoe UI", 11)
        self.font_bold = ("Segoe UI Semibold", 11)

        # Top control frame (white background)
        control_frame = tk.Frame(self.root, bg="#FFFFFF", bd=0)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=25, pady=20)

        top_row = tk.Frame(control_frame, bg="#FFFFFF")
        top_row.pack(fill=tk.X, pady=(5,10))
        bottom_row = tk.Frame(control_frame, bg="#FFFFFF")
        bottom_row.pack(fill=tk.X, pady=(0,10))

        # Load and add logo image on the left
        try:
            logo_path = "logo.png"
            logo_img = Image.open(logo_path)
            logo_img = logo_img.resize((200, 100), Image.Resampling.LANCZOS)  # Resize here
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(top_row, image=self.logo_photo, bg="#FFFFFF")
            logo_label.grid(row=0, column=0, padx=(0,12), pady=2, sticky='w')
        except Exception as e:
            self.logo_photo = None
            print(f"[Warning] Could not load logo.png: {e}")

        # Model label + combobox, shifted one column right to col=1
        lbl_model = tk.Label(top_row, text="Select Model:", fg="#000000", bg="#FFFFFF", font=self.font_normal)
        lbl_model.grid(row=0, column=1, padx=(0,8), pady=2, sticky='w')

        self.model_combo = ttk.Combobox(top_row, values=SUPPORTED_MODELS, width=38, state='readonly',
                                        font=self.font_normal)
        self.model_combo.grid(row=0, column=2, padx=8, pady=2, sticky='w')
        if SUPPORTED_MODELS:
            self.model_combo.current(0)

        # Button styles
        btn_style = {
            "bg": "#F0F0F0",  # Light gray
            "fg": "#000000",  # Black text
            "activebackground": "#D9D9D9",
            "activeforeground": "#000000",
            "font": ("Segoe UI Semibold", 11),
            "bd": 0,
            "relief": tk.FLAT,
            "cursor": "hand2",
            "width": 14,
            "height": 1
        }

        self.load_btn = tk.Button(top_row, text="Load Model", command=self.load_model, **btn_style)
        self.load_btn.grid(row=0, column=3, padx=8, pady=2)

        self.open_btn = tk.Button(top_row, text="Open Video File", command=self.open_video, **btn_style)
        self.open_btn.grid(row=0, column=4, padx=8, pady=2)

        self.webcam_btn = tk.Button(top_row, text="Use Webcam", command=self.use_webcam, **btn_style)
        self.webcam_btn.grid(row=0, column=5, padx=8, pady=2)

        # Stop button with red style
        stop_btn_style = btn_style.copy()
        stop_btn_style.update({
            "bg": "#F44336",           # Red
            "activebackground": "#E53935",
            "fg": "#FFFFFF",
            "activeforeground": "#FFFFFF",
        })
        self.stop_btn = tk.Button(top_row, text="Stop", command=self.stop_detection, **stop_btn_style)
        self.stop_btn.grid(row=0, column=6, padx=8, pady=2)

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

        # Message log frame at the bottom
        log_frame = tk.Frame(self.root, bg="#FFFFFF", bd=2, relief=tk.SOLID)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=25, pady=(0,20), ipady=5)

        log_label = tk.Label(log_frame, text="Message Log:", bg="#FFFFFF", fg="#000000", font=self.font_bold)
        log_label.pack(anchor='w', padx=5, pady=(5,0))

        self.log_text = tk.Text(log_frame, height=8, bg="#F7F7F7", fg="#000000", font=self.font_normal,
                                state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0), pady=5)

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        self.log_text.config(yscrollcommand=scrollbar.set)

    def resize_canvas(self, event):
        pass

    def update_sensitivity(self, val):
        self.conf_threshold = round(float(val), 2)
        self.sensitivity_value_label.config(text=f"{self.conf_threshold:.2f}")

    def log_message(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}\n"
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def load_model(self):
        selected = self.model_combo.get()
        model_path = os.path.join(MODEL_PATH, selected)
        try:
            self.model = YOLO(model_path)
            self.log_message(f"Model '{selected}' loaded successfully!", "SUCCESS")
        except Exception as e:
            self.log_message(f"Failed to load model: {str(e)}", "ERROR")

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if path:
            self.start_detection(path)

    def use_webcam(self):
        self.start_detection(0)

    def start_rtsp_stream(self):
        url = self.rtsp_entry.get().strip()
        if not url:
            self.log_message("RTSP URL is missing. Please enter a valid RTSP URL.", "WARNING")
            return
        self.start_detection(url)

    def start_detection(self, source):
        if not self.model:
            self.log_message("No model loaded! Please load a model first.", "WARNING")
            return
        self.running = True
        self.cap = cv2.VideoCapture(source)
        self.log_message(f"Started detection on source: {source}", "INFO")
        threading.Thread(target=self.detect_loop, daemon=True).start()

    def detect_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.log_message("Video stream ended or failed to read frame.", "INFO")
                break

            results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
            annotated_frame = results[0].plot()

            tk_img = get_tk_image(annotated_frame, size=(self.canvas.winfo_width(), self.canvas.winfo_height()))
            self.canvas.configure(image=tk_img)
            self.canvas.image = tk_img

        if self.cap:
            self.cap.release()
            self.log_message("Video capture released.", "INFO")

    def stop_detection(self):
        if self.running:
            self.running = False
            self.log_message("Detection stopped.", "INFO")
        self.canvas.configure(image=None)
        self.canvas.image = None


if __name__ == "__main__":
    if not os.path.exists("models") or not SUPPORTED_MODELS:
        print("[ERROR] No models found in the 'models/' folder.")
        exit()

    root = tk.Tk()
    app = CrimeDetectionApp(root)
    root.mainloop()

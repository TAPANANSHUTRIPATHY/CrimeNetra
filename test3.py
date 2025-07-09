import tkinter as tk
from tkinter import ttk, filedialog
import threading
import cv2
import os
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO
from utils.helper import get_tk_image

# For voice abuse detection
import speech_recognition as sr
from transformers import pipeline
import nltk
import re

# Setup
MODEL_PATH = 'models'
SUPPORTED_MODELS = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pt')] + ["Voice Abuse Detection"]
CAPTURE_FOLDER = "captures"
RECORD_FOLDER = "records"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)
os.makedirs(RECORD_FOLDER, exist_ok=True)
nltk.download('punkt')

class CrimeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crime Detection — Modern Crime Detector")
        self.root.geometry("1280x880")
        self.root.configure(bg="#FFFFFF")

        self.model = None
        self.cap = None
        self.running = False
        self.conf_threshold = 0.5
        self.recording = False
        self.video_writer = None
        self.current_frame = None
        self.record_path = ""

        self.voice_classifier = None
        self.sr_recognizer = sr.Recognizer()

        self.setup_ui()

    def setup_ui(self):
        self.font_normal = ("Segoe UI", 11)
        self.font_bold = ("Segoe UI Semibold", 11)

        control_frame = tk.Frame(self.root, bg="#FFFFFF")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=25, pady=20)

        top_row = tk.Frame(control_frame, bg="#FFFFFF")
        top_row.pack(fill=tk.X, pady=(5, 10))
        bottom_row = tk.Frame(control_frame, bg="#FFFFFF")
        bottom_row.pack(fill=tk.X, pady=(0, 10))

        try:
            logo_img = Image.open("logo.png").resize((200, 100), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            tk.Label(top_row, image=self.logo_photo, bg="#FFFFFF").grid(row=0, column=0, padx=(0, 12), pady=2, sticky='w')
        except Exception as e:
            print(f"[Warning] Could not load logo.png: {e}")

        tk.Label(top_row, text="Select Model:", fg="#000000", bg="#FFFFFF", font=self.font_normal)\
            .grid(row=0, column=1, padx=(0, 8), pady=2, sticky='w')

        self.model_combo = ttk.Combobox(top_row, values=SUPPORTED_MODELS, width=38, state='readonly', font=self.font_normal)
        self.model_combo.grid(row=0, column=2, padx=8, pady=2, sticky='w')
        if SUPPORTED_MODELS:
            self.model_combo.current(0)

        btn_style = {
            "bg": "#F0F0F0", "fg": "#000000", "activebackground": "#D9D9D9",
            "activeforeground": "#000000", "font": self.font_bold,
            "bd": 0, "relief": tk.FLAT, "cursor": "hand2", "width": 14, "height": 1
        }

        tk.Button(top_row, text="Load Model", command=self.load_model, **btn_style).grid(row=0, column=3, padx=8, pady=2)
        tk.Button(top_row, text="Open Video File", command=self.open_video, **btn_style).grid(row=0, column=4, padx=8, pady=2)
        tk.Button(top_row, text="Use Webcam", command=self.use_webcam, **btn_style).grid(row=0, column=5, padx=8, pady=2)

        stop_btn_style = btn_style.copy()
        stop_btn_style.update({"bg": "#F44336", "activebackground": "#E53935", "fg": "#FFFFFF", "activeforeground": "#FFFFFF"})
        tk.Button(top_row, text="Stop", command=self.stop_detection, **stop_btn_style).grid(row=0, column=6, padx=8, pady=2)

        tk.Label(bottom_row, text="RTSP URL:", fg="#000000", bg="#FFFFFF", font=self.font_normal).grid(row=0, column=0, padx=(0, 8), pady=6, sticky='w')
        self.rtsp_entry = tk.Entry(bottom_row, bg="#F7F7F7", fg="#000000", insertbackground="#000000", font=self.font_normal, relief=tk.FLAT, width=60)
        self.rtsp_entry.grid(row=0, column=1, padx=8, pady=6, sticky='w')
        tk.Button(bottom_row, text="Start RTSP Stream", command=self.start_rtsp_stream, **btn_style).grid(row=0, column=2, padx=8, pady=6)

        # Dialog box for detection result
        self.dialog_frame = tk.Frame(self.root, bg="#FFF8DC", bd=1, relief=tk.SOLID)
        self.dialog_frame.pack(fill=tk.X, padx=25, pady=(0, 10))
        self.dialog_label = tk.Label(self.dialog_frame, text="System ready.", fg="#000000", bg="#FFF8DC",
                                     font=("Segoe UI", 11, "bold"), anchor="w", justify="left")
        self.dialog_label.pack(fill=tk.X, padx=10, pady=5)

        sensitivity_frame = tk.Frame(self.root, bg="#FFFFFF")
        sensitivity_frame.pack(side=tk.TOP, fill=tk.X, padx=25)
        tk.Label(sensitivity_frame, text="Sensitivity (Confidence Threshold):", fg="#000000", bg="#FFFFFF", font=self.font_normal)\
            .pack(side=tk.LEFT, padx=(0, 10), pady=10)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TScale', troughcolor='#DDDDDD', background='#4CAF50', thickness=14, sliderlength=20)
        self.sensitivity_slider = ttk.Scale(sensitivity_frame, from_=0.1, to=1.0, value=self.conf_threshold, command=self.update_sensitivity, style='TScale', length=280)
        self.sensitivity_slider.pack(side=tk.LEFT)
        self.sensitivity_value_label = tk.Label(sensitivity_frame, text=f"{self.conf_threshold:.2f}", fg="#4CAF50", bg="#FFFFFF", font=self.font_bold)
        self.sensitivity_value_label.pack(side=tk.LEFT, padx=10)

        record_row = tk.Frame(self.root, bg="#FFFFFF")
        record_row.pack(fill=tk.X, padx=25, pady=(0, 10))
        tk.Button(record_row, text="Capture Frame", command=self.capture_frame, **btn_style).pack(side=tk.LEFT, padx=8)
        self.record_btn = tk.Button(record_row, text="Start Recording", command=self.toggle_recording, **btn_style)
        self.record_btn.pack(side=tk.LEFT, padx=8)
        tk.Button(record_row, text="Start Voice Abuse", command=lambda: threading.Thread(target=self.listen_voice_abuse).start(), **btn_style).pack(side=tk.LEFT, padx=8)

        video_frame = tk.Frame(self.root, bg="#FFFFFF", bd=2, relief=tk.SOLID, width=1280, height=720)
        video_frame.pack(padx=25, pady=20)
        video_frame.pack_propagate(False)
        self.canvas = tk.Label(video_frame, bg="#000000", bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self.resize_canvas)

        # Message log
        log_frame = tk.Frame(self.root, bg="#FFFFFF", bd=2, relief=tk.SOLID)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=25, pady=(0, 20), ipady=5)
        tk.Label(log_frame, text="Message Log:", bg="#FFFFFF", fg="#000000", font=self.font_bold)\
            .pack(anchor='w', padx=5, pady=(5, 0))

        self.log_text = tk.Text(log_frame, height=8, bg="#F7F7F7", fg="#000000", font=self.font_normal,
                                state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0), pady=5)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        self.log_text.config(yscrollcommand=scrollbar.set)

    def resize_canvas(self, event): pass

    def update_sensitivity(self, val):
        self.conf_threshold = round(float(val), 2)
        self.sensitivity_value_label.config(text=f"{self.conf_threshold:.2f}")

    def log_message(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {level}: {message}\n"

        color_map = {
            "INFO": "#2196F3",
            "SUCCESS": "#4CAF50",
            "WARNING": "#FF9800",
            "ERROR": "#F44336"
        }
        self.dialog_label.config(text=message, fg=color_map.get(level, "#000000"))

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def load_model(self):
        selected = self.model_combo.get()
        if selected == "Voice Abuse Detection":
            self.voice_classifier = pipeline("text-classification", model="unitary/toxic-bert")
            self.log_message("Voice Abuse Detection model loaded.", "SUCCESS")
        else:
            try:
                self.model = YOLO(os.path.join(MODEL_PATH, selected))
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
            self.log_message("RTSP URL is missing.", "WARNING")
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
                self.log_message("Stream ended or failed to read frame.", "INFO")
                break

            self.current_frame = frame.copy()
            results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
            annotated = results[0].plot()

            try:
                classes = results[0].names
                detected_classes = [classes[int(cls)] for cls in results[0].boxes.cls]
                if 'theft' in detected_classes:
                    self.log_message("⚠️ ALERT: Theft Detected!", "WARNING")
            except Exception as e:
                print(f"[Detection Warning] {e}")

            if self.recording and self.video_writer:
                self.video_writer.write(annotated)

            tk_img = get_tk_image(annotated, size=(self.canvas.winfo_width(), self.canvas.winfo_height()))
            self.canvas.configure(image=tk_img)
            self.canvas.image = tk_img

        if self.cap:
            self.cap.release()
            self.log_message("Video capture released.", "INFO")
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            self.record_btn.config(text="Start Recording")
            self.log_message(f"Recording saved: {self.record_path}", "SUCCESS")

    def stop_detection(self):
        if self.running:
            self.running = False
            self.log_message("Detection stopped.", "INFO")
        self.canvas.configure(image=None)
        self.canvas.image = None

    def capture_frame(self):
        if self.current_frame is None:
            self.log_message("No frame to capture!", "WARNING")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_combo.get().replace(".pt", "")
        filename = f"{model_name}_capture_{timestamp}.jpg"
        path = os.path.join(CAPTURE_FOLDER, filename)
        cv2.imwrite(path, self.current_frame)
        self.log_message(f"Captured frame saved: {path}", "SUCCESS")

    def toggle_recording(self):
        if not self.cap or not self.cap.isOpened():
            self.log_message("No active source to record!", "WARNING")
            return

        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_combo.get().replace(".pt", "")
            filename = f"{model_name}_record_{timestamp}.avi"
            self.record_path = os.path.join(RECORD_FOLDER, filename)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(self.record_path, fourcc, fps, (width, height))
            self.recording = True
            self.record_btn.config(text="Stop Recording")
            self.log_message(f"Recording started: {self.record_path}", "INFO")
        else:
            self.recording = False
            self.record_btn.config(text="Start Recording")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.log_message(f"Recording stopped: {self.record_path}", "SUCCESS")

    def listen_voice_abuse(self):
        if not self.voice_classifier:
            self.log_message("Load 'Voice Abuse Detection' model first.", "WARNING")
            return
        self.log_message("🎤 Listening for abusive language...", "INFO")
        with sr.Microphone() as source:
            self.sr_recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.sr_recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = self.sr_recognizer.recognize_google(audio).lower()
                self.log_message(f"📝 Recognized: {text}", "INFO")

                cleaned = re.sub(r'\s+', ' ', text.strip())
                result = self.voice_classifier(cleaned)[0]
                label = result['label']
                score = result['score']

                if label in ['toxic', 'insult', 'threat'] and score > 0.7:
                    self.log_message("⚠️ Abusive Language Detected!", "WARNING")
                else:
                    self.log_message("✅ No abuse detected.", "SUCCESS")

            except sr.UnknownValueError:
                self.log_message("❌ Could not understand audio.", "ERROR")
            except sr.WaitTimeoutError:
                self.log_message("⏳ Timeout: No voice detected.", "WARNING")
            except Exception as e:
                self.log_message(f"Error: {str(e)}", "ERROR")

if __name__ == "__main__":
    if not os.path.exists("models") or not SUPPORTED_MODELS:
        print("[ERROR] No models found in the 'models/' folder.")
        exit()

    root = tk.Tk()
    app = CrimeDetectionApp(root)
    root.mainloop()

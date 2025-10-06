import math
import os
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, filedialog, messagebox
import cv2
import mediapipe as mp
import nltk
import speech_recognition as sr
import winsound  # For playing alert.wav on Windows; for Linux/Mac, different method needed
from PIL import Image, ImageTk
from transformers import pipeline
from ultralytics import YOLO

from utils.helper import get_tk_image  # Your helper to convert cv2 to Tk image

# Setup
MODEL_PATH = 'models'
SUPPORTED_MODELS = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pt')]
# Add our special modes:
SUPPORTED_MODES = SUPPORTED_MODELS + ["Voice Abuse Detection", "Harassment Detection"]

CAPTURE_FOLDER = "captures"
RECORD_FOLDER = "records"
ALERT_SOUND = "alert.wav"

os.makedirs(CAPTURE_FOLDER, exist_ok=True)
os.makedirs(RECORD_FOLDER, exist_ok=True)

nltk.download('punkt')


class CrimeHarassmentApp:
    def __init__(self, root):
        self.logo_photo = None
        self.root = root
        self.root.title("Crime Netra - Crime and Harassment Detection System")
        self.root.geometry("1280x880")
        self.root.configure(bg="#FFFFFF")

        # For YOLO model crime detection
        self.model = None
        self.cap = None
        self.running = False
        self.conf_threshold = 0.5
        self.recording = False
        self.video_writer = None
        self.current_frame = None
        self.record_path = ""

        # For Voice Abuse Detection
        self.voice_classifier = None
        self.sr_recognizer = sr.Recognizer()

        # For Harassment Detection (MediaPipe)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.is_analyzing_harassment = False

        # Harassment detection vars
        self.TOUCH_DISTANCE = 50
        self.STALKING_DISTANCE = 100
        self.STALKING_TIME = 3
        self.INAPPROPRIATE_TOUCH_FRAMES = 10

        self.saved_alert_frames = []

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
            logo_img = Image.open("logo1.png").resize((200, 100), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            tk.Label(top_row, image=self.logo_photo, bg="#FFFFFF").grid(row=0, column=0, padx=(0, 12), pady=2, sticky='w')
        except Exception as e:
            print(f"[Warning] Could not load logo.png: {e}")
            
        try:
           logo1_img = Image.open("logo.png").resize((200, 100), Image.Resampling.LANCZOS)
           self.logo1_photo = ImageTk.PhotoImage(logo1_img)
           # Place it in column 7 (assuming columns 1-6 are used for controls), sticky to east/right
           tk.Label(top_row, image=self.logo1_photo, bg="#FFFFFF").grid(row=0, column=7, padx=(12, 0), pady=2, sticky='e')
        except Exception as e:
           print(f"[Warning] Could not load logo1.png: {e}")


        tk.Label(top_row, text="Select Mode:", fg="#000000", bg="#FFFFFF", font=self.font_normal)\
            .grid(row=0, column=1, padx=(0, 8), pady=2, sticky='w')

        self.model_combo = ttk.Combobox(top_row, values=SUPPORTED_MODES, width=38, state='readonly', font=self.font_normal)
        self.model_combo.grid(row=0, column=2, padx=8, pady=2, sticky='w')
        if SUPPORTED_MODES:
            self.model_combo.current(0)

        btn_style = {
            "bg": "#F0F0F0", "fg": "#000000", "activebackground": "#D9D9D9",
            "activeforeground": "#000000", "font": self.font_bold,
            "bd": 0, "relief": tk.FLAT, "cursor": "hand2", "width": 14, "height": 1
        }

        tk.Button(top_row, text="Load Mode", command=self.load_mode, **btn_style).grid(row=0, column=3, padx=8, pady=2)
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

    def resize_canvas(self, event): 
        pass

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

    def load_mode(self):
        selected = self.model_combo.get()
        if selected == "Voice Abuse Detection":
            self.voice_classifier = pipeline("text-classification", model="unitary/toxic-bert")
            self.log_message("Voice Abuse Detection model loaded.", "SUCCESS")
            self.model = None
        elif selected == "Harassment Detection":
            self.log_message("Harassment Detection mode selected.", "SUCCESS")
            self.model = None
        else:
            try:
                self.model = YOLO(os.path.join(MODEL_PATH, selected))
                self.log_message(f"Model '{selected}' loaded successfully!", "SUCCESS")
            except Exception as e:
                self.log_message(f"Failed to load model: {str(e)}", "ERROR")

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if path:
            if self.model_combo.get() == "Harassment Detection":
                self.start_harassment_analysis(path)
            else:
                self.start_detection(path)

    def use_webcam(self):
        if self.model_combo.get() == "Harassment Detection":
            messagebox.showinfo("Info", "Harassment Detection currently supports only video files.")
        else:
            self.start_detection(0)

    def start_rtsp_stream(self):
        url = self.rtsp_entry.get().strip()
        if not url:
            self.log_message("RTSP URL is missing.", "WARNING")
            return
        if self.model_combo.get() == "Harassment Detection":
            messagebox.showinfo("Info", "Harassment Detection does not support RTSP streams.")
            return
        self.start_detection(url)

    # --- Crime Detection / YOLO ---

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

    # --- Harassment Detection ---

    def start_harassment_analysis(self, video_path):
        if self.is_analyzing_harassment:
            self.log_message("Harassment analysis already running.", "WARNING")
            return

        self.is_analyzing_harassment = True
        self.saved_alert_frames.clear()
        self.log_message(f"Started harassment analysis on {video_path}", "INFO")
        threading.Thread(target=self.analyze_harassment_video, args=(video_path,), daemon=True).start()

 # --- Harassment Detection ---

    def analyze_harassment_video(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            stalking_alert_triggered = False
            inappropriate_alert_triggered = False
            close_start_time = None
            inappropriate_touch_count = 0

            # Create folder for harassment pictures if not exists
            harassment_folder = "harassment_pictures"
            os.makedirs(harassment_folder, exist_ok=True)
            self.log_message(f"Harassment pictures folder: {os.path.abspath(harassment_folder)}", "INFO")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while cap.isOpened() and self.is_analyzing_harassment:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % 30 == 0:
                    progress_percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    self.root.after(0, lambda: self.log_message(f"Processing frame {frame_count}/{total_frames} ({progress_percent:.1f}%)", "INFO"))

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb)

                if results.pose_landmarks:
                    self.root.after(0, lambda: self.log_message("Evaluating pose for harassment...", "INFO"))
                    self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    lm = results.pose_landmarks.landmark

                    def to_pixel(point):
                        return int(point.x * w), int(point.y * h)

                    lw = to_pixel(lm[self.mp_pose.PoseLandmark.LEFT_WRIST])
                    rw = to_pixel(lm[self.mp_pose.PoseLandmark.RIGHT_WRIST])
                    lh = to_pixel(lm[self.mp_pose.PoseLandmark.LEFT_HIP])
                    rh = to_pixel(lm[self.mp_pose.PoseLandmark.RIGHT_HIP])
                    nose = to_pixel(lm[self.mp_pose.PoseLandmark.NOSE])
                    left_shoulder = to_pixel(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
                    right_shoulder = to_pixel(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])

                    hip_center = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)

                    lw_to_hip_dist = math.dist(lw, hip_center)
                    rw_to_hip_dist = math.dist(rw, hip_center)

                    torso_height = abs(left_shoulder[1] - hip_center[1])
                    lower_torso_threshold = hip_center[1] - (torso_height * 0.2)

                    hand_near_private_area = (lw_to_hip_dist < self.TOUCH_DISTANCE or rw_to_hip_dist < self.TOUCH_DISTANCE)
                    hands_in_lower_region = (lw[1] > lower_torso_threshold or rw[1] > lower_torso_threshold)

                    hands_crossed = (lw[0] > hip_center[0] and rw[0] < hip_center[0]) or \
                                    (lw[0] < hip_center[0] and rw[0] > hip_center[0])

                    inappropriate_gesture = (
                        hand_near_private_area and
                        hands_in_lower_region and
                        (hands_crossed or (lw_to_hip_dist < self.TOUCH_DISTANCE and rw_to_hip_dist < self.TOUCH_DISTANCE))
                    )

                    # Immediate detection and saving for inappropriate touch
                    if inappropriate_gesture and not inappropriate_alert_triggered:
                        inappropriate_alert_triggered = True
                        self.play_alert_sound()
                        self.root.after(0, lambda: self.log_message("\ud83d\udea8 ALERT: Inappropriate touch detected!", "WARNING"))
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"inappropriate_touch_{timestamp}.jpg"
                            filepath = os.path.join(harassment_folder, filename)
                            cv2.imwrite(filepath, frame)
                            self.log_message(f"Saved inappropriate touch frame immediately: {filepath}", "SUCCESS")
                        except Exception as e:
                            self.log_message(f"Error saving inappropriate touch frame: {e}", "ERROR")

                    # Reset inappropriate_touch_count since we're saving immediately
                    else:
                        inappropriate_touch_count = 0

                    stalk_dist = math.dist(nose, rh)

                    # Immediate detection and saving for stalking
                    if stalk_dist < self.STALKING_DISTANCE and not stalking_alert_triggered:
                        stalking_alert_triggered = True
                        self.play_alert_sound()
                        self.root.after(0, lambda: self.log_message("\ud83d\udea8 ALERT: Stalking behavior detected!", "WARNING"))
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"stalking_{timestamp}.jpg"
                            filepath = os.path.join(harassment_folder, filename)
                            cv2.imwrite(filepath, frame)
                            self.log_message(f"Saved stalking frame immediately: {filepath}", "SUCCESS")
                        except Exception as e:
                            self.log_message(f"Error saving stalking frame: {e}", "ERROR")
                    else:
                        close_start_time = None

                tk_img = get_tk_image(frame, size=(self.canvas.winfo_width(), self.canvas.winfo_height()))
                self.root.after(0, lambda img=tk_img: self.canvas.configure(image=img))
                self.canvas.image = tk_img

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            self.is_analyzing_harassment = False

            self.root.after(0, lambda: self.log_message("Harassment analysis completed.", "SUCCESS"))

        except Exception as e:
            self.log_message(f"Harassment analysis error: {e}", "ERROR")

    def save_alert_frame(self, frame, alert_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{alert_type}_{timestamp}.jpg"
        filepath = os.path.join(CAPTURE_FOLDER, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def play_alert_sound(self):
        try:
            winsound.PlaySound(ALERT_SOUND, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Error playing alert sound: {e}")

    def show_harassment_alerts(self, alert_images):
        # Popup window showing saved alert images
        win = tk.Toplevel(self.root)
        win.title("Harassment Alerts")
        win.geometry("600x600")
        win.configure(bg="#FFFFFF")

        lbl = tk.Label(win, text="Detected Harassment Alerts", font=("Segoe UI Semibold", 14), bg="#FFFFFF")
        lbl.pack(pady=10)

        for img_path in alert_images:
            try:
                img = Image.open(img_path)
                img.thumbnail((550, 350))
                photo = ImageTk.PhotoImage(img)
                panel = tk.Label(win, image=photo)
                panel.image = photo
                panel.pack(pady=10)
            except Exception as e:
                print(f"Error loading alert image: {e}")

    # --- Voice Abuse Detection ---

    def listen_voice_abuse(self):
        if self.model_combo.get() != "Voice Abuse Detection":
            self.log_message("Please select 'Voice Abuse Detection' mode to use this feature.", "WARNING")
            return

        self.log_message("Listening for voice abuse detection... Please speak.", "INFO")
        with sr.Microphone() as source:
            audio = self.sr_recognizer.listen(source, phrase_time_limit=5)

        try:
            text = self.sr_recognizer.recognize_google(audio)
            self.log_message(f"Recognized speech: {text}", "INFO")
            result = self.voice_classifier(text)[0]
            label = result['label']
            score = result['score']
            if label == 'toxic' and score > 0.7:
                self.log_message("🚨 ALERT: Voice abuse detected!", "WARNING")
                self.play_alert_sound()
            else:
                self.log_message("No abuse detected in voice.", "SUCCESS")
        except Exception as e:
            self.log_message(f"Voice recognition error: {e}", "ERROR")

    def capture_frame(self):
        if self.current_frame is None:
            self.log_message("No frame to capture!", "WARNING")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(CAPTURE_FOLDER, filename)
        cv2.imwrite(filepath, self.current_frame)
        self.log_message(f"Frame captured: {filename}", "SUCCESS")

    def toggle_recording(self):
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.config(text="Start Recording")
            self.log_message(f"Stopped recording. Saved: {self.record_path}", "SUCCESS")
        else:
            if self.current_frame is None:
                self.log_message("No video feed to record!", "WARNING")
                return
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.record_path = os.path.join(RECORD_FOLDER, f"recording_{timestamp}.mp4")
            height, width = self.current_frame.shape[:2]
            self.video_writer = cv2.VideoWriter(self.record_path, fourcc, 20.0, (width, height))
            self.recording = True
            self.record_btn.config(text="Stop Recording")
            self.log_message("Started recording video.", "INFO")

    def stop_detection(self):
        self.running = False
        self.is_analyzing_harassment = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.record_btn.config(text="Start Recording")
        self.canvas.configure(image=None)
        self.canvas.image = None
        self.log_message("Detection stopped.", "INFO")


if __name__ == "__main__":
    root = tk.Tk()
    app = CrimeHarassmentApp(root)
    root.mainloop()

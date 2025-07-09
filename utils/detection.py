from ultralytics import YOLO
import cv2

def run_live_detection(model_path, stream_source, window_name="Detection"):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open stream: {stream_source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to grab frame.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

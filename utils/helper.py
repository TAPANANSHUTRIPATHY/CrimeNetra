import cv2
from PIL import Image, ImageTk

def get_tk_image(frame, size=(640, 480)):
    frame = cv2.resize(frame, size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    return ImageTk.PhotoImage(img)

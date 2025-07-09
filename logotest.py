import tkinter as tk
from PIL import Image, ImageTk
import os

print("Current working directory:", os.getcwd())
print("Files:", os.listdir())

root = tk.Tk()
root.geometry("200x100")

try:
    logo_img = Image.open("logo.png")
    print(f"Image size before resize: {logo_img.size}")
    logo_img = logo_img.resize((200, 100), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_img)
    label = tk.Label(root, image=logo_photo)
    label.image = logo_photo  # keep reference
    label.pack()
except Exception as e:
    print("Error loading logo.png:", e)

root.mainloop()

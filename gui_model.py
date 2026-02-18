import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import os

DEVICE = "cpu"
NUM_CLASSES = 10

MODEL_PATH = r"C:\Users\Atieksh\Documents\h\model.pth"

def get_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )

model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        return None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256))

    img_norm = img_resized / 255.0
    tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.argmax(pred, dim=1).squeeze().numpy()

    color_mask = np.zeros((256, 256, 3), dtype=np.uint8)

    colors = [
        (0,0,0), (0,255,0), (255,0,0), (0,0,255), (255,255,0),
        (255,0,255), (0,255,255), (128,128,0), (128,0,128), (0,128,128)
    ]

    for i in range(NUM_CLASSES):
        color_mask[pred == i] = colors[i]

    return img, color_mask

def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    original, predicted = predict_image(file_path)

    if original is None:
        return

    original = Image.fromarray(original).resize((300, 300))
    predicted = Image.fromarray(predicted).resize((300, 300))

    original_tk = ImageTk.PhotoImage(original)
    predicted_tk = ImageTk.PhotoImage(predicted)

    panel_original.config(image=original_tk)
    panel_original.image = original_tk

    panel_pred.config(image=predicted_tk)
    panel_pred.image = predicted_tk

root = tk.Tk()
root.title("Offroad AI Segmentation")

btn = tk.Button(root, text="Select Image", command=open_image)
btn.pack(pady=10)

panel_original = tk.Label(root)
panel_original.pack(side="left", padx=10)

panel_pred = tk.Label(root)
panel_pred.pack(side="right", padx=10)

root.mainloop()

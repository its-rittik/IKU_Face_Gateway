import os
import numpy as np
import cv2
from PIL import Image, ImageFilter
from tensorflow.keras.preprocessing import image
from config import IMAGE_INPUT_SIZE

def equalize_histogram(pil_img):
    img = np.array(pil_img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_eq)

def preprocess_image(pil_img):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMAGE_INPUT_SIZE)
    pil_img = equalize_histogram(pil_img)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return pil_img

def predict_with_tta(pil_img, model):
    base = image.img_to_array(pil_img) / 255.0
    base = np.expand_dims(base, axis=0)

    flipped = np.flip(base, axis=2)
    bright = np.clip(base * 1.1, 0, 1)

    preds = [
        model.predict(base)[0][0],
        model.predict(flipped)[0][0],
        model.predict(bright)[0][0]
    ]
    return np.mean(preds)

def process_images(images, model, folder_path):
    results = []
    os.makedirs(folder_path, exist_ok=True)

    for i, img_file in enumerate(images):
        try:
            pil_img = Image.open(img_file.stream)
            pil_img = preprocess_image(pil_img)
            prediction = predict_with_tta(pil_img, model)
            label = "REAL" if prediction >= 0.5 else "FAKE"
            results.append(label)
        except Exception:
            results.append("FAKE")
    return results
import hashlib

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

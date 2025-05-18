import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFilter
import cv2

# CONFIGURATION
local_model_path = r"D:\IKU\Deep Leaning\LAB\Project\IKU-Face-Gateway\image_detaction_model.h5"
test_folder = r"D:\IKU\Deep Leaning\LAB\Project\IKU-Face-Gateway\uploads\76eec7e9-1dc6-4553-a886-be7355ee05ee\images"
keras_input_size = (128, 128)

# Load model
model = load_model(local_model_path)

# Histogram Equalization
def equalize_histogram(pil_img):
    img = np.array(pil_img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_eq)

# Test Time Augmentation (TTA)
def predict_with_tta(img_array):
    preds = []
    preds.append(model.predict(img_array)[0][0])  # original
    preds.append(model.predict(np.flip(img_array, axis=2))[0][0])  # horizontal flip
    bright = np.clip(img_array * 1.1, 0, 1)
    preds.append(model.predict(bright)[0][0])  # slight brightness
    return np.mean(preds)
real = 0 
fake = 0
# Run predictions
for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            file_path = os.path.join(test_folder, filename)
            img = Image.open(file_path).convert('RGB')
            img = img.resize(keras_input_size)
            img = equalize_histogram(img)
            img = img.filter(ImageFilter.GaussianBlur(radius=0.3))

            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = predict_with_tta(img_array)
            label = "REAL" if prediction >= 0.5 else "FAKE"
            if label == "REAL":
                real += 1
            else:
                fake += 1
            print(f"{filename} → my model: {label} (score: {prediction:.4f})")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
print(f"Total REAL: {real}, Total FAKE: {fake}")
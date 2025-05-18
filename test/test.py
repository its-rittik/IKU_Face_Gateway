import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFilter
import cv2

# CONFIGURATION
model_path = r"D:\IKU\Deep Leaning\LAB\Project\IKU-Face-Gateway\deepfake_model.h5"
test_folder = r"D:\IKU\Deep Leaning\LAB\Project\IKU-Face-Gateway\uploads\ad976450-7613-400f-9284-e889ab0ecf3c\images"
input_size = (224, 224)
 # Match Test 1 model's input

# Load trained model
model = load_model(model_path)

# Histogram Equalization
def equalize_histogram(pil_img):
    img = np.array(pil_img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_eq)

# Preprocessing for a single image
def preprocess_image(pil_img):
    pil_img = pil_img.convert('RGB')
    pil_img = pil_img.resize(input_size)
    pil_img = equalize_histogram(pil_img)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return pil_img

# Test Time Augmentation (like Test 1)
def predict_with_tta(pil_img):
    base = image.img_to_array(pil_img) / 255.0
    base = np.expand_dims(base, axis=0)

    flipped = np.flip(base, axis=2)  # horizontal flip
    bright = np.clip(base * 1.1, 0, 1)  # +10% brightness

    preds = [
        model.predict(base)[0][0],
        model.predict(flipped)[0][0],
        model.predict(bright)[0][0]
    ]
    return np.mean(preds)

# Counters
real_count = 0
fake_count = 0

# Run predictions
for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            file_path = os.path.join(test_folder, filename)
            pil_img = Image.open(file_path)
            pil_img = preprocess_image(pil_img)

            prediction = predict_with_tta(pil_img)
            label = "REAL" if prediction >= 0.5 else "FAKE"

            if label == "REAL":
                real_count += 1
            else:
                fake_count += 1

            print(f"{filename} → Prediction: {label} (score: {prediction:.4f})")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# Summary
print(f"\nTotal REAL: {real_count}, Total FAKE: {fake_count}")

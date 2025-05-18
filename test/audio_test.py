import os
import subprocess
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

# === Paths ===
MODEL_PATH = r"D:\IKU\Deep Leaning\LAB\Project\IKU-Face-Gateway\cnn_spectrogram_model.h5"
WEBM_INPUT = r"D:\IKU\Deep Leaning\LAB\Project\IKU-Face-Gateway\uploads\d37fec2b-1afb-426f-ab99-ec3a17ebe75f\audio.webm"
TEMP_WAV = "temp_audio.wav"
TEMP_IMG = "temp_spec.png"

# === Step 1: Convert .webm to .wav using FFmpeg ===
def convert_webm_to_wav_ffmpeg(webm_path, wav_path):
    command = [
        'ffmpeg', '-y', '-i', webm_path,
        '-ar', '16000', '-ac', '1', wav_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Converted using ffmpeg: {wav_path}")

# === Step 2: Create mel spectrogram image ===
def create_spectrogram(wav_path, image_path):
    y, sr = librosa.load(wav_path, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_DB, sr=sr)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"🖼️ Saved spectrogram to {image_path}")

# === Step 3: Load model and predict from image ===
def predict_from_image(image_path, model_path):
    img = Image.open(image_path).resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))

    model = tf.keras.models.load_model(model_path)
    prob = model.predict(img_array)[0][0]
    label = "REAL" if prob > 0.5 else "FAKE"
    print(f"\n🗣️ Prediction: {label} (Confidence: {prob:.4f})")
    return label, prob

# === Run the pipeline ===
convert_webm_to_wav_ffmpeg(WEBM_INPUT, TEMP_WAV)
create_spectrogram(TEMP_WAV, TEMP_IMG)
predict_from_image(TEMP_IMG, MODEL_PATH)

# === Clean up ===
os.remove(TEMP_WAV)
os.remove(TEMP_IMG)

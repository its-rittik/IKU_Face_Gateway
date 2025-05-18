import os
import io
import hashlib
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from PIL import Image
from config import AUDIO_SAMPLE_RATE, AUDIO_DURATION, AUDIO_IMAGE_SIZE

# === 1. Calculate SHA-256 Hash ===
def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

# === 2. Convert webm to wav ===
import subprocess

def convert_webm_to_wav(webm_path, wav_path):
    try:
        command = [
            'ffmpeg', '-y', '-i', webm_path,
            '-ar', '16000', '-ac', '1', wav_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(wav_path)
    except Exception as e:
        print(f"[FFmpeg Conversion Error] {e}")
        return False


# === 3. Create mel spectrogram ===
def create_spectrogram(wav_path, image_path):
    try:
        y, sr = librosa.load(wav_path, sr=AUDIO_SAMPLE_RATE, duration=AUDIO_DURATION)
        if len(y) == 0:
            raise ValueError("Empty audio signal")

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_DB, sr=sr)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return True
    except Exception as e:
        print(f"[Spectrogram Error] {e}")
        return False

# === 4. Load image and predict ===
def predict_audio(model, spectrogram_path):
    try:
        img = Image.open(spectrogram_path).resize(AUDIO_IMAGE_SIZE).convert("RGB")
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1, *AUDIO_IMAGE_SIZE, 3))
        prob = model.predict(img_array, verbose=0)[0][0]
        label = "REAL" if prob >= 0.8 else "FAKE"
        return label, prob
    except Exception as e:
        print(f"[Prediction Error] {e}")
        return "FAKE", 0.0

# === 5. For Flask Uploads ===
def process_audio(audio_file, model, folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
        ext = os.path.splitext(audio_file.filename)[1].lower()
        ext = ext if ext in ['.webm', '.ogg'] else '.webm'

        audio_path = os.path.join(folder_path, f'audio{ext}')
        wav_path = os.path.join(folder_path, "audio.wav")
        spec_path = os.path.join(folder_path, "audio_spec.png")

        audio_file.stream.seek(0)
        with open(audio_path, 'wb') as f:
            f.write(audio_file.read())

        if os.path.getsize(audio_path) < 1000:
            print("[Error] Uploaded audio file too small or empty.")
            return "FAKE"

        print("📦 Audio file hash:", calculate_file_hash(audio_path))

        if not convert_webm_to_wav(audio_path, wav_path):
            return "FAKE"
        if not create_spectrogram(wav_path, spec_path):
            return "FAKE"

        label, confidence = predict_audio(model, spec_path)
        print(f"🧠 Audio prediction: {label} (confidence: {confidence:.4f})")
        return label

    except Exception as e:
        print(f"[Audio Processing Error] {e}")
        return "FAKE"

# === 6. For Manual Testing (path-based) ===
def process_audio_file(audio_path, model, folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
        wav_path = os.path.join(folder_path, "audio.wav")
        spec_path = os.path.join(folder_path, "audio_spec.png")

        print("📦 Audio file hash:", calculate_file_hash(audio_path))

        if not convert_webm_to_wav(audio_path, wav_path):
            return "FAKE"
        if not create_spectrogram(wav_path, spec_path):
            return "FAKE"

        label, confidence = predict_audio(model, spec_path)
        print(f"🧠 Audio prediction: {label} (confidence: {confidence:.4f})")
        return label

    except Exception as e:
        print(f"[Audio File Error] {e}")
        return "FAKE"

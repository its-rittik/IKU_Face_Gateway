from tensorflow.keras.models import load_model
from config import IMAGE_MODEL_PATH, AUDIO_MODEL_PATH

def load_models():
    image_model = load_model(IMAGE_MODEL_PATH)
    audio_model = load_model(AUDIO_MODEL_PATH)
    return image_model, audio_model

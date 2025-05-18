import os
import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATABASE = os.path.join(BASE_DIR, "access_log.db")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model.h5")
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "cnn_spectrogram_model.h5")

# === Image Settings ===
IMAGE_INPUT_SIZE = (224, 224)
MAX_IMAGE_SIZE_MB = 10
MAX_CONTENT_LENGTH = 16 * 1024 * 1024 

AUDIO_MIN_IMAGES = 15
# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'webm', 'ogg'}



# === Audio Settings ===
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 5
AUDIO_IMAGE_SIZE = (224, 224)
MAX_AUDIO_SIZE_MB = 20

# === Security ===
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-fallback-do-not-use-in-production")
DEBUG = True


LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(BASE_DIR, 'app.log')  

from flask import Flask, request, render_template, jsonify, session
from datetime import datetime
import os
import logging
import uuid
from PIL import Image
import tensorflow as tf

from config import (
    UPLOAD_FOLDER, AUDIO_MIN_IMAGES,
    ALLOWED_IMAGE_EXTENSIONS, ALLOWED_AUDIO_EXTENSIONS,
    DEBUG, SECRET_KEY,
    IMAGE_MODEL_PATH, AUDIO_MODEL_PATH,
    LOG_LEVEL, LOG_FORMAT, LOG_FILE,
    MAX_CONTENT_LENGTH
)
from models.loader import load_models
from utils.image_processing import process_images, calculate_file_hash
from utils.audio_processing import process_audio, process_audio_file
from utils.database import init_db, log_access, log_file_hash
from utils.security import get_user_ip, is_valid_location

# === Logging Setup ===
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# === Flask Setup ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = SECRET_KEY

# === DB and Models ===
init_db()
try:
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
    logging.info("✅ Models loaded successfully")
except Exception as e:
    logging.error(f"❌ Model loading error: {e}")
    raise

# === Utility ===
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# === Home Route ===
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        client_ip = get_user_ip(request)

        if not is_valid_location(client_ip):
            logging.warning(f"🚫 Blocked IP: {client_ip}")
            return jsonify({'error': 'Access denied due to invalid network origin.'}), 403

        try:
            images = request.files.getlist('images')
            audio = request.files.get('audio')
            logging.info(f"📥 Received {len(images)} images and audio: {audio is not None}")

            session_id = str(uuid.uuid4())
            folder_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
            images_path = os.path.join(folder_path, 'images')
            os.makedirs(images_path, exist_ok=True)

            image_results = process_images(images, image_model, folder_path)
            image_hash = None

            for idx, label in enumerate(image_results):
                try:
                    img_filename = f'image_{idx}_{int(datetime.now().timestamp())}.jpg'
                    img_path = os.path.join(images_path, img_filename)
                    img = Image.open(images[idx].stream)
                    img.save(img_path, format='JPEG', quality=95)

                    img_hash = calculate_file_hash(img_path)
                    log_file_hash(img_hash, 'image', session_id, label)
                    if idx == 0:
                        image_hash = img_hash
                    logging.info(f"🖼️ Image {idx} saved: {label} → {img_path}")
                except Exception as e:
                    logging.error(f"❌ Image {idx} failed: {e}")

            # === Audio Processing ===
            audio_result = "FAKE"
            audio_hash = None
            try:
                if audio and hasattr(audio, 'filename') and audio.filename:
                    ext = os.path.splitext(audio.filename)[1].lower()
                    audio_ext = ext if ext in ['.webm', '.ogg', '.wav', '.mp3'] else '.webm'
                    audio_path = os.path.join(folder_path, f'audio{audio_ext}')
                    audio.stream.seek(0)
                    with open(audio_path, 'wb') as f:
                        f.write(audio.read())

                    if os.path.getsize(audio_path) >= 1000:
                        audio_hash = calculate_file_hash(audio_path)
                        audio_result = process_audio_file(audio_path, audio_model, folder_path)
                        log_file_hash(audio_hash, 'audio', session_id, audio_result)
                        logging.info(f"🎤 Audio processed: {audio_result} → {audio_path}")
                    else:
                        logging.warning("⚠️ Audio too small or empty.")
            except Exception as e:
                logging.error(f"❌ Audio error: {e}")

            # === Final Result Decision ===
            if image_results.count("REAL") >= 5:
                image_final = "REAL"
            else:
                image_final = "FAKE"
            final_result = "REAL" if image_final == "REAL" and audio_result == "REAL" else "FAKE"

            log_access(
                session_id=session_id,
                ip_address=client_ip,
                face_result=image_final,
                audio_result=audio_result,
                image_hash=image_hash,
                audio_hash=audio_hash,
                status="success" if final_result == "REAL" else "denied"
            )

            if final_result == "REAL":
                return jsonify({'redirect': 'https://cats.iku.edu.tr/portal'})
            return jsonify({
                'error': 'Access Denied',
                'image_results': image_results,
                'audio_result': audio_result
            }), 403

        except Exception as e:
            error_msg = str(e)
            logging.error(f"❌ Unexpected error: {error_msg}")
            log_access(
                session_id=session.get('session_id', 'unknown'),
                ip_address=client_ip,
                face_result="UNKNOWN",
                audio_result="UNKNOWN",
                image_hash=None,
                audio_hash=None,
                status="error",
                error_message=error_msg
            )
            return jsonify({'error': f'Unexpected server error: {error_msg}'}), 500

    return render_template('index.html')

# === Lightweight Verification Endpoint ===
@app.route('/verify', methods=['POST'])
def verify():
    try:
        session_id = session.get('session_id') or str(uuid.uuid4())
        session['session_id'] = session_id
        folder_path = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(folder_path, exist_ok=True)

        # Face
        if 'face' not in request.files:
            return jsonify({'error': 'No face image provided'}), 400
        face_file = request.files['face']
        face_path = os.path.join(folder_path, 'face.jpg')
        face_file.save(face_path)
        face_hash = calculate_file_hash(face_path)
        log_file_hash(face_hash, 'face', session_id, 'processed')

        # Audio
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio provided'}), 400
        audio_file = request.files['audio']
        audio_result = process_audio(audio_file, audio_model, folder_path)

        audio_path = os.path.join(folder_path, 'audio.webm')
        audio_hash = calculate_file_hash(audio_path) if os.path.exists(audio_path) else None
        log_file_hash(audio_hash, 'audio', session_id, audio_result)

        log_access(
            session_id=session_id,
            ip_address=get_user_ip(request),
            face_result="REAL",
            audio_result=audio_result,
            image_hash=face_hash,
            audio_hash=audio_hash,
            status="success" if audio_result == "REAL" else "denied"
        )

        return jsonify({'result': audio_result, 'session_id': session_id})

    except Exception as e:
        error_msg = str(e)
        logging.error(f"❌ Verification error: {error_msg}")
        log_access(
            session_id=session.get('session_id', 'unknown'),
            ip_address=request.remote_addr,
            face_result="UNKNOWN",
            audio_result="UNKNOWN",
            image_hash=None,
            audio_hash=None,
            status="error",
            error_message=error_msg
        )
        return jsonify({'error': error_msg}), 500

# === Error Handlers ===
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# === Launch App ===
if __name__ == '__main__':
    port = 5000
    print(f"\nApp running in debug mode: {DEBUG}")
    print(f"Visit: http://127.0.0.1:{port}/\n")
    app.run(debug=DEBUG, port=port)

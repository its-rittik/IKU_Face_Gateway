import sqlite3
import logging
from datetime import datetime
from config import DATABASE

# === Configure Logging ===
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Create DB connection ===
def get_db_connection():
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {str(e)}")
        raise

# === Initialize DB and create tables ===
def init_db():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # access_log table with image_hash included
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                session_id TEXT NOT NULL,
                ip_address TEXT,
                face_result TEXT,
                audio_result TEXT,
                image_hash TEXT,
                audio_hash TEXT,
                status TEXT,
                error_message TEXT
            )
        ''')

        # file_hashes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                session_id TEXT NOT NULL,
                status TEXT NOT NULL
            )
        ''')

        # optional logs table (used in insert_log/get_recent_logs)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip TEXT,
                image_result TEXT,
                audio_result TEXT,
                timestamp TEXT,
                folder_hash TEXT,
                created_at TEXT
            )
        ''')

        conn.commit()
        logging.info("✅ Database initialized successfully")
    except Exception as e:
        logging.error(f"❌ Database initialization error: {str(e)}")
        raise
    finally:
        conn.close()

# === Log full access entry ===
def log_access(session_id, ip_address, face_result, audio_result, image_hash, audio_hash, status, error_message=None):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO access_log 
            (timestamp, session_id, ip_address, face_result, audio_result, image_hash, audio_hash, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            session_id,
            ip_address,
            face_result,
            audio_result,
            image_hash,
            audio_hash,
            status,
            error_message
        ))

        conn.commit()
        logging.info(f"✅ Access log inserted for session: {session_id}")
    except Exception as e:
        logging.error(f"❌ Error logging access: {str(e)}")
    finally:
        conn.close()

# === Log hash per file ===
def log_file_hash(file_hash, file_type, session_id, status):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO file_hashes 
            (file_hash, file_type, timestamp, session_id, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            file_hash,
            file_type,
            datetime.now(),
            session_id,
            status
        ))

        conn.commit()
        logging.info(f"✅ File hash logged: {file_hash}")
    except sqlite3.IntegrityError:
        logging.warning(f"⚠️ Duplicate file hash: {file_hash}")
    except Exception as e:
        logging.error(f"❌ Error logging file hash: {str(e)}")
    finally:
        conn.close()

# === Get recent access logs ===
def get_access_history(limit=100):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM access_log 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()
    except Exception as e:
        logging.error(f"❌ Error retrieving access history: {str(e)}")
        return []
    finally:
        conn.close()

# === Legacy or Custom Log Table Entry ===
def insert_log(ip, image_result, audio_result, folder_hash):
    try:
        timestamp = datetime.now().isoformat()
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO logs 
                (ip, image_result, audio_result, timestamp, folder_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                ip, image_result, audio_result, timestamp, folder_hash, timestamp
            ))
            conn.commit()
            logging.info(f"📘 Log entry created for IP: {ip}")
    except sqlite3.Error as e:
        logging.error(f"❌ Error inserting log: {str(e)}")
        raise

# === Fetch recent logs ===
def get_recent_logs(limit=10):
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM logs ORDER BY created_at DESC LIMIT ?',
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logging.error(f"❌ Error retrieving logs: {str(e)}")
        return []

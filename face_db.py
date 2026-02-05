# face_db.py
import face_recognition
import numpy as np
import os

DB_PATH = "data/embeddings.npy"


def get_embedding(frame):
    encodings = face_recognition.face_encodings(frame)

    if not encodings:
        print("❌ No face encoding found")
        return None

    return encodings[0]


def load_db():
    if os.path.exists(DB_PATH):
        return list(np.load(DB_PATH, allow_pickle=True))
    return []


def save_db(db):
    os.makedirs("data", exist_ok=True)
    np.save(DB_PATH, db)


def is_duplicate(new_emb, db, threshold=0.55):
    for i, emb in enumerate(db):
        dist = np.linalg.norm(emb - new_emb)
        print(f"🔎 Comparing with DB face {i+1} → distance: {dist:.3f}")

        if dist < threshold:
            return True

    return False


def add_face(embedding):
    db = load_db()
    db.append(embedding)
    save_db(db)

import numpy as np
import os
import cv2
import uuid

DB_PATH = "stored_embeddings.npy"
IMAGE_DIR = "stored_images"


def load_db():
    if os.path.exists(DB_PATH):
        return list(np.load(DB_PATH, allow_pickle=True))
    return []


def save_db(db):
    np.save(DB_PATH, db)


def store_face(frame, embedding):
    """
    Save image + embedding
    """

    os.makedirs(IMAGE_DIR, exist_ok=True)

    # generate unique filename
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(IMAGE_DIR, filename)

    # save face image
    cv2.imwrite(path, frame)

    # save embedding
    db = load_db()
    db.append(embedding)
    save_db(db)

    return filename


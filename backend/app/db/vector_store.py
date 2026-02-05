import numpy as np
import os

DB_PATH = "backend/stored_embeddings.npy"

def load_db():
    if os.path.exists(DB_PATH):
        return list(np.load(DB_PATH, allow_pickle=True))
    return []

def save_db(db):
    np.save(DB_PATH, db)

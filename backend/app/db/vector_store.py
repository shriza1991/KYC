import numpy as np
import os

# Save inside backend folder
DB_PATH = "stored_embeddings.npy"


def load_db():
    if os.path.exists(DB_PATH):
        return list(np.load(DB_PATH, allow_pickle=True))
    return []


def save_db(db):
    # ensure directory exists (safe even if already exists)
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    np.save(DB_PATH, db)

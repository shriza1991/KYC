import numpy as np
from app.db.vector_store import load_db, save_db

THRESHOLD = 0.55

def check_duplicate(new_emb):
    db = load_db()

    for emb in db:
        dist = np.linalg.norm(emb - new_emb)
        if dist < THRESHOLD:
            return True

    db.append(new_emb)
    save_db(db)

    return False

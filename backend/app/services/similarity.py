import numpy as np
from app.db.vector_store import load_db

THRESHOLD = 0.55


def check_duplicate(new_emb):
    db = load_db()

    for emb in db:
        if np.linalg.norm(emb - new_emb) < THRESHOLD:
            return True

    return False


def search_face(new_emb):
    """
    Returns (match_found, similarity_score)
    """
    db = load_db()

    best_score = float("inf")

    for emb in db:
        dist = np.linalg.norm(emb - new_emb)
        best_score = min(best_score, dist)

    if best_score < THRESHOLD:
        return True, best_score

    return False, best_score


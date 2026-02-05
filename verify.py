# verify.py
from camera import capture_frame
from liveness import liveness_check
from face_db import get_embedding, is_duplicate, add_face, load_db


def verify_user():
    print("\n=== STEP 1: Capture First Frame ===")
    frame1 = capture_frame("Capture Frame 1")

    if frame1 is None:
        print("❌ Capture cancelled")
        return

    print("\n👉 Move your head slightly...")

    print("\n=== STEP 2: Capture Second Frame ===")
    frame2 = capture_frame("Capture Frame 2")

    if frame2 is None:
        print("❌ Capture cancelled")
        return

    print("\n=== STEP 3: Liveness Check ===")

    if not liveness_check(frame1, frame2):
        print("\n🚨 Liveness FAILED — Possible spoof attempt")
        return

    print("✅ Liveness passed")

    print("\n=== STEP 4: Face Encoding ===")

    embedding = get_embedding(frame2)

    if embedding is None:
        print("❌ Could not encode face")
        return

    db = load_db()

    print("\n=== STEP 5: Duplicate Check ===")

    if is_duplicate(embedding, db):
        print("\n⚠ Duplicate identity detected — REJECTED")
    else:
        add_face(embedding)
        print("\n✅ Identity approved & stored")

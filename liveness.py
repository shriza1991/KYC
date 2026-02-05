# liveness.py
import face_recognition
import numpy as np


def liveness_check(frame1, frame2):
    loc1 = face_recognition.face_locations(frame1)
    loc2 = face_recognition.face_locations(frame2)

    if not loc1 or not loc2:
        print("❌ Face not detected during liveness check")
        return False

    # compare horizontal face position
    x1 = loc1[0][3]
    x2 = loc2[0][3]

    movement = abs(x2 - x1)

    print(f"🔍 Detected movement: {movement}")

    # threshold for movement
    return movement > 15

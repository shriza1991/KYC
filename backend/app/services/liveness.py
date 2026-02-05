import face_recognition

def liveness_check(f1, f2):
    loc1 = face_recognition.face_locations(f1)
    loc2 = face_recognition.face_locations(f2)

    if not loc1 or not loc2:
        return False

    movement = abs(loc2[0][3] - loc1[0][3])

    return movement > 15

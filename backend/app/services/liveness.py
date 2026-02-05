from app.services.face_model import face_app

def liveness_check(f1, f2):
    faces1 = face_app.get(f1)
    faces2 = face_app.get(f2)

    if not faces1 or not faces2:
        return False

    x1 = faces1[0].bbox[0]
    x2 = faces2[0].bbox[0]

    movement = abs(x2 - x1)

    return movement > 10


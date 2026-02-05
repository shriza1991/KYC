import face_recognition

def detect_face(frame):
    faces = face_recognition.face_locations(frame)
    return len(faces) == 1

import face_recognition

def get_embedding(frame):
    enc = face_recognition.face_encodings(frame)
    return enc[0] if enc else None

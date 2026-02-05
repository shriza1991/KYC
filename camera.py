# camera.py
import cv2


def capture_frame(window_name="Camera"):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access camera")
        return None

    print("\n👉 Press 'C' to capture | 'Q' to cancel")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read camera")
            break

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            return frame

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

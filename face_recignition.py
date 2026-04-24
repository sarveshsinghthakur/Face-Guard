import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from datetime import datetime
from deepface import DeepFace

# ── Config ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "Facenet"       # Deep learning model for face embeddings
MATCH_THRESHOLD = 0.55            # Cosine similarity threshold (higher = stricter)
FACE_PAD_RATIO = 0.25             # Padding around detected face for better recognition

# Initialize MediaPipe Face Detector
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.6)
face_detection = vision.FaceDetector.create_from_options(options)


# ── Load known faces with deep embeddings ────────────────────────────────
def load_known_faces(path="known_faces"):
    known_encodings = []
    known_names = []

    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath)
        if img is None:
            continue

        try:
            results = DeepFace.represent(
                img_path=filepath,
                model_name=EMBEDDING_MODEL,
                enforce_detection=False,
                detector_backend="opencv",
            )
            if results:
                known_encodings.append(np.array(results[0]["embedding"]))
                known_names.append(os.path.splitext(file)[0])
                print(f"  Loaded face: {os.path.splitext(file)[0]}")
        except Exception as e:
            print(f"  Warning: Could not process {file}: {e}")

    return known_encodings, known_names


print("Loading face embedding model (first run downloads ~95MB)...")
known_encodings, known_names = load_known_faces()
print(f"Loaded {len(known_names)} known faces")

students = known_names.copy()

# CSV setup
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

os.makedirs("attendance", exist_ok=True)
f = open(f"attendance/{current_date}.csv", "w+", newline="")
writer = csv.writer(f)


def get_face_encoding(face_img):
    """Get deep face embedding for a cropped face image (numpy array)."""
    try:
        results = DeepFace.represent(
            img_path=face_img,
            model_name=EMBEDDING_MODEL,
            enforce_detection=False,
            detector_backend="skip",
        )
        if results:
            return np.array(results[0]["embedding"])
    except Exception:
        pass
    return None


def match_face(encoding, known_encodings):
    """Match a face encoding against known faces using cosine similarity."""
    if len(known_encodings) == 0 or encoding is None:
        return "Unknown"

    # Cosine similarity: higher = more similar
    similarities = []
    for known_enc in known_encodings:
        cos_sim = np.dot(encoding, known_enc) / (
            np.linalg.norm(encoding) * np.linalg.norm(known_enc) + 1e-10
        )
        similarities.append(cos_sim)

    best_idx = int(np.argmax(similarities))
    best_sim = similarities[best_idx]

    if best_sim > MATCH_THRESHOLD:
        return known_names[best_idx]
    return "Unknown"


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_detection.detect(mp_image)

    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            bw = bbox.width
            bh = bbox.height

            # Pad face crop for better recognition accuracy
            pad = int(max(bw, bh) * FACE_PAD_RATIO)
            fy1, fy2 = max(0, y - pad), min(h, y + bh + pad)
            fx1, fx2 = max(0, x - pad), min(w, x + bw + pad)
            face_img = frame[fy1:fy2, fx1:fx2]

            if face_img.size == 0:
                continue

            encoding = get_face_encoding(face_img)
            name = match_face(encoding, known_encodings)

            # Draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            # Text
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Attendance
            if name != "Unknown" and name in students:
                current_time = datetime.now().strftime("%H:%M:%S")
                writer.writerow([name, current_time])
                students.remove(name)
                print(f"Attendance recorded: {name} at {current_time}")

    cv2.imshow("Face Attendance (DeepFace + MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
f.close()
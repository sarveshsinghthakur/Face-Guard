import cv2
import numpy as np
from deepface import DeepFace
import os
import csv
import json
import base64
import threading
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from mail import send_attendance_mail

# ── Config ───────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"
USERS_CSV = "users.csv"
ATTENDANCE_DIR = "attendance"
MATCH_THRESHOLD = 0.55  # cosine similarity threshold for Facenet
EMBEDDING_MODEL = "Facenet"  # deep learning face embedding model

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ── In-memory state ─────────────────────────────────────────────────────
known_encodings: list[np.ndarray] = []
known_names: list[str] = []
user_emails: dict[str, str] = {}
recorded_today: set[str] = set()
current_date: str = datetime.now().strftime("%d-%m-%Y")


# ── Helpers ──────────────────────────────────────────────────────────────
def load_users():
    global user_emails
    user_emails = {}
    if os.path.exists(USERS_CSV):
        with open(USERS_CSV, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                user_emails[row["Name"]] = row["Email"]


def save_user(name: str, email: str):
    exists = os.path.exists(USERS_CSV)
    with open(USERS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Name", "Email"])
        if not exists:
            w.writeheader()
        w.writerow({"Name": name, "Email": email})
    user_emails[name] = email


def load_known_faces():
    global known_encodings, known_names
    known_encodings, known_names = [], []
    for file in os.listdir(KNOWN_FACES_DIR):
        filepath = os.path.join(KNOWN_FACES_DIR, file)
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


def get_face_encoding(face_img):
    """Get deep face embedding for a face image (numpy array or filepath)."""
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


def match_face(encoding):
    if not known_encodings or encoding is None:
        return "Unknown"
    # Cosine similarity matching
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


def record_attendance(name: str) -> bool:
    global current_date, recorded_today
    today = datetime.now().strftime("%d-%m-%Y")
    if today != current_date:
        current_date = today
        recorded_today = set()
    if name in recorded_today:
        return False
    recorded_today.add(name)
    att_file = os.path.join(ATTENDANCE_DIR, f"{current_date}.csv")
    exists = os.path.exists(att_file)
    with open(att_file, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Name", "Time", "Date"])
        if not exists:
            w.writeheader()
        w.writerow({
            "Name": name,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Date": current_date,
        })
    if name in user_emails:
        threading.Thread(
            target=send_attendance_mail,
            args=(user_emails[name], name),
            daemon=True,
        ).start()
    return True


def load_today_attendance():
    global current_date, recorded_today
    current_date = datetime.now().strftime("%d-%m-%Y")
    recorded_today = set()
    att_file = os.path.join(ATTENDANCE_DIR, f"{current_date}.csv")
    if os.path.exists(att_file):
        with open(att_file, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                recorded_today.add(row["Name"])


# ── FastAPI App ──────────────────────────────────────────────────────────
app = FastAPI(title="FaceGuard")


@app.on_event("startup")
def startup():
    load_users()
    print("Loading face embedding model (first run downloads ~95MB)...")
    load_known_faces()
    load_today_attendance()
    print(f"Loaded {len(known_names)} faces, {len(user_emails)} users")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── REST endpoints ───────────────────────────────────────────────────────
class RegisterReq(BaseModel):
    name: str
    email: str
    face_image: str  # base64-encoded JPEG


@app.post("/api/register")
async def register(req: RegisterReq):
    img_data = base64.b64decode(req.face_image)
    arr = np.frombuffer(img_data, np.uint8)
    face_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if face_img is None:
        raise HTTPException(400, "Invalid image")
    filepath = os.path.join(KNOWN_FACES_DIR, f"{req.name}.jpg")
    cv2.imwrite(filepath, face_img)
    try:
        results = DeepFace.represent(
            img_path=filepath,
            model_name=EMBEDDING_MODEL,
            enforce_detection=False,
            detector_backend="opencv",
        )
        if results:
            known_encodings.append(np.array(results[0]["embedding"]))
            known_names.append(req.name)
        else:
            os.remove(filepath)
            raise HTTPException(400, "No face detected in image")
    except HTTPException:
        raise
    except Exception as e:
        os.remove(filepath)
        raise HTTPException(400, f"Face processing failed: {e}")
    save_user(req.name, req.email)
    record_attendance(req.name)
    return {"status": "ok", "message": f"Registered {req.name}"}


@app.get("/api/attendance")
async def get_attendance():
    records = []
    att_file = os.path.join(ATTENDANCE_DIR, f"{current_date}.csv")
    if os.path.exists(att_file):
        with open(att_file, "r", newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
    return {"date": current_date, "records": records}


@app.get("/api/users")
async def get_users():
    return {
        "users": [{"name": n, "email": e} for n, e in user_emails.items()]
    }


# ── WebSocket for real-time detection ────────────────────────────────────
@app.websocket("/ws/video")
async def video_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            payload = json.loads(raw)
            frame_b64 = payload.get("frame", "")
            if not frame_b64:
                continue
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]

            img_data = base64.b64decode(frame_b64)
            arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Use DeepFace for face detection
            try:
                faces = DeepFace.extract_faces(img_path=frame, detector_backend='opencv', enforce_detection=False)
                if not faces:
                    continue
                
                h, w, _ = frame.shape
                detections = []
                
                for face_data in faces:
                    # Get face coordinates from DeepFace result
                    x, y, bw, bh = face_data['facial_area']['x'], face_data['facial_area']['y'], \
                                   face_data['facial_area']['w'], face_data['facial_area']['h']
                    
                    # Pad face crop for better recognition accuracy
                    pad = int(max(bw, bh) * 0.25)
                    fy1, fy2 = max(0, y - pad), min(h, y + bh + pad)
                    fx1, fx2 = max(0, x - pad), min(w, x + bw + pad)
                    face_img = frame[fy1:fy2, fx1:fx2]
                    if face_img.size == 0:
                        continue

                    enc = get_face_encoding(face_img)
                    name = match_face(enc)

                    status = "unknown"
                    face_b64 = ""
                    if name != "Unknown":
                        if name in recorded_today:
                            status = "present"
                        else:
                            record_attendance(name)
                            status = "recorded"
                    else:
                        _, buf = cv2.imencode(".jpg", face_img)
                        face_b64 = base64.b64encode(buf).decode()

                    detections.append({
                        "name": name,
                        "status": status,
                        "bbox": {
                            "x": x / w,
                            "y": y / h,
                            "w": bw / w,
                            "h": bh / h,
                        },
                        "face_image": face_b64,
                    })
                
                await ws.send_json({"detections": detections})
            except Exception as e:
                print(f"Face detection error: {e}")
                await ws.send_json({"detections": []})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WS error: {e}")

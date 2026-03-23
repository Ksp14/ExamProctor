import os
import base64
import datetime
import cv2
import numpy as np

_DEEPFACE_OK = False

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR  = os.path.join(_BASE_DIR, "registered_faces")
FACE_PATH = os.path.join(FACE_DIR,  "reference.jpg")


class FaceAuthenticator:

    def __init__(self):
        os.makedirs(FACE_DIR, exist_ok=True)
        self.status = {
            "registered":      os.path.exists(FACE_PATH),
            "authenticated":   False,
            "auth_attempts":   0,
            "last_confidence": None,
            "last_message":    "Waiting for registration" if not os.path.exists(FACE_PATH) else "Ready to authenticate",
            "registered_at":   None,
        }

    def has_registered_face(self):
        return os.path.exists(FACE_PATH)

    def get_registered_face_b64(self):
        if not os.path.exists(FACE_PATH):
            return None
        with open(FACE_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _detect_face(self, frame):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return len(faces) > 0

    def _draw_face_box(self, frame):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        out = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 100), 2)
            cv2.putText(out, "Face detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
        return out, len(faces) > 0

    def capture_and_register(self, frame):
        if not self._detect_face(frame):
            msg = "No face detected. Please look directly at the camera."
            self.status["last_message"] = msg
            return {"ok": False, "message": msg}

        os.makedirs(FACE_DIR, exist_ok=True)
        cv2.imwrite(FACE_PATH, frame)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status.update({
            "registered":    True,
            "authenticated": False,
            "registered_at": ts,
            "last_message":  f"Face registered at {ts}.",
        })
        return {"ok": True, "message": f"Face registered successfully at {ts}."}

    def authenticate(self, frame):
        self.status["auth_attempts"] += 1

        if not os.path.exists(FACE_PATH):
            msg = "No registered face found. Please register first."
            self.status["last_message"] = msg
            return {"verified": False, "confidence": None, "message": msg}

        if not self._detect_face(frame):
            msg = "No face detected. Please centre your face."
            self.status["last_message"] = msg
            return {"verified": False, "confidence": None, "message": msg}

        ref_img     = cv2.imread(FACE_PATH)
        ref_gray    = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        liv_gray    = cv2.cvtColor(frame,   cv2.COLOR_BGR2GRAY)
        ref_resized = cv2.resize(ref_gray, (200, 200))
        liv_resized = cv2.resize(liv_gray, (200, 200))
        ref_hist    = cv2.calcHist([ref_resized], [0], None, [256], [0, 256])
        liv_hist    = cv2.calcHist([liv_resized], [0], None, [256], [0, 256])
        cv2.normalize(ref_hist, ref_hist)
        cv2.normalize(liv_hist, liv_hist)

        score      = cv2.compareHist(ref_hist, liv_hist, cv2.HISTCMP_CORREL)
        confidence = round(float(max(0.0, score)), 3)
        verified   = confidence > 0.3

        if verified:
            msg = f"Authentication successful. Confidence: {confidence:.2%}"
            self.status.update({
                "authenticated":   True,
                "last_confidence": confidence,
                "last_message":    msg,
            })
        else:
            msg = f"Authentication failed. Confidence: {confidence:.2%}. Try again."
            self.status.update({
                "authenticated":   False,
                "last_confidence": confidence,
                "last_message":    msg,
            })

        return {"verified": verified, "confidence": confidence, "message": msg}

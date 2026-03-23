import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import time
import base64
import threading
import random
import string
import numpy as np
from datetime import datetime

from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

from modules.object_detection  import ObjectDetector
from modules.face_monitor       import FaceMonitor
from modules.audio_detection    import AudioDetector
from modules.evidence_recorder  import EvidenceRecorder
from modules.eye_tracking       import EyeTracker
from modules.face_auth          import FaceAuthenticator

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# ── Shared camera ─────────────────────────────────────────────────────────────
_cam_lock = threading.Lock()
_cap      = None

def _get_cap():
    global _cap
    if _cap is None or not _cap.isOpened():
        _cap = cv2.VideoCapture(0)
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        _cap.set(cv2.CAP_PROP_FPS,          30)
        _cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)
        _cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
    return _cap

def _read_frame():
    with _cam_lock:
        cap = _get_cap()
        if not cap.isOpened():
            return False, None
        ret, frame = cap.read()
    if not ret or frame is None:
        return False, None
    return True, frame

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "phase":            "register",
    "auth_message":     "",
    "auth_confidence":  None,
    "running":          False,
    "cheat_score":      0,
    "head_direction":   "CENTER",
    "gaze_direction":   "CENTER",
    "audio_rms":        0.0,
    "person_count":     1,
    "events": {
        "mobile_phone":    0,
        "book":            0,
        "laptop":          0,
        "multiple_person": 0,
        "head_turn":       0,
        "eye_movement":    0,
        "speech":          0,
    },
    "log":         [],
    "session_id":  "",
    "start_time":  None,
    "latest_frame":None,
}
state_lock = threading.Lock()
_auth      = FaceAuthenticator()

# ── Logging ───────────────────────────────────────────────────────────────────
def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _log(message, level="info"):
    entry = {"time": _ts(), "msg": message, "level": level}
    with state_lock:
        state["log"].append(entry)
        if len(state["log"]) > 200:
            state["log"].pop(0)
    os.makedirs("logs", exist_ok=True)
    with open("logs/cheating_log.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level.upper()}] {message}\n")

# ── Proctoring system ─────────────────────────────────────────────────────────
class ExamMonitoringSystem:

    def __init__(self):
        self.object_detector = ObjectDetector()
        self.face_monitor    = FaceMonitor()
        self.eye_tracker     = EyeTracker()
        self.audio_detector  = AudioDetector()
        self.recorder        = EvidenceRecorder()

        self.cheat_score  = 0
        self.cooldown     = 2
        self.phone_cooldown = 1

        self.last_event_time = {
            "phone": 0, "book": 0, "laptop": 0,
            "person": 0, "head": 0, "eye": 0, "speech": 0,
        }
        self.audio_alert = None
        self._start_audio_thread()

    def _start_audio_thread(self):
        def audio_loop():
            while True:
                try:
                    susp, _, alert = self.audio_detector.detect_audio()
                    if susp and alert == "speech":
                        self.audio_alert = "speech"
                except Exception:
                    pass
                if not state["running"]:
                    time.sleep(0.5)
        t = threading.Thread(target=audio_loop, daemon=True)
        t.start()

    def can_trigger(self, event):
        now = time.time()
        if now - self.last_event_time[event] > self.cooldown:
            self.last_event_time[event] = now
            return True
        return False

    def _push_score(self, delta, event_key, log_msg, level="warning"):
        self.cheat_score += delta
        with state_lock:
            state["cheat_score"]       = self.cheat_score
            state["events"][event_key] += 1
        _log(log_msg, level)

    def run(self):
        with state_lock:
            state["running"]    = True
            state["start_time"] = time.time()
        _log("Proctoring started", "info")

        try:
            while state["running"]:
                ret, frame = _read_frame()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                # ── Object detection ──────────────────────────────────────────
                detections, _, persons = self.object_detector.detect_objects(frame)

                if persons > 1 and self.can_trigger("person"):
                    self._push_score(6, "multiple_person", "Multiple persons detected", "alert")
                    self.recorder.auto_record(frame, "multiple_person")

                for obj in detections:
                    lbl = obj["label"]
                    now = time.time()
                    if lbl == "cell phone" and (now - self.last_event_time["phone"]) > self.phone_cooldown:
                        self.last_event_time["phone"] = now
                        self._push_score(5, "mobile_phone", "Mobile phone detected", "alert")
                        self.recorder.auto_record(frame, "phone")
                    elif lbl == "book" and self.can_trigger("book"):
                        self._push_score(4, "book", "Book detected", "warning")
                        self.recorder.auto_record(frame, "book")
                    elif lbl == "laptop" and self.can_trigger("laptop"):
                        self._push_score(3, "laptop", "Laptop detected", "warning")
                        self.recorder.auto_record(frame, "laptop")

                frame = self.object_detector.draw_detections(frame, detections)

                # ── Head ──────────────────────────────────────────────────────
                frame, head_cheat, head_dir = self.face_monitor.process_frame(frame)
                with state_lock:
                    state["head_direction"] = head_dir
                if head_cheat and self.can_trigger("head"):
                    self._push_score(2, "head_turn", f"Head turn: {head_dir}", "warning")
                    self.recorder.auto_record(frame, "head_turn")

                # ── Eyes ──────────────────────────────────────────────────────
                frame, eye_cheat, gaze_dir = self.eye_tracker.process_frame(frame)
                with state_lock:
                    state["gaze_direction"] = gaze_dir
                if eye_cheat and self.can_trigger("eye"):
                    self._push_score(3, "eye_movement", f"Eye movement: {gaze_dir}", "warning")
                    self.recorder.auto_record(frame, "eye_movement")

                if head_cheat and eye_cheat:
                    self.cheat_score += 2
                    with state_lock:
                        state["cheat_score"] = self.cheat_score
                    _log("Combined violation: head + eye", "alert")

                # ── Audio ─────────────────────────────────────────────────────
                rms = self.audio_detector.status.get("rms", 0.0)
                with state_lock:
                    state["audio_rms"]    = round(float(rms), 4)
                    state["person_count"] = persons

                if self.audio_alert == "speech" and self.can_trigger("speech"):
                    self._push_score(4, "speech", "Continuous speech detected", "alert")
                    self.recorder.record_audio(5)
                    self.audio_alert = None

                # ── Encode frame ──────────────────────────────────────────────
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with state_lock:
                    state["latest_frame"] = buf.tobytes()

        except Exception as exc:
            _log(f"Detection error: {exc}", "error")
        finally:
            with state_lock:
                state["running"]       = False
                state["latest_frame"]  = None
            _log("Proctoring stopped", "info")

# ── Monitor thread ────────────────────────────────────────────────────────────
_monitor_thread = None
_monitor_system = None

def _start_monitoring():
    global _monitor_thread, _monitor_system
    if _monitor_thread and _monitor_thread.is_alive():
        return
    _monitor_system = ExamMonitoringSystem()
    _monitor_thread = threading.Thread(target=_monitor_system.run, daemon=True)
    _monitor_thread.start()

def _stop_monitoring():
    with state_lock:
        state["running"] = False

# ── Frame decoder ─────────────────────────────────────────────────────────────
def _frame_from_request():
    data = request.get_json(silent=True) or {}
    b64  = data.get("image", "")
    if b64:
        try:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            arr   = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return True, frame
        except Exception:
            pass
    return _read_frame()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/api/auth_status")
def api_auth_status():
    try:
        with state_lock:
            phase = state["phase"]
        s = _auth.status
        ref_b64 = None
        try:
            ref_b64 = _auth.get_registered_face_b64()
        except Exception:
            pass
        return jsonify({
            "phase":           phase,
            "registered":      s.get("registered",      False),
            "authenticated":   s.get("authenticated",   False),
            "auth_attempts":   s.get("auth_attempts",   0),
            "last_confidence": s.get("last_confidence", None),
            "last_message":    s.get("last_message",    ""),
            "registered_at":   s.get("registered_at",   None),
            "reference_face":  ref_b64,
        })
    except Exception as e:
        return jsonify({"error": str(e), "phase": "register"}), 500

@app.route("/api/frame_preview")
def api_frame_preview():
    ret, frame = _read_frame()
    if not ret or frame is None:
        return Response(status=204)
    frame, _ = _auth._draw_face_box(frame)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store"})

@app.route("/api/register", methods=["POST", "OPTIONS"])
def api_register():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
    ok, frame = _frame_from_request()
    if not ok or frame is None:
        return jsonify({"ok": False, "message": "Could not read camera."}), 500
    result = _auth.capture_and_register(frame)
    if result["ok"]:
        with state_lock:
            state["phase"] = "authenticate"
        _log("Face registered", "info")
    return jsonify(result)

@app.route("/api/authenticate", methods=["POST", "OPTIONS"])
def api_authenticate():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
    ok, frame = _frame_from_request()
    if not ok or frame is None:
        return jsonify({"verified": False, "message": "Could not read camera."}), 500
    result = _auth.authenticate(frame)
    if result["verified"]:
        with state_lock:
            state["phase"]           = "exam"
            state["auth_confidence"] = result["confidence"]
        _log("Authentication passed", "info")
    else:
        _log(f"Authentication failed: {result['message']}", "warning")
    return jsonify(result)

@app.route("/api/start", methods=["POST"])
def api_start():
    with state_lock:
        if state["phase"] != "exam":
            return jsonify({"ok": False, "message": "Authentication required."}), 403
    sid = "EX-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    with state_lock:
        state["session_id"]  = sid
        state["cheat_score"] = 0
        state["log"]         = []
        for k in state["events"]:
            state["events"][k] = 0
    _start_monitoring()
    _log(f"Exam session started: {sid}", "info")
    return jsonify({"ok": True, "session_id": sid})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    _stop_monitoring()
    return jsonify({"ok": True})

@app.route("/api/status")
def api_status():
    with state_lock:
        elapsed = int(time.time() - state["start_time"]) if state["start_time"] else 0
        score   = state["cheat_score"]
        verdict = ("CHEATING DETECTED" if score >= 15
                   else "SUSPICIOUS"   if score >= 8
                   else "NORMAL")
        return jsonify({
            "phase":           state["phase"],
            "running":         state["running"],
            "cheat_score":     score,
            "verdict":         verdict,
            "head_direction":  state["head_direction"],
            "gaze_direction":  state["gaze_direction"],
            "audio_rms":       state["audio_rms"],
            "person_count":    state["person_count"],
            "events":          dict(state["events"]),
            "elapsed_seconds": elapsed,
            "session_id":      state["session_id"],
        })

@app.route("/api/log")
def api_log():
    with state_lock:
        return jsonify({"log": list(state["log"][-100:])})

@app.route("/api/frame")
def api_frame():
    with state_lock:
        data = state["latest_frame"]
    if data is None:
        return Response(status=204)
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store"})

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting camera...")
    _get_cap()
    if _auth.has_registered_face():
        with state_lock:
            state["phase"] = "authenticate"
    print("Dashboard -> http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

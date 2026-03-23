# AI Exam Proctoring System

## Project Structure
```
project/
├── main.py                    ← Run this to start the server
├── requirements.txt           ← All dependencies
├── templates/
│   └── index.html             ← Web dashboard UI
├── modules/
│   ├── __init__.py
│   ├── face_auth.py           ← Face registration & authentication
│   ├── face_monitor.py        ← Head direction detection
│   ├── eye_tracking.py        ← Eye gaze detection
│   ├── audio_detection.py     ← Speech detection
│   ├── object_detection.py    ← Phone/book/person detection (YOLO)
│   ├── evidence_recorder.py   ← Screenshots & audio evidence
│   └── system_control.py      ← Exam start/stop control
├── registered_faces/          ← Auto-created, stores reference.jpg
├── evidence/
│   ├── screenshots/           ← Violation screenshots saved here
│   └── audio/                 ← Speech recordings saved here
└── logs/
    └── cheating_log.txt       ← All events logged here
```

## Setup (Windows)

### Step 1 — Create virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 2 — Install dependencies in order
```powershell
pip install numpy==1.26.4
pip install protobuf==4.25.3
pip install tensorflow==2.15.0
pip install tf-keras==2.15.0
pip install opencv-python ultralytics flask==2.3.3 flask-cors==4.0.0 werkzeug==2.3.7
pip install sounddevice webrtcvad-wheels librosa scipy
```

### Step 3 — Run
```powershell
python main.py
```

### Step 4 — Open browser
```
http://localhost:5000
```

---

## How it works

### Phase 1 — Register
- Camera opens with live preview
- Student centres face in the oval guide
- Click "Capture & Register" to save reference face

### Phase 2 — Authenticate
- System compares live face to registered face
- Must pass confidence check to proceed
- Click "Re-register" if authentication keeps failing

### Phase 3 — Exam Dashboard
- Click "Start Monitoring" to begin proctoring
- Live camera feed with detection overlays
- 6 violation detectors running simultaneously
- Evidence automatically saved on violations

---

## Violation Scoring

| Violation         | Points |
|-------------------|--------|
| Multiple persons  | +6     |
| Mobile phone      | +5     |
| Speech detected   | +4     |
| Book detected     | +4     |
| Eye movement      | +3     |
| Laptop detected   | +3     |
| Head turn         | +2     |

**Verdict:**
- Score 0–7  → NORMAL
- Score 8–14 → SUSPICIOUS
- Score 15+  → CHEATING DETECTED

---

## Evidence Files
Every violation automatically saves:
- **Screenshot** → `evidence/screenshots/event_name_timestamp.jpg`
- **Audio clip** → `evidence/audio/speech_timestamp.wav` (speech only)
- **Log entry**  → `logs/cheating_log.txt`

---

## Notes
- DeepFace is NOT required — authentication uses OpenCV histogram matching
- mediapipe is NOT required — face/eye detection uses OpenCV Haar cascades
- YOLOv8n model (`yolov8n.pt`) downloads automatically on first run (~6MB)
- TensorFlow 2.15 is required for deepface compatibility if you choose to enable it

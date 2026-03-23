import cv2
from ultralytics import YOLO


class ObjectDetector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.target_classes = {"person", "cell phone", "laptop", "book"}
        self.conf_threshold = 0.4

    def detect_objects(self, frame):
        results          = self.model(frame, conf=self.conf_threshold, verbose=False)
        detected_objects = []
        person_count     = 0

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label    = self.model.names[class_id]
                conf     = float(box.conf[0])
                if label in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_objects.append({
                        "label":      label,
                        "confidence": conf,
                        "bbox":       (x1, y1, x2, y2),
                    })
                    if label == "person":
                        person_count += 1

        suspicious = person_count > 1
        return detected_objects, suspicious, person_count

    def draw_detections(self, frame, detections):
        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]
            conf  = obj["confidence"]
            text  = f"{label} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

import cv2
import time


class FaceMonitor:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.turn_start_time     = None
        self.cheat_threshold     = 5
        self.violation_triggered = False

    def get_head_direction(self, x, y, w, h, frame_w, frame_h):
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        if face_center_x < frame_w * 0.35:
            return "LEFT"
        elif face_center_x > frame_w * 0.65:
            return "RIGHT"
        elif face_center_y < frame_h * 0.35:
            return "UP"
        elif face_center_y > frame_h * 0.65:
            return "DOWN"
        return "CENTER"

    def process_frame(self, frame):
        direction = "CENTER"
        cheating  = False
        h, w      = frame.shape[:2]
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            direction    = self.get_head_direction(x, y, fw, fh, w, h)
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
            cv2.putText(frame, f"Head: {direction}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            current_time = time.time()
            if direction in ["LEFT", "RIGHT"]:
                if self.turn_start_time is None:
                    self.turn_start_time = current_time
                duration = current_time - self.turn_start_time
                if duration >= self.cheat_threshold and not self.violation_triggered:
                    cheating = True
                    self.violation_triggered = True
            else:
                self.turn_start_time     = None
                self.violation_triggered = False

        return frame, cheating, direction

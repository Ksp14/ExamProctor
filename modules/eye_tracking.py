import cv2
import time
import numpy as np


class EyeTracker:

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.gaze_start_time     = None
        self.cheat_threshold     = 4
        self.violation_triggered = False

    def get_gaze_direction(self, eye_frame):
        try:
            gray_eye  = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
            h, w      = gray_eye.shape
            _, thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)
            left_part  = np.sum(thresh[:, :w//2])
            right_part = np.sum(thresh[:, w//2:])
            if left_part > right_part * 1.5:
                return "LEFT"
            elif right_part > left_part * 1.5:
                return "RIGHT"
        except Exception:
            pass
        return "CENTER"

    def process_frame(self, frame):
        gaze_text = "CENTER"
        cheating  = False
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) > 0:
            x, y, fw, fh  = faces[0]
            roi_gray  = gray[y:y+fh, x:x+fw]
            roi_color = frame[y:y+fh, x:x+fw]

            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )

            if len(eyes) >= 1:
                ex, ey, ew, eh = eyes[0]
                eye_frame  = roi_color[ey:ey+eh, ex:ex+ew]
                gaze_text  = self.get_gaze_direction(eye_frame)
                cv2.rectangle(roi_color, (ex, ey),
                              (ex+ew, ey+eh), (0, 255, 255), 1)

            current_time = time.time()
            if gaze_text in ["LEFT", "RIGHT"]:
                if self.gaze_start_time is None:
                    self.gaze_start_time = current_time
                duration = current_time - self.gaze_start_time
                if duration >= self.cheat_threshold and not self.violation_triggered:
                    cheating = True
                    self.violation_triggered = True
            else:
                self.gaze_start_time     = None
                self.violation_triggered = False

        return frame, cheating, gaze_text

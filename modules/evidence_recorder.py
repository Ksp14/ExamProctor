import cv2
import os
import time
import datetime
import wave

try:
    import sounddevice as sd
    import numpy as np
    _SD_OK = True
except Exception:
    _SD_OK = False


class EvidenceRecorder:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.screenshot_folder = os.path.join(base_dir, "evidence", "screenshots")
        self.audio_folder      = os.path.join(base_dir, "evidence", "audio")
        self.log_file          = os.path.join(base_dir, "logs", "cheating_log.txt")

        os.makedirs(self.screenshot_folder, exist_ok=True)
        os.makedirs(self.audio_folder,      exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        self.video_writer    = None
        self.recording       = False
        self.start_time      = None
        self.last_event_time = {}
        self.cooldown        = 10

    def log_event(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} - {message}\n")

    def save_screenshot(self, frame, event_name):
        """Save a screenshot immediately when a violation is detected."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"{event_name}_{timestamp}.jpg"
            filepath  = os.path.join(self.screenshot_folder, filename)
            cv2.imwrite(filepath, frame)
            self.log_event(f"Screenshot saved: {filename}")
            print(f"[Evidence] Screenshot saved: {filepath}")
            return filepath
        except Exception as e:
            self.log_event(f"Screenshot failed: {e}")
            return None

    def start_video_record(self, frame, event_name):
        if self.recording:
            return
        self.recording  = True
        self.start_time = time.time()
        h, w, _         = frame.shape
        timestamp       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename        = f"{event_name}_{timestamp}.avi"
        filepath        = os.path.join(self.screenshot_folder, filename)
        fourcc          = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(filepath, fourcc, 20, (w, h))
        self.log_event(f"Video recording started: {filename}")

    def write_frame(self, frame):
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def stop_video_record(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        self.log_event("Video recording stopped")

    def record_audio(self, duration=5, sample_rate=16000):
        if not _SD_OK:
            return
        try:
            import numpy as np
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"speech_{timestamp}.wav"
            filepath  = os.path.join(self.audio_folder, filename)
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16"
            )
            sd.wait()
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())
            self.log_event(f"Audio evidence recorded: {filename}")
            print(f"[Evidence] Audio saved: {filepath}")
        except Exception as e:
            self.log_event(f"Audio recording failed: {e}")

    def auto_record(self, frame, event_name, duration=6):
        """Save screenshot + start video clip on violation."""
        now = time.time()
        if event_name in self.last_event_time:
            if now - self.last_event_time[event_name] < self.cooldown:
                return
        self.last_event_time[event_name] = now

        # Always save screenshot immediately
        self.save_screenshot(frame, event_name)

        # Also record a short video clip
        if not self.recording:
            self.start_video_record(frame, event_name)
        self.write_frame(frame)
        if self.start_time and time.time() - self.start_time >= duration:
            self.stop_video_record()

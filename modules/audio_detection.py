import sounddevice as sd
import numpy as np
import webrtcvad
import librosa
import datetime
import os
import time
from scipy.signal import butter, lfilter


class AudioDetector:

    def __init__(self):
        self.sample_rate    = 16000
        self.frame_duration = 30
        self.frame_size     = int(self.sample_rate * self.frame_duration / 1000)

        self.vad = webrtcvad.Vad(3)

        self.speech_counter   = 0
        self.speech_threshold = 5       # lowered for faster detection
        self.last_alert_time  = 0
        self.cooldown         = 3       # seconds between alerts
        self.min_rms          = 0.001
        self.min_speech_energy= 0.001
        self.energy_history   = []

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_file = os.path.join(base_dir, "logs", "cheating_log.txt")
        self.alert    = None

        self.status = {
            "rms":            0.0,
            "vad_active":     False,
            "speech_energy":  0.0,
            "speech_counter": 0,
            "alert":          None,
        }

    def bandpass_filter(self, data, low=300, high=3400):
        nyquist = 0.5 * self.sample_rate
        low  = low  / nyquist
        high = high / nyquist
        b, a = butter(4, [low, high], btype='band')
        return lfilter(b, a, data)

    def float_to_pcm(self, audio):
        audio = np.clip(audio, -1, 1)
        audio = (audio * 32767).astype(np.int16)
        return audio.tobytes()

    def log_event(self, message):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} - {message}\n")

    def record_audio(self):
        audio = sd.rec(
            self.frame_size,
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return audio.flatten()

    def speech_frequency_check(self, audio):
        stft        = np.abs(librosa.stft(audio))
        freqs       = librosa.fft_frequencies(sr=self.sample_rate)
        speech_band = (freqs > 300) & (freqs < 3400)
        return np.mean(stft[speech_band])

    def is_constant_noise(self, rms):
        self.energy_history.append(rms)
        if len(self.energy_history) > 40:
            self.energy_history.pop(0)
        if len(self.energy_history) < 20:
            return False
        return np.var(self.energy_history) < 0.000002

    def detect_audio(self):
        audio = self.record_audio()

        if np.max(np.abs(audio)) == 0:
            self.status.update({"rms": 0.0, "vad_active": False,
                                 "speech_energy": 0.0, "alert": None})
            return False, None, None

        audio = self.bandpass_filter(audio)
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        rms   = float(np.sqrt(np.mean(audio ** 2)))
        self.status["rms"] = round(rms, 4)

        if rms < self.min_rms:
            self.status.update({"vad_active": False, "speech_energy": 0.0, "alert": None})
            return False, None, None

        if self.is_constant_noise(rms):
            self.status.update({"vad_active": False, "alert": None})
            return False, None, None

        # Use RMS directly for faster detection
        if rms > self.min_rms:
            self.speech_counter += 1
        else:
            self.speech_counter = max(0, self.speech_counter - 1)

        self.status["speech_counter"] = self.speech_counter
        self.status["vad_active"]     = rms > self.min_rms

        suspicious = False
        if self.speech_counter >= self.speech_threshold:
            now = time.time()
            if now - self.last_alert_time > self.cooldown:
                message = "Continuous speech detected"
                print("⚠ AUDIO ALERT:", message)
                self.log_event(message)
                self.alert          = "speech"
                self.status["alert"]= "speech"
                suspicious          = True
                self.last_alert_time= now
                self.speech_counter = 0

        return suspicious, None, self.alert

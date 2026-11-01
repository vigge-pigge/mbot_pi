import cv2
import time
import numpy as np
from threading import Thread
from typing import Tuple


class VideoStream:
    """Kameratråd för Raspberry Pi 5 med nativt BGR888-format för OpenCV."""

    def __init__(self, resolution: Tuple[int, int] = (640, 480), src: int = 0) -> None:
        self.resolution = resolution
        self.src = src
        self.stopped: bool = False
        self.picam2 = None
        self.cap = None

        self.fallback_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        cv2.putText(self.fallback_frame, "NO CAMERA FEED", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        self.frame = self.fallback_frame.copy()

        # Starta Picamera2 med nativt BGR888-format (OpenCV-standard)
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": self.resolution, "format": "BGR888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1.0)
            print("[CAMERA] Picamera2 startades i nativt BGR888-läge.")
        except Exception as e:
            print(f"[CAMERA] Picamera2 kunde inte startas: {e}. Försöker med OpenCV...")
            try:
                self.cap = cv2.VideoCapture(self.src)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                if self.cap.isOpened():
                    print(f"[CAMERA] Startades med OpenCV på index {self.src}.")
            except Exception as e2:
                print(f"[CAMERA] OpenCV misslyckades: {e2}")

    def start(self) -> "VideoStream":
        thread = Thread(target=self.update, args=(), daemon=True)
        thread.start()
        return self

    def update(self) -> None:
        while not self.stopped:
            if self.picam2 is not None:
                try:
                    frame = self.picam2.capture_array("main")
                    if frame is not None and frame.size > 0:
                        # Nativ BGR888: Vi roterar enbart bilden 180 grader
                        self.frame = cv2.rotate(frame, cv2.ROTATE_180)[:, :, ::-1]
                        # Debug: Print color of a specific pixel
                        pixel_color = self.frame[100, 100]
                        #print(f"[CAMERA DEBUG] Pixel at (100, 100) has color: {pixel_color}")
                except Exception as e:
                    print(f"[CAMERA ERROR] Fel vid bildhämtning: {e}")
            elif self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.frame = cv2.rotate(frame, cv2.ROTATE_180)[:, :, ::-1]
                    # Debug: Print color of a specific pixel
                    pixel_color = self.frame[100, 100]
                    #print(f"[CAMERA DEBUG] Pixel at (100, 100) has color: {pixel_color}")
            time.sleep(0.01)

    def read(self) -> np.ndarray:
        return self.frame

    def stop(self) -> None:
        self.stopped = True
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        print("[CAMERA] Kameran stoppades.")
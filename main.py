import os
import time
import queue
import cv2
import numpy as np
from yolo_detector import YoloDetector
from datetime import datetime
from multiprocessing import Queue
from camera import VideoStream
from motor_driver import MegaPiDriver
from ai_agent import CognitiveAgent


class PIDController:
    def __init__(self, kp: float = 200.0, ki: float = 5.0, kd: float = 20.0) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_sum = 0.0
        self.prev_diff = 0.0

    def compute(self, x_ort: float) -> tuple[float, float]:
        ref = 0.5
        diff = ref - x_ort
        self.integral_sum += diff
        d_value = diff - self.prev_diff
        self.prev_diff = diff

        correction = (self.kp * abs(diff)) + (self.ki * self.integral_sum) - (self.kd * d_value)

        if x_ort < 0.5:
            left_speed = max(0.0, min(100.0, 100.0 - correction))
            right_speed = 100.0
        else:
            left_speed = 100.0
            right_speed = max(0.0, min(100.0, 100.0 - correction))

        return left_speed, right_speed


def main(command_queue: Queue, telemetry_queue: Queue, frame_queue: Queue) -> None:
    print("[MAIN] Starting robot control process...")

    yolo = YoloDetector()
    vs = VideoStream(resolution=(640, 480)).start()
    pid = PIDController(kp=200.0, ki=5.0, kd=20.0)
    motors = MegaPiDriver()
    agent = CognitiveAgent()

    mode = "AUTONOMOUS"
    last_ai_query_time = 0.0
    current_ai_decision = {"action": "SEARCH", "target_label": "none", "reasoning": "Initializing"}

    while True:
        try:
            loop_start = time.time()

            # 1. Säker läsning från command_queue
            try:
                cmd = command_queue.get_nowait()
                if isinstance(cmd, dict) and "mode" in cmd:
                    mode = cmd["mode"]
            except queue.Empty:
                pass

            frame = vs.read()
            detections = yolo.detect(frame)
            x_ort = 0.5

            # 2. Hämta beslut från AI-agenten (Icke-blockerande)
            if time.time() - last_ai_query_time > 1.0:
                current_ai_decision = agent.decide_action(detections)
                last_ai_query_time = time.time()

            # 3. Motorstyrning
            if mode == "AUTONOMOUS":
                if current_ai_decision.get("action") == "TRACK_TARGET":
                    left_spd, right_spd = pid.compute(x_ort)
                    motors.set_motors(left_spd, right_spd)
                else:
                    motors.set_motors(-30.0, 30.0)  # Sökrotering

           # 4. HÄR RITAS YOLO-RUTORNA UT OCH SKICKAS TILL WEBBPLATSEN
            if frame is not None and frame.size > 0:
                # Rita rutorna från senaste YOLO-detekteringen på webb-bilden
                annotated_frame = yolo.draw_boxes(frame, detections)

                success, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                    try:
                        frame_queue.put_nowait(buffer.tobytes())
                    except queue.Full:
                        pass

            # 5. Uppdatera telemetri med svensk tidsstämpel
            telemetry_payload = {
                "status": "running",
                "mode": mode,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ai_decision": current_ai_decision,
                "x_ort": x_ort
            }
            try:
                telemetry_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                telemetry_queue.put_nowait(telemetry_payload)
            except queue.Full:
                pass

            # 6. Håll ~60 Hz loop-hastighet
            elapsed = time.time() - loop_start
            time.sleep(max(0.001, (1.0 / 60.0) - elapsed))

        except Exception as e:
            print(f"[MAIN ERROR] Unexpected error in control loop: {e}")
            time.sleep(0.05)
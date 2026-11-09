import os
import time
import queue
import cv2
import numpy as np
from hailo_detector import HailoDetector
from datetime import datetime
from multiprocessing import Queue
from camera import VideoStream
from motor_driver import MegaPiDriver
from ai_agent import CognitiveAgent


AREA_STANN = 0.63 #2m: 0,0735 1,5m: 0,1288 1m: 0,2208 0,5m: 0,6350
AREA_REF = 0.40
K_DIST = 200.0
MAX_FORWARD = 60.0


class PIDController:
    def __init__(self, kp: float = 200.0, ki: float = 5.0, kd: float = 20.0) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_sum = 0.0
        self.prev_diff = 0.0

    def compute(self, x_ort: float, forward: float = 100.0) -> tuple[float, float]:
        ref = 0.5
        diff = ref - x_ort
        self.integral_sum += diff
        self.integral_sum = max(-20.0, min(20.0, self.integral_sum))  # anti-windup
        d_value = diff - self.prev_diff
        self.prev_diff = diff

        correction = (self.kp * abs(diff)) + (self.ki * self.integral_sum) - (self.kd * d_value)

        if x_ort < 0.5:
            left_speed = max(0.0, min(100.0, forward - correction))
            right_speed = max(0.0, min(100.0, forward))
        else:
            left_speed = max(0.0, min(100.0, forward))
            right_speed = max(0.0, min(100.0, forward - correction))

        return left_speed, right_speed


def main(command_queue: Queue, telemetry_queue: Queue, frame_queue: Queue) -> None:
    print("[MAIN] Starting robot control process...")

    yolo = HailoDetector(hef_path="yolov8s.hef", conf_threshold=0.45)
    vs = VideoStream(resolution=(640, 480)).start()
    pid = PIDController(kp=200.0, ki=5.0, kd=20.0)
    motors = MegaPiDriver()
    agent = CognitiveAgent()

    mode = "AUTONOMOUS"
    state = "IDLE"
    last_ai_query_time = 0.0
    last_motor_time = 0.0
    current_ai_decision = {"action": "IDLE", "target_label": "none", "reasoning": "Waiting for AI assignment"}
    reached_target_label = None
    pending_event = None
    completed_labels = set()
    prev_state = None

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

            # Välj ut målet som AI-agenten pekat på (annars ingen styrning)
            label_wanted = current_ai_decision.get("target_label")
            target = None
            for det in detections:
                if label_wanted in (None, "none") or det["label"] == label_wanted:
                    target = det
                    break
            x_ort = target["x_ort"] if target is not None else 0.5

            # 2. Hämta beslut från AI-agenten (Icke-blockerande)
            if time.time() - last_ai_query_time > 1.0:
                dets_for_agent = [d for d in detections if d.get("label") not in completed_labels]
                new_decision = agent.decide_action(dets_for_agent, event=pending_event)
                pending_event = None  # agenten har fatt händelsen, rensa har
                action = new_decision.get("action")
                print(f"[AI] action={new_decision.get('action')} target={new_decision.get('target_label')} | {str(new_decision.get('reasoning'))[:80]}")
                if state != "STOP" or action != "TRACK_TARGET" or (
                    reached_target_label is not None
                    and new_decision.get("target_label") != reached_target_label
                ):
                    current_ai_decision = new_decision
                last_ai_query_time = time.time()

            # 3. Motorstyrning (max ett kommando var 40 ms -> undviker serieflood)
            action = current_ai_decision.get("action")
            if mode == "AUTONOMOUS" and time.time() - last_motor_time >= 0.04:
                last_motor_time = time.time()
                if state == "IDLE":
                    if action == "SEARCH":
                        state = "SEARCH"
                    elif action == "TRACK_TARGET":
                        state = "TRACK"
                    elif action == "STOP":
                        state = "STOP"
                    motors.set_motors(0.0, 0.0)
                elif state == "SEARCH":
                    if action == "TRACK_TARGET":
                        state = "TRACK"
                    elif action == "STOP":
                        state = "STOP"
                    if state == "SEARCH":
                        motors.set_motors(-20.0, 20.0)
                    else:
                        motors.set_motors(0.0, 0.0)
                elif state == "TRACK":
                    if action == "SEARCH":
                        state = "SEARCH"
                        motors.set_motors(-20.0, 20.0)
                    elif action == "STOP":
                        state = "STOP"
                        motors.set_motors(0.0, 0.0)
                    elif action != "TRACK_TARGET":
                        state = "IDLE"
                        motors.set_motors(0.0, 0.0)
                    elif target is None:
                        motors.set_motors(0.0, 0.0)
                    elif target.get("area", 0.0) < AREA_STANN:
                        state = "APPROACH"
                        forward = max(0.0, min(MAX_FORWARD, K_DIST * (AREA_REF - target["area"])))
                        left_spd, right_spd = pid.compute(x_ort, forward)
                        motors.set_motors(left_spd, right_spd)
                    else:  # area >= AREA_STANN -> mattekt, redan nara nog
                        state = "STOP"
                        reached_target_label = target.get("label")
                        completed_labels.add(target.get("label"))
                        pending_event = f"TARGET_REACHED {target.get('label')}"
                        print("[MAIN] TARGET_REACHED")
                        motors.set_motors(0.0, 0.0)
                elif state == "APPROACH":
                    if action == "SEARCH":
                        state = "SEARCH"
                        motors.set_motors(-20.0, 20.0)
                    elif action == "STOP":
                        state = "STOP"
                        motors.set_motors(0.0, 0.0)
                    elif action != "TRACK_TARGET":
                        state = "IDLE"
                        motors.set_motors(0.0, 0.0)
                    elif target is not None and target.get("area", 0.0) >= AREA_STANN:
                        state = "STOP"
                        reached_target_label = target.get("label")
                        completed_labels.add(target.get("label"))
                        pending_event = f"TARGET_REACHED {target.get('label')}"
                        print("[MAIN] TARGET_REACHED")
                        motors.set_motors(0.0, 0.0)
                    elif target is None:
                        motors.set_motors(0.0, 0.0)
                    else:
                        forward = max(0.0, min(MAX_FORWARD, K_DIST * (AREA_REF - target["area"])))
                        left_spd, right_spd = pid.compute(x_ort, forward)
                        motors.set_motors(left_spd, right_spd)
                elif state == "STOP":
                    if action != "TRACK_TARGET" or (
                        reached_target_label is not None
                        and current_ai_decision.get("target_label") != reached_target_label
                    ):
                        reached_target_label = None
                        if action == "SEARCH":
                            completed_labels.clear()
                            state = "SEARCH"
                        elif action == "TRACK_TARGET":
                            state = "TRACK"
                        elif action == "STOP":
                            state = "STOP"
                        else:
                            state = "IDLE"
                    motors.set_motors(0.0, 0.0)

            # 4. HÄR RITAS YOLO-RUTORNA UT OCH SKICKAS TILL WEBBPLATSEN
            if state != prev_state:
                print(f"[STATE] {prev_state} -> {state} | target={current_ai_decision.get('target_label')} x_ort={x_ort:.2f}")
                prev_state = state
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
                "state": state,
                "x_ort": x_ort,
                "area": target.get("area") if target is not None else None,
                "event": pending_event,
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
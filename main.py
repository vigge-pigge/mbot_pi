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


AREA_REF = 0.30
K_DIST = 200.0
MIN_FORWARD = 50.0
MAX_FORWARD = 60.0
TARGET_ACQUIRE_STOP_S = 1.0  # stopp innan roboten börjar följa vald target
MIN_TARGET_IOU = 0.10


def box_iou(box_a: list[int], box_b: list[int]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def choose_target(
    detections: list[dict], label: str | None, locked_box: list[int] | None
) -> dict | None:
    candidates = [
        detection for detection in detections
        if detection.get("label") == label and detection.get("box")
    ]
    if not candidates:
        return None

    if locked_box is None:
        return max(candidates, key=lambda detection: detection.get("confidence", 0.0))

    target = max(
        candidates,
        key=lambda detection: box_iou(locked_box, detection["box"]),
    )
    if box_iou(locked_box, target["box"]) < MIN_TARGET_IOU:
        return None
    return target


class SteeringController:
    def __init__(self, turn_gain: float = 50.0) -> None:
        self.turn_gain = turn_gain

    def compute(self, x_ort: float, forward: float) -> tuple[float, float]:
        turn = self.turn_gain * (x_ort - 0.5)
        left_speed = max(0.0, min(100.0, forward + turn))
        right_speed = max(0.0, min(100.0, forward - turn))
        return left_speed, right_speed


def main(command_queue: Queue, telemetry_queue: Queue, frame_queue: Queue) -> None:
    print("[MAIN] Starting robot control process...")

    yolo = HailoDetector(hef_path="yolov8s.hef", conf_threshold=0.45)
    vs = VideoStream(resolution=(640, 480)).start()
    steering = SteeringController()
    motors = MegaPiDriver()
    agent = CognitiveAgent()

    mode = "AUTONOMOUS"
    state = "IDLE"
    last_ai_query_time = 0.0
    last_motor_time = 0.0
    current_ai_decision = {"action": "IDLE", "target_label": "none", "reasoning": "Waiting for AI assignment"}
    prev_state = None
    target_stop_until = 0.0
    locked_target_label = None
    locked_target_box = None

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

            # Select by class initially, then follow the matching bounding box.
            label_wanted = current_ai_decision.get("target_label")
            target = choose_target(
                detections,
                locked_target_label or label_wanted,
                locked_target_box,
            )
            if target is not None and locked_target_label is not None:
                locked_target_box = target["box"]
            x_ort = target["x_ort"] if target is not None else 0.5

            # 2. Hämta beslut från AI-agenten (Icke-blockerande)
            if time.time() - last_ai_query_time > 1.0:
                new_decision = agent.decide_action(detections)
                action = new_decision.get("action")
                print(f"[AI] action={new_decision.get('action')} target={new_decision.get('target_label')} | {str(new_decision.get('reasoning'))[:80]}")
                selected_label = new_decision.get("target_label")
                selected_target_visible = any(
                    det.get("label") == selected_label for det in detections
                )
                if locked_target_label is None:
                    current_ai_decision = new_decision
                    if action == "TRACK_TARGET" and selected_target_visible:
                        locked_target_label = selected_label
                        target = choose_target(detections, selected_label, None)
                        locked_target_box = target["box"] if target is not None else None
                        if target is not None:
                            x_ort = target["x_ort"]
                        target_stop_until = time.time() + TARGET_ACQUIRE_STOP_S
                        state = "STOP"
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
                    motors.set_motors(0.0, 0.0)
                elif state == "TRACK":
                    if action == "SEARCH":
                        state = "SEARCH"
                        motors.set_motors(0.0, 0.0)
                    elif action == "STOP":
                        state = "STOP"
                        motors.set_motors(0.0, 0.0)
                    elif action != "TRACK_TARGET":
                        state = "IDLE"
                        motors.set_motors(0.0, 0.0)
                    elif target is None:
                        motors.set_motors(0.0, 0.0)
                    else:
                        state = "APPROACH"
                        forward = max(MIN_FORWARD, min(MAX_FORWARD, K_DIST * (AREA_REF - target["area"])))
                        left_spd, right_spd = steering.compute(x_ort, forward)
                        motors.set_motors(left_spd, right_spd)
                elif state == "APPROACH":
                    if action == "SEARCH":
                        state = "SEARCH"
                        motors.set_motors(0.0, 0.0)
                    elif action == "STOP":
                        state = "STOP"
                        motors.set_motors(0.0, 0.0)
                    elif action != "TRACK_TARGET":
                        state = "IDLE"
                        motors.set_motors(0.0, 0.0)
                    elif target is None:
                        motors.set_motors(0.0, 0.0)
                    else:
                        forward = max(MIN_FORWARD, min(MAX_FORWARD, K_DIST * (AREA_REF - target["area"])))
                        left_spd, right_spd = steering.compute(x_ort, forward)
                        motors.set_motors(left_spd, right_spd)
                elif state == "STOP":
                    now = time.time()
                    if target_stop_until > now:
                        motors.set_motors(0.0, 0.0)
                    elif target_stop_until > 0.0:
                        target_stop_until = 0.0
                        state = "TRACK"
                        motors.set_motors(0.0, 0.0)
                    elif action != "TRACK_TARGET":
                        if action == "SEARCH":
                            state = "SEARCH"
                            motors.set_motors(0.0, 0.0)
                        elif action == "TRACK_TARGET":
                            state = "TRACK"
                        elif action == "STOP":
                            state = "STOP"
                        else:
                            state = "IDLE"
                    else:
                        state = "TRACK"
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
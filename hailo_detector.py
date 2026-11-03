import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Any

import cv2

# Gor hailo-apps importerbart (annars hittas inte paketen under venv)
HAILO_APPS_ROOT = Path.home() / "hailo-apps"
if str(HAILO_APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(HAILO_APPS_ROOT))

from hailo_apps.python.core.common.hailo_inference import HailoInfer  # pyright: ignore[reportMissingImports]
from hailo_apps.python.core.common.toolbox import get_labels  # pyright: ignore[reportMissingImports]
from hailo_apps.python.standalone_apps.object_detection.object_detection_post_process import extract_detections  # pyright: ignore[reportMissingImports]


class HailoDetector:
    """YOLOv8s pa Hailo NPU (AI HAT+).
    Samma granssnitt som gamla YoloDetector, sa main.py behover knappt andras.
    """

    def __init__(self, hef_path: str, conf_threshold: float = 0.45) -> None:
        print(f"[HAILO] Laddar HEF: {hef_path}")
        self.conf_threshold = conf_threshold
        self.infer = HailoInfer(hef_path, batch_size=1)
        self.height, self.width = self.infer.get_input_shape()[:2]
        self.latest_detections: List[Dict[str, Any]] = []
        self.is_processing: bool = False
        self.last_inference_ms: float = 0.0
        self._labels = get_labels(None)   # COCO-etiketter, samma som gamla modellen
        self._config = {
            "visualization_params": {
                "score_thres": conf_threshold,
                "max_boxes_to_draw": 30,
            }
        }
        print(f"[HAILO] NPU redo! Modellinput: {self.width}x{self.height}")

    # --- preprocess: letterbox till modellens input (padded resize) ---
    def _preprocess(self, frame: Any) -> Any:
        h, w = frame.shape[:2]
        scale = min(self.width / w, self.height / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h))
        pad_top = (self.height - new_h) // 2
        pad_left = (self.width - new_w) // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, self.height - new_h - pad_top,
            pad_left, self.width - new_w - pad_left,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        # Hailo-modellen vill ha RGB; kameran levererar BGR
        return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    def _run_inference(self, frame: Any) -> None:
        try:
            start_time = time.time()
            prepped = self._preprocess(frame)

            holder: Dict[str, Any] = {}
            done = threading.Event()

            def _on_done(completion_info, bindings_list) -> None:
                if completion_info.exception:
                    print(f"[HAILO ERROR] {completion_info.exception}")
                else:
                    bindings = bindings_list[0]
                    # yolov8s-HEF:en har inbyggd NMS-postprocess (en utdata)
                    holder["raw"] = bindings.output().get_buffer()
                done.set()

            job = self.infer.run([prepped], _on_done)
            job.wait(10000)
            done.wait(timeout=5.0)
            self.last_inference_ms = (time.time() - start_time) * 1000

            if "raw" not in holder:
                self.latest_detections = []
                return

            result = extract_detections(frame, holder["raw"], self._config)

            detections: List[Dict[str, Any]] = []
            h, w = frame.shape[:2]
            for box, cls_id, score in zip(
                result["detection_boxes"],
                result["detection_classes"],
                result["detection_scores"],
            ):
                if score < self.conf_threshold:
                    continue
                label = (self._labels[int(cls_id)]
                         if int(cls_id) < len(self._labels) else str(cls_id))
                x1, y1, x2, y2 = [int(v) for v in box]
                x_ort = ((x1 + x2) / 2.0) / w
                y_ort = ((y1 + y2) / 2.0) / h
                area = ((x2 - x1) * (y2 - y1)) / float(w * h)
                detections.append({
                    "label": label,
                    "confidence": round(float(score), 2),
                    "x_ort": round(x_ort, 3),
                    "y_ort": round(y_ort, 3),
                    "area": round(area, 4),
                    "box": [x1, y1, x2, y2],
                })

            self.latest_detections = detections
            summary = ", ".join(
                f"{d['label']} ({int(d['confidence']*100)}%, "
                f"x={d['x_ort']:.2f}, y={d['y_ort']:.2f}, area={d['area']:.4f})"
                for d in detections
            )
            print(f"[HAILO {self.last_inference_ms:.1f}ms] Ser: {summary}")
        except Exception as e:
            print(f"[HAILO ERROR] {e}")
        finally:
            self.is_processing = False

    def detect(self, frame: Any) -> List[Dict[str, Any]]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return self.latest_detections
        if not self.is_processing:
            self.is_processing = True
            thread = threading.Thread(
                target=self._run_inference, args=(frame.copy(),), daemon=True
            )
            thread.start()
        return self.latest_detections

    def draw_boxes(self, frame: Any, detections: List[Dict[str, Any]]) -> Any:
        """Ritar ut detektionsrutor for webbvisning (samma som gamla YoloDetector)."""
        if frame is None or not detections:
            return frame
        annotated = frame.copy()
        for det in detections:
            box = det.get("box")
            if box:
                x1, y1, x2, y2 = box
                label = f"{det['label']} {int(det['confidence'] * 100)}%"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, max(y1 - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated
import time
import cv2
import threading
from typing import List, Dict, Any
from ultralytics import YOLO


class YoloDetector:
    """Icke-blockerande realtidsdetektering med YOLOv8 för Raspberry Pi 5."""

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.45) -> None:
        print(f"[YOLO] Laddar modell: {model_name}...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.latest_detections: List[Dict[str, Any]] = []
        self.is_processing: bool = False
        print("[YOLO] Modell laddad och redo!")

    def _run_inference(self, frame: Any) -> None:
        try:
            start_time = time.time()
            results_list: Any = self.model(frame, imgsz=320, stream=False, verbose=False)
            if not results_list:
                self.latest_detections = []
                return

            result = results_list[0]
            inference_time_ms = (time.time() - start_time) * 1000

            detections: List[Dict[str, Any]] = []
            terminal_summary: List[str] = []
            h, w = frame.shape[:2]

            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf >= self.conf_threshold:
                        cls_id = int(box.cls[0])
                        names: Dict[int, str] = getattr(self.model, "names", {})
                        label = names.get(cls_id, str(cls_id))

                        xyxy = box.xyxy[0].tolist()
                        x_center = ((xyxy[0] + xyxy[2]) / 2.0) / w

                        detections.append({
                            "label": label,
                            "confidence": round(conf, 2),
                            "x_ort": round(x_center, 3),
                            "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        })
                        terminal_summary.append(f"{label} ({int(conf*100)}%, x={x_center:.2f})")

            self.latest_detections = detections
            #if terminal_summary:
                #print(f"[YOLO {inference_time_ms:.1f}ms] Ser: {', '.join(terminal_summary)}")
        except Exception as e:
            print(f"[YOLO ERROR] {e}")
        finally:
            self.is_processing = False

    def detect(self, frame: Any) -> List[Dict[str, Any]]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return self.latest_detections

        if not self.is_processing:
            self.is_processing = True
            thread = threading.Thread(
                target=self._run_inference, 
                args=(frame.copy(),), 
                daemon=True
            )
            thread.start()

        return self.latest_detections

    def draw_boxes(self, frame: Any, detections: List[Dict[str, Any]]) -> Any:
        """Ritar ut YOLO-rutor och etiketter på bildrutan för webbvisning."""
        if frame is None or not detections:
            return frame

        annotated_frame = frame.copy()
        for det in detections:
            box = det.get("box")
            if box:
                x1, y1, x2, y2 = box
                label = f"{det['label']} {int(det['confidence'] * 100)}%"
                
                # Rita grön rektangel
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Rita textetikett ovanför boxen
                cv2.putText(annotated_frame, label, (x1, max(y1 - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_frame
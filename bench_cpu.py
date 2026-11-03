# Jämförelsetest: yolov8n pa CPU (gamla konfigurationen)
import time
from camera import VideoStream
from yolo_detector import YoloDetector

vs = VideoStream(resolution=(640, 480)).start()
det = YoloDetector(model_name="yolov8n.pt", conf_threshold=0.45)
time.sleep(2)  # lat kameran komma igang

DURATION = 20.0
n = 0
times = []
print(f"[BENCH] Startar {DURATION}s test med yolov8n pa CPU...")
t0 = time.time()
while time.time() - t0 < DURATION:
    frame = vs.read()
    det.is_processing = True      # stoppar detect():s traddlogik
    t1 = time.time()
    det._run_inference(frame)     # kor synkront: en bild i taget
    times.append((time.time() - t1) * 1000)
    n += 1
elapsed = time.time() - t0

fps = n / elapsed
mean_ms = sum(times) / len(times)
print(f"[BENCH] Bilder: {n}, Tid: {elapsed:.1f}s")
print(f"[BENCH] FPS: {fps:.2f}")
print(f"[BENCH] Inferenstid: medel {mean_ms:.1f} ms, "
      f"min {min(times):.1f}, max {max(times):.1f} (n={n})")

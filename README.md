# mbot_pi

Pi-sidan av den autonoma mBot Mega-roboten (gymnasiearbete).
Objektdetektering med YOLOv8s pa Hailo NPU (Raspberry Pi AI HAT+),
PID-styrning och kognitiv AI-agent via Ollama.

## Installation

1. Kamera + AI HAT+ monterat, `sudo apt install hailo-all` installerat.
2. `hailo-apps` klonad till `~/hailo-apps`:
   `git clone https://github.com/hailo-ai/hailo-apps.git`
3. Venv skapas med systempaket:
   `python3 -m venv --system-site-packages venv && source venv/bin/activate`
4. `pip install flask flask-socketio requests opencv-python numpy scipy`
   `pip install -r ~/hailo-apps/hailo_apps/python/standalone_apps/object_detection/requirements.txt`
5. **Ladda ner modellen** (yolov8s.hef ar gitignorerad, den foljer inte med klonen):
cd ~/hailo-apps/hailo_apps/python/standalone_apps/object_detection
python object_detection.py -n yolov8s -i rpi --no-display -t 5
cp $(find ~ -iname "yolov8s*.hef" 2>/dev/null | head -1) ~/robot_project/yolov8s.hef

## Kora

 source venv/bin/activate
 python app.py

Styr via webbgranssnittet. Stang alla andra program som anvander NPU:n forst.

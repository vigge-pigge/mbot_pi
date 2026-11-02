import time
import queue
from typing import Generator
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from multiprocessing import Process, Queue
from main import main

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

command_queue: Queue = Queue(maxsize=10)
telemetry_queue: Queue = Queue(maxsize=5)
frame_queue: Queue = Queue(maxsize=2)


def telemetry_loop() -> None:
    while True:
        try:
            data = telemetry_queue.get(timeout=0.1)
            socketio.emit('telemetry', data)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[SOCKETIO] Error in telemetry_loop: {e}")
        time.sleep(0.05)


def gen_frames() -> Generator[bytes, None, None]:
    frame_count = 0
    last_fps_update = time.monotonic()

    while True:
        try:
            # Töm kön helt för att BARA plocka den absolut senaste bildrutan!
            frame_bytes = None
            while not frame_queue.empty():
                frame_bytes = frame_queue.get_nowait()

            if frame_bytes is not None:
                frame_count += 1
                now = time.monotonic()
                elapsed = now - last_fps_update
                if elapsed >= 1:
                    socketio.emit('video_fps', {
                        'fps': round(frame_count / elapsed),
                    })
                    frame_count = 0
                    last_fps_update = now

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[STREAM ERROR] {e}")

        # Begränsa webbflödet till ~30 FPS så att nätverket inte blir överbelastat
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on_error_default
def default_error_handler(e):
    """Global SocketIO error handler (Punkt 2)."""
    print(f"[SOCKETIO ERROR] Captured default error: {e}")


@socketio.on('manual_control')
def handle_control(data: dict) -> None:
    try:
        command_queue.put_nowait(data)
    except queue.Full:
        pass


if __name__ == '__main__':
    robot_process = Process(target=main, args=(command_queue, telemetry_queue, frame_queue))
    robot_process.start()

    socketio.start_background_task(telemetry_loop)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
import glob
import time
import serial
from typing import Optional


class MegaPiDriver:
    """Serial communication driver with auto-port detection and simulation fallback."""

    def __init__(self, baudrate: int = 115200) -> None:
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self._connect()

    def _connect(self) -> None:
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        for port in ports:
            try:
                self.ser = serial.Serial(port, self.baudrate, timeout=1.0)
                time.sleep(2.0)
                print(f"[MOTOR] Connected to MegaPi on port: {port}")
                return
            except Exception as e:
                print(f"[MOTOR] Failed to connect on {port}: {e}")

        print("[MOTOR] WARNING: No MegaPi hardware found. Running in SIMULATION MODE.")
        self.ser = None

    def set_motors(self, left_speed: float, right_speed: float) -> None:
        left_clamped = max(-100.0, min(100.0, left_speed))
        right_clamped = max(-100.0, min(100.0, right_speed))

        left_pwm = int((left_clamped / 100.0) * 255)
        right_pwm = int((right_clamped / 100.0) * 255)

        command_str = f"M {left_pwm} {right_pwm}\n"

        if self.ser and self.ser.is_open:
            try:
                self.ser.write(command_str.encode("utf-8"))
            except Exception as e:
                print(f"[MOTOR] Serial write error: {e}")
                self.ser = None

    def stop(self) -> None:
        self.set_motors(0.0, 0.0)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[MOTOR] Serial connection closed.")
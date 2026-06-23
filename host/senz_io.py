#!/usr/bin/env python3
"""
senz_io.py
==========
Shared I/O and math for the senz glove host apps (Qt and matplotlib viz).

Provides:
  - parse_frame(line)            CSV line -> (fingers, quaternion, cal)
  - quat_to_matrix / euler_to_quat / continuous_quat / rotate
  - SerialSource / SimSource / BLESource / ThreadedReader
  - open_source(...)             pick a source from CLI-style flags

CSV frame format (must match firmware):
    f0[,f1...],qw,qx,qy,qz,cal
"""

import math
import threading
import time

import numpy as np

# Must match FINGER_PINS[] in the firmware.
NUM_FINGERS = 1
FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"]
ADC_MAX = 4095

# Nordic UART Service UUIDs (must match the firmware) for BLE mode.
BLE_NAME = "senz-glove"
NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # device -> host (notify)
NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # host -> device (write)


# ----------------------------------------------------------------------------
# Parsing + math
# ----------------------------------------------------------------------------
def parse_frame(line):
    """Parse one CSV line into (fingers, (w,x,y,z), cal) or None."""
    parts = line.split(",")
    if len(parts) < NUM_FINGERS + 4:
        return None
    try:
        fingers = [int(parts[i]) for i in range(NUM_FINGERS)]
        q = tuple(float(parts[NUM_FINGERS + i]) for i in range(4))
        cal = int(parts[NUM_FINGERS + 4]) if len(parts) > NUM_FINGERS + 4 else None
    except ValueError:
        return None
    return fingers, q, cal


def quat_to_matrix(w, x, y, z):
    """Rotation matrix from a quaternion (no gimbal lock)."""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_to_quat(roll, pitch, yaw):
    """Euler (deg) -> quaternion (w,x,y,z). Used only by the simulator."""
    r, p, y = (math.radians(a) / 2 for a in (roll, pitch, yaw))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def continuous_quat(q, prev):
    """Flip q to the same hemisphere as prev (q and -q are the same rotation)."""
    if prev is not None and sum(a * b for a, b in zip(q, prev)) < 0:
        return tuple(-c for c in q)
    return tuple(q)


def rotate(R, pts):
    """Rotate an (N,3) array of points by matrix R."""
    return (R @ np.asarray(pts).T).T


# ----------------------------------------------------------------------------
# Data sources
# ----------------------------------------------------------------------------
class SerialSource:
    """Reads CSV frames from the glove over serial."""

    def __init__(self, port, baud=115200):
        import serial  # pyserial

        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)  # let the ESP32-C3 reset/boot
        self.ser.reset_input_buffer()

    def read(self):
        try:
            self.ser.write(b".")  # heartbeat -> OLED LINK: OK
        except Exception:
            pass
        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return None
        return parse_frame(line)

    def close(self):
        self.ser.close()


class SimSource:
    """Fake but plausible data so the viz works without hardware."""

    def __init__(self):
        self.t0 = time.time()

    def read(self):
        t = time.time() - self.t0
        fingers = []
        for i in range(NUM_FINGERS):
            phase = t * 1.5 - i * 0.6
            fingers.append(int((math.sin(phase) + 1) / 2 * ADC_MAX))
        roll = 30 * math.sin(t * 0.7)
        pitch = 80 * math.sin(t * 0.5 + 1)  # large pitch to exercise up/down
        yaw = (t * 20) % 360 - 180
        q = euler_to_quat(roll, pitch, yaw)
        time.sleep(1 / 100)  # mimic ~100 Hz
        return fingers, q, 0xFF

    def close(self):
        pass


class BLESource:
    """Connects to the glove over BLE (Nordic UART Service) using bleak.

    Runs an asyncio loop on a background thread, subscribes to TX notifications,
    keeps the latest parsed frame. Exposes get()/close() like ThreadedReader.
    """

    def __init__(self, name=BLE_NAME):
        self._name = name
        self._latest = None
        self._buf = b""
        self._lock = threading.Lock()
        self._running = True
        import asyncio

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import asyncio

        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            print("BLE error:", e)

    async def _main(self):
        import asyncio

        from bleak import BleakClient, BleakScanner

        print(f"Scanning for BLE device '{self._name}' ...")
        device = await BleakScanner.find_device_by_name(self._name, timeout=15.0)
        if device is None:
            print(f"Could not find '{self._name}'. Powered and in BLE mode?")
            return
        async with BleakClient(device) as client:
            print(f"Connected to {self._name}")

            def on_notify(_char, data):
                self._feed(bytes(data))

            await client.start_notify(NUS_TX, on_notify)
            while self._running and client.is_connected:
                try:
                    await client.write_gatt_char(NUS_RX, b".", response=False)
                except Exception:
                    pass
                await asyncio.sleep(0.5)  # heartbeat -> OLED LINK: OK

    def _feed(self, data):
        self._buf += data
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            frame = parse_frame(raw.decode("utf-8", errors="ignore").strip())
            if frame is not None:
                with self._lock:
                    self._latest = frame

    def get(self):
        with self._lock:
            return self._latest

    def close(self):
        self._running = False


class ThreadedReader:
    """Wraps a pull-based source (serial/sim) so reads never block the GUI."""

    def __init__(self, source):
        self.source = source
        self._latest = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                d = self.source.read()
            except Exception:
                d = None
            if d is not None:
                with self._lock:
                    self._latest = d

    def get(self):
        with self._lock:
            return self._latest

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.source.close()


def open_source(port=None, ble=False, name=BLE_NAME, simulate=False, baud=115200):
    """Return a reader exposing get()/close() for the chosen transport."""
    if simulate:
        return ThreadedReader(SimSource())
    if ble:
        return BLESource(name)  # manages its own thread
    if port:
        return ThreadedReader(SerialSource(port, baud))
    raise ValueError("provide port=..., ble=True, or simulate=True")

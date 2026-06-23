#!/usr/bin/env python3
"""
live_hand_viz.py
================
Live 3D visualization for the senz glove prototype.

Reads CSV frames from the ESP32-C3 over USB serial:

    f0[,f1...],roll,pitch,yaw,cal

  f0.. : raw 12-bit ADC (0..4095), one per finger
  roll,pitch,yaw : degrees (fused on-chip by the BNO055)
  cal : BNO055 calibration status byte (optional; sys/gyro/accel/mag each 0..3)

It draws a 3D hand: each finger curls based on its calibrated pot value, and the
whole hand is rotated by the IMU orientation (roll/pitch/yaw). You can orbit the
view with the mouse. The IMU gives orientation, not position -- see
docs/PROTOTYPE.md.

Usage:
    # Live over USB:
    python live_hand_viz.py --port COM5

    # Live over Bluetooth (BLE; flash firmware with USE_BLE true):
    python live_hand_viz.py --ble

    # No hardware yet? Develop the viz against fake data:
    python live_hand_viz.py --simulate

Calibration (per finger range):
    Press 'o' with fingers fully EXTENDED (open hand),
    then 'c' with fingers fully FLEXED (closed fist).

Dependencies:
    pip install pyserial matplotlib numpy
    pip install bleak        # only needed for --ble
"""

import argparse
import math
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Must match the number of entries in FINGER_PINS[] in the firmware.
# Currently 1 pot; bump to 4 (and extend FINGER_NAMES) when you add more.
NUM_FINGERS = 1
FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"]
ADC_MAX = 4095

# Nordic UART Service UUIDs (must match the firmware) for BLE mode.
BLE_NAME = "senz-glove"
NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # device -> host (notify)
NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # host -> device (write)


def parse_frame(line):
    """Parse one CSV line into (fingers, quaternion, cal) or None.

    quaternion is (w, x, y, z). cal is the BNO055 status byte or None.
    """
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


# ----------------------------------------------------------------------------
# Data sources
# ----------------------------------------------------------------------------
class SerialSource:
    """Reads CSV frames from the glove over serial."""

    def __init__(self, port, baud=115200):
        import serial  # pyserial, imported lazily so --simulate needs no install

        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)  # let the ESP32-C3 reset/boot
        self.ser.reset_input_buffer()

    def read(self):
        # Heartbeat so the glove's OLED shows LINK: OK while we're reading.
        try:
            self.ser.write(b".")
        except Exception:
            pass
        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return None
        return parse_frame(line)

    def close(self):
        self.ser.close()


class BLESource:
    """Connects to the glove over BLE (Nordic UART Service) using bleak.

    Runs an asyncio event loop on a background thread, subscribes to TX
    notifications, and keeps the most recent parsed frame. Exposes get()/close()
    like ThreadedReader, so it plugs straight into the renderer.
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
        from bleak import BleakClient, BleakScanner

        print(f"Scanning for BLE device '{self._name}' ...")
        device = await BleakScanner.find_device_by_name(self._name, timeout=15.0)
        if device is None:
            print(
                f"Could not find '{self._name}'. Is the glove powered and in BLE mode?"
            )
            return
        async with BleakClient(device) as client:
            print(f"Connected to {self._name}")

            def on_notify(_char, data):
                self._feed(bytes(data))

            await client.start_notify(NUS_TX, on_notify)
            import asyncio

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
    """Reads the source on a background thread so the GUI never blocks.

    The renderer just grabs the most recent frame; serial I/O latency no longer
    stalls the animation, which makes the view much smoother.
    """

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


class SimSource:
    """Generates fake but plausible data so the viz can be built without hardware."""

    def __init__(self):
        self.t0 = time.time()

    def read(self):
        t = time.time() - self.t0
        fingers = []
        for i in range(NUM_FINGERS):
            phase = t * 1.5 - i * 0.6
            v = (math.sin(phase) + 1) / 2  # 0..1
            fingers.append(int(v * ADC_MAX))
        roll = 30 * math.sin(t * 0.7)
        pitch = 80 * math.sin(t * 0.5 + 1)  # large pitch to exercise up/down
        yaw = (t * 20) % 360 - 180
        q = euler_to_quat(roll, pitch, yaw)
        cal = 0xFF
        time.sleep(1 / 50)  # mimic ~50 Hz
        return fingers, q, cal

    def close(self):
        pass


# ----------------------------------------------------------------------------
# 3D math
# ----------------------------------------------------------------------------
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


def rotate(R, pts):
    """Rotate an (N,3) array of points by matrix R."""
    return (R @ np.asarray(pts).T).T


# ----------------------------------------------------------------------------
# 3D hand
# ----------------------------------------------------------------------------
class HandView3D:
    """Palm + fingers drawn in 3D. Fingers curl; whole hand rotates by the IMU.

    Local frame: X = across palm, Y = finger direction, Z = back of hand (up).
    Fingers point +Y when extended and curl toward -Z (toward the palm).
    """

    def __init__(self, ax):
        self.ax = ax
        self.finger_x = np.linspace(-1.5, 1.5, NUM_FINGERS)
        self.seg_len = [1.0, 0.8, 0.6]

        # Palm as a 3D box (slab) with thickness in Z.
        t = 0.4  # half-thickness
        x0, x1, y0, y1 = -2.0, 2.0, -2.0, 0.0
        self.box_corners = np.array(
            [
                [x0, y0, -t],
                [x1, y0, -t],
                [x1, y1, -t],
                [x0, y1, -t],  # 0-3 bottom
                [x0, y0, t],
                [x1, y0, t],
                [x1, y1, t],
                [x0, y1, t],  # 4-7 top
            ]
        )
        # Six faces of the box (vertex index quads) -> filled solid.
        self.box_faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [3, 2, 6, 7],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5],  # right
        ]
        self.palm_poly = Poly3DCollection(
            [], facecolor="tab:blue", edgecolor="k", linewidths=1, alpha=0.6
        )
        ax.add_collection3d(self.palm_poly)

        # Palm normal vector (perpendicular to the back of the hand, +Z).
        self.palm_center = np.array([0.0, -1.0, 0.0])
        self.normal_tip = self.palm_center + np.array([0.0, 0.0, 2.5])
        (self.normal_line,) = ax.plot([], [], [], "-o", lw=2, ms=5, color="tab:red")

        self.finger_lines = []
        for _ in range(NUM_FINGERS):
            (ln,) = ax.plot([], [], [], "-o", lw=4, ms=5)
            self.finger_lines.append(ln)

        lim = 4
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_title("senz glove - live 3D hand")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=-60)

    def _finger_points(self, x0, flex):
        """Chain of joints for one finger, curling with flexion."""
        bend = math.radians(flex * 75.0)  # per-joint bend
        pts = [[x0, 0.0, 0.0]]
        cur = np.array([x0, 0.0, 0.0])
        angle = 0.0
        for seg in self.seg_len:
            angle += bend
            cur = cur + np.array([0.0, seg * math.cos(angle), -seg * math.sin(angle)])
            pts.append(cur.copy())
        return np.array(pts)

    def update(self, flex, R):
        """flex: list of 0..1 per finger. R: 3x3 rotation matrix to apply."""
        # Palm box as filled faces.
        corners = rotate(R, self.box_corners)
        self.palm_poly.set_verts([corners[face] for face in self.box_faces])

        # Palm normal vector.
        nrm = rotate(R, np.array([self.palm_center, self.normal_tip]))
        self.normal_line.set_data(nrm[:, 0], nrm[:, 1])
        self.normal_line.set_3d_properties(nrm[:, 2])

        for i, ln in enumerate(self.finger_lines):
            pts = self._finger_points(self.finger_x[i], flex[i])
            pts = rotate(R, pts)
            ln.set_data(pts[:, 0], pts[:, 1])
            ln.set_3d_properties(pts[:, 2])


# ----------------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="senz glove live 3D visualization")
    ap.add_argument("--port", help="Serial port, e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--ble",
        action="store_true",
        help="Connect wirelessly over BLE (needs USE_BLE in firmware)",
    )
    ap.add_argument("--name", default=BLE_NAME, help="BLE device name to connect to")
    ap.add_argument(
        "--simulate",
        action="store_true",
        help="Use fake data instead of a real connection",
    )
    args = ap.parse_args()

    if args.simulate:
        # Read on a background thread so the GUI stays smooth.
        reader = ThreadedReader(SimSource())
    elif args.ble:
        # BLESource manages its own background thread + latest frame.
        reader = BLESource(args.name)
    elif args.port:
        reader = ThreadedReader(SerialSource(args.port, args.baud))
    else:
        ap.error("provide --port PORT, use --ble, or use --simulate")

    cal_open = [0] * NUM_FINGERS
    cal_closed = [ADC_MAX] * NUM_FINGERS
    last_raw = [0] * NUM_FINGERS

    # Orientation tare: 'offset' is the baseline; displayed = offset^T * current.
    orient = {"offset": np.eye(3), "current": np.eye(3)}

    # Previous quaternion, used to keep the sign continuous (q and -q are the
    # same rotation; the BNO055 flips between them, which flips A1/A2/A3).
    prev_q = [None]

    # Axis remapper. Inputs are ambiguously named Axis 1/2/3 (= roll/pitch/yaw).
    # For each model rotation axis (X, Y, Z) we pick which input drives it and a
    # sign. 'src' indexes the input list; 'sign' inverts it.
    mods = {
        "src": [0, 1, 2],  # model X<-Axis1, Y<-Axis2, Z<-Axis3 by default
        "sign": [1, 1, 1],  # per model axis inversion
        "inv_finger": False,
    }
    AXIS_LABELS = ["A1", "A2", "A3"]

    fig = plt.figure(figsize=(11, 6))
    fig.subplots_adjust(bottom=0.16)
    ax_hand = fig.add_subplot(1, 2, 1, projection="3d")
    ax_info = fig.add_subplot(1, 2, 2)
    hand = HandView3D(ax_hand)

    ax_info.axis("off")
    info_text = ax_info.text(
        0.02, 0.98, "", va="top", family="monospace", transform=ax_info.transAxes
    )

    def normalize(raw):
        out = []
        for i in range(NUM_FINGERS):
            lo, hi = cal_open[i], cal_closed[i]
            if hi == lo:
                out.append(0.0)
            else:
                out.append(float(np.clip((raw[i] - lo) / (hi - lo), 0.0, 1.0)))
        return out

    def reset_orientation(event=None):
        orient["offset"] = orient["current"].copy()
        print("Orientation reset - current pose is now level baseline.")

    def on_key(event):
        nonlocal cal_open, cal_closed
        if event.key == "o":
            cal_open = list(last_raw)
            print("Captured OPEN (extended):", cal_open)
        elif event.key == "c":
            cal_closed = list(last_raw)
            print("Captured CLOSED (flexed):", cal_closed)
        elif event.key == "r":
            reset_orientation()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # --- Button bar (two rows) ---
    buttons = []  # keep references so they aren't garbage-collected
    MODEL_AXES = ["X", "Y", "Z"]

    def add_src_button(x, y, i):
        ax = fig.add_axes([x, y, 0.12, 0.05])
        b = Button(ax, f"Rot{MODEL_AXES[i]}<{AXIS_LABELS[mods['src'][i]]}")

        def cb(_evt):
            mods["src"][i] = (mods["src"][i] + 1) % 3  # cycle A1->A2->A3
            b.label.set_text(f"Rot{MODEL_AXES[i]}<{AXIS_LABELS[mods['src'][i]]}")
            fig.canvas.draw_idle()

        b.on_clicked(cb)
        buttons.append(b)

    def add_sign_button(x, y, i):
        ax = fig.add_axes([x, y, 0.12, 0.05])
        b = Button(ax, f"Rot{MODEL_AXES[i]} +")

        def cb(_evt):
            mods["sign"][i] *= -1
            sgn = "-" if mods["sign"][i] < 0 else "+"
            b.label.set_text(f"Rot{MODEL_AXES[i]} {sgn}")
            fig.canvas.draw_idle()

        b.on_clicked(cb)
        buttons.append(b)

    # Row 1 (y=0.09): reset + source-remap buttons.
    reset_ax = fig.add_axes([0.20, 0.09, 0.12, 0.05])
    reset_btn = Button(reset_ax, "Reset (r)")
    reset_btn.on_clicked(reset_orientation)
    buttons.append(reset_btn)
    add_src_button(0.34, 0.09, 0)
    add_src_button(0.48, 0.09, 1)
    add_src_button(0.62, 0.09, 2)

    # Row 2 (y=0.02): finger invert + per-axis sign invert.
    finger_ax = fig.add_axes([0.20, 0.02, 0.12, 0.05])
    finger_btn = Button(finger_ax, "Finger:off")

    def finger_cb(_evt):
        mods["inv_finger"] = not mods["inv_finger"]
        finger_btn.label.set_text(f"Finger:{'ON' if mods['inv_finger'] else 'off'}")
        fig.canvas.draw_idle()

    finger_btn.on_clicked(finger_cb)
    buttons.append(finger_btn)
    add_sign_button(0.34, 0.02, 0)
    add_sign_button(0.48, 0.02, 1)
    add_sign_button(0.62, 0.02, 2)

    def update(_frame):
        data = reader.get()  # most recent frame; never blocks on I/O
        if data is None:
            return
        fingers, q, cal = data
        for i in range(NUM_FINGERS):
            last_raw[i] = fingers[i]
        flex = normalize(fingers)
        if mods["inv_finger"]:
            flex = [1.0 - f for f in flex]

        # Enforce sign continuity: if this quaternion points the opposite way
        # from the last one, negate it (same rotation, stable components).
        q = list(q)
        if prev_q[0] is not None and sum(a * b for a, b in zip(q, prev_q[0])) < 0:
            q = [-c for c in q]
        prev_q[0] = q

        # Quaternion: w + vector (x,y,z). Treat the 3 vector parts as Axis 1/2/3
        # and remap/invert them onto the model axes (fixes mirroring/swaps with
        # no gimbal lock).
        qw, qv = q[0], [q[1], q[2], q[3]]
        mx = mods["sign"][0] * qv[mods["src"][0]]
        my = mods["sign"][1] * qv[mods["src"][1]]
        mz = mods["sign"][2] * qv[mods["src"][2]]

        # Apply orientation tare: show pose relative to the saved baseline.
        orient["current"] = quat_to_matrix(qw, mx, my, mz)
        R_disp = orient["offset"].T @ orient["current"]
        hand.update(flex, R_disp)

        lines = ["FINGER FLEXION (0=open 1=fist)", ""]
        for i in range(NUM_FINGERS):
            bar = "#" * int(flex[i] * 20)
            lines.append(f"{FINGER_NAMES[i]:>7}: {flex[i]:.2f} |{bar:<20}|")
        mapping = "  ".join(
            f"Rot{MODEL_AXES[i]}<{AXIS_LABELS[mods['src'][i]]}"
            f"{'-' if mods['sign'][i] < 0 else '+'}"
            for i in range(3)
        )
        lines += [
            "",
            "ORIENTATION (quaternion)",
            "",
            f"  w {q[0]:6.3f}",
            f"  A1 (x) {q[1]:6.3f}",
            f"  A2 (y) {q[2]:6.3f}",
            f"  A3 (z) {q[3]:6.3f}",
            "",
            "MAPPING",
            f"  {mapping}",
        ]
        if cal is not None:
            lines += [
                "",
                "BNO055 CALIBRATION (3=best)",
                f"  sys {(cal >> 6) & 3}  gyro {(cal >> 4) & 3}"
                f"  acc {(cal >> 2) & 3}  mag {cal & 3}",
            ]
        lines += [
            "",
            "Keys: 'o'=open 'c'=fist 'r'=reset",
            "Buttons: Rot?<Ax = remap, +/- = invert",
            "Red arrow = palm normal. Orbit by drag.",
        ]
        info_text.set_text("\n".join(lines))

    # ~60 fps target; matplotlib 3D is CPU-rendered so actual fps may be lower.
    anim = FuncAnimation(fig, update, interval=16, cache_frame_data=False)
    try:
        plt.show()
    finally:
        reader.close()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
live_hand_qt.py
===============
GPU-accelerated live 3D hand visualization for the senz glove (pyqtgraph/OpenGL).

This is the recommended viz: it renders on the GPU, so it stays smooth (60-120
fps) where the matplotlib version (live_hand_viz.py) is CPU-bound (~20-40 fps).

Reads quaternion frames from the glove over USB serial or BLE (or a simulator),
draws a filled palm box, finger(s) that curl, and a palm normal vector. Controls
let you remap/invert the orientation axes (Axis 1/2/3), invert finger direction,
and zero the orientation baseline (tare).

Usage:
    python live_hand_qt.py --port COM5      # USB
    python live_hand_qt.py --ble            # Bluetooth (firmware USE_BLE true)
    python live_hand_qt.py --simulate       # no hardware

Dependencies:
    pip install pyqtgraph PyOpenGL PyQt5 pyserial numpy
    pip install bleak        # only for --ble
"""

import argparse
import sys

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import senz_io
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from senz_io import (
    ADC_MAX,
    FINGER_NAMES,
    NUM_FINGERS,
    continuous_quat,
    quat_to_matrix,
    rotate,
)

AXIS_LABELS = ["A1", "A2", "A3"]
MODEL_AXES = ["X", "Y", "Z"]
FINGER_COLORS = [
    (1.0, 0.6, 0.1, 1.0),
    (0.2, 0.9, 0.4, 1.0),
    (0.3, 0.6, 1.0, 1.0),
    (0.9, 0.3, 0.8, 1.0),
]


class HandGL:
    """Filled palm box + curling fingers + normal vector, drawn in OpenGL."""

    def __init__(self, view):
        t = 0.4  # palm half-thickness
        x0, x1, y0, y1 = -2.0, 2.0, -2.0, 0.0
        self.box_local = np.array(
            [
                [x0, y0, -t],
                [x1, y0, -t],
                [x1, y1, -t],
                [x0, y1, -t],
                [x0, y0, t],
                [x1, y0, t],
                [x1, y1, t],
                [x0, y1, t],
            ]
        )
        self.box_faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # bottom
                [4, 5, 6],
                [4, 6, 7],  # top
                [0, 1, 5],
                [0, 5, 4],  # front
                [3, 2, 6],
                [3, 6, 7],  # back
                [0, 3, 7],
                [0, 7, 4],  # left
                [1, 2, 6],
                [1, 6, 5],  # right
            ]
        )
        self.palm = gl.GLMeshItem(
            vertexes=self.box_local,
            faces=self.box_faces,
            color=(0.25, 0.5, 1.0, 0.55),
            drawEdges=True,
            edgeColor=(0, 0, 0, 1),
            smooth=False,
            glOptions="translucent",
        )
        view.addItem(self.palm)

        self.finger_x = np.linspace(-1.5, 1.5, NUM_FINGERS)
        self.seg_len = [1.0, 0.8, 0.6]
        self.finger_items = []
        for i in range(NUM_FINGERS):
            it = gl.GLLinePlotItem(
                width=6, antialias=True, color=FINGER_COLORS[i % len(FINGER_COLORS)]
            )
            view.addItem(it)
            self.finger_items.append(it)

        self.palm_center = np.array([0.0, -1.0, 0.0])
        self.normal_tip = self.palm_center + np.array([0.0, 0.0, 2.5])
        self.normal = gl.GLLinePlotItem(
            width=4, antialias=True, color=(1.0, 0.1, 0.1, 1.0)
        )
        view.addItem(self.normal)

    def _finger_points(self, x0, flex):
        bend = np.radians(flex * 75.0)
        pts = [np.array([x0, 0.0, 0.0])]
        cur = pts[0].copy()
        angle = 0.0
        for seg in self.seg_len:
            angle += bend
            cur = cur + np.array([0.0, seg * np.cos(angle), -seg * np.sin(angle)])
            pts.append(cur.copy())
        return np.array(pts)

    def update(self, flex, R):
        self.palm.setMeshData(vertexes=rotate(R, self.box_local), faces=self.box_faces)
        for i, it in enumerate(self.finger_items):
            it.setData(pos=rotate(R, self._finger_points(self.finger_x[i], flex[i])))
        self.normal.setData(
            pos=rotate(R, np.array([self.palm_center, self.normal_tip]))
        )


def main():
    ap = argparse.ArgumentParser(description="senz glove GPU 3D visualization")
    ap.add_argument("--port", help="Serial port, e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--ble", action="store_true", help="Connect over BLE")
    ap.add_argument("--name", default=senz_io.BLE_NAME, help="BLE device name")
    ap.add_argument("--simulate", action="store_true", help="Use fake data")
    args = ap.parse_args()

    try:
        reader = senz_io.open_source(
            port=args.port,
            ble=args.ble,
            name=args.name,
            simulate=args.simulate,
            baud=args.baud,
        )
    except ValueError as e:
        ap.error(str(e))

    # State
    mods = {"src": [0, 1, 2], "sign": [1, 1, 1], "inv_finger": False}
    cal = {"open": [0] * NUM_FINGERS, "closed": [ADC_MAX] * NUM_FINGERS}
    last_raw = [0] * NUM_FINGERS
    offset = {"R": np.eye(3), "cur": np.eye(3)}
    prev_q = [None]

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.setWindowTitle("senz glove - live 3D hand (GPU)")
    win.resize(1100, 640)
    root = QtWidgets.QHBoxLayout(win)

    # --- 3D view ---
    view = gl.GLViewWidget()
    view.setCameraPosition(distance=14, elevation=20, azimuth=-60)
    grid = gl.GLGridItem()
    grid.setSize(12, 12)
    grid.setSpacing(1, 1)
    view.addItem(grid)
    hand = HandGL(view)
    root.addWidget(view, stretch=3)

    # --- Control panel ---
    panel = QtWidgets.QWidget()
    pcol = QtWidgets.QVBoxLayout(panel)
    panel.setFixedWidth(300)
    root.addWidget(panel, stretch=1)

    def normalize(raw):
        out = []
        for i in range(NUM_FINGERS):
            lo, hi = cal["open"][i], cal["closed"][i]
            out.append(
                0.0 if hi == lo else float(np.clip((raw[i] - lo) / (hi - lo), 0, 1))
            )
        return out

    def reset_level():
        offset["R"] = offset["cur"].copy()

    # Buttons
    src_btns, sign_btns = [], []

    def make_src_btn(i):
        b = QtWidgets.QPushButton(f"Rot{MODEL_AXES[i]} < {AXIS_LABELS[mods['src'][i]]}")

        def cb():
            mods["src"][i] = (mods["src"][i] + 1) % 3
            b.setText(f"Rot{MODEL_AXES[i]} < {AXIS_LABELS[mods['src'][i]]}")

        b.clicked.connect(cb)
        return b

    def make_sign_btn(i):
        b = QtWidgets.QPushButton(f"Rot{MODEL_AXES[i]}  +")

        def cb():
            mods["sign"][i] *= -1
            b.setText(f"Rot{MODEL_AXES[i]}  {'-' if mods['sign'][i] < 0 else '+'}")

        b.clicked.connect(cb)
        return b

    reset_btn = QtWidgets.QPushButton("Reset Level (tare)")
    reset_btn.clicked.connect(reset_level)
    pcol.addWidget(reset_btn)

    pcol.addWidget(QtWidgets.QLabel("Axis source (which input drives each axis):"))
    for i in range(3):
        b = make_src_btn(i)
        src_btns.append(b)
        pcol.addWidget(b)

    pcol.addWidget(QtWidgets.QLabel("Axis invert:"))
    for i in range(3):
        b = make_sign_btn(i)
        sign_btns.append(b)
        pcol.addWidget(b)

    finger_btn = QtWidgets.QPushButton("Invert finger: OFF")

    def toggle_finger():
        mods["inv_finger"] = not mods["inv_finger"]
        finger_btn.setText(f"Invert finger: {'ON' if mods['inv_finger'] else 'OFF'}")

    finger_btn.clicked.connect(toggle_finger)
    pcol.addWidget(finger_btn)

    cal_row = QtWidgets.QHBoxLayout()
    open_btn = QtWidgets.QPushButton("Set Open")
    fist_btn = QtWidgets.QPushButton("Set Fist")
    open_btn.clicked.connect(lambda: cal.__setitem__("open", list(last_raw)))
    fist_btn.clicked.connect(lambda: cal.__setitem__("closed", list(last_raw)))
    cal_row.addWidget(open_btn)
    cal_row.addWidget(fist_btn)
    pcol.addLayout(cal_row)

    info = QtWidgets.QLabel()
    info.setFont(QtGui.QFont("Courier New", 9))
    info.setAlignment(QtCore.Qt.AlignTop)
    info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    pcol.addWidget(info, stretch=1)

    def tick():
        data = reader.get()
        if data is None:
            return
        fingers, q, calbyte = data
        for i in range(NUM_FINGERS):
            last_raw[i] = fingers[i]
        flex = normalize(fingers)
        if mods["inv_finger"]:
            flex = [1.0 - f for f in flex]

        q = continuous_quat(q, prev_q[0])
        prev_q[0] = q
        qw, qv = q[0], [q[1], q[2], q[3]]
        mx = mods["sign"][0] * qv[mods["src"][0]]
        my = mods["sign"][1] * qv[mods["src"][1]]
        mz = mods["sign"][2] * qv[mods["src"][2]]
        offset["cur"] = quat_to_matrix(qw, mx, my, mz)
        R = offset["R"].T @ offset["cur"]
        hand.update(flex, R)

        rows = ["FINGER FLEXION", ""]
        for i in range(NUM_FINGERS):
            bar = "#" * int(flex[i] * 18)
            rows.append(f"{FINGER_NAMES[i]:>7}: {flex[i]:.2f} |{bar:<18}|")
        rows += [
            "",
            "QUATERNION",
            f"  w {q[0]:6.3f}  A1 {q[1]:6.3f}",
            f"  A2 {q[2]:6.3f}  A3 {q[3]:6.3f}",
        ]
        if calbyte is not None:
            rows += [
                "",
                "BNO055 CAL (3=best)",
                f"  sys {(calbyte >> 6) & 3} gyro {(calbyte >> 4) & 3}"
                f" acc {(calbyte >> 2) & 3} mag {calbyte & 3}",
            ]
        rows += ["", "Drag view to orbit. Scroll to zoom."]
        info.setText("\n".join(rows))

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(10)  # ~100 fps target; GPU keeps up

    win.show()
    try:
        app.exec_() if hasattr(app, "exec_") else app.exec()
    finally:
        reader.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
camera_setup.py  --  camera setup & alignment interface for the senz labeler
============================================================================
A native PyQt5 tool to dial in the **phone camera** BEFORE recording a
ground-truth labeling session (see docs/senz_camera_hld.md). It shows the live
feed with a **MediaPipe hand-skeleton overlay** (what the tracker actually
understands) plus an **actionable warnings HUD** ("too dark", "move closer",
"blurry", "change angle", ...) and a **readiness** banner, so you can frame and
light the rig well and repeatably.

It matches senz_v3_qt.py's **light/dark themes** and uses a **toggle button per
overlay/panel**. When the framing is good, **Copy recorder command** hands off to
dataset_recorder.py with the chosen source.

    python camera_setup.py                       # default webcam (index 0)
    python camera_setup.py --source 1            # another camera index
    python camera_setup.py --source http://phone-ip:port/video   # phone IP camera
    python camera_setup.py --demo                # no webcam/mediapipe (synthetic)

Design: the metric/assessment core is **pure numpy and importable headless** (Qt,
OpenCV, MediaPipe and senz_v3_qt are all imported lazily), mirroring senz_v3_qt.py
so the warning logic is unit-testable without a camera.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# Hand topology (21 MediaPipe landmarks) -- shared with the 3D visualizer via
# hand_model, so both programs draw the SAME per-finger colored skeleton.
# ----------------------------------------------------------------------------
import hand_model as hmod

HAND_CONNECTIONS = hmod.HAND_CONNECTIONS
FINGERTIPS = hmod.FINGERTIPS
N_LANDMARKS = hmod.N_LANDMARKS


def _qcolor(rgb, a=230):
    """hand_model rgb (0..1) -> QColor (imported lazily so this stays headless)."""
    from pyqtgraph.Qt import QtGui
    r, g, b = rgb
    return QtGui.QColor(int(r * 255), int(g * 255), int(b * 255), a)

# Default quality thresholds (all tunable live). Normalized image coords [0,1].
DEFAULT_THRESHOLDS = {
    "bright_low": 0.25,     # mean luminance below -> too dark
    "bright_high": 0.85,    # above -> too bright
    "overexp_frac": 0.08,   # fraction of near-white pixels -> glare
    "sharp_min": 45.0,      # Laplacian variance below -> blurry
    "det_conf_min": 0.60,   # detection score below -> weak detection
    "hand_size_min": 0.18,  # hand max dimension (frac of frame) below -> too far
    "hand_size_max": 0.85,  # above -> too close
    "edge_margin": 0.03,    # bbox within this of an edge -> clipped
    "center_tol": 0.28,     # center offset from image center beyond -> off-center
    "vel_max": 0.055,       # normalized landmark velocity above -> moving too fast
    "occl_spread_min": 0.22,  # fingertip spread / hand size below -> fingers hidden
    "fps_min": 12.0,        # below -> low frame rate
}

# Toggle-able overlay/panel layers (button label -> key).
LAYERS = [
    ("Skeleton", "skeleton"), ("Joints", "joints"), ("Bounding box", "bbox"),
    ("Framing guide", "guide"), ("Thirds grid", "grid"),
    ("Occlusion tint", "occlusion"), ("Mirror", "mirror"),
    ("Warnings HUD", "warnings"), ("Metrics", "metrics"),
]
DEFAULT_LAYERS = {k: (k not in ("grid", "mirror")) for _, k in LAYERS}

# Detection-aid video filters (applied to the frame fed to MediaPipe + shown).
FILTERS = [("Auto-contrast", "auto_contrast"), ("Brighten", "brighten"),
           ("Sharpen", "sharpen")]
DEFAULT_FILTERS = {k: False for _, k in FILTERS}

DEFAULT_CONFIG = {
    "source": "0",
    "theme": "dark",
    "expected_hand": "right",
    "label": "",
    "width": None,
    "height": None,
    "fps": None,
    "thresholds": dict(DEFAULT_THRESHOLDS),
    "layers": dict(DEFAULT_LAYERS),
    "filters": dict(DEFAULT_FILTERS),
}


# ----------------------------------------------------------------------------
# Pure image / landmark metrics (numpy only -- headless-testable)
# ----------------------------------------------------------------------------
def to_gray(frame_rgb):
    """HxWx3 RGB (or HxW) -> float grayscale 0..255."""
    a = np.asarray(frame_rgb, dtype=np.float64)
    if a.ndim == 3:
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    return a


def frame_brightness(gray):
    """Mean luminance, 0..1."""
    return float(gray.mean()) / 255.0 if gray.size else 0.0


def overexposed_fraction(gray, thresh=250.0):
    """Fraction of near-white (blown-out) pixels, 0..1."""
    return float((gray >= thresh).mean()) if gray.size else 0.0


def frame_sharpness(gray):
    """Variance of the Laplacian (focus/blur measure). Higher = sharper."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    g = gray
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def hand_bbox(landmarks):
    """21 (x,y,z) normalized -> (xmin,ymin,xmax,ymax), or None."""
    if not landmarks:
        return None
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_size(bbox):
    """Max of width/height as a fraction of the frame."""
    if bbox is None:
        return 0.0
    return max(bbox[2] - bbox[0], bbox[3] - bbox[1])


def bbox_center(bbox):
    if bbox is None:
        return (0.5, 0.5)
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_clipped(bbox, margin):
    if bbox is None:
        return False
    return (bbox[0] < margin or bbox[1] < margin
            or bbox[2] > 1.0 - margin or bbox[3] > 1.0 - margin)


def landmark_velocity(prev, cur):
    """Mean per-landmark (x,y) displacement between two frames (normalized)."""
    if not prev or not cur or len(prev) != len(cur):
        return 0.0
    d = [(a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for a, b in zip(cur, prev)]
    return float(np.sqrt(np.array(d)).mean())


def fingertip_spread(landmarks, size):
    """Mean fingertip distance to their centroid / hand size. Low -> hidden/fist."""
    if not landmarks or size <= 1e-6:
        return 1.0
    tips = np.array([landmarks[i][:2] for i in FINGERTIPS], dtype=float)
    c = tips.mean(axis=0)
    return float(np.linalg.norm(tips - c, axis=1).mean()) / size


def compute_metrics(frame_rgb, landmarks, present, det_conf=1.0, handed="Unknown",
                    prev_landmarks=None, fps=30.0, latency_ms=0.0, stride=4):
    """Bundle every raw metric into one dict (pure numpy)."""
    if frame_rgb is not None:
        gray = to_gray(np.asarray(frame_rgb)[::stride, ::stride])
    else:
        gray = np.zeros((1, 1))
    bbox = hand_bbox(landmarks) if (present and landmarks) else None
    size = bbox_size(bbox)
    return {
        "brightness": frame_brightness(gray),
        "overexp": overexposed_fraction(gray),
        "sharpness": frame_sharpness(gray),
        "present": int(present),
        "det_conf": float(det_conf),
        "handed": handed,
        "hand_size": size,
        "center": bbox_center(bbox),
        "bbox": bbox,
        "velocity": landmark_velocity(prev_landmarks, landmarks) if present else 0.0,
        "spread": fingertip_spread(landmarks, size) if bbox else 1.0,
        "fps": float(fps),
        "latency_ms": float(latency_ms),
    }


def assess(metrics, thresholds, expected_hand=None):
    """Metrics -> {warnings:[{id,severity,text,hint}], readiness: green|amber|red}.

    The single source of truth for every warning sign. severity in
    error/warn/info; readiness is red if any error, amber if any warn, else green.
    """
    t = thresholds
    w = []

    def add(id_, sev, text, hint):
        w.append({"id": id_, "severity": sev, "text": text, "hint": hint})

    b = metrics["brightness"]
    if b < t["bright_low"]:
        add("dark", "error", "Too dark", "Add light or brighten the room")
    elif b > t["bright_high"] or metrics["overexp"] > t["overexp_frac"]:
        add("bright", "warn", "Too bright / glare", "Reduce light or change the angle")
    if metrics["sharpness"] < t["sharp_min"]:
        add("blur", "warn", "Blurry / out of focus", "Hold steady, clean the lens, add light")

    if not metrics["present"]:
        add("nohand", "error", "No hand detected", "Move your hand into the frame")
    else:
        if metrics["det_conf"] < t["det_conf_min"]:
            add("weak", "warn", "Weak hand detection",
                "Improve lighting/angle; use a plain background")
        hs = metrics["hand_size"]
        if hs < t["hand_size_min"]:
            add("far", "warn", "Hand too far", "Move your hand closer to the camera")
        elif hs > t["hand_size_max"] or bbox_clipped(metrics["bbox"], t["edge_margin"]):
            add("close", "warn", "Hand too close / cut off",
                "Move back so the whole hand is in frame")
        cx, cy = metrics["center"]
        if max(abs(cx - 0.5), abs(cy - 0.5)) > t["center_tol"]:
            add("center", "info", "Hand off-center", "Center your hand in the frame")
        if metrics["velocity"] > t["vel_max"]:
            add("fast", "info", "Moving too fast", "Slow down for cleaner tracking")
        if metrics["spread"] < t["occl_spread_min"]:
            add("occl", "info", "Fingers may be hidden",
                "Rotate your hand so the fingers face the camera")
        if expected_hand and metrics["handed"] != "Unknown" \
                and metrics["handed"].lower() != expected_hand.lower():
            add("handed", "warn", f"Detected {metrics['handed']} hand, expected "
                f"{expected_hand.capitalize()}",
                "Mirror flip or wrong hand -- labels may be left/right swapped")

    if metrics["fps"] < t["fps_min"]:
        add("fps", "info", f"Low frame rate ({metrics['fps']:.0f} fps)",
            "Close other apps or lower the camera resolution")

    sev = {x["severity"] for x in w}
    readiness = "red" if "error" in sev else ("amber" if "warn" in sev else "green")
    return {"warnings": w, "readiness": readiness}


# ----------------------------------------------------------------------------
# Config persistence (repo json idiom: guarded load, indent=2 dump, dict.get)
# ----------------------------------------------------------------------------
def load_config(path):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy of defaults
    if path and os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        for k in ("source", "theme", "expected_hand", "label", "width", "height", "fps"):
            if k in data:
                cfg[k] = data[k]
        cfg["thresholds"].update(data.get("thresholds", {}))
        cfg["layers"].update(data.get("layers", {}))
        cfg["filters"].update(data.get("filters", {}))
    return cfg


def save_config(path, cfg):
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)


def recorder_command(source, label=""):
    """The dataset_recorder.py invocation for the chosen source."""
    cmd = f"python dataset_recorder.py --camera {source}"
    if label:
        cmd += f" --label {label}"
    return cmd


# ----------------------------------------------------------------------------
# Synthetic camera (--demo): drives the whole UI with no webcam / mediapipe.
# Returns the same shape as HandTracker.get_frame().
# ----------------------------------------------------------------------------
def _synthetic_hand(t):
    """21 procedural landmarks (x,y,z) that sway + curl gently over time t."""
    sway = 0.03 * np.sin(t * 0.8)
    curl = 0.5 - 0.5 * np.cos(t * 0.6)          # 0 open .. 1 curled
    wrist = np.array([0.5 + sway, 0.78])
    pts = [(*wrist, 0.0)]
    # finger base offsets (x) across the knuckle line, and base y.
    bases_x = [-0.11, -0.05, 0.02, 0.08, 0.13]  # thumb..pinky
    base_y = [0.66, 0.55, 0.53, 0.55, 0.60]
    lengths = [0.09, 0.14, 0.15, 0.13, 0.10]
    for f in range(5):
        bx = 0.5 + sway + bases_x[f]
        by = base_y[f]
        seg = lengths[f] / 3.0
        # 4 joints per finger, bending by curl (fingers point up = -y)
        for j in range(4):
            frac = j / 3.0
            bend = curl * 0.35 * frac
            x = bx + bend * 0.12 * (1 if f != 0 else -1)
            y = by - lengths[f] * frac + bend * 0.10
            pts.append((float(x), float(y), 0.0))
    return pts  # 1 + 5*4 = 21


class SyntheticCamera:
    """No-hardware source with HandTracker.get_frame()'s interface."""

    def __init__(self, w=640, h=480, realtime=True):
        self.w, self.h = w, h
        self.realtime = realtime
        self._n = 0
        # A fixed sharp sine texture (gives realistic non-zero sharpness).
        yy, xx = np.mgrid[0:h, 0:w]
        self._tex = (0.5 + 0.25 * np.sin(xx / 6.0) * np.sin(yy / 6.0))

    def start(self):
        return self

    def get_frame(self):
        t = self._n / 30.0
        self._n += 1
        # Moving mid-brightness gradient + sharp texture -> plausible frame.
        base = 90 + 60 * (0.5 + 0.5 * np.sin(t * 0.5))
        img = np.clip(base * self._tex + 40, 0, 255).astype(np.uint8)
        frame = np.stack([img, img, img], axis=-1)          # gray RGB
        lms = _synthetic_hand(t)
        if self.realtime:
            time.sleep(1.0 / 30.0)
        return (frame, lms, 0.92, 1, time.time(), "Right")

    def stop(self):
        pass


# ----------------------------------------------------------------------------
# Qt UI (composition, like senz_v3_qt -- Qt imported lazily so the core is headless)
# ----------------------------------------------------------------------------
_SEV = {"error": ("✕", (210, 70, 70)), "warn": ("!", (220, 160, 60)),
        "info": ("i", (90, 150, 210))}
_READY = {"green": ("READY TO RECORD", (60, 170, 90)),
          "amber": ("ADJUST", (210, 160, 60)),
          "red": ("NOT READY", (200, 70, 70))}


class VideoPanel:
    """QLabel that shows the frame as a pixmap with QPainter overlays."""

    def __init__(self, theme_bg="#181b20"):
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.label = QtWidgets.QLabel()
        self.label.setMinimumSize(640, 480)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet(f"background-color: {theme_bg};")
        self.widget = self.label
        self._buf = None

    def update_view(self, frame_rgb, landmarks, layers, occluded=False):
        from pyqtgraph.Qt import QtCore, QtGui

        h, w = frame_rgb.shape[:2]
        buf = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        self._buf = buf  # keep a ref so Qt doesn't read freed memory
        img = QtGui.QImage(buf.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        mirror = layers.get("mirror", False)
        if mirror:
            img = img.mirrored(True, False)
        pix = QtGui.QPixmap.fromImage(img).scaled(
            self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        W, H = pix.width(), pix.height()

        def px(pt):
            x = (1.0 - pt[0]) if mirror else pt[0]
            return QtCore.QPointF(x * W, pt[1] * H)

        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if layers.get("grid"):
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 60), 1))
            for f in (1 / 3, 2 / 3):
                p.drawLine(int(f * W), 0, int(f * W), H)
                p.drawLine(0, int(f * H), W, int(f * H))
        if layers.get("guide"):
            gw, gh = 0.55 * W, 0.7 * H
            p.setPen(QtGui.QPen(QtGui.QColor(120, 200, 255, 130), 2, QtCore.Qt.DashLine))
            p.drawRect(int((W - gw) / 2), int((H - gh) / 2), int(gw), int(gh))
        if landmarks:
            if layers.get("bbox"):
                bb = hand_bbox(landmarks)
                tl, br = px((bb[0], bb[1])), px((bb[2], bb[3]))
                p.setPen(QtGui.QPen(QtGui.QColor(240, 220, 60, 200), 2))
                p.drawRect(QtCore.QRectF(tl, br))
            if layers.get("skeleton"):
                for a, b in HAND_CONNECTIONS:
                    p.setPen(QtGui.QPen(_qcolor(hmod.connection_color(a, b)), 3))
                    p.drawLine(px(landmarks[a]), px(landmarks[b]))
            if layers.get("joints"):
                for i in range(min(N_LANDMARKS, len(landmarks))):
                    tip = i in FINGERTIPS
                    if tip and occluded and layers.get("occlusion"):
                        col = QtGui.QColor(230, 70, 70, 240)
                    else:
                        col = _qcolor(hmod.joint_color(i), a=240)
                    p.setBrush(col)
                    p.setPen(QtGui.QPen(col, 1))
                    r = 5 if tip else 3
                    p.drawEllipse(px(landmarks[i]), r, r)
        p.end()
        self.label.setPixmap(pix)


class SetupPanel:
    """Right-hand control panel (composition; self.widget is the QWidget)."""

    def __init__(self, cfg, callbacks):
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

        self.cfg = cfg
        self.cb = callbacks
        self._theme = cfg["theme"]
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(330)
        lay = QtWidgets.QVBoxLayout(panel)

        self.theme_btn = QtWidgets.QPushButton(f"Theme: {self._theme.capitalize()}")
        self.theme_btn.clicked.connect(self._toggle_theme)
        lay.addWidget(self.theme_btn)

        # Camera source.
        lay.addWidget(QtWidgets.QLabel("<b>Camera source</b> (index or URL)"))
        srow = QtWidgets.QHBoxLayout()
        self.source_edit = QtWidgets.QLineEdit(str(cfg["source"]))
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(
            lambda: self.cb["connect"](self.source_edit.text().strip()))
        self.scan_btn = QtWidgets.QPushButton("Scan")
        self.scan_btn.setToolTip("Probe camera indices 0-5 for USB / built-in cameras")
        self.scan_btn.clicked.connect(self.cb["scan"])
        srow.addWidget(self.source_edit)
        srow.addWidget(self.connect_btn)
        srow.addWidget(self.scan_btn)
        lay.addLayout(srow)
        lay.addWidget(QtWidgets.QLabel(
            "<i>USB / built-in: a number (0,1,2). Phone: an http URL.</i>"))

        hrow = QtWidgets.QHBoxLayout()
        hrow.addWidget(QtWidgets.QLabel("Expected hand:"))
        self.hand_box = QtWidgets.QComboBox()
        self.hand_box.addItems(["right", "left"])
        self.hand_box.setCurrentText(cfg.get("expected_hand", "right"))
        self.hand_box.currentTextChanged.connect(self.cb["expected_hand"])
        hrow.addWidget(self.hand_box)
        lay.addLayout(hrow)

        # Readiness banner.
        self.ready = QtWidgets.QLabel("...")
        self.ready.setAlignment(QtCore.Qt.AlignCenter)
        self.ready.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.ready.setFixedHeight(34)
        lay.addWidget(self.ready)

        # Overlay/panel toggle buttons (checkable).
        lay.addWidget(QtWidgets.QLabel("<b>Overlays / panels</b>"))
        grid = QtWidgets.QGridLayout()
        self.layer_btns = {}
        for i, (label, key) in enumerate(LAYERS):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setChecked(bool(cfg["layers"].get(key, True)))
            b.toggled.connect(lambda on, k=key: self.cb["layer"](k, on))
            grid.addWidget(b, i // 2, i % 2)
            self.layer_btns[key] = b
        lay.addLayout(grid)

        # Detection-aid filters (help MediaPipe in poor light; not saved to data).
        lay.addWidget(QtWidgets.QLabel("<b>Filters</b> (detection aid)"))
        frow = QtWidgets.QHBoxLayout()
        self.filter_btns = {}
        for label, key in FILTERS:
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setChecked(bool(cfg["filters"].get(key, False)))
            b.toggled.connect(lambda on, k=key: self.cb["filter"](k, on))
            frow.addWidget(b)
            self.filter_btns[key] = b
        lay.addLayout(frow)

        # A few live threshold sliders.
        self.sliders = {}
        for key, lo, hi, label in (("bright_low", 0, 60, "Min brightness"),
                                   ("sharp_min", 0, 200, "Min sharpness"),
                                   ("hand_size_min", 5, 60, "Min hand size")):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(label))
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(int(cfg["thresholds"][key] * (100 if key != "sharp_min" else 1)))
            s.valueChanged.connect(lambda v, k=key: self.cb["threshold"](k, v))
            row.addWidget(s)
            lay.addLayout(row)
            self.sliders[key] = s

        # Warnings HUD.
        self.hud_title = QtWidgets.QLabel("<b>Warnings</b>")
        lay.addWidget(self.hud_title)
        self.hud = QtWidgets.QWidget()
        self.hud_lay = QtWidgets.QVBoxLayout(self.hud)
        self.hud_lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.hud)

        # Metrics readout.
        self.metrics = QtWidgets.QLabel()
        self.metrics.setFont(QtGui.QFont("Courier New", 9))
        self.metrics.setAlignment(QtCore.Qt.AlignTop)
        lay.addWidget(self.metrics)

        lay.addStretch(1)

        # Recorder handoff.
        lrow = QtWidgets.QHBoxLayout()
        lrow.addWidget(QtWidgets.QLabel("Label:"))
        self.label_edit = QtWidgets.QLineEdit(cfg.get("label", ""))
        lrow.addWidget(self.label_edit)
        lay.addLayout(lrow)
        self.cmd_edit = QtWidgets.QLineEdit()
        self.cmd_edit.setReadOnly(True)
        lay.addWidget(self.cmd_edit)
        crow = QtWidgets.QHBoxLayout()
        copy_btn = QtWidgets.QPushButton("Copy recorder command")
        copy_btn.clicked.connect(self.cb["copy_cmd"])
        save_btn = QtWidgets.QPushButton("Save settings")
        save_btn.clicked.connect(self.cb["save"])
        crow.addWidget(copy_btn)
        crow.addWidget(save_btn)
        lay.addLayout(crow)

        self.status = QtWidgets.QLabel("")
        lay.addWidget(self.status)

        self.widget = panel
        self._QtGui = QtGui
        self._QtWidgets = QtWidgets

    def _toggle_theme(self):
        self._theme = "light" if self._theme == "dark" else "dark"
        self.theme_btn.setText(f"Theme: {self._theme.capitalize()}")
        self.cb["theme"](self._theme)

    def set_readiness(self, readiness):
        text, (r, g, b) = _READY[readiness]
        self.ready.setText(text)
        self.ready.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: white; border-radius: 4px;")

    def set_warnings(self, warnings, show):
        QtWidgets = self._QtWidgets
        while self.hud_lay.count():
            item = self.hud_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.hud_title.setVisible(show)
        self.hud.setVisible(show)
        if not show:
            return
        if not warnings:
            ok = QtWidgets.QLabel("✓  looks good")
            ok.setStyleSheet("color: rgb(90,180,110);")
            self.hud_lay.addWidget(ok)
            return
        for wn in warnings:
            icon, (r, g, b) = _SEV[wn["severity"]]
            row = QtWidgets.QLabel(f"{icon}  {wn['text']} — {wn['hint']}")
            row.setWordWrap(True)
            row.setStyleSheet(f"color: rgb({r},{g},{b});")
            self.hud_lay.addWidget(row)

    def set_metrics(self, text, show):
        self.metrics.setVisible(show)
        self.metrics.setText(text if show else "")

    def set_command(self, cmd):
        self.cmd_edit.setText(cmd)


def main():
    ap = argparse.ArgumentParser(description="senz camera setup & alignment UI")
    ap.add_argument("--source", default=None, help="camera index or URL (phone cam)")
    ap.add_argument("--demo", action="store_true", help="synthetic feed, no webcam")
    ap.add_argument("--config", default="camera_setup.json", help="settings file")
    ap.add_argument("--label", default=None, help="preset session label")
    ap.add_argument("--width", type=int, default=None, help="requested capture width")
    ap.add_argument("--height", type=int, default=None, help="requested capture height")
    ap.add_argument("--fps", type=int, default=None, help="requested capture fps")
    args = ap.parse_args()

    from pyqtgraph.Qt import QtCore, QtWidgets
    from senz_v3_qt import THEMES

    cfg = load_config(args.config)
    if args.source is not None:
        cfg["source"] = args.source
    if args.label is not None:
        cfg["label"] = args.label
    for k in ("width", "height", "fps"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.setObjectName("camsetup")
    win.setWindowTitle("senz camera setup")
    win.resize(1120, 700)
    root = QtWidgets.QHBoxLayout(win)

    state = {"source": None, "prev_lms": None, "tlast": None, "fps": 0.0,
             "theme": cfg["theme"], "occluded": False}

    def open_source(src):
        if state["source"] is not None:
            try:
                state["source"].stop()
            except Exception:
                pass
        if args.demo:
            state["source"] = SyntheticCamera().start()
            return
        from camera_tracker import HandTracker
        state["source"] = HandTracker(
            camera=src, keep_frame=True, show=False,
            width=cfg.get("width"), height=cfg.get("height"), req_fps=cfg.get("fps"),
            filters=cfg["filters"]).start()

    video = VideoPanel(THEMES[cfg["theme"]]["gl_bg"])
    root.addWidget(video.widget, stretch=3)

    def on_connect(src):
        cfg["source"] = src
        open_source(src)
        panel.status.setText(f"connected: {src}")

    def on_layer(key, on):
        cfg["layers"][key] = on

    def on_threshold(key, v):
        cfg["thresholds"][key] = v / (100.0 if key != "sharp_min" else 1.0)

    def on_theme(mode):
        state["theme"] = mode
        cfg["theme"] = mode
        apply_theme(mode)

    def on_copy_cmd():
        cmd = recorder_command(cfg["source"], panel.label_edit.text().strip())
        panel.set_command(cmd)
        QtWidgets.QApplication.clipboard().setText(cmd)
        panel.status.setText("recorder command copied to clipboard")

    def on_save():
        cfg["label"] = panel.label_edit.text().strip()
        cfg["expected_hand"] = panel.hand_box.currentText()
        save_config(args.config, cfg)
        panel.status.setText(f"saved -> {args.config}")

    def on_filter(key, on):
        cfg["filters"][key] = on
        src = state["source"]
        if src is not None and hasattr(src, "set_filters"):
            src.set_filters(cfg["filters"])   # live, no reconnect

    def on_scan():
        panel.status.setText("scanning cameras...")
        QtWidgets.QApplication.processEvents()
        from camera_tracker import list_cameras
        cams = list_cameras()
        panel.status.setText("cameras: " + (", ".join(map(str, cams)) or "none found"))
        if cams:
            panel.source_edit.setText(str(cams[0]))

    callbacks = {"connect": on_connect, "layer": on_layer, "threshold": on_threshold,
                 "theme": on_theme, "copy_cmd": on_copy_cmd, "save": on_save,
                 "filter": on_filter, "scan": on_scan,
                 "expected_hand": lambda h: cfg.__setitem__("expected_hand", h)}
    panel = SetupPanel(cfg, callbacks)
    root.addWidget(panel.widget, stretch=1)

    def apply_theme(mode):
        th = THEMES[mode]
        panel.widget.setStyleSheet(th["qss"])
        win.setStyleSheet(f'QWidget#camsetup {{ background-color: {th["win_bg"]}; }}')
        video.label.setStyleSheet(f"background-color: {th['gl_bg']};")

    apply_theme(cfg["theme"])
    open_source(cfg["source"])

    def tick():
        got = state["source"].get_frame() if state["source"] else None
        now = time.time()
        if got is None:
            src = state["source"]
            err = src.get_error() if (src and hasattr(src, "get_error")) else None
            hint = err or ("Downloading the hand model, or check the source / that the "
                           "phone-cam app is running")
            panel.set_readiness("red")
            panel.set_warnings([{"id": "nostream", "severity": "error",
                                 "text": "No camera stream", "hint": hint}],
                               cfg["layers"].get("warnings", True))
            return
        # Only re-render on a NEW frame (t_cap changed); poll fast, work only when
        # there's something new. fps measures the REAL camera/inference throughput.
        if got[4] == state.get("last_tcap"):
            return
        if state["tlast"] is not None:
            inst = 1.0 / max(1e-6, got[4] - state["tlast"])
            state["fps"] += 0.25 * (inst - state["fps"])
        state["tlast"] = got[4]
        state["last_tcap"] = got[4]
        frame, lms, score, present, t_cap, handed = got
        latency = max(0.0, (now - t_cap) * 1e3)
        m = compute_metrics(frame, lms, present, det_conf=score, handed=handed,
                            prev_landmarks=state["prev_lms"], fps=state["fps"],
                            latency_ms=latency)
        state["prev_lms"] = lms
        res = assess(m, cfg["thresholds"], cfg.get("expected_hand"))
        state["occluded"] = any(x["id"] == "occl" for x in res["warnings"])
        video.update_view(frame, lms, cfg["layers"], occluded=state["occluded"])
        panel.set_readiness(res["readiness"])
        panel.set_warnings(res["warnings"], cfg["layers"].get("warnings", True))
        panel.set_metrics(
            f"fps        {m['fps']:5.1f}\n"
            f"latency    {m['latency_ms']:5.0f} ms\n"
            f"brightness {m['brightness']:.2f}\n"
            f"sharpness  {m['sharpness']:6.0f}\n"
            f"det conf   {m['det_conf']:.2f}  ({handed})\n"
            f"hand size  {m['hand_size']:.2f}\n"
            f"velocity   {m['velocity']:.3f}",
            cfg["layers"].get("metrics", True))
        panel.set_command(recorder_command(cfg["source"], panel.label_edit.text().strip()))

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(10)   # poll ~100 Hz; tick only re-renders on a new frame

    win.show()
    try:
        (app.exec_ if hasattr(app, "exec_") else app.exec)()
    finally:
        if state["source"]:
            try:
                state["source"].stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()

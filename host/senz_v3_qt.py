#!/usr/bin/env python3
"""
senz_v3_qt.py  --  native GPU 3D hand visualizer for the v3 prototype
=====================================================================
Renders the v3 glove in a **native desktop OpenGL window** (pyqtgraph + PyOpenGL
+ PyQt5) at ~100 fps -- NOT a browser. It is SCHEMA-DRIVEN, so it serves both v3
builds from the same code (it reads nimu/nforce from the firmware banner):

  - Hardware Sprint v3 TACTILE (default): 1 dorsum IMU + BNO055 wrist + 15-taxel
    velostat force array. Gross hand movement + tactile; fingers ride in a neutral
    rest pose (they come from the camera later).
  - v3 PROTO: index 2 IMUs, middle 2, thumb 3 + dorsum + BNO055 + 12-15 taxels
    (fingers articulate from their own IMUs).

By convention the **dorsum (hand/palm frame) is the LAST IMU**; the fingers hang
off it, and the wrist between the forearm (BNO055) and the hand (dorsum) is drawn
as a **flexing polygon** that bends with wrist flex/deviation -- not a rigid box.
When a finger's IMU isn't provided by the firmware it renders dimmed in rest pose.

The firmware streams RAW accel/gyro, so this runs Madgwick fusion per finger IMU
on the host (the wrist BNO055 is already fused). Live over serial or hardware-free
via the base simulators (`--sim tactile|proto`):

    python senz_v3_qt.py --simulate --hand right              # tactile sim (default)
    python senz_v3_qt.py --simulate --sim proto --hand right   # 8-IMU proto sim
    python senz_v3_qt.py --port COM5 --hand right               # wired glove @ 921600

`--hand right|left` mirrors the palm geometry. Controls (right panel): enable /
disable a sensor, axis-remap (invert/swap to match its mount), Zero-hand (tare the
current pose flat), Clear zero, Zero force.

The pure-numpy core imports without Qt/OpenGL so it is unit-testable headless.
"""

import argparse
import math
import sys
import threading

import numpy as np

import hand_model as hmod
from senz_io import quat_to_matrix

# ----------------------------------------------------------------------------
# Sensors: index 0 = wrist BNO055 (forearm), 1..nimu = imu0..imu(nimu-1). By
# CONVENTION the DORSUM (hand/palm frame) is the LAST IMU, so its sensor index
# is `nimu`. The proto build (nimu=8) -> dorsum sensor 8 (imu7); the tactile
# build (nimu=1) -> dorsum sensor 1 (imu0, the only IMU). Finger IMUs occupy
# sensors 1..nimu-1; when an fewer-IMU firmware doesn't provide a finger's IMU,
# that finger is rendered in a neutral rest pose (see compute_skeleton).
# ----------------------------------------------------------------------------
NUM_SENSORS = 9        # default (proto); main() sizes SensorConfigV3 from nimu
DORSUM_SENSOR = 8      # default (proto); main() passes schema.nimu at runtime
_PROTO_LABELS = ["wrist (BNO055 / forearm)", "idx-prox", "idx-dist", "mid-prox",
                 "mid-dist", "thmb-base", "thmb-tip", "thmb-meta", "hand-dorsum"]
SENSOR_LABELS = _PROTO_LABELS


def make_sensor_labels(nimu):
    """Combo-box labels for a wrist + `nimu` IMU set (dorsum = last IMU)."""
    if nimu == 8:
        return list(_PROTO_LABELS)                     # nice named proto set
    labels = ["wrist (BNO055 / forearm)"]
    labels += [f"imu{i}" for i in range(max(0, nimu - 1))]
    if nimu >= 1:
        labels.append("hand-dorsum")
    return labels

# Force channel -> finger (firmware order: thumb C0-3, index C4-7, middle C8-11)
FORCE_BASE = {"thumb": 0, "index": 4, "middle": 8}

# Palm taxels (channels C12-C14) as (label, channel, hand-local position). Right-hand
# canonical (thumb on -X); make_palm() mirrors x for the left hand. Needs nforce >= 15.
_PALM_BASE = [
    ("palm-center", 12, (0.0, 1.0, -0.25)),
    ("palm-thenar", 13, (-0.95, 0.6, -0.25)),   # thumb-side mound
    ("palm-hypo",   14, (0.95, 0.6, -0.25)),     # pinky-side mound
]
PALM_CHANNELS = [ch for _, ch, _ in _PALM_BASE]


def make_palm(hand="right"):
    """Palm taxel local positions for the given hand ('left' mirrors X)."""
    s = 1.0 if hand == "right" else -1.0
    return [(lbl, ch, (x * s, y, z)) for lbl, ch, (x, y, z) in _PALM_BASE]

BONE_RADIUS = 0.16
JOINT_RADIUS = 0.22
WRIST_RADIUS = 0.30
PALM_LEN = 2.0      # wrist pivot (origin) up to the knuckle line
FOREARM_LEN = 1.6

AXIS_OPTIONS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
_AXIS_VEC = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0),
             "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}

_EX, _EY, _EZ = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))

# Light / dark themes for the control panel (Qt stylesheet) and the 3D viewbox
# (GL background + grid). `gl_bg` -> GLViewWidget.setBackgroundColor; `grid` is a
# 0-255 RGBA tuple for GLGridItem.setColor; `qss` styles the control panel.
_DARK_QSS = """
QWidget { background-color: #23262b; color: #e6e6e6; }
QPushButton { background-color: #3a3f47; border: 1px solid #4a4f57; padding: 4px; border-radius: 3px; }
QPushButton:hover { background-color: #474d56; }
QComboBox { background-color: #3a3f47; border: 1px solid #4a4f57; padding: 2px; }
QComboBox QAbstractItemView { background-color: #2c3036; color: #e6e6e6; selection-background-color: #4a90d9; }
"""
_LIGHT_QSS = """
QWidget { background-color: #f0f2f5; color: #1a1a1a; }
QPushButton { background-color: #e2e5ea; border: 1px solid #b8bcc4; padding: 4px; border-radius: 3px; }
QPushButton:hover { background-color: #d6dae1; }
QComboBox { background-color: #ffffff; border: 1px solid #b8bcc4; padding: 2px; }
QComboBox QAbstractItemView { background-color: #ffffff; color: #1a1a1a; selection-background-color: #4a90d9; }
"""
THEMES = {
    "dark":  {"gl_bg": "#181b20", "win_bg": "#202329",
              "grid": (130, 130, 145, 90),  "qss": _DARK_QSS},
    "light": {"gl_bg": "#dfe3e8", "win_bg": "#eceff2",
              "grid": (90, 90, 110, 130),   "qss": _LIGHT_QSS},
}


# The hand is drawn as the MediaPipe 21-landmark, 5-finger skeleton (see
# hand_model.py) so it matches the camera program and reads orientation clearly.
# Force taxels live at the fingertip landmarks of thumb/index/middle.
FORCE_FINGERTIP = {"thumb": 4, "index": 8, "middle": 12}


def finger_imu_map(nimu):
    """Which sensor bends which finger joint, for builds that have finger IMUs.

    Returns a list of ``(pivot_landmark, downstream_landmarks, sensor_index)`` for
    ``hand_model.bend_fingers``. Sensor 0 = wrist; the finger IMUs are sensors 1..
    and the dorsum (hand frame) is the LAST IMU (= nimu). Only the 8-IMU proto /
    pinch layout carries finger IMUs; the tactile build (nimu=1, dorsum only)
    returns [] -> the fingers render as the canonical open hand.

    proto / pinch sensor order (1-based): 1 idx-prox, 2 idx-dist, 3 mid-prox,
    4 mid-dist, 5 thmb-base, 6 thmb-tip, 7 thmb-meta (base MCP, 9-axis), 8 dorsum.
    """
    if nimu >= 8:
        return [
            (1, [2, 3, 4], 7),        # thumb CMC   <- thmb-meta (base MCP)
            (2, [3, 4], 5),           # thumb MCP   <- thmb-base
            (3, [4], 6),              # thumb IP    <- thmb-tip
            (5, [6, 7, 8], 1),        # index MCP   <- idx-prox
            (6, [7, 8], 2),           # index PIP   <- idx-dist
            (9, [10, 11, 12], 3),     # middle MCP  <- mid-prox
            (10, [11, 12], 4),        # middle PIP  <- mid-dist
        ]
    return []


# ----------------------------------------------------------------------------
# Pure-numpy math (headless-testable)
# ----------------------------------------------------------------------------
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def quat_to_R(q):
    return quat_to_matrix(*q)


def build_remap(sx, sy, sz):
    return np.array([_AXIS_VEC[sx], _AXIS_VEC[sy], _AXIS_VEC[sz]], dtype=float)


def is_valid_remap(M):
    return np.allclose(M @ M.T, np.eye(3), atol=1e-6)


def matrix_to_axes(M):
    out = []
    for row in M:
        best, bestd = "+X", -9
        for name, vec in _AXIS_VEC.items():
            d = float(np.dot(row, vec))
            if d > bestd:
                best, bestd = name, d
        out.append(best)
    return out


def align_z_to(d):
    """Rotation mapping +Z onto unit vector d (Rodrigues)."""
    d = _unit(np.asarray(d, dtype=float))
    v = np.cross(_EZ, d)
    c = float(np.dot(_EZ, d))
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class SensorConfigV3:
    """Per-sensor enable / axis-remap / zero-tare (pure numpy).

    `n_sensors` = 1 (wrist) + nimu; defaults to the proto count for back-compat.
    """

    def __init__(self, n_sensors=NUM_SENSORS):
        self.n = n_sensors
        self.enabled = [True] * n_sensors
        self.remap = [np.eye(3) for _ in range(n_sensors)]
        self.zinv = [None] * n_sensors

    def set_enabled(self, idx, on):
        self.enabled[idx] = bool(on)

    def set_remap(self, idx, M):
        self.remap[idx] = M

    def reset_remap(self, idx):
        self.remap[idx] = np.eye(3)

    def _remapped(self, idx, qraw):
        M = self.remap[idx]
        return M @ quat_to_R(qraw) @ M.T

    def sensor_R(self, idx, qraw):
        if not self.enabled[idx]:
            return np.eye(3)
        R = self._remapped(idx, qraw)
        if self.zinv[idx] is not None:
            R = R @ self.zinv[idx]
        return R

    def capture_zero(self, raw_quats):
        for i in range(min(self.n, len(raw_quats))):
            self.zinv[i] = self._remapped(i, raw_quats[i]).T

    def clear_zero(self):
        self.zinv = [None] * self.n


def compute_skeleton(raw_quats, cfg, dorsum_sensor=DORSUM_SENSOR, hand="right",
                     world_lms=None, imu_map=None):
    """21-landmark hand posed in 3D. Dorsum IMU = hand/palm frame orientation;
    BNO055 = forearm frame (wrist-flex).

    raw_quats: [wrist_bno, q_imu0 .. q_imu(nimu-1)]; the dorsum is the LAST IMU at
    sensor index `dorsum_sensor` (= nimu). The finger SHAPE comes from the first
    available of three sources (highest authority first):
      - Fusion (world_lms given): the camera's live world landmarks, orientation-
        normalized via hand_model.pose_from_world -- the CAMERA articulates fingers.
      - Finger IMUs (imu_map non-empty, proto/pinch build): each finger's landmarks
        bend about their joints from that finger's IMU vs the hand frame, so the
        glove's own IMUs curl the fingers (see hand_model.bend_fingers).
      - Canonical (tactile build, no finger IMUs): the open hand; only gross
        orientation from the dorsum.
    The resulting hand-local shape is rotated by R_hand and placed at the wrist.
    Returns the forearm/hand frames, wrist-flex angle, 21 world-space `points`,
    a `fused` flag (camera on), and a `driven` label for the readout.
    """
    R_forearm = cfg.sensor_R(0, raw_quats[0])
    R_hand = cfg.sensor_R(dorsum_sensor, raw_quats[dorsum_sensor])
    wrist_pos = np.zeros(3)
    if world_lms is not None:
        local = hmod.pose_from_world(world_lms)       # camera shape, orientation-free
        fused, driven = True, "IMU+camera"
    elif imu_map:
        bends = []
        for pivot, downstream, s in imu_map:          # finger IMUs bend the fingers
            if s < len(raw_quats) and cfg.enabled[s]:
                R_rel = R_hand.T @ cfg.sensor_R(s, raw_quats[s])
                bends.append((pivot, downstream, R_rel))
        local = hmod.bend_fingers(hmod.canonical(hand), bends)
        fused, driven = False, "IMU fingers"
    else:
        local = hmod.canonical(hand)                  # canonical open hand
        fused, driven = False, "IMU gross"
    points = wrist_pos + (R_hand @ local.T).T         # 21x3 world positions
    flex = rot_angle_deg(R_forearm.T @ R_hand)
    return dict(R_forearm=R_forearm, R_hand=R_hand, wrist_pos=wrist_pos,
                flex_deg=flex, points=points, fused=fused, driven=driven)


def force_color(t):
    """Relative grip 0..1 -> (r,g,b) 0..1, dim-blue -> bright red."""
    t = max(0.0, min(1.0, float(t)))
    r = 0.15 + 0.85 * t
    g = 0.15 + 0.35 * (1.0 - abs(2.0 * t - 1.0))
    b = 0.6 * (1.0 - t) + 0.12
    return (r, g, b)


def build_wrist_palm(wrist_pos, R_forearm, R_hand, rows=6, length=PALM_LEN,
                     half_w=1.5, half_t=0.28, arch=0.45):
    """Lofted palm/wrist slab that bends from the forearm frame to the hand frame.

    The proximal (wrist) cross-section is oriented by the forearm; the distal
    (knuckle) one by the hand; intermediate rings blend between them, so the
    surface flexes at the wrist. Intermediate rings also lift toward +Z (back of
    hand) by a sine bump, giving the palm a visible longitudinal ARCH (domed back
    of hand, not a flat box). Returns (vertexes, faces).
    """
    f0, u0 = R_forearm @ _EY, R_forearm @ _EZ
    f1, u1 = R_hand @ _EY, R_hand @ _EZ
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    verts = []
    pos = np.asarray(wrist_pos, dtype=float).copy()
    for j in range(rows + 1):
        t = j / rows
        fwd = _unit((1 - t) * f0 + t * f1)
        up = _unit((1 - t) * u0 + t * u1)
        right = _unit(np.cross(fwd, up))
        up = _unit(np.cross(right, fwd))
        center = pos + up * (arch * math.sin(math.pi * t))   # longitudinal dome
        for sx, sz in corners:
            verts.append(center + right * sx * half_w + up * sz * half_t)
        if j < rows:
            pos = pos + fwd * (length / rows)
    verts = np.array(verts)
    faces = []
    for j in range(rows):
        a, b = j * 4, (j + 1) * 4
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append([a + k, a + k2, b + k2])
            faces.append([a + k, b + k2, b + k])
    faces += [[0, 1, 2], [0, 2, 3]]           # proximal cap
    m = rows * 4
    faces += [[m, m + 2, m + 1], [m, m + 3, m + 2]]  # distal cap
    return verts, np.array(faces)


# ----------------------------------------------------------------------------
# Threaded reader (non-blocking GUI tick; drains to freshest frame)
# ----------------------------------------------------------------------------
class ThreadedMultiReader:
    def __init__(self, source):
        self.source = source
        self.schema = source.schema
        self._latest = None
        self._lock = threading.Lock()
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._run:
            try:
                f = self.source.read()
            except Exception:
                f = None
            if f is not None:
                with self._lock:
                    self._latest = f

    def get(self):
        with self._lock:
            return self._latest

    def send(self, cmd):
        try:
            self.source.send(cmd)
        except Exception:
            pass

    def close(self):
        self._run = False
        try:
            self.source.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# GPU renderer (pyqtgraph OpenGL) -- imported lazily so headless import works
# ----------------------------------------------------------------------------
class HandGL:
    """Renders the MediaPipe 21-landmark, 5-finger skeleton -- the same hand the
    camera program draws -- as GPU spheres (joints) + cylinders (bones), colored
    per finger from ``hand_model`` so orientation reads at a glance. Force taxels
    sit at the thumb/index/middle fingertips and on the palm surface.
    """

    def __init__(self, view, palm=None, fingers=None):
        import pyqtgraph.opengl as gl

        self.gl = gl
        self.palm_taxels = palm or []
        # Which fingers to draw. The pinch build passes thumb/index/middle so the
        # ring + pinky are omitted entirely (no joints, no bones, no force pads).
        self.fingers = list(fingers) if fingers else list(hmod.FINGERS)
        self._conns = hmod.active_connections(self.fingers)
        self._lms = hmod.active_landmarks(self.fingers)
        cyl = gl.MeshData.cylinder(rows=1, cols=16,
                                   radius=[BONE_RADIUS, BONE_RADIUS], length=1.0)
        self._cv, self._cf = cyl.vertexes(), cyl.faces()
        sph = gl.MeshData.sphere(rows=10, cols=16, radius=1.0)
        self._sv, self._sf = sph.vertexes(), sph.faces()
        mesh_kw = dict(smooth=True, shader="shaded", drawEdges=False,
                       glOptions="opaque")

        # Flexing wrist/palm slab + forearm stub + wrist joint.
        v, f = build_wrist_palm(np.zeros(3), np.eye(3), np.eye(3))
        # Brighter, edge-free palm so the shaded underside isn't a dark "shadow"
        # that hides the mesh (was drawEdges=True with a near-black edge color).
        self.palm = gl.GLMeshItem(vertexes=v, faces=f, color=(0.58, 0.7, 0.93, 1.0),
                                  smooth=True, shader="shaded", drawEdges=False,
                                  glOptions="opaque")
        view.addItem(self.palm)
        self.forearm = gl.GLMeshItem(vertexes=self._cv, faces=self._cf,
                                     color=(0.55, 0.55, 0.62, 1.0), **mesh_kw)
        view.addItem(self.forearm)
        self.wrist = gl.GLMeshItem(vertexes=self._sv, faces=self._sf,
                                   color=(0.7, 0.7, 0.8, 1.0), **mesh_kw)
        view.addItem(self.wrist)

        # One bone cylinder per ACTIVE connection, one sphere per ACTIVE landmark,
        # each colored from the shared hand model (camera-style per-finger colors).
        self.bones = []
        for a, b in self._conns:
            col = hmod.connection_color(a, b) + (1.0,)
            it = gl.GLMeshItem(vertexes=self._cv, faces=self._cf, color=col, **mesh_kw)
            view.addItem(it)
            self.bones.append(it)
        self.joints = []
        for i in self._lms:
            col = hmod.joint_color(i) + (1.0,)
            jt = gl.GLMeshItem(vertexes=self._sv, faces=self._sf, color=col, **mesh_kw)
            view.addItem(jt)
            self.joints.append(jt)

        # LOW-POLY style: one flat-shaded box per bone (12 tris vs a cylinder + two
        # joint spheres) -- much cheaper to rebuild each frame, so it holds fps far
        # better than the smooth "noodle" hand. Hidden until selected.
        lp_kw = dict(smooth=False, shader="shaded", drawEdges=False, glOptions="opaque")
        self.lp_bones = []
        for a, b in self._conns:
            col = hmod.connection_color(a, b) + (1.0,)
            it = gl.GLMeshItem(vertexes=hmod.BOX_CORNERS, faces=hmod.BOX_FACES,
                               color=col, **lp_kw)
            view.addItem(it)
            self.lp_bones.append(it)

        # Force patches: a 2x2 scatter at each active thumb/index/middle fingertip.
        self.force_pts = {}
        for name in FORCE_FINGERTIP:
            if name not in self.fingers:
                continue
            sp = gl.GLScatterPlotItem(pos=np.zeros((4, 3)), size=0.001, pxMode=False)
            view.addItem(sp)
            self.force_pts[name] = sp

        # Palm taxels (single scatter of N points on the palm surface).
        self.palm_scatter = None
        if self.palm_taxels:
            self.palm_scatter = gl.GLScatterPlotItem(
                pos=np.zeros((len(self.palm_taxels), 3)), size=0.001, pxMode=False)
            view.addItem(self.palm_scatter)

        # Render style: "noodle" (smooth cylinders + spheres) or "lowpoly" (boxes).
        self.style = "noodle"
        self._apply_style()

    def _apply_style(self):
        noodle = self.style == "noodle"
        for it in self.bones + self.joints:
            it.setVisible(noodle)
        for it in self.lp_bones:
            it.setVisible(not noodle)

    def set_style(self, style):
        """'noodle' (smooth, prettier) or 'lowpoly' (blocky, cheaper -> higher fps)."""
        self.style = "lowpoly" if style == "lowpoly" else "noodle"
        self._apply_style()

    def _sphere_at(self, center, radius):
        return self._sv * radius + np.asarray(center)

    def _cylinder(self, base, tip):
        d = np.asarray(tip) - np.asarray(base)
        L = max(1e-6, float(np.linalg.norm(d)))
        v = self._cv.copy()
        v[:, 2] *= L
        return (align_z_to(d) @ v.T).T + np.asarray(base)

    def update(self, skel, fp):
        wp, Rf, Rh = skel["wrist_pos"], skel["R_forearm"], skel["R_hand"]
        pts = skel["points"]
        v, f = build_wrist_palm(wp, Rf, Rh)
        self.palm.setMeshData(vertexes=v, faces=f)
        self.forearm.setMeshData(vertexes=self._cylinder(wp, wp + Rf @ (-_EY * FOREARM_LEN)),
                                 faces=self._cf)
        self.wrist.setMeshData(vertexes=self._sphere_at(wp, WRIST_RADIUS), faces=self._sf)
        if self.style == "noodle":
            for (a, b), it in zip(self._conns, self.bones):
                it.setMeshData(vertexes=self._cylinder(pts[a], pts[b]), faces=self._cf)
            for lm, jt in zip(self._lms, self.joints):
                r = JOINT_RADIUS * (1.35 if lm in hmod.FINGERTIPS else 1.0)
                jt.setMeshData(vertexes=self._sphere_at(pts[lm], r), faces=self._sf)
        else:
            for (a, b), it in zip(self._conns, self.lp_bones):
                d = pts[b] - pts[a]
                L = max(1e-6, float(np.linalg.norm(d)))
                w = BONE_RADIUS * 1.5
                verts, faces = hmod.oriented_box((pts[a] + pts[b]) * 0.5,
                                                 align_z_to(d), (w, w, L * 0.5))
                it.setMeshData(vertexes=verts, faces=faces)
        for name, sp in self.force_pts.items():
            self._force_patch(name, Rh, pts[FORCE_FINGERTIP[name]], fp)
        self._palm_patch(wp, Rh, fp)

    def _palm_patch(self, wp, Rh, fp):
        if self.palm_scatter is None or not fp:
            return
        pts, cols, sizes = [], [], []
        for _lbl, ch, local in self.palm_taxels:
            rel = fp[ch]["relative_grip"] if ch < len(fp) else 0.0
            pts.append(np.asarray(wp) + Rh @ np.array(local, dtype=float))
            r, g, b = force_color(rel)
            cols.append((r, g, b, 1.0))
            sizes.append(0.18 + 0.25 * rel)
        self.palm_scatter.setData(pos=np.array(pts), color=np.array(cols),
                                  size=np.array(sizes))

    def _force_patch(self, name, Rh, tip, fp):
        base = FORCE_BASE[name]
        ex, ez = Rh @ _EX, Rh @ _EZ
        offs = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
        pts, cols, sizes = [], [], []
        for c, (dx, dz) in enumerate(offs):
            rel = fp[base + c]["relative_grip"] if fp else 0.0
            pts.append(tip + ex * dx * 0.28 + ez * dz * 0.28 + ez * 0.25)
            r, g, b = force_color(rel)
            cols.append((r, g, b, 1.0))
            sizes.append(0.16 + 0.22 * rel)
        self.force_pts[name].setData(pos=np.array(pts), color=np.array(cols),
                                     size=np.array(sizes))


# ----------------------------------------------------------------------------
# Control panel + telemetry (Qt)
# ----------------------------------------------------------------------------
class ControlPanel:
    def __init__(self, cfg, on_zero_force, palm_channels=None, sensor_labels=None):
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

        self.cfg = cfg
        self.active = 0
        sensor_labels = sensor_labels or SENSOR_LABELS
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(320)
        lay = QtWidgets.QVBoxLayout(panel)

        self._theme = "dark"
        self._theme_cb = None
        self.theme_btn = QtWidgets.QPushButton("Theme: Dark")
        self.theme_btn.clicked.connect(self._on_theme)
        lay.addWidget(self.theme_btn)

        # Render style: noodle (smooth, prettier) vs low-poly (blocky, higher fps).
        self._style = "noodle"
        self._style_cb = None
        self.style_btn = QtWidgets.QPushButton("Style: Noodle")
        self.style_btn.setToolTip("Switch to low-poly (blocky) for higher fps")
        self.style_btn.clicked.connect(self._on_style)
        lay.addWidget(self.style_btn)

        lay.addWidget(QtWidgets.QLabel("<b>Sensor</b>"))
        self.sensor_box = QtWidgets.QComboBox()
        self.sensor_box.addItems(sensor_labels)
        self.sensor_box.currentIndexChanged.connect(self._on_select)
        lay.addWidget(self.sensor_box)

        self.enable_cb = QtWidgets.QCheckBox("Enabled")
        self.enable_cb.setChecked(True)
        self.enable_cb.stateChanged.connect(self._on_enable)
        lay.addWidget(self.enable_cb)

        lay.addWidget(QtWidgets.QLabel("Axis remap (output ← source):"))
        self.axis_boxes = []
        for lbl in ("X", "Y", "Z"):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(f" {lbl} ←"))
            cb = QtWidgets.QComboBox()
            cb.addItems(AXIS_OPTIONS)
            cb.currentIndexChanged.connect(self._on_axis)
            row.addWidget(cb)
            lay.addLayout(row)
            self.axis_boxes.append(cb)

        for text, cb in (("Reset axes", self._on_reset_axes),
                         ("Zero hand", self._on_zero),
                         ("Clear zero", self._on_clear_zero),
                         ("Zero force", on_zero_force)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(cb)
            lay.addWidget(b)

        lay.addWidget(QtWidgets.QLabel("<b>Force (2×2 per finger)</b>"))
        self.cells = {}
        grid = QtWidgets.QGridLayout()
        for col, fname in enumerate(("index", "middle", "thumb")):
            grid.addWidget(QtWidgets.QLabel(fname), 0, col)
            sub = QtWidgets.QGridLayout()
            base = FORCE_BASE[fname]
            for c in range(4):
                cell = QtWidgets.QFrame()
                cell.setFixedSize(28, 28)
                cell.setFrameShape(QtWidgets.QFrame.Box)
                sub.addWidget(cell, c // 2, c % 2)
                self.cells[base + c] = cell
            holder = QtWidgets.QWidget()
            holder.setLayout(sub)
            grid.addWidget(holder, 1, col)
        lay.addLayout(grid)

        if palm_channels:
            lay.addWidget(QtWidgets.QLabel("palm (center / thenar / hypo)"))
            prow = QtWidgets.QHBoxLayout()
            for ch in palm_channels:
                cell = QtWidgets.QFrame()
                cell.setFixedSize(28, 28)
                cell.setFrameShape(QtWidgets.QFrame.Box)
                prow.addWidget(cell)
                self.cells[ch] = cell
            prow.addStretch(1)
            lay.addLayout(prow)

        self.info = QtWidgets.QLabel()
        self.info.setFont(QtGui.QFont("Courier New", 9))
        self.info.setAlignment(QtCore.Qt.AlignTop)
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.info, stretch=1)

        self.widget = panel
        self._refresh_axis_widgets()

    def _on_select(self, idx):
        self.active = idx
        self._refresh_axis_widgets()

    def _on_enable(self, _s):
        self.cfg.set_enabled(self.active, self.enable_cb.isChecked())

    def _on_axis(self, _i):
        M = build_remap(self.axis_boxes[0].currentText(),
                        self.axis_boxes[1].currentText(),
                        self.axis_boxes[2].currentText())
        if is_valid_remap(M):
            self.cfg.set_remap(self.active, M)

    def _on_reset_axes(self):
        self.cfg.reset_remap(self.active)
        self._refresh_axis_widgets()

    def _on_zero(self):
        if hasattr(self, "_zero_cb"):
            self._zero_cb()

    def _on_clear_zero(self):
        self.cfg.clear_zero()

    def set_zero_callback(self, cb):
        self._zero_cb = cb

    def set_theme_callback(self, cb):
        self._theme_cb = cb

    def set_style_callback(self, cb):
        self._style_cb = cb

    def _on_style(self):
        self._style = "lowpoly" if self._style == "noodle" else "noodle"
        self.style_btn.setText(f"Style: {'Low-poly' if self._style == 'lowpoly' else 'Noodle'}")
        if self._style_cb:
            self._style_cb(self._style)

    def _on_theme(self):
        self._theme = "light" if self._theme == "dark" else "dark"
        self.theme_btn.setText(f"Theme: {self._theme.capitalize()}")
        if self._theme_cb:
            self._theme_cb(self._theme)

    def _refresh_axis_widgets(self):
        self.enable_cb.blockSignals(True)
        self.enable_cb.setChecked(self.cfg.enabled[self.active])
        self.enable_cb.blockSignals(False)
        for cb, name in zip(self.axis_boxes, matrix_to_axes(self.cfg.remap[self.active])):
            cb.blockSignals(True)
            cb.setCurrentText(name)
            cb.blockSignals(False)

    def update_force(self, fp):
        if not fp:
            return
        for m, cell in self.cells.items():
            if m >= len(fp):
                continue
            r, g, b = force_color(fp[m]["relative_grip"])
            cell.setStyleSheet(
                f"background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)});")

    def set_info(self, text):
        self.info.setText(text)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _wrist_quat(frame):
    q = (frame.get("bno_qw", 1.0), frame.get("bno_qx", 0.0),
         frame.get("bno_qy", 0.0), frame.get("bno_qz", 0.0))
    if abs(q[0]) + abs(q[1]) + abs(q[2]) + abs(q[3]) < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    return q


def main():
    import time

    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtWidgets

    import senz_multi_io as mio
    import pinch as pinchmod
    from fusion.madgwick import MadgwickAHRS
    from force_pipeline import ForceArray, process_frame

    ap = argparse.ArgumentParser(description="senz v3 native GPU hand visualizer")
    ap.add_argument("--port", help="serial port, e.g. COM5")
    ap.add_argument("--simulate", action="store_true", help="no hardware")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--beta", type=float, default=0.1, help="Madgwick gain")
    ap.add_argument("--hand", choices=["right", "left"], default="right",
                    help="hand layout (default right)")
    ap.add_argument("--sim", choices=["tactile", "proto", "pinch"], default="tactile",
                    help="which simulator for --simulate. tactile: 1 dorsum IMU + "
                         "15-taxel force. proto: 8-IMU + 15 taxel. pinch: 8-IMU + "
                         "12 fingertip taxels, animates index/middle-thumb pinches")
    ap.add_argument("--camera", metavar="SRC",
                    help="camera source (index or URL) to FUSE: the IMU supplies "
                         "hand orientation, the camera articulates the fingers")
    ap.add_argument("--ble", metavar="NAME",
                    help="connect over Bluetooth LE to this device name (Nordic "
                         "UART Service) instead of a serial --port")
    ap.add_argument("--fingers", metavar="LIST",
                    help="comma list of fingers to draw (thumb,index,middle,ring,"
                         "pinky). Default: all, or thumb,index,middle for the pinch "
                         "build")
    args = ap.parse_args()

    if args.ble:
        from senz_ble_io import open_ble_source
        source = open_ble_source(args.ble)
    elif args.simulate or not args.port:
        if args.sim == "proto":
            from senz_v3_sim import SimV3Source
            source = SimV3Source()
        elif args.sim == "pinch":
            from senz_v3_pinch_sim import SimV3PinchSource
            source = SimV3PinchSource()
        else:
            from senz_v3_tactile_sim import SimV3TactileSource
            source = SimV3TactileSource()
    else:
        source = mio.open_multi_source(port=args.port, baud=args.baud)
    reader = ThreadedMultiReader(source)
    schema = reader.schema

    # The dorsum (hand frame) is the LAST IMU by convention -> sensor index nimu.
    n_imu = max(0, schema.nimu)
    dorsum_sensor = n_imu if n_imu >= 1 else 0
    if n_imu == 0:
        print("WARNING: firmware reports nimu=0 (banner not parsed?) -- hand pose "
              "falls back to the wrist frame; check the serial banner.")
    sensor_labels = make_sensor_labels(n_imu)
    # Builds with finger IMUs (proto/pinch, nimu>=8) articulate the fingers from
    # them; the tactile build (nimu=1) gets [] -> canonical open hand.
    imu_map = finger_imu_map(n_imu)

    # Which fingers to draw. Explicit --fingers wins; otherwise the pinch build
    # (via --sim pinch or a "pinch" firmware banner) omits the ring + pinky.
    is_pinch = args.sim == "pinch" or "pinch" in (schema.build or "")
    if args.fingers:
        active_fingers = hmod.parse_fingers(args.fingers)
    elif is_pinch:
        active_fingers = list(hmod.PINCH_FINGERS)
    else:
        active_fingers = list(hmod.FINGERS)

    cfg = SensorConfigV3(1 + n_imu)
    filters = [MadgwickAHRS(beta=args.beta) for _ in range(n_imu)]
    forces = ForceArray(schema.nforce, rate=schema.rate or 200)
    state = {"prev_t": None, "fps": 0.0, "tlast": None, "raw_quats": None}

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.setObjectName("senzmain")
    win.setWindowTitle(f"senz v3 - live 3D hand ({args.hand}) [GPU]")
    win.resize(1180, 720)
    root = QtWidgets.QHBoxLayout(win)

    view = gl.GLViewWidget()
    view.setCameraPosition(distance=13, elevation=22, azimuth=-65)
    grid = gl.GLGridItem()
    grid.setSize(14, 14)
    grid.setSpacing(1, 1)
    grid.translate(0, 0, -3)
    view.addItem(grid)
    has_palm = schema.nforce >= 15
    palm = make_palm(args.hand) if has_palm else None
    hand = HandGL(view, palm, fingers=active_fingers)
    root.addWidget(view, stretch=3)

    # Optional camera fusion: the tracker runs its own thread and exposes the
    # latest world landmarks; when present the fingers articulate from the camera
    # while the dorsum IMU keeps supplying orientation (see compute_skeleton).
    tracker = None
    if args.camera:
        try:
            from camera_tracker import HandTracker
            src = int(args.camera) if str(args.camera).isdigit() else args.camera
            tracker = HandTracker(camera=src, keep_frame=False)
            tracker.start()
            print(f"camera fusion ON (source {args.camera!r})")
        except Exception as e:
            print(f"camera fusion unavailable ({e}); IMU-only")
            tracker = None

    def zero_force():
        forces.__init__(schema.nforce, rate=schema.rate or 200)

    ctrl = ControlPanel(cfg, zero_force,
                        palm_channels=PALM_CHANNELS if has_palm else None,
                        sensor_labels=sensor_labels)
    ctrl.set_zero_callback(lambda: cfg.capture_zero(state["raw_quats"])
                           if state["raw_quats"] else None)
    root.addWidget(ctrl.widget, stretch=1)

    def apply_theme(mode):
        th = THEMES[mode]
        view.setBackgroundColor(th["gl_bg"])
        try:
            grid.setColor(th["grid"])
        except Exception:
            pass
        ctrl.widget.setStyleSheet(th["qss"])
        win.setStyleSheet(f'QWidget#senzmain {{ background-color: {th["win_bg"]}; }}')

    ctrl.set_theme_callback(apply_theme)
    ctrl.set_style_callback(hand.set_style)
    apply_theme("dark")

    def tick():
        frame = reader.get()
        if frame is None:
            return
        t_us = frame.get("t_us", 0)
        if state["prev_t"] is None:
            dt = 1.0 / (schema.rate or 200)
        else:
            dt = max(1e-4, min(0.05, (t_us - state["prev_t"]) * 1e-6))
        state["prev_t"] = t_us

        raw_quats = [_wrist_quat(frame)]
        d2r = math.pi / 180.0
        for k in range(n_imu):
            ax, ay, az = mio.imu_accel_g(frame, k)
            gx, gy, gz = mio.imu_gyro_dps(frame, k)
            raw_quats.append(filters[k].update(gx * d2r, gy * d2r, gz * d2r,
                                               ax, ay, az, dt))
        state["raw_quats"] = raw_quats

        world = tracker.get_world() if tracker is not None else None
        skel = compute_skeleton(raw_quats, cfg, dorsum_sensor, args.hand,
                                world_lms=world, imu_map=imu_map)
        fp = process_frame(frame, forces)
        hand.update(skel, fp)
        ctrl.update_force(fp)

        now = time.time()
        if state["tlast"] is not None:
            state["fps"] += 0.1 * (1.0 / max(1e-6, now - state["tlast"]) - state["fps"])
        state["tlast"] = now
        lines = [f"fps ~ {state['fps']:5.1f}   dt {dt*1e3:4.1f} ms   hand={args.hand}",
                 f"drive: {skel['driven']}   wrist flex {skel['flex_deg']:5.1f} deg", ""]
        for c in ("thumb", "index", "middle"):
            b = FORCE_BASE[c]
            bars = " ".join(f"{fp[b+i]['relative_grip']:.2f}" for i in range(4))
            lines.append(f"{c:6s} {bars}")
        if has_palm:
            bars = " ".join(f"{fp[c]['relative_grip']:.2f}" for c in PALM_CHANNELS)
            lines.append(f"palm   {bars}")
        # Pinch readout (ML focus): tip gaps + pad pressure + detected state.
        pf = pinchmod.pinch_features(skel["points"], fp)
        lines += ["",
                  f"pinch  idx d={pf['dist_index']:.2f} f={pf['force_index']:.2f}",
                  f"       mid d={pf['dist_middle']:.2f} f={pf['force_middle']:.2f}",
                  f"  -> {pf['state'].upper()}"]
        ctrl.set_info("\n".join(lines))

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(10)

    win.show()
    try:
        (app.exec_ if hasattr(app, "exec_") else app.exec)()
    finally:
        reader.close()
        if tracker is not None:
            tracker.stop()


if __name__ == "__main__":
    main()

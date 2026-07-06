#!/usr/bin/env python3
"""
senz_visualizer.py
==================
Live 3D hand skeleton for the senz_glove_v2 stream (HLD v2, deliverable #14).

Reads the 180-byte binary frame via ``senz_parser`` (11 quaternions: 1 BNO055
wrist + 10 per-finger MPU-6500) and draws an anatomically-simplified hand as
VPython cylinders (bones) and spheres (joints): 5 fingers, each a proximal +
distal segment, hung off a wrist/palm block driven by the BNO055.

Bone directions come purely from the IMU quaternions; positions chain forward
(the tip of each proximal bone is the base of its distal bone). There is no
position tracking -- the hand stays centered in the scene, which is correct for
an IMU-only system (HLD "Known Limitation").

The firmware runs a per-finger Madgwick filter and sends each quaternion RELATIVE
to the wrist frame (q_rel = conj(q_wrist) * q_finger); the composition here,
Rw @ R(q_rel), recovers world orientation (validated to 1e-15 against the math).

Interactive controls (in the browser window, no restart needed):
  * Active sensor menu -- pick wrist or any of the 10 finger IMUs to edit.
  * Enabled checkbox   -- temporarily disable a sensor (its bone freezes
                          straight and dims); handy to isolate a misbehaving IMU.
  * Axis remap menus   -- re-map each output axis X/Y/Z to any signed source
                          axis (invert or swap), applied as M @ R @ M^T so the
                          result stays a valid rotation. Reference finger accel
                          frame: +Y away from hand, +X toward thumb, +Z up.
  * Zero hand          -- capture the current pose as the neutral/tare for the
                          wrist and every finger (live calibration).
  * Clear zero / Reset axes for the active sensor.

Quaternion->rotation reuses ``senz_io.quat_to_matrix`` (pure numpy, normalized,
no gimbal lock) -- no scipy needed.

Run (calibrate first for a tidy flat pose, optional):
    python senz_calibrate_pose.py --simulate --yes
    python senz_visualizer.py --simulate       # no hardware
    python senz_visualizer.py --port COM5       # wired USB serial (921600)
"""

import argparse
import json
import os

import numpy as np

import senz_io       # quat_to_matrix (shared, pure-numpy quaternion math)
import senz_parser   # v2 binary frame source

# ----------------------------------------------------------------------------
# Skeleton geometry (mm). Lengths from the HLD "Starting bone lengths" table;
# knuckle positions and thumb rest angle are starting values, adjustable.
# ----------------------------------------------------------------------------
PALM_LEN = 40.0        # wrist block: origin -> knuckle line, along +y
PALM_WIDTH = 34.0
PALM_THICK = 12.0

# One entry per finger. prox/dist index into frame.mpu (HLD finger-placement
# table). knuckle is the finger's base in the wrist-local frame; rest is the
# bone's rest direction (fingers forward +y; thumb splayed out).
FINGERS = [
    dict(name="index",  color=(1.0, 0.2, 0.2), prox=0, dist=1,
         knuckle=(15.0, PALM_LEN, 0.0), rest=(0.0, 1.0, 0.0), plen=45.0, dlen=25.0),
    dict(name="middle", color=(0.2, 1.0, 0.2), prox=2, dist=3,
         knuckle=(5.0, PALM_LEN + 2, 0.0), rest=(0.0, 1.0, 0.0), plen=45.0, dlen=25.0),
    dict(name="ring",   color=(0.3, 0.5, 1.0), prox=4, dist=5,
         knuckle=(-5.0, PALM_LEN, 0.0), rest=(0.0, 1.0, 0.0), plen=45.0, dlen=25.0),
    dict(name="pinky",  color=(1.0, 1.0, 0.2), prox=6, dist=7,
         knuckle=(-15.0, PALM_LEN - 4, 0.0), rest=(0.0, 1.0, 0.0), plen=45.0, dlen=25.0),
    dict(name="thumb",  color=(1.0, 1.0, 1.0), prox=8, dist=9,
         knuckle=(20.0, 12.0, 0.0), rest=(0.7, 0.55, 0.0), plen=35.0, dlen=25.0),
]

BONE_RADIUS = 4.0
JOINT_RADIUS = 5.5
DISABLED_OPACITY = 0.2

# Sensor indexing used everywhere below: 0 = wrist (BNO055), 1..10 = MPU finger
# IMUs (index j+1 == frame.mpu[j]).
NUM_SENSORS = 1 + senz_parser.NUM_IMU
_MPU_LABELS = [None] * senz_parser.NUM_IMU
for _f in FINGERS:
    _seg = ("base", "tip") if _f["name"] == "thumb" else ("prox", "dist")
    _MPU_LABELS[_f["prox"]] = f'{_f["name"]} {_seg[0]}'
    _MPU_LABELS[_f["dist"]] = f'{_f["name"]} {_seg[1]}'
SENSOR_LABELS = ["wrist (BNO055)"] + _MPU_LABELS

# Axis remap options and helpers.
AXIS_OPTIONS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
_AXIS_VEC = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0),
             "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def quat_to_R(q):
    """(w, x, y, z) -> 3x3 rotation matrix (reuses the project's helper)."""
    return senz_io.quat_to_matrix(*q)


def build_remap(sx, sy, sz):
    """Signed-permutation matrix from three axis choices (rows = X,Y,Z sources)."""
    return np.array([_AXIS_VEC[sx], _AXIS_VEC[sy], _AXIS_VEC[sz]], dtype=float)


def is_valid_remap(M):
    """True iff M is orthogonal (a genuine signed permutation, no dup axes)."""
    return np.allclose(M @ M.T, np.eye(3), atol=1e-9)


def matrix_to_axes(M):
    """Inverse of build_remap: matrix -> ('+X','+Y','+Z')-style labels."""
    names = ["X", "Y", "Z"]
    out = []
    for i in range(3):
        j = int(np.argmax(np.abs(M[i])))
        out.append(("+" if M[i, j] >= 0 else "-") + names[j])
    return out


# ----------------------------------------------------------------------------
# Per-sensor configuration: enable flag, axis remap, and zero-pose tare.
# Pure numpy -- no VPython -- so it is unit-testable headless.
# ----------------------------------------------------------------------------
class SensorConfig:
    def __init__(self):
        self.enabled = [True] * NUM_SENSORS
        self.remap = [np.eye(3) for _ in range(NUM_SENSORS)]
        self.zinv = [None] * NUM_SENSORS   # zero-tare inverse matrices, or None

    def set_enabled(self, idx, on):
        self.enabled[idx] = bool(on)

    def set_remap(self, idx, M):
        self.remap[idx] = M

    def reset_remap(self, idx):
        self.remap[idx] = np.eye(3)

    def _remapped(self, idx, qraw):
        """Orientation with the axis remap applied but NOT the zero tare."""
        M = self.remap[idx]
        return M @ quat_to_R(qraw) @ M.T

    def sensor_R(self, idx, qraw):
        """Final orientation matrix for a sensor: remap, tare, or identity."""
        if not self.enabled[idx]:
            return np.eye(3)
        R = self._remapped(idx, qraw)
        if self.zinv[idx] is not None:
            R = R @ self.zinv[idx]
        return R

    def capture_zero(self, frame):
        """Tare wrist + all fingers so the current pose becomes neutral."""
        self.zinv[0] = self._remapped(0, frame.bno).T
        for j in range(senz_parser.NUM_IMU):
            self.zinv[j + 1] = self._remapped(j + 1, frame.mpu[j]).T

    def clear_zero(self):
        self.zinv = [None] * NUM_SENSORS

    def set_finger_zinv(self, matrices):
        """Seed finger zero-tares from senz_calibrate_pose.py's pose_offsets."""
        for j, m in enumerate(matrices[: senz_parser.NUM_IMU]):
            self.zinv[j + 1] = np.asarray(m, dtype=float)


# ----------------------------------------------------------------------------
# Optional zero-pose calibration file (from senz_calibrate_pose.py).
# ----------------------------------------------------------------------------
def load_pose_offsets(path="pose_offsets.json"):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    quats = data.get("offsets", [])
    if len(quats) < senz_parser.NUM_IMU:
        return None
    # Inverse of a rotation matrix is its transpose.
    return [quat_to_R(q).T for q in quats[: senz_parser.NUM_IMU]]


# ----------------------------------------------------------------------------
# Forward kinematics -- pure numpy, no VPython. Returns world-space points.
# ----------------------------------------------------------------------------
def compute_skeleton(frame, cfg=None):
    """Frame -> dict with palm segment and per-finger geometry + enable flags."""
    if cfg is None:
        cfg = SensorConfig()
    Rw = cfg.sensor_R(0, frame.bno)          # wrist -> world
    origin = np.zeros(3)
    palm_tip = Rw @ np.array([0.0, PALM_LEN, 0.0])

    fingers = []
    for f in FINGERS:
        knuckle = Rw @ np.asarray(f["knuckle"], dtype=float)
        rest = _unit(f["rest"])
        Rp = Rw @ cfg.sensor_R(f["prox"] + 1, frame.mpu[f["prox"]])
        Rd = Rw @ cfg.sensor_R(f["dist"] + 1, frame.mpu[f["dist"]])
        mid = knuckle + (Rp @ rest) * f["plen"]     # proximal tip = distal base
        tip = mid + (Rd @ rest) * f["dlen"]          # fingertip
        fingers.append(dict(knuckle=knuckle, mid=mid, tip=tip, color=f["color"],
                            prox_on=cfg.enabled[f["prox"] + 1],
                            dist_on=cfg.enabled[f["dist"] + 1]))

    return dict(palm=(origin, palm_tip), wrist_on=cfg.enabled[0], fingers=fingers)


# ----------------------------------------------------------------------------
# VPython renderer (imported lazily so headless imports/tests don't need it).
# ----------------------------------------------------------------------------
class HandRenderer:
    """Builds the skeleton once, then repositions/dims bones+joints each frame."""

    def __init__(self):
        from vpython import canvas, box, cylinder, sphere, vector

        self._vec = vector
        self.scene = canvas(title="senz — live hand (v2)", width=900, height=640,
                            background=vector(0.08, 0.08, 0.10))
        self.scene.forward = vector(0, -0.4, -1)
        self.scene.up = vector(0, 1, 0)

        z = vector(0, 0, 0)
        self.palm = box(pos=z, axis=vector(0, PALM_LEN, 0), height=PALM_WIDTH,
                        width=PALM_THICK, color=vector(0.6, 0.6, 0.62))
        self.wrist_joint = sphere(pos=z, radius=JOINT_RADIUS,
                                  color=vector(0.7, 0.7, 0.72))

        self.prox, self.dist = [], []
        self.knuckle_j, self.mid_j = [], []
        for f in FINGERS:
            col = vector(*f["color"])
            self.prox.append(cylinder(pos=z, axis=vector(0, f["plen"], 0),
                                      radius=BONE_RADIUS, color=col))
            self.dist.append(cylinder(pos=z, axis=vector(0, f["dlen"], 0),
                                      radius=BONE_RADIUS, color=col))
            self.knuckle_j.append(sphere(pos=z, radius=JOINT_RADIUS, color=col))
            self.mid_j.append(sphere(pos=z, radius=JOINT_RADIUS * 0.9, color=col))

    def _v(self, a):
        return self._vec(float(a[0]), float(a[1]), float(a[2]))

    def update(self, skel):
        base, ptip = skel["palm"]
        self.palm.pos = self._v(base)
        self.palm.axis = self._v(ptip - base)
        self.palm.opacity = 1.0 if skel["wrist_on"] else DISABLED_OPACITY
        self.wrist_joint.pos = self._v(base)
        for i, fg in enumerate(skel["fingers"]):
            self.prox[i].pos = self._v(fg["knuckle"])
            self.prox[i].axis = self._v(fg["mid"] - fg["knuckle"])
            self.prox[i].opacity = 1.0 if fg["prox_on"] else DISABLED_OPACITY
            self.dist[i].pos = self._v(fg["mid"])
            self.dist[i].axis = self._v(fg["tip"] - fg["mid"])
            self.dist[i].opacity = 1.0 if fg["dist_on"] else DISABLED_OPACITY
            self.knuckle_j[i].pos = self._v(fg["knuckle"])
            self.mid_j[i].pos = self._v(fg["mid"])


# ----------------------------------------------------------------------------
# Interactive control panel (VPython widgets) wired to a SensorConfig.
# ----------------------------------------------------------------------------
def _build_controls(scene, cfg, state):
    from vpython import menu, checkbox, button, wtext

    w = {}
    status = wtext(text="")

    def set_status(msg):
        status.text = f"\n<b>{msg}</b>\n"

    def refresh():
        """Sync widgets to the active sensor's stored config."""
        a = state["active"]
        w["en"].checked = cfg.enabled[a]
        ax = matrix_to_axes(cfg.remap[a])
        w["mx"].selected, w["my"].selected, w["mz"].selected = ax
        set_status(f"editing {SENSOR_LABELS[a]}  |  enabled={cfg.enabled[a]}  "
                   f"axes {ax[0]}/{ax[1]}/{ax[2]}")

    def on_select(m):
        state["active"] = SENSOR_LABELS.index(m.selected)
        refresh()

    def on_enable(c):
        cfg.set_enabled(state["active"], c.checked)
        set_status(f"{SENSOR_LABELS[state['active']]} "
                   f"{'ENABLED' if c.checked else 'DISABLED'}")

    def on_axis(_=None):
        a = state["active"]
        M = build_remap(w["mx"].selected, w["my"].selected, w["mz"].selected)
        if not is_valid_remap(M):
            set_status("invalid: X/Y/Z must each use a different source axis")
            ax = matrix_to_axes(cfg.remap[a])           # revert menus
            w["mx"].selected, w["my"].selected, w["mz"].selected = ax
            return
        cfg.set_remap(a, M)
        set_status(f"{SENSOR_LABELS[a]} axes -> "
                   f"{w['mx'].selected}/{w['my'].selected}/{w['mz'].selected}")

    def on_reset_axes(_):
        cfg.reset_remap(state["active"])
        refresh()

    def on_zero(_):
        fr = state["latest"]
        if fr is None:
            set_status("no frame yet — wait for data, then Zero")
            return
        cfg.capture_zero(fr)
        set_status("zeroed: current pose is now neutral for all sensors")

    def on_clear_zero(_):
        cfg.clear_zero()
        set_status("zero tare cleared (raw orientations)")

    scene.append_to_caption("\n\nActive sensor:  ")
    w["sel"] = menu(choices=SENSOR_LABELS, bind=on_select)
    scene.append_to_caption("   ")
    w["en"] = checkbox(text=" enabled", bind=on_enable, checked=True)

    scene.append_to_caption("\n\nAxis remap (output <- source):   X ")
    w["mx"] = menu(choices=AXIS_OPTIONS, bind=on_axis)
    scene.append_to_caption("  Y ")
    w["my"] = menu(choices=AXIS_OPTIONS, bind=on_axis)
    scene.append_to_caption("  Z ")
    w["mz"] = menu(choices=AXIS_OPTIONS, bind=on_axis)
    scene.append_to_caption("   ")
    button(text="Reset axes", bind=on_reset_axes)

    scene.append_to_caption("\n\nCalibration:   ")
    button(text="Zero hand", bind=on_zero)
    scene.append_to_caption("  ")
    button(text="Clear zero", bind=on_clear_zero)
    scene.append_to_caption("\n")

    refresh()
    return w


def main():
    ap = argparse.ArgumentParser(description="senz v2 live 3D hand visualizer")
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--simulate", action="store_true", help="no hardware")
    ap.add_argument("--baud", type=int, default=senz_parser.BAUD)
    ap.add_argument("--pose", default="pose_offsets.json",
                    help="zero-pose calibration file (optional)")
    args = ap.parse_args()

    from vpython import rate  # fail here with a clear message if vpython missing

    src = senz_parser.open_frame_source(args.port, simulate=args.simulate,
                                        baud=args.baud)
    cfg = SensorConfig()
    offs = load_pose_offsets(args.pose)
    if offs is not None:
        cfg.set_finger_zinv(offs)
        print(f"# loaded finger zero-pose from {args.pose}")
    elif os.path.exists(args.pose):
        print(f"# {args.pose} present but unusable; ignoring")

    print("# visualizer running; use the on-screen controls. Ctrl-C to stop.")
    renderer = HandRenderer()
    state = {"active": 0, "latest": None}
    _build_controls(renderer.scene, cfg, state)

    try:
        while True:
            rate(60)  # cap render at ~60 Hz; drain to the freshest frame
            frame = src.read(block=False)
            if frame is not None:
                state["latest"] = frame
                renderer.update(compute_skeleton(frame, cfg))
    except KeyboardInterrupt:
        pass
    finally:
        src.close()


if __name__ == "__main__":
    main()

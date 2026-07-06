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

STATUS: rendering + forward kinematics, driven by real per-finger orientation.
The firmware now runs a Madgwick filter per finger and sends each quaternion
RELATIVE to the wrist frame (q_rel = conj(q_wrist) * q_finger), so the
composition here -- Rw @ R(q_rel) -- recovers world orientation. That contract
was validated numerically (Rw @ R(q_rel) == R(q_finger) to 1e-15). Fingers
articulate on hardware and in --simulate. Remaining pending piece: a clean
flat-hand zero pose needs senz_calibrate_pose.py (writes pose_offsets.json,
auto-loaded below); without it, fingers rest at the sensors' mounted
orientation rather than a tidy straight pose.

Quaternion->rotation reuses ``senz_io.quat_to_matrix`` (pure numpy, normalized,
no gimbal lock) -- no scipy needed.

Run:
    python senz_visualizer.py --simulate       # no hardware (wrist rocks)
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


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def quat_to_R(q):
    """(w, x, y, z) -> 3x3 rotation matrix (reuses the project's helper)."""
    return senz_io.quat_to_matrix(*q)


# ----------------------------------------------------------------------------
# Optional zero-pose calibration (from senz_calibrate_pose.py, deferred).
# If pose_offsets.json is present it holds each sensor's flat-hand quaternion;
# we right-multiply by its inverse so a calibrated flat hand renders flat.
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
# Forward kinematics -- pure numpy, no VPython, so it is unit-testable headless.
# Returns world-space points for every bone and joint.
# ----------------------------------------------------------------------------
def compute_skeleton(frame, offsets=None):
    """Frame -> dict with palm segment and per-finger (knuckle, mid, tip, color)."""
    Rw = quat_to_R(frame.bno)          # wrist -> world
    origin = np.zeros(3)
    palm_tip = Rw @ np.array([0.0, PALM_LEN, 0.0])

    fingers = []
    for f in FINGERS:
        knuckle = Rw @ np.asarray(f["knuckle"], dtype=float)
        rest = _unit(f["rest"])

        # Each finger quaternion is expressed relative to the wrist frame (that
        # is the firmware's step-6 contract; identity until fusion lands), so
        # world orientation = wrist-to-world @ finger-relative. <-- validate here
        Rp = Rw @ quat_to_R(frame.mpu[f["prox"]])
        Rd = Rw @ quat_to_R(frame.mpu[f["dist"]])
        if offsets is not None:
            Rp = Rw @ (quat_to_R(frame.mpu[f["prox"]]) @ offsets[f["prox"]])
            Rd = Rw @ (quat_to_R(frame.mpu[f["dist"]]) @ offsets[f["dist"]])

        mid = knuckle + (Rp @ rest) * f["plen"]     # proximal tip = distal base
        tip = mid + (Rd @ rest) * f["dlen"]          # fingertip
        fingers.append(dict(knuckle=knuckle, mid=mid, tip=tip, color=f["color"]))

    return dict(palm=(origin, palm_tip), fingers=fingers)


# ----------------------------------------------------------------------------
# VPython renderer (imported lazily so headless imports/tests don't need it).
# ----------------------------------------------------------------------------
class HandRenderer:
    """Builds the skeleton once, then repositions bones/joints each frame."""

    def __init__(self):
        from vpython import canvas, box, cylinder, sphere, vector, color

        self._vec = vector
        self.scene = canvas(title="senz — live hand (v2)", width=900, height=700,
                            background=vector(0.08, 0.08, 0.10))
        self.scene.forward = vector(0, -0.4, -1)
        self.scene.up = vector(0, 1, 0)

        z = vector(0, 0, 0)
        self.palm = box(pos=z, axis=vector(0, PALM_LEN, 0), height=PALM_WIDTH,
                        width=PALM_THICK, color=vector(0.6, 0.6, 0.62))
        self.wrist_joint = sphere(pos=z, radius=JOINT_RADIUS,
                                  color=vector(0.7, 0.7, 0.72))

        # 10 bone cylinders (2 per finger) + 10 knuckle/mid joint spheres.
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
        self.wrist_joint.pos = self._v(base)
        for i, fg in enumerate(skel["fingers"]):
            self.prox[i].pos = self._v(fg["knuckle"])
            self.prox[i].axis = self._v(fg["mid"] - fg["knuckle"])
            self.dist[i].pos = self._v(fg["mid"])
            self.dist[i].axis = self._v(fg["tip"] - fg["mid"])
            self.knuckle_j[i].pos = self._v(fg["knuckle"])
            self.mid_j[i].pos = self._v(fg["mid"])


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
    offsets = load_pose_offsets(args.pose)
    if offsets is None and os.path.exists(args.pose):
        print(f"# {args.pose} present but unusable; ignoring")
    print("# visualizer running; close the window or Ctrl-C to stop")
    if not args.simulate:
        print("# NOTE: finger fusion is not in firmware yet -> fingers render "
              "flat; wrist orientation is live")

    renderer = HandRenderer()
    try:
        while True:
            rate(60)  # cap render at ~60 Hz; drain to the freshest frame
            frame = src.read(block=False)
            if frame is not None:
                renderer.update(compute_skeleton(frame, offsets))
    except KeyboardInterrupt:
        pass
    finally:
        src.close()


if __name__ == "__main__":
    main()

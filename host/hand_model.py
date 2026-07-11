#!/usr/bin/env python3
"""
hand_model.py  --  shared 21-landmark hand topology + geometry
==============================================================
One source of truth for the MediaPipe hand skeleton, imported by both the camera
UI (``camera_setup.py``) and the 3D visualizer (``senz_v3_qt.py``) so they look
the same. Pure numpy -- no Qt/OpenCV/mediapipe -- so it is headless-testable.

Landmark order = MediaPipe (``camera_tracker.LANDMARK_NAMES``): 0 wrist, then
thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20 (mcp->pip->dip->tip).

Coordinate convention (hand-local, right hand, palm down):
  +X -> toward the thumb..pinky axis (thumb at -X, pinky at +X)
  +Y -> finger direction (away from the wrist)
  +Z -> out the back of the hand
This matches the palm frame senz_v3_qt places with the dorsum IMU, so posing the
hand is just ``wrist_pos + R_hand @ local``.
"""

import numpy as np

from camera_tracker import LANDMARK_NAMES   # 21 names, MediaPipe order

# 21-landmark connectivity (MediaPipe HAND_CONNECTIONS) + fingertip indices.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]
FINGERTIPS = [4, 8, 12, 16, 20]
N_LANDMARKS = 21

# Landmark -> finger (0/wrist + the 5-9-13-17 cross-links render as "palm").
FINGER_OF = ["palm"] + ["thumb"] * 4 + ["index"] * 4 + ["middle"] * 4 + \
            ["ring"] * 4 + ["pinky"] * 4
FINGER_LANDMARKS = {
    "thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20],
}

# Distinct per-finger colors (rgb 0..1) -- distinct colors make hand orientation
# read at a glance (you can tell thumb from pinky).
FINGER_COLORS = {
    "thumb":  (0.96, 0.47, 0.35),
    "index":  (0.36, 0.74, 1.00),
    "middle": (0.42, 0.90, 0.52),
    "ring":   (0.97, 0.80, 0.32),
    "pinky":  (0.82, 0.52, 0.96),
    "palm":   (0.78, 0.78, 0.84),
}
PALM_COLOR = FINGER_COLORS["palm"]

# Canonical flat OPEN right hand in hand-local coords (wrist at origin). Middle
# knuckle sits highest in +Z -> the transverse metacarpal arch (domed back).
CANONICAL_HAND = np.array([
    (0.00, 0.00, 0.00),    # 0  wrist
    (-0.90, 0.40, 0.00),   # 1  thumb_cmc
    (-1.50, 1.00, 0.00),   # 2  thumb_mcp
    (-1.92, 1.60, 0.02),   # 3  thumb_ip
    (-2.24, 2.10, 0.04),   # 4  thumb_tip
    (-1.00, 2.00, 0.10),   # 5  index_mcp
    (-1.05, 2.95, 0.10),   # 6  index_pip
    (-1.08, 3.55, 0.10),   # 7  index_dip
    (-1.10, 4.05, 0.10),   # 8  index_tip
    (-0.35, 2.20, 0.12),   # 9  middle_mcp
    (-0.35, 3.25, 0.12),   # 10 middle_pip
    (-0.35, 3.95, 0.12),   # 11 middle_dip
    (-0.35, 4.50, 0.12),   # 12 middle_tip
    (0.35, 2.10, 0.10),    # 13 ring_mcp
    (0.40, 3.05, 0.10),    # 14 ring_pip
    (0.43, 3.65, 0.10),    # 15 ring_dip
    (0.45, 4.15, 0.10),    # 16 ring_tip
    (1.02, 1.92, 0.05),    # 17 pinky_mcp
    (1.12, 2.62, 0.05),    # 18 pinky_pip
    (1.18, 3.10, 0.05),    # 19 pinky_dip
    (1.22, 3.48, 0.05),    # 20 pinky_tip
], dtype=float)

# Reference length (wrist -> middle_mcp) used to normalize a fused hand's scale.
CANON_REF = float(np.linalg.norm(CANONICAL_HAND[9] - CANONICAL_HAND[0]))


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def mirror_hand(pts, hand="right"):
    """Flip X for the left hand (thumb switches side). Returns a copy."""
    pts = np.array(pts, dtype=float, copy=True)
    if hand == "left":
        pts[:, 0] *= -1.0
    return pts


def joint_color(i):
    return FINGER_COLORS[FINGER_OF[i]]


def connection_color(a, b):
    """Bone color: the finger's color if both ends share a finger; the finger's
    color for a wrist->mcp link; palm gray for the mcp-to-mcp cross links."""
    fa, fb = FINGER_OF[a], FINGER_OF[b]
    if fa == fb and fa != "palm":
        return FINGER_COLORS[fa]
    if a == 0:
        return FINGER_COLORS.get(fb, PALM_COLOR)
    return PALM_COLOR


def hand_local_frame(lms):
    """Orthonormal palm frame (columns = local X/Y/Z in the input frame) from
    wrist(0), index_mcp(5), pinky_mcp(17), middle_mcp(9) -- Gram-Schmidt. This is
    the rotation that carries the hand's own frame into the input (camera) frame;
    its transpose removes the camera's orientation."""
    lms = np.asarray(lms, dtype=float)
    origin = lms[0]
    across = lms[17] - lms[5]        # index_mcp -> pinky_mcp  (+X)
    fwd = lms[9] - origin            # wrist -> middle_mcp      (+Y)
    x = _unit(across)
    y = _unit(fwd - np.dot(fwd, x) * x)
    z = _unit(np.cross(x, y))
    y = _unit(np.cross(z, x))        # re-orthogonalize
    return np.column_stack([x, y, z])


def pose_from_world(world_lms, hand="right"):
    """Camera world landmarks (21x3, metric, any camera orientation) -> the hand's
    shape in ITS OWN local frame, scaled to CANONICAL_HAND's size. Orientation is
    removed here so a downstream IMU rotation (R_hand) supplies it (the fusion:
    IMU = orientation, camera = finger articulation)."""
    w = np.asarray(world_lms, dtype=float)
    R = hand_local_frame(w)
    local = (w - w[0]) @ R           # each landmark expressed in the hand frame
    ref = float(np.linalg.norm(local[9])) or 1.0
    return local * (CANON_REF / ref)


def canonical(hand="right"):
    """The canonical open hand for the given handedness (IMU-only render)."""
    return mirror_hand(CANONICAL_HAND, hand)


def bend_fingers(pts, bends):
    """Articulate a canonical hand from per-joint rotations (finger IMUs).

    ``pts`` is a 21x3 hand-local pose (e.g. ``canonical(hand)``). ``bends`` is a
    list of ``(pivot_landmark, downstream_landmarks, R_rel)`` applied in order,
    proximal joint first: every landmark in ``downstream_landmarks`` is rotated by
    the hand-frame rotation ``R_rel`` about the *current* position of
    ``pivot_landmark``. Because it walks proximal->distal and re-reads the pivot,
    a distal bend rides on top of the proximal one (nested curl), so two IMUs per
    finger reproduce MCP + PIP flexion. Returns a new array; ``pts`` is untouched.
    """
    out = np.array(pts, dtype=float, copy=True)
    for pivot, downstream, R in bends:
        idx = list(downstream)
        if not idx:
            continue
        piv = out[pivot]
        out[idx] = piv + (np.asarray(R, dtype=float) @ (out[idx] - piv).T).T
    return out

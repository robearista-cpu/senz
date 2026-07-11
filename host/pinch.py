#!/usr/bin/env python3
"""
pinch.py  --  pinch-gesture features (pure numpy, headless-testable)
====================================================================
The v3 **pinch** build focuses its ML on pinching, so this turns the two live
signals -- the posed 21-landmark hand and the fingertip force pads -- into a small
set of pinch features the recorder can log and the visualizer can show:

  - **distance**: normalized 3D gap between the thumb tip and the index / middle
    tip (scaled by the hand's own size so it is camera/scale independent).
  - **force**: aggregate relative grip of the two fingertip pads involved in a
    pinch (thumb pad + that finger's pad), 0..1.
  - **state**: a simple label -- ``open`` / ``index`` / ``middle`` / ``both`` --
    from distance + force thresholds.

Landmark indices are MediaPipe (thumb tip 4, index tip 8, middle tip 12); force
channels follow the firmware order (thumb C0-3, index C4-7, middle C8-11).
"""

import numpy as np

# Fingertip landmark pairs that form a pinch with the thumb (tip 4).
PINCH_PAIRS = {"index": (4, 8), "middle": (4, 12)}
# Force channel groups (2x2 pads) per finger, firmware order.
FORCE_GROUP = {"thumb": (0, 1, 2, 3), "index": (4, 5, 6, 7), "middle": (8, 9, 10, 11)}

# Default thresholds. distance is normalized by wrist->middle_mcp (~1 hand unit);
# a relaxed open pinch sits well above ~0.6, a closed pinch near ~0.2.
DIST_CLOSE = 0.55      # normalized tip gap below this = fingers together
FORCE_CONTACT = 0.15   # aggregate relative grip above this = actually pressing


def _hand_scale(points):
    """Reference length (wrist landmark 0 -> middle_mcp landmark 9) for scale
    normalization. Falls back to 1.0 for a degenerate hand."""
    ref = float(np.linalg.norm(np.asarray(points[9], float) - np.asarray(points[0], float)))
    return ref if ref > 1e-6 else 1.0


def pinch_distances(points):
    """Normalized thumb->finger tip distances from a 21x3 posed hand.

    Returns ``{"index": d, "middle": d}`` where d is the 3D tip gap divided by the
    hand's own size, so it means the same at any scale or camera distance."""
    pts = np.asarray(points, dtype=float)
    scale = _hand_scale(pts)
    out = {}
    for name, (a, b) in PINCH_PAIRS.items():
        out[name] = float(np.linalg.norm(pts[a] - pts[b])) / scale
    return out


def _group_grip(fp, group):
    """Mean relative grip (0..1) over a channel group, robust to a short force list."""
    vals = [fp[c]["relative_grip"] for c in group if c < len(fp)]
    return float(np.mean(vals)) if vals else 0.0


def pinch_forces(fp):
    """Aggregate pinch pressure per finger from the force pipeline output.

    A pinch requires the thumb pad AND that finger's pad to BOTH press, so the
    pinch force is the ``min`` of the two groups -- it is high only when the thumb
    and that specific finger are in contact together. (A mean would leak the shared
    thumb pad's pressure onto the idle finger.) ``fp`` is the
    ``force_pipeline.process_frame`` list. Returns ``{"index": f, "middle": f}``."""
    if not fp:
        return {"index": 0.0, "middle": 0.0}
    thumb = _group_grip(fp, FORCE_GROUP["thumb"])
    out = {}
    for name in ("index", "middle"):
        out[name] = min(thumb, _group_grip(fp, FORCE_GROUP[name]))
    return out


def pinch_state(dists, forces, dist_close=DIST_CLOSE, force_contact=FORCE_CONTACT):
    """Classify the pinch. For a tactile glove the ground-truth pinch signal is
    thumb+finger CONTACT (both pads pressing -> pinch_forces), so state is force-
    driven. Distance is reported separately as a supporting/ML feature (it is only
    meaningful once the fingers are articulated by IMUs or the camera). Returns
    'open' / 'index' / 'middle' / 'both'."""
    idx = forces["index"] > force_contact
    mid = forces["middle"] > force_contact
    if idx and mid:
        return "both"
    if idx:
        return "index"
    if mid:
        return "middle"
    return "open"


def pinch_features(points, fp):
    """Convenience bundle: distances + forces + state in one dict (for logging)."""
    dists = pinch_distances(points)
    forces = pinch_forces(fp)
    return {
        "dist_index": dists["index"], "dist_middle": dists["middle"],
        "force_index": forces["index"], "force_middle": forces["middle"],
        "state": pinch_state(dists, forces),
    }

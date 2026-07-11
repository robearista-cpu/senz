#!/usr/bin/env python3
"""
senz_v3_pinch_sim.py  --  simulator for Hardware Sprint v3 (PINCH)
=================================================================
No-hardware source for the **pinch** build: the index/middle/thumb glove focused
on pinching gestures for ML. Sensor set matches the ``senz_glove_v3_pinch``
firmware exactly (nimu=8, nforce=12):

  - 8 IMUs: thumb 3 (base MCP 9-axis + 2x 6-axis), index 2, middle 2, dorsum 1.
    By convention the DORSUM is the LAST IMU (sensor index nimu).
  - BNO055 wrist (fused quaternion) = forearm frame.
  - 12 velostat taxels = three fingertip 2x2 pads (thumb C0-3, index C4-7,
    middle C8-11). NO palm taxels (that is the tactile build).

Like the other v3 sims it scripts a real per-IMU orientation animation and emits
the **raw accel + gyro that would produce it**, so the host's Madgwick fusion +
the visualizer's finger articulation reconstruct the motion. The scripted motion
is a repeating **pinch cycle**: an index-thumb pinch, release, a middle-thumb
pinch, release -- and the fingertip forces spike on contact for the two fingers
that are pinching, so the tactile signal reads as a real pinch.

Use:
    from senz_v3_pinch_sim import SimV3PinchSource   # viz --sim pinch uses this
    python senz_v3_pinch_sim.py                        # print a few frames
"""

import math
import time

import numpy as np

import senz_multi_io as mio
from senz_io import quat_to_matrix

NUM_IMU = 8
NUM_FORCE = 12   # thumb C0-3, index C4-7, middle C8-11 (three fingertip 2x2 pads)

_PINCH_PERIOD = 6.0   # seconds: index-pinch (first half) then middle-pinch
_GRAVITY_W = np.array([0.0, 0.0, 1.0])   # world "up"; a still IMU reads +1g on Z
_DEG2RAD = math.pi / 180.0

# Which IMUs belong to which finger (indices into the 8-IMU set, dorsum = 7).
_THUMB = (4, 5, 6)     # thmb-base, thmb-tip, thmb-meta
_INDEX = (0, 1)        # idx-prox, idx-dist
_MIDDLE = (2, 3)       # mid-prox, mid-dist
_DORSUM = 7

# Per-IMU flexion axis + peak curl amplitude (deg). Index/middle curl about X
# (finger flexion); the thumb opposes about a tilted horizontal axis so it swings
# toward the fingers. Axes are horizontal (z=0) so the motion stays observable to
# a 6-axis IMU (rotation about the vertical/gravity axis is invisible to fusion).
_IMU_AXIS_AMP = {
    0: ((1.0, 0.0, 0.0), 58.0),   # idx-prox
    1: ((1.0, 0.0, 0.0), 50.0),   # idx-dist
    2: ((1.0, 0.0, 0.0), 58.0),   # mid-prox
    3: ((1.0, 0.0, 0.0), 50.0),   # mid-dist
    4: ((1.0, 0.5, 0.0), 42.0),   # thmb-base
    5: ((1.0, 0.5, 0.0), 34.0),   # thmb-tip
    6: ((0.8, 0.6, 0.0), 30.0),   # thmb-meta (base MCP)
    7: ((1.0, 0.0, 0.0), 16.0),   # hand-dorsum (gentle pitch = slight wrist flex)
}


def _quat_about_axis(axis, angle):
    """Unit quaternion (w,x,y,z) for a rotation of `angle` rad about `axis`."""
    n = math.sqrt(sum(a * a for a in axis)) or 1.0
    ax, ay, az = (a / n for a in axis)
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), ax * s, ay * s, az * s)


def _quat_mul(a, b):
    """Hamilton product of two (w,x,y,z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _raised(x):
    """Smooth 0..1 raised-cosine bump over the half-open window x in [0,1)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0, 0.0
    env = 0.5 - 0.5 * math.cos(2.0 * math.pi * x)         # 0 at ends, 1 mid
    denv = math.pi * math.sin(2.0 * math.pi * x)          # d(env)/dx
    return env, denv


def _finger_envelope(k, t):
    """Curl envelope (0..1) and its time-derivative for IMU k at time t.

    First half of the cycle = index-thumb pinch (index + thumb curl); second half
    = middle-thumb pinch (middle + thumb curl). The thumb curls in BOTH halves.
    """
    u = (t % _PINCH_PERIOD) / _PINCH_PERIOD    # 0..1 within a cycle
    idx_env, idx_d = _raised(u * 2.0)          # active in first half
    mid_env, mid_d = _raised((u - 0.5) * 2.0)  # active in second half
    du = 1.0 / _PINCH_PERIOD
    if k in _INDEX:
        return idx_env, idx_d * 2.0 * du
    if k in _MIDDLE:
        return mid_env, mid_d * 2.0 * du
    if k in _THUMB:
        # Thumb opposes for whichever pinch is active (take the stronger one).
        if idx_env >= mid_env:
            return idx_env, idx_d * 2.0 * du
        return mid_env, mid_d * 2.0 * du
    return 0.0, 0.0                            # dorsum handled separately


class SimV3PinchSource:
    """No-hardware source with the ``open_multi_source`` interface (pinch build)."""

    def __init__(self, rate=200, realtime=True):
        cols = ["t_us", "bno_cal", "bno_qw", "bno_qx", "bno_qy", "bno_qz"]
        for i in range(NUM_IMU):
            cols += [f"imu{i}_ok"] + [f"imu{i}_{a}{x}" for a in "ag" for x in "xyz"]
        cols += [f"force{m}" for m in range(NUM_FORCE)]
        self.schema = mio.Schema(cols, NUM_IMU, NUM_FORCE, rate)
        self._rate = rate
        self._realtime = realtime
        self._n = 0

    def _imu_theta(self, k, t):
        """Curl angle (rad) and its rate (rad/s) for IMU k at time t."""
        amp = math.radians(_IMU_AXIS_AMP[k][1])
        if k == _DORSUM:
            # gentle standalone pitch, not tied to the pinch envelope
            theta = amp * math.sin(0.6 * t)
            return theta, amp * 0.6 * math.cos(0.6 * t)
        env, denv = _finger_envelope(k, t)
        return amp * env, amp * denv

    def imu_quat(self, k, t):
        """The scripted orientation quaternion of IMU k at time t (for tests)."""
        axis = _IMU_AXIS_AMP[k][0]
        theta, _ = self._imu_theta(k, t)
        return _quat_about_axis(axis, theta)

    def _imu_accel_gyro(self, k, t):
        axis = _IMU_AXIS_AMP[k][0]
        theta, dtheta = self._imu_theta(k, t)
        q = _quat_about_axis(axis, theta)
        R = quat_to_matrix(*q)                              # body -> world
        n = math.sqrt(sum(a * a for a in axis)) or 1.0
        axis_u = np.array(axis) / n
        gyro = dtheta * axis_u                              # rad/s, body frame
        accel = R.T @ _GRAVITY_W                            # g, body frame
        return accel, gyro / _DEG2RAD                       # accel(g), gyro(dps)

    def _force_counts(self, t):
        """12 raw ADC counts (0..4095). The two fingers of the ACTIVE pinch press
        against the thumb; the idle finger stays near baseline. Higher = harder."""
        u = (t % _PINCH_PERIOD) / _PINCH_PERIOD
        idx_env, _ = _raised(u * 2.0)
        mid_env, _ = _raised((u - 0.5) * 2.0)
        thumb_press = max(idx_env, mid_env)                 # thumb feels both pinches
        press_by_ch = (
            [thumb_press] * 4 +                             # C0-3  thumb
            [idx_env] * 4 +                                 # C4-7  index
            [mid_env] * 4                                   # C8-11 middle
        )
        out = []
        for ch, p in enumerate(press_by_ch):
            wobble = 0.10 * math.sin(1.9 * t + ch * 1.3)    # 2x2 pads not identical
            press = max(0.0, min(1.0, p + wobble * (p > 0.05)))
            out.append(int(250 + 2600 * press))             # ~250 open .. ~2850 hard
        return out

    def read(self):
        t = self._n / self._rate
        self._n += 1
        f = {"t_us": int(t * 1e6), "bno_cal": 0xFF}
        # Wrist: gentle rock about X and Y, emitted as a fused quaternion.
        qw = _quat_about_axis((1.0, 0.0, 0.0), math.radians(14 * math.sin(t * 0.5)))
        qy = _quat_about_axis((0.0, 1.0, 0.0), math.radians(9 * math.sin(t * 0.35)))
        w_q = _quat_mul(qw, qy)
        f["bno_qw"], f["bno_qx"], f["bno_qy"], f["bno_qz"] = w_q
        for k in range(NUM_IMU):
            accel, gyro = self._imu_accel_gyro(k, t)
            f[f"imu{k}_ok"] = 1
            f[f"imu{k}_ax"] = int(accel[0] * mio.ACC_LSB_PER_G)
            f[f"imu{k}_ay"] = int(accel[1] * mio.ACC_LSB_PER_G)
            f[f"imu{k}_az"] = int(accel[2] * mio.ACC_LSB_PER_G)
            f[f"imu{k}_gx"] = int(gyro[0] * mio.GYR_LSB_PER_DPS)
            f[f"imu{k}_gy"] = int(gyro[1] * mio.GYR_LSB_PER_DPS)
            f[f"imu{k}_gz"] = int(gyro[2] * mio.GYR_LSB_PER_DPS)
        counts = self._force_counts(t)
        for m in range(NUM_FORCE):
            f[f"force{m}"] = counts[m]
        if self._realtime:
            time.sleep(1.0 / self._rate)
        return f

    def send(self, cmd):
        pass

    def close(self):
        pass


def main():
    """Standalone sanity check: print the header + a few synthetic frames."""
    src = SimV3PinchSource(realtime=False)
    print("# " + ",".join(src.schema.columns))
    for _ in range(5):
        f = src.read()
        print(",".join(str(f[c]) for c in src.schema.columns))
    src.close()


if __name__ == "__main__":
    main()

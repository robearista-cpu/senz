#!/usr/bin/env python3
"""
senz_v3_tactile_sim.py  --  base simulator for Hardware Sprint v3 (TACTILE)
==========================================================================
No-hardware source for the tactile-first v3 build: **1 dorsum IMU** (gross hand
movement) + BNO055 wrist + a **15-taxel velostat force array** (the deliverable).
It matches the ``senz_glove_v3_tactile`` firmware schema exactly (nimu=1,
nforce=15), so ``senz_v3_qt.py --simulate`` (default ``--sim tactile``) exercises
the real Madgwick + force pipeline with nothing plugged in.

Like ``senz_v3_sim`` it scripts a real orientation animation for the dorsum and
emits the **raw accel + gyro that would produce it** (so host fusion reconstructs
the pose), and it drives the force channels with a **lifelike grasp cycle** so the
tactile overlay looks alive: fingertips engage first, the palm encloses a beat
later, then release -- looping.

Use:
    from senz_v3_tactile_sim import SimV3TactileSource   # viz --simulate uses this
    python senz_v3_tactile_sim.py                          # print a few frames
"""

import math
import time

import numpy as np

import senz_multi_io as mio
from senz_io import quat_to_matrix

NUM_IMU = 1
NUM_FORCE = 15   # C0..C11 fingers + C12..C14 palm (center/thenar/hypothenar)

# Dorsum IMU animation (axis, amplitude deg, angular freq, phase). A single
# horizontal (z=0, gravity-OBSERVABLE) tilted axis gives a combined pitch/roll
# rock that reads as gross hand movement and round-trips cleanly through 6-axis
# fusion (rotation about the vertical/gravity axis is unobservable, so we avoid it).
_DORSUM_ANIM = ((1.0, 0.5, 0.0), 26.0, 0.5, 0.0)

# Force channel -> grasp group; each group engages at a slightly different phase
# so the array closes like a real hand (fingertips -> palm) instead of all at once.
_FORCE_GROUP = (
    ["thumb"] * 4 +      # C0..C3
    ["index"] * 4 +      # C4..C7
    ["middle"] * 4 +     # C8..C11
    ["palm"] * 3         # C12..C14
)
_GROUP_PHASE = {"thumb": 0.00, "index": 0.08, "middle": 0.05, "palm": 0.22}
_GRASP_PERIOD = 7.0   # seconds per grasp/release cycle

_GRAVITY_W = np.array([0.0, 0.0, 1.0])   # world "up"; a still IMU reads +1g on Z
_DEG2RAD = math.pi / 180.0


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


def _force_counts(t):
    """15 raw ADC counts (0..4095) with a lifelike grasp cycle. Higher = harder
    press (velostat resistance drops -> divider node rises -> ADC rises)."""
    out = []
    for ch in range(NUM_FORCE):
        grp = _FORCE_GROUP[ch]
        # Per-group grasp envelope 0..1, phase-shifted so the hand closes in order.
        phase = _GROUP_PHASE[grp]
        env = 0.5 - 0.5 * math.cos(2.0 * math.pi * (t / _GRASP_PERIOD - phase))
        # A little per-taxel variation so the 2x2 pads aren't identical.
        wobble = 0.12 * math.sin(1.7 * t + ch * 1.3)
        press = max(0.0, min(1.0, env + wobble))
        out.append(int(250 + 2600 * press))   # ~250 open .. ~2850 hard press
    return out


class SimV3TactileSource:
    """No-hardware source with the ``open_multi_source`` interface (tactile build)."""

    def __init__(self, rate=200, realtime=True):
        cols = ["t_us", "bno_cal", "bno_qw", "bno_qx", "bno_qy", "bno_qz"]
        for i in range(NUM_IMU):
            cols += [f"imu{i}_ok"] + [f"imu{i}_{a}{x}" for a in "ag" for x in "xyz"]
        cols += [f"force{m}" for m in range(NUM_FORCE)]
        self.schema = mio.Schema(cols, NUM_IMU, NUM_FORCE, rate)
        self._rate = rate
        self._realtime = realtime
        self._n = 0

    def imu_quat(self, k, t):
        """The scripted orientation quaternion of IMU k at time t (for tests)."""
        axis, amp_deg, w, phase = _DORSUM_ANIM
        theta = math.radians(amp_deg) * (0.5 - 0.5 * math.cos(w * t + phase))
        return _quat_about_axis(axis, theta)

    def _imu_accel_gyro(self, k, t):
        axis, amp_deg, w, phase = _DORSUM_ANIM
        amp = math.radians(amp_deg)
        theta = amp * (0.5 - 0.5 * math.cos(w * t + phase))
        dtheta = amp * 0.5 * w * math.sin(w * t + phase)   # d/dt of theta, rad/s
        q = _quat_about_axis(axis, theta)
        R = quat_to_matrix(*q)                              # body -> world
        n = math.sqrt(sum(a * a for a in axis)) or 1.0
        axis_u = np.array(axis) / n
        gyro = dtheta * axis_u                              # rad/s, body frame
        accel = R.T @ _GRAVITY_W                            # g, body frame
        return accel, gyro / _DEG2RAD                       # accel(g), gyro(dps)

    def read(self):
        t = self._n / self._rate
        self._n += 1
        f = {"t_us": int(t * 1e6), "bno_cal": 0xFF}
        # Wrist: gentle rock about X and Y, emitted as a fused quaternion.
        qw = _quat_about_axis((1.0, 0.0, 0.0), math.radians(16 * math.sin(t * 0.5)))
        qy = _quat_about_axis((0.0, 1.0, 0.0), math.radians(10 * math.sin(t * 0.35)))
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
        counts = _force_counts(t)
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
    src = SimV3TactileSource(realtime=False)
    print("# " + ",".join(src.schema.columns))
    for _ in range(5):
        f = src.read()
        print(",".join(str(f[c]) for c in src.schema.columns))
    src.close()


if __name__ == "__main__":
    main()

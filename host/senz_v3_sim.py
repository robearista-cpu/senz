#!/usr/bin/env python3
"""
senz_v3_sim.py  --  base simulator for the v3 hand (no hardware)
================================================================
Produces a lifelike, deterministic synthetic stream in the exact
``senz_multi_io`` frame shape (dict keyed by column name) for the v3 sensor set:
7 finger IMUs (index 2, middle 2, thumb 3) + BNO055 wrist + 12 velostat taxels.

Unlike ``senz_multi_io.SimMultiSource`` (which just wiggles the gyro), this
scripts a real per-IMU orientation animation and then emits the **raw accel +
gyro that would produce it**, so the visualizer's real Madgwick fusion path
reconstructs the intended pose. That makes ``--simulate`` both pretty *and* an
end-to-end check of the fusion pipeline.

Each finger IMU rotates about a fixed axis by a smooth curl angle
``theta(t) = amp*(0.5 - 0.5*cos(w*t + phase))``:
  - gyro  (body) = dtheta/dt about that axis          (constant axis => body == world axis)
  - accel (body) = gravity (world +Z) rotated into the sensor frame = R(q)^T @ [0,0,1]
Both are packed to int16 raw counts with the firmware LSB constants.

Use:
    from senz_v3_sim import SimV3Source        # viz --simulate uses this
    python senz_v3_sim.py                        # standalone: print a few frames
"""

import math
import time

import numpy as np

import senz_multi_io as mio
from senz_io import quat_to_matrix

NUM_IMU = 8
NUM_FORCE = 15   # C0..C11 fingers + C12..C14 palm (center/thenar/hypothenar)


def _quat_about_axis(axis, angle):
    """Unit quaternion (w,x,y,z) for a rotation of `angle` rad about `axis`."""
    n = math.sqrt(sum(a * a for a in axis)) or 1.0
    ax, ay, az = (a / n for a in axis)
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), ax * s, ay * s, az * s)


# Per-IMU animation: (rotation axis, amplitude deg, angular freq, phase).
# Index/middle curl about X (finger flexion); the thumb swings on a horizontal
# axis tilted toward Y so it moves in a different plane (opposition-like) while
# staying gravity-OBSERVABLE. Rotation about the vertical (gravity) axis is
# invisible to a 6-axis IMU, so all animation axes are kept horizontal (z=0) --
# otherwise the filter would hold a permanent yaw offset. Distal segments lag.
_IMU_ANIM = [
    ((1.0, 0.0, 0.0), 50.0, 1.2, 0.0),   # 0 idx-prox
    ((1.0, 0.0, 0.0), 42.0, 1.2, 0.5),   # 1 idx-dist
    ((1.0, 0.0, 0.0), 55.0, 1.1, 0.9),   # 2 mid-prox
    ((1.0, 0.0, 0.0), 46.0, 1.1, 1.4),   # 3 mid-dist
    ((1.0, 0.4, 0.0), 34.0, 0.9, 2.0),   # 4 thmb-base
    ((1.0, 0.4, 0.0), 28.0, 0.9, 2.4),   # 5 thmb-tip
    ((0.7, 0.7, 0.0), 24.0, 0.8, 1.6),   # 6 thmb-meta
    ((1.0, 0.0, 0.0), 22.0, 0.6, 0.5),   # 7 hand-dorsum (palm-frame pitch = wrist flex)
]

_GRAVITY_W = np.array([0.0, 0.0, 1.0])   # world "up"; a still IMU reads +1g on Z
_DEG2RAD = math.pi / 180.0


class SimV3Source:
    """No-hardware source with the ``open_multi_source`` interface."""

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
        axis, amp_deg, w, phase = _IMU_ANIM[k]
        theta = math.radians(amp_deg) * (0.5 - 0.5 * math.cos(w * t + phase))
        return _quat_about_axis(axis, theta)

    def _imu_accel_gyro(self, k, t):
        axis, amp_deg, w, phase = _IMU_ANIM[k]
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
        qw = _quat_about_axis((1.0, 0.0, 0.0), math.radians(18 * math.sin(t * 0.5)))
        qy = _quat_about_axis((0.0, 1.0, 0.0), math.radians(12 * math.sin(t * 0.35)))
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
        for m in range(NUM_FORCE):
            press = max(0.0, math.sin(t * 0.9 - m * 0.5))
            f[f"force{m}"] = int(200 + 1900 * press)
        if self._realtime:
            time.sleep(1.0 / self._rate)
        return f

    def send(self, cmd):
        pass

    def close(self):
        pass


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


def main():
    """Standalone sanity check: print the header + a few synthetic frames."""
    src = SimV3Source(realtime=False)
    print("# " + ",".join(src.schema.columns))
    for _ in range(5):
        f = src.read()
        print(",".join(str(f[c]) for c in src.schema.columns))
    src.close()


if __name__ == "__main__":
    main()

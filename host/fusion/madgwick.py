#!/usr/bin/env python3
"""
fusion/madgwick.py  --  6-axis Madgwick AHRS (HLD objective 6, baseline)
========================================================================
A small, dependency-light Madgwick filter for turning a finger IMU's raw
accel + gyro into an orientation quaternion offline. The finger IMUs are
6-axis (no magnetometer -- a mag on a finger segment is dominated by hand/body
ferrous cross-talk), so this is the IMU (accel+gyro) variant: gravity corrects
pitch/roll drift; absolute heading (yaw) is provided by the wrist BNO055, not
by the finger IMUs.

This is intentionally a *baseline* for the fusion research objective. Swappable
with an EKF or a learned model later; the interface (update -> quaternion) is
what the dataset tooling depends on.

Quaternion convention: (w, x, y, z), earth-to-sensor, unit norm.
"""

import math

import numpy as np


class MadgwickAHRS:
    def __init__(self, beta=0.05, q=(1.0, 0.0, 0.0, 0.0)):
        self.beta = beta  # filter gain; higher trusts accel more
        self.q = np.array(q, dtype=float)

    def update(self, gx, gy, gz, ax, ay, az, dt):
        """Advance the filter one step.

        gx,gy,gz : angular rate in rad/s
        ax,ay,az : acceleration (any consistent unit; normalized internally)
        dt       : timestep in seconds
        Returns the updated quaternion (w, x, y, z).
        """
        q = self.q
        qw, qx, qy, qz = q

        # Normalize accelerometer; if it's ~0 (free-fall/garbage) skip correction.
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 1e-9:
            ax, ay, az = ax / norm, ay / norm, az / norm

            # Gradient (objective function = gravity error) for 6-axis.
            f1 = 2 * (qx * qz - qw * qy) - ax
            f2 = 2 * (qw * qx + qy * qz) - ay
            f3 = 2 * (0.5 - qx * qx - qy * qy) - az
            j11, j12, j13, j14 = -2 * qy, 2 * qz, -2 * qw, 2 * qx
            j21, j22, j23, j24 = 2 * qx, 2 * qw, 2 * qz, 2 * qy
            j31, j32, j33, j34 = 0.0, -4 * qx, -4 * qy, 0.0
            grad = np.array([
                j11 * f1 + j21 * f2 + j31 * f3,
                j12 * f1 + j22 * f2 + j32 * f3,
                j13 * f1 + j23 * f2 + j33 * f3,
                j14 * f1 + j24 * f2 + j34 * f3,
            ])
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-9:
                grad = grad / gnorm
        else:
            grad = np.zeros(4)

        # Rate of change from gyro.
        qdot = 0.5 * np.array([
            -qx * gx - qy * gy - qz * gz,
            qw * gx + qy * gz - qz * gy,
            qw * gy - qx * gz + qz * gx,
            qw * gz + qx * gy - qy * gx,
        ])

        qdot = qdot - self.beta * grad
        q = q + qdot * dt
        q = q / np.linalg.norm(q)
        self.q = q
        return tuple(q)


def fuse_finger_imu(accel_g, gyro_dps, t_us, beta=0.05):
    """Run Madgwick over one finger IMU's time series.

    accel_g  : (N,3) acceleration in g
    gyro_dps : (N,3) angular rate in deg/s
    t_us     : (N,) timestamps in microseconds
    Returns  : (N,4) quaternions (w, x, y, z).
    """
    accel_g = np.asarray(accel_g, dtype=float)
    gyro_dps = np.asarray(gyro_dps, dtype=float)
    t_us = np.asarray(t_us, dtype=float)
    n = len(t_us)
    out = np.zeros((n, 4))
    ahrs = MadgwickAHRS(beta=beta)
    deg2rad = math.pi / 180.0
    prev_t = t_us[0] if n else 0.0
    for i in range(n):
        dt = max(1e-4, (t_us[i] - prev_t) * 1e-6)
        prev_t = t_us[i]
        gx, gy, gz = (gyro_dps[i] * deg2rad)
        ax, ay, az = accel_g[i]
        out[i] = ahrs.update(gx, gy, gz, ax, ay, az, dt)
    return out


if __name__ == "__main__":
    # Sanity: a still IMU reading 1 g on +Z should converge to ~identity and stay.
    n = 500
    accel = np.tile([0.0, 0.0, 1.0], (n, 1))
    gyro = np.zeros((n, 3))
    t = np.arange(n) * 5000.0  # 200 Hz in microseconds
    q = fuse_finger_imu(accel, gyro, t)
    print("final quaternion:", np.round(q[-1], 4))
    print("norm(q) =", round(float(np.linalg.norm(q[-1])), 6))

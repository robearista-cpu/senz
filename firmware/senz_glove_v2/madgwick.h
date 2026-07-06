/*
 * madgwick.h
 * ----------
 * 6-axis Madgwick AHRS (accel + gyro, no magnetometer) for the per-finger
 * MPU-6500 IMUs. One MadgwickImu instance per finger holds that finger's
 * orientation quaternion and advances it each 200 Hz sample.
 *
 * This is a direct C++ port of the project's reference filter,
 * host/fusion/madgwick.py -- SAME gradient/Jacobian formulation, SAME
 * (w, x, y, z) earth-to-sensor convention -- so the firmware's real-time
 * quaternions agree numerically with the host's offline fusion of the same
 * raw stream. Keep the two in sync if either changes.
 *
 * 6-axis means gravity corrects pitch/roll drift but heading (yaw) is not
 * anchored -- absolute yaw comes from the BNO055 wrist reference, and per-finger
 * yaw drift is an accepted limitation of a mag-free finger IMU (see HLD).
 *
 * Units: gyro in rad/s, accel in any consistent unit (normalized internally),
 * dt in seconds. beta is the filter gain (HLD: start at 0.1).
 */
#pragma once

#include <math.h>

class MadgwickImu {
 public:
  explicit MadgwickImu(float beta = 0.1f)
      : beta_(beta), q0_(1.0f), q1_(0.0f), q2_(0.0f), q3_(0.0f) {}

  void setBeta(float beta) { beta_ = beta; }
  void reset() { q0_ = 1.0f; q1_ = q2_ = q3_ = 0.0f; }

  // Advance the filter one step. Mirrors MadgwickAHRS.update() in the host ref.
  void update(float gx, float gy, float gz, float ax, float ay, float az,
              float dt) {
    const float qw = q0_, qx = q1_, qy = q2_, qz = q3_;

    // Gradient of the gravity-error objective (only if accel is usable).
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float norm = sqrtf(ax * ax + ay * ay + az * az);
    if (norm > 1e-9f) {
      ax /= norm;
      ay /= norm;
      az /= norm;

      // f = R^T(q) * [0,0,1] - a_measured   (gravity residual)
      const float f1 = 2.0f * (qx * qz - qw * qy) - ax;
      const float f2 = 2.0f * (qw * qx + qy * qz) - ay;
      const float f3 = 2.0f * (0.5f - qx * qx - qy * qy) - az;

      // Jacobian J of f w.r.t. q, then grad = J^T * f.
      const float j11 = -2.0f * qy, j12 = 2.0f * qz, j13 = -2.0f * qw, j14 = 2.0f * qx;
      const float j21 = 2.0f * qx, j22 = 2.0f * qw, j23 = 2.0f * qz, j24 = 2.0f * qy;
      const float j31 = 0.0f, j32 = -4.0f * qx, j33 = -4.0f * qy, j34 = 0.0f;
      g0 = j11 * f1 + j21 * f2 + j31 * f3;
      g1 = j12 * f1 + j22 * f2 + j32 * f3;
      g2 = j13 * f1 + j23 * f2 + j33 * f3;
      g3 = j14 * f1 + j24 * f2 + j34 * f3;

      const float gn = sqrtf(g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3);
      if (gn > 1e-9f) {
        g0 /= gn;
        g1 /= gn;
        g2 /= gn;
        g3 /= gn;
      }
    }

    // Quaternion rate of change from the gyro.
    float qd0 = 0.5f * (-qx * gx - qy * gy - qz * gz);
    float qd1 = 0.5f * (qw * gx + qy * gz - qz * gy);
    float qd2 = 0.5f * (qw * gy - qx * gz + qz * gx);
    float qd3 = 0.5f * (qw * gz + qx * gy - qy * gx);

    // Apply the gradient-descent feedback step.
    qd0 -= beta_ * g0;
    qd1 -= beta_ * g1;
    qd2 -= beta_ * g2;
    qd3 -= beta_ * g3;

    // Integrate and renormalize.
    float nw = qw + qd0 * dt;
    float nx = qx + qd1 * dt;
    float ny = qy + qd2 * dt;
    float nz = qz + qd3 * dt;
    const float qn = sqrtf(nw * nw + nx * nx + ny * ny + nz * nz);
    if (qn > 1e-9f) {
      q0_ = nw / qn;
      q1_ = nx / qn;
      q2_ = ny / qn;
      q3_ = nz / qn;
    }
  }

  // Current orientation as (w, x, y, z).
  void quat(float q[4]) const {
    q[0] = q0_;
    q[1] = q1_;
    q[2] = q2_;
    q[3] = q3_;
  }

 private:
  float beta_;
  float q0_, q1_, q2_, q3_;
};

// --- Quaternion helpers (Hamilton product, (w,x,y,z)) ----------------------
// Used to express a finger quaternion relative to the wrist frame:
//   q_rel = conj(q_wrist) * q_finger    (so q_wrist * q_rel == q_finger)
static inline void quatConj(const float q[4], float out[4]) {
  out[0] = q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = -q[3];
}

static inline void quatMul(const float a[4], const float b[4], float out[4]) {
  out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

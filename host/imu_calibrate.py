#!/usr/bin/env python3
"""
imu_calibrate.py  --  Multi-IMU calibration application (HLD objective 2)
=========================================================================
Standalone host tool that calibrates the finger-IMU array and persists the
result to the glove's flash, so the array starts with reduced drift on every
boot.

Workflow:
  1. Discovery   -- read the firmware header; report which IMUs responded.
  2. Capture     -- put the firmware in raw mode (XR) and collect N samples
                    with the glove held still.
  3. Generate    -- per IMU: gyro bias = mean(raw gyro at rest). Accel offset
                    is left at 0 for v1 (a 6-point tumble belongs in a later
                    revision; the field is reserved end-to-end).
  4. Export      -- write a calibration profile JSON for the record.
  5. Push + save -- send O,... offsets per IMU, then S to store them in NVS.
  6. Validate    -- back in normal mode (XN), confirm the residual gyro rate at
                    rest is near zero.

Usage:
    python imu_calibrate.py --port COM7
    python imu_calibrate.py --simulate          # dry run, no hardware
    python imu_calibrate.py --port COM7 --validate-only
"""

import argparse
import json
import time

import senz_multi_io as io


def collect(src, n, settle=0.3):
    """Grab n parsed frames, skipping a short settle window first."""
    t_end = time.time() + settle
    while time.time() < t_end:
        src.read()
    frames = []
    while len(frames) < n:
        f = src.read()
        if f is not None:
            frames.append(f)
    return frames


def gyro_bias(frames, k):
    """Mean raw gyro (LSB) for IMU k across frames -> (gx, gy, gz)."""
    n = len(frames)
    return tuple(
        round(sum(f[f"imu{k}_g{ax}"] for f in frames) / n) for ax in "xyz"
    )


def residual_rate(frames, k):
    """Mean |gyro| in deg/s for IMU k -- should be ~0 after calibration."""
    mags = []
    for f in frames:
        gx, gy, gz = io.imu_gyro_dps(f, k)
        mags.append((gx * gx + gy * gy + gz * gz) ** 0.5)
    return sum(mags) / len(mags)


def present_imus(src, frames):
    return [k for k in src.schema.imu_indices() if any(f[f"imu{k}_ok"] for f in frames)]


def main():
    ap = argparse.ArgumentParser(description="senz multi-IMU calibration tool")
    ap.add_argument("--port", help="serial port (e.g. COM7, /dev/ttyACM0)")
    ap.add_argument("--simulate", action="store_true", help="no hardware")
    ap.add_argument("--samples", type=int, default=400, help="samples per stage")
    ap.add_argument("--out", default="imu_calibration.json", help="profile path")
    ap.add_argument("--validate-only", action="store_true",
                    help="skip calibration, just report residual rest rate")
    args = ap.parse_args()

    src = io.open_multi_source(port=args.port, simulate=args.simulate)
    print(f"Discovered {src.schema.nimu} IMU slots, {src.schema.nforce} force "
          f"regions @ {src.schema.rate} Hz")

    print("Hold the glove COMPLETELY STILL for the next few seconds...")
    src.send("XR")  # raw mode: measure true gyro, not already-corrected gyro
    time.sleep(0.2)
    rest = collect(src, args.samples)
    imus = present_imus(src, rest)
    print(f"Responding IMUs: {imus or '(none)'}")
    if not imus:
        print("No IMUs responding -- check wiring / chip-select expander.")
        src.close()
        return

    if args.validate_only:
        for k in imus:
            print(f"  imu{k}: residual rest rate {residual_rate(rest, k):.2f} deg/s")
        src.close()
        return

    # --- Generate + push offsets ---
    profile = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "imus": {}}
    for k in imus:
        gx, gy, gz = gyro_bias(rest, k)
        ax, ay, az = 0, 0, 0  # accel offset reserved for a later 6-point tumble
        profile["imus"][str(k)] = {
            "gyro_bias_lsb": [gx, gy, gz],
            "accel_offset_lsb": [ax, ay, az],
            "rest_rate_before_dps": round(residual_rate(rest, k), 3),
        }
        src.send(f"O,{k},{gx},{gy},{gz},{ax},{ay},{az}")
        print(f"  imu{k}: gyro bias = ({gx}, {gy}, {gz}) LSB -> pushed")
        time.sleep(0.05)

    src.send("S")  # persist to NVS
    time.sleep(0.2)

    with open(args.out, "w") as fh:
        json.dump(profile, fh, indent=2)
    print(f"Saved profile -> {args.out}; offsets written to glove flash.")

    # --- Validate ---
    src.send("XN")  # normal (offset-corrected) mode
    time.sleep(0.2)
    check = collect(src, args.samples)
    print("Validation (residual rest rate, lower is better):")
    for k in imus:
        before = profile["imus"][str(k)]["rest_rate_before_dps"]
        after = residual_rate(check, k)
        print(f"  imu{k}: {before:.2f} -> {after:.2f} deg/s")

    src.close()


if __name__ == "__main__":
    main()

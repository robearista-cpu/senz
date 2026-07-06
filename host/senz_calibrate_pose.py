#!/usr/bin/env python3
"""
senz_calibrate_pose.py
======================
Zero-pose capture for the v2 hand visualizer (HLD v2, deliverable #13).

The per-finger MPU-6500s are mounted at whatever orientation the glove sewing
put them, so at a real flat hand the fused finger quaternions are NOT identity.
This tool captures each finger's quaternion while you hold a known reference
pose (hand flat, palm down, fingers straight) and writes them to
``pose_offsets.json``. ``senz_visualizer.py`` auto-loads that file and divides it
out, so a flat hand then renders as a flat hand.

What is stored is exactly what the visualizer expects:
    {"offsets": [[w,x,y,z], ... 10 ...], "wrist": [w,x,y,z], ...}
The visualizer applies each finger's inverse:  R_shown = Rw @ R(q_raw) @ R(q0)^-1,
so at the reference pose (q_raw == q0) the fingers sit straight along the palm.

Run BEFORE the visualizer, once per session (re-donning the glove shifts the
sensors):
    python senz_calibrate_pose.py --port COM5        # wired
    python senz_calibrate_pose.py --simulate --yes    # no hardware, no prompt
"""

import argparse
import json
import time

import numpy as np

import senz_parser


def average_quats(samples):
    """Robust mean of unit quaternions held nearly still (sign-aligned)."""
    acc = np.zeros(4)
    ref = None
    for q in samples:
        v = np.asarray(q, dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-9:
            continue
        v = v / n
        if ref is None:
            ref = v
        if np.dot(v, ref) < 0.0:  # q and -q are the same rotation; align hemis.
            v = -v
        acc += v
    n = np.linalg.norm(acc)
    return (acc / n).tolist() if n > 1e-9 else [1.0, 0.0, 0.0, 0.0]


def capture(src, n_samples, settle):
    """Collect n_samples frames after a short settle, return per-sensor lists."""
    if settle > 0:
        print(f"Hold the pose... capturing in {settle:.0f}s")
        time.sleep(settle)
    print(f"Capturing {n_samples} frames — keep still.")
    finger_samples = [[] for _ in range(senz_parser.NUM_IMU)]
    wrist_samples = []
    got = 0
    deadline = time.time() + max(5.0, n_samples * 0.05)  # generous timeout
    while got < n_samples and time.time() < deadline:
        frame = src.read(block=True, timeout=1.0)
        if frame is None:
            continue
        wrist_samples.append(frame.bno)
        for i in range(senz_parser.NUM_IMU):
            finger_samples[i].append(frame.mpu[i])
        got += 1
    if got == 0:
        raise RuntimeError("no frames captured — check the port/connection")
    if got < n_samples:
        print(f"# only got {got}/{n_samples} frames before timeout; using those")
    return finger_samples, wrist_samples, got


def main():
    ap = argparse.ArgumentParser(description="senz v2 zero-pose calibration")
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--simulate", action="store_true", help="no hardware")
    ap.add_argument("--baud", type=int, default=senz_parser.BAUD)
    ap.add_argument("--out", default="pose_offsets.json", help="output file")
    ap.add_argument("--samples", type=int, default=200, help="frames to average")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds to hold before capture starts")
    ap.add_argument("--yes", action="store_true",
                    help="skip the Enter prompt (auto-capture; for scripting)")
    args = ap.parse_args()

    print("=== senz zero-pose calibration ===")
    print("Hold your hand FLAT, palm DOWN, fingers STRAIGHT and together.")
    if not args.yes:
        input("Press Enter when you're holding the pose... ")

    src = senz_parser.open_frame_source(args.port, simulate=args.simulate,
                                        baud=args.baud)
    try:
        finger_samples, wrist_samples, got = capture(src, args.samples,
                                                     0.0 if args.yes else args.settle)
    finally:
        src.close()

    offsets = [average_quats(s) for s in finger_samples]
    wrist = average_quats(wrist_samples)

    data = {
        "note": "zero-pose finger quaternions (relative-to-wrist); "
                "senz_visualizer.py divides these out so a flat hand is flat",
        "num_imu": senz_parser.NUM_IMU,
        "samples": got,
        "offsets": offsets,
        "wrist": wrist,
    }
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=2)

    print(f"\nSaved {senz_parser.NUM_IMU} finger offsets to {args.out} "
          f"({got} frames averaged).")
    print("Run the visualizer now — it loads this automatically:")
    src_arg = "--simulate" if args.simulate else f"--port {args.port or 'COMx'}"
    print(f"    python senz_visualizer.py {src_arg}")


if __name__ == "__main__":
    main()

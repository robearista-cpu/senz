#!/usr/bin/env python3
"""
record.py
=========
Capture the glove's sensor stream to a labeled CSV for machine learning.

Each row is one timestamped frame:
    t_s, label, f0[,f1...], qw, qx, qy, qz, cal

  t_s   : seconds since recording started (host clock)
  label : the gesture/class you're recording (from --label)
  f0..  : raw 12-bit finger ADC value(s)
  qw..  : orientation quaternion
  cal   : BNO055 calibration byte

Records over USB serial (lossless full rate). For an ML dataset, USB is the
right choice -- BLE can drop samples. Run one session per gesture:

    python record.py --port COM5 --label fist
    python record.py --port COM5 --label open
    python record.py --port COM5 --label point

Press Ctrl+C to stop; the file is saved to data/<label>_<timestamp>.csv.

Dependencies: pip install pyserial numpy
"""

import argparse
import csv
import datetime
import os
import time

import senz_io


def main():
    ap = argparse.ArgumentParser(description="Record glove data for ML")
    ap.add_argument("--port", required=True, help="Serial port, e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", default="unlabeled", help="Gesture/class name")
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="Auto-stop after N seconds (0 = until Ctrl+C)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.outdir, f"{args.label}_{stamp}.csv")

    header = (
        ["t_s", "label"]
        + [f"f{i}" for i in range(senz_io.NUM_FINGERS)]
        + ["qw", "qx", "qy", "qz", "cal"]
    )

    src = senz_io.SerialSource(args.port, args.baud)
    n = 0
    t0 = time.monotonic()
    print(f"Recording '{args.label}' -> {path}")
    print("Perform the gesture now. Ctrl+C to stop.")
    try:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            while True:
                frame = src.read()
                if frame is None:
                    continue
                fingers, q, cal = frame
                t = time.monotonic() - t0
                w.writerow(
                    [f"{t:.4f}", args.label, *fingers]
                    + [f"{c:.4f}" for c in q]
                    + [cal if cal is not None else ""]
                )
                n += 1
                if args.seconds and t >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        dur = time.monotonic() - t0
        hz = n / dur if dur > 0 else 0
        print(f"\nSaved {n} samples ({dur:.1f}s, ~{hz:.0f} Hz) -> {path}")


if __name__ == "__main__":
    main()

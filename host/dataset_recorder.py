#!/usr/bin/env python3
"""
dataset_recorder.py  --  Unified multi-modal recorder (HLD objective 5)
=======================================================================
Records one synchronized, timestamped dataset combining every modality:

  - BNO055 wrist orientation        (serial)
  - finger IMU array accel + gyro   (serial)
  - Velostat force regions          (serial, processed via force_pipeline)
  - MediaPipe hand landmarks        (camera, optional)

The serial stream is the timing master (it carries the firmware ``t_us``). Each
recorded row is one serial frame, tagged with a host wall-clock time and joined
to the most recent camera landmark frame (nearest-previous sample-and-hold).
This is deliberately simple and lossless on the sensor side; precise offline
resampling/alignment is the job of dataset_prep.py (objective 7).

Outputs, under data/<session>/:
  - frames.csv   one row per serial frame, all modalities flattened
  - meta.json    session metadata (schema, rate, columns, counts, timing)

Usage:
    python dataset_recorder.py --port COM7 --camera 0 --label grasp_cup
    python dataset_recorder.py --simulate --duration 5     # no hardware/camera
"""

import argparse
import csv
import json
import os
import time

import senz_multi_io as io
from force_pipeline import ForceArray, process_frame


def _session_dir(root, label):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{label}" if label else stamp
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    return path, name


def run(port=None, camera=None, simulate=False, label="", duration=0.0,
        root="data", nimu=3, nforce=5):
    src = io.open_multi_source(port=port, simulate=simulate, nimu=nimu, nforce=nforce)
    schema = src.schema
    forces = ForceArray(schema.nforce, rate=schema.rate or 200)

    # Optional camera modality.
    tracker = None
    cam_cols = []
    if camera is not None:
        from camera_tracker import HandTracker

        tracker = HandTracker(camera=camera).start()
        cam_cols = HandTracker.landmark_columns()
        time.sleep(1.0)  # let the camera warm up

    # Build the output column list: serial schema + processed force + camera.
    force_proc_cols = []
    for m in range(schema.nforce):
        force_proc_cols += [f"force{m}_grip", f"force{m}_rel", f"force{m}_contact"]
    columns = ["t_host"] + list(schema.columns) + force_proc_cols + \
        [f"cam_{c}" for c in cam_cols]

    sess_dir, sess_name = _session_dir(root, label)
    csv_path = os.path.join(sess_dir, "frames.csv")

    print(f"Recording -> {csv_path}")
    print(f"  modalities: serial(imu={schema.nimu}, force={schema.nforce})"
          f"{', camera' if tracker else ''}")
    print("  Ctrl-C to stop." if not duration else f"  for {duration:.0f}s.")

    n_rows = 0
    t_start = time.time()
    t_end = t_start + duration if duration else None
    try:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            while True:
                frame = src.read()
                if frame is None:
                    if t_end and time.time() > t_end:
                        break
                    continue
                row = {"t_host": time.time()}
                row.update(frame)

                # Processed force features.
                fp = process_frame(frame, forces)
                for m, res in enumerate(fp):
                    row[f"force{m}_grip"] = round(res["grip"], 5)
                    row[f"force{m}_rel"] = round(res["relative_grip"], 5)
                    row[f"force{m}_contact"] = int(res["contact"])

                # Camera sample-and-hold.
                if tracker:
                    cam = tracker.get_latest()
                    if cam:
                        for c in cam_cols:
                            if c in cam:
                                row[f"cam_{c}"] = cam[c]

                writer.writerow(row)
                n_rows += 1
                if n_rows % 200 == 0:
                    print(f"  {n_rows} frames...", end="\r")
                if t_end and time.time() > t_end:
                    break
    except KeyboardInterrupt:
        print("\n  stopped by user.")
    finally:
        if tracker:
            tracker.stop()
        src.close()

    elapsed = time.time() - t_start
    meta = {
        "session": sess_name,
        "label": label,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "simulate" if (simulate or not port) else port,
        "camera": camera,
        "schema": {
            "nimu": schema.nimu,
            "nforce": schema.nforce,
            "rate_hz": schema.rate,
            "serial_columns": list(schema.columns),
        },
        "columns": columns,
        "rows": n_rows,
        "elapsed_s": round(elapsed, 2),
        "effective_hz": round(n_rows / elapsed, 1) if elapsed > 0 else 0,
    }
    with open(os.path.join(sess_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nDone: {n_rows} rows in {elapsed:.1f}s "
          f"(~{meta['effective_hz']} Hz) -> {sess_dir}")
    return sess_dir


def main():
    ap = argparse.ArgumentParser(description="senz unified dataset recorder")
    ap.add_argument("--port", help="glove serial port (omit for --simulate)")
    ap.add_argument("--camera", type=int, default=None,
                    help="camera index to also record landmarks (omit to skip)")
    ap.add_argument("--simulate", action="store_true", help="no hardware")
    ap.add_argument("--label", default="", help="session label, e.g. grasp_cup")
    ap.add_argument("--duration", type=float, default=0, help="seconds (0 = until Ctrl-C)")
    ap.add_argument("--root", default="data", help="dataset root directory")
    args = ap.parse_args()
    run(port=args.port, camera=args.camera, simulate=args.simulate,
        label=args.label, duration=args.duration, root=args.root)


if __name__ == "__main__":
    main()

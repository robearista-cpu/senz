# senz host software

Host-side Python for the senz glove: live 3D hand visualization, calibration,
dataset recording, and the sensor-stream I/O layers.

## Install

```
pip install -r requirements.txt
```

Not everything is needed for every task. The essentials for the visualizer are
`numpy`, `pyserial`, `pyqtgraph`, `PyOpenGL`, `PyQt5`; Bluetooth adds `bleak`;
the legacy viz adds `matplotlib`. Dataset/camera tooling pulls in `pandas`,
`mediapipe`, and `opencv-python` (heavy, optional).

---

## Running the visualization

The visualizer is **`live_hand_qt.py`** (GPU-accelerated, 60–120 fps). It runs
three ways — no hardware, wired USB, or Bluetooth — via the transport layer in
`senz_io.py`. Pick one:

### Without any hardware (simulate)
Synthetic data, so you can develop and check the 3D view with no glove attached:
```
python live_hand_qt.py --simulate
```

### Wired (USB serial)
Plug the glove into USB, then pass its serial port:
```
python live_hand_qt.py --port COM5
```
Add `--baud 115200` to override the default baud rate. Find your port with:
```
python -m serial.tools.list_ports -v
```
On Linux/macOS the port looks like `/dev/ttyACM0` or `/dev/cu.usbmodemXXXX`.

### Bluetooth (BLE)
Wireless, no cable. The firmware must be built with BLE enabled and advertising:
```
python live_hand_qt.py --ble
```
Uses the Nordic UART Service via `bleak`; the default device name is
`senz-glove` (override with `--name <device-name>`). Requires `pip install bleak`.

> Fallback viz: if `pyqtgraph`/OpenGL won't install, `live_hand_viz.py`
> (matplotlib, ~20–40 fps) takes the same `--simulate` / `--port` / `--ble` flags.

### Controls
- **Drag** to orbit, **scroll** to zoom.
- **Reset Level (tare)** — hold your hand flat, click to zero the baseline.
- **RotX/Y/Z < A?** — cycle which quaternion axis (x/y/z) drives each model
  rotation axis (fixes mirroring/swaps). **RotX/Y/Z +/-** inverts an axis.
- **Invert finger** — flip finger open/closed direction.
- **Set Open / Set Fist** — calibrate the finger flexion range.

---

## Hardware Sprint v2 stream (binary frame)

The v2 firmware (`firmware/senz_glove_v2/`, 10× MPU-6500 + BNO055) streams a
fixed **180-byte binary frame** instead of CSV. Its host reader is
**`senz_parser.py`** and its 3D viewer is **`senz_visualizer.py`**.

**Validate the stream** (`senz_parser.py`) — wired or hardware-free:
```
python senz_parser.py --simulate        # no hardware: synthetic frames + live Hz
python senz_parser.py --port COM5        # wired: prints frame rate + drop count
```

**Multi-sensor 3D hand** (`senz_visualizer.py`) — VPython skeleton, one bone per
IMU (11 total: wrist + 10 finger segments):
```
python senz_visualizer.py --simulate     # no hardware (wrist rocks)
python senz_visualizer.py --port COM5     # wired USB serial
```
(v2 serial is 921600 baud by default; override with `--baud`. Needs `vpython`.)

> **Fingers currently render flat.** The per-finger Madgwick fusion is not in the
> firmware yet (supervised, max-effort), so the 10 finger quaternions arrive as
> identity — the wrist orientation is live and moves the whole hand, but the
> fingers don't articulate until fusion lands. The renderer + forward kinematics
> are already correct and length-preserving; only the quaternion source is
> pending. **v2 Bluetooth** and **zero-pose calibration** (`pose_offsets.json`,
> auto-loaded if present) are likewise deferred. For a fully articulating hand
> today, use `live_hand_qt.py` above (single-IMU pipeline).

---

## Files

### Visualization & I/O
- **`live_hand_qt.py`** — **recommended** GPU 3D hand viz (pyqtgraph/OpenGL).
- `live_hand_viz.py` — legacy matplotlib viz (CPU fallback).
- `senz_io.py` — shared serial / BLE / simulate sources + quaternion math for
  the viz. Single-hand quaternion stream; BLE device name `senz-glove`.
- `senz_multi_io.py` — self-describing CSV I/O for the multi-IMU firmware
  (learns the column schema from the device banner at runtime).
- `senz_parser.py` — v2 binary-frame reader (180-byte frame, daemon thread +
  freshest-frame queue). Serial + `--simulate`.
- `senz_visualizer.py` — v2 VPython hand skeleton, one bone per IMU (rendering +
  forward kinematics). Serial + `--simulate`. Fingers flat until fusion lands.

### Calibration
- `imu_calibrate.py` — multi-IMU calibration application (HLD objective 2).
- `fusion/madgwick.py` — 6-axis Madgwick AHRS baseline (HLD objective 6).

### Dataset & recording
- `record.py` — capture the sensor stream to a labeled CSV for ML.
- `dataset.py` — load/preprocess recorded sessions into arrays (NumPy only).
- `dataset_prep.py` — ML dataset preparation (HLD objective 7).
- `dataset_recorder.py` — unified multi-modal recorder (HLD objective 5).
- `force_pipeline.py` — Velostat force-sensor processing (HLD objective 4).
- `camera_tracker.py` — MediaPipe Hands camera tracking (HLD objective 3).

## Notes
- The glove streams quaternions (no gimbal lock), so the hand won't flip when
  pointing up or down.
- The legacy stream tops out at 100 Hz (the BNO055 fusion ceiling); the viz
  drains to the latest frame, so it never lags behind the stream.

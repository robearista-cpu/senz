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
**`senz_parser.py`**. Validate the stream — wired or hardware-free:

```
python senz_parser.py --simulate        # no hardware: synthetic frames + live Hz
python senz_parser.py --port COM5        # wired: prints frame rate + drop count
```
(v2 serial runs at 921600 baud by default; override with `--baud`.)

> **v2 Bluetooth and the v2 3D visualizer are not built yet.** BLE transport
> (MTU/conn-param/PHY) and the binary-frame visualizer + zero-pose calibration
> are the supervised, max-effort tasks in the HLD. Until the firmware's Madgwick
> filter lands, the 10 finger quaternions arrive as identity (a flat hand), so
> `senz_parser.py` today is for **stream validation**, not finger animation. For
> a moving 3D hand right now, use `live_hand_qt.py` above.

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

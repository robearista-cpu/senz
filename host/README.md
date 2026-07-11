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

**Zero-pose calibration** (`senz_calibrate_pose.py`) — run once per session,
before the visualizer, so a flat hand renders flat. Hold your hand flat/palm
down/fingers straight; it averages each finger's quaternion and writes
`pose_offsets.json` (which the visualizer auto-loads):
```
python senz_calibrate_pose.py --port COM5        # wired; prompts, then captures
python senz_calibrate_pose.py --simulate --yes    # no hardware, no prompt
```

**Multi-sensor 3D hand** (`senz_visualizer.py`) — VPython skeleton, one bone per
IMU (11 total: wrist + 10 finger segments):
```
python senz_visualizer.py --simulate     # no hardware (wrist rocks, fingers curl)
python senz_visualizer.py --port COM5     # wired USB serial
```
(v2 serial is 921600 baud by default; override with `--baud`. Needs `vpython`.)

In-window controls (no restart needed):
- **Active sensor** menu — pick the wrist or any of the 10 finger IMUs to edit.
- **Enabled** checkbox — temporarily disable a sensor; its bone freezes straight
  and dims (useful to isolate a flaky IMU).
- **Axis remap** (X/Y/Z ← source) — invert or swap that sensor's axes to match how
  it's physically mounted. Reference finger accel frame: **+Y** away from hand,
  **+X** toward thumb, **+Z** up. Applied as `M·R·Mᵀ`, so it stays a valid
  rotation; the three axes must be distinct or the change is rejected.
- **Zero hand** — capture the current pose as neutral for the wrist and every
  finger (live tare). **Clear zero** / **Reset axes** undo per sensor.

> **Fingers articulate.** The firmware runs a per-finger Madgwick filter and
> sends each quaternion relative to the wrist, so fingers move independently in
> both `--simulate` and on hardware (the composition was validated to 1e-15
> against the visualizer's math). Run `senz_calibrate_pose.py` first for a tidy
> flat-hand pose; the remaining pending piece is **v2 Bluetooth** (still
> serial-only). Run `senz_visualizer.py --simulate` to see the full pipeline
> move with no hardware.

---

## Hardware Sprint v3 — tactile-first (current)

This sprint is scoped to **getting the tactile (velostat force) data down solid**
plus **gross hand movement** — *not* every finger accelerometer. Fine finger motion
is deferred to **camera** capture (`camera_tracker.py`) in a later fusion pass. The
firmware (`firmware/senz_glove_v3_tactile/`) is minimal:

- **BNO055** wrist (I2C) — forearm frame.
- **1× MPU-9250 dorsum IMU** (SPI, single direct `CS=GPIO4`) — back-of-hand/palm
  frame. Together they give hand orientation + wrist-flex.
- **15-taxel velostat array** via the CD74HC4067 (the deliverable): thumb/index/
  middle fingertip 2×2 + palm center/thenar/hypothenar.

No 74HC595, no I2C mux, no analog mux for the IMU — the CD74HC4067 is used **only**
for the velostat pads. Wiring: `docs/PINOUT_v3_tactile.txt`.

The viewer is the same **`senz_v3_qt.py`** (it reads `nimu`/`nforce` from the
banner). With the tactile build the dorsum drives gross hand pose and the **fingers
render dimmed in a neutral rest pose** (they'll come from the camera later); the
**tactile overlay + force grids are the focus**.

```
python senz_v3_qt.py --simulate --hand right     # no hardware (tactile sim, default)
python senz_v3_qt.py --port COM5 --hand right      # wired glove @ 921600
python senz_v3_tactile_sim.py                       # sim alone: print a few frames
```

---

## Hardware Sprint v3 prototype — native GPU visualizer

The v3 prototype firmware (`firmware/senz_glove_v3_proto/`) is a trimmed build —
wrist (BNO055, forearm) + index (2 IMUs) + middle (2 IMUs) + thumb (3 IMUs) + a
**back-of-hand dorsum IMU** + a **12-taxel velostat force array** — streaming a
self-describing CSV of **raw** accel/gyro + force (`senz_multi_io.py`). Its viewer
is **`senz_v3_qt.py`**: a **native OpenGL window** (pyqtgraph + PyOpenGL + PyQt5),
*not* a browser — shaded cylinder bones + sphere joints at ~100 fps, on-hand force
patches, and a side panel of the three 2×2 force grids. Because the firmware sends
raw data, the host runs **Madgwick fusion per finger IMU** (the wrist is already
fused on-device).

The **dorsum IMU drives the hand/palm frame** (fingers hang off it); the wrist
BNO055 is the **forearm** frame, and the wrist between them renders as a **flexing
polygon** that bends with wrist flex/deviation (not a rigid box). Wiring:
`docs/PINOUT_v3_proto.txt` (left hand) and `docs/PINOUT_v3_proto_RIGHT.txt`.

### Run it
```
python senz_v3_qt.py --simulate --hand right   # no hardware (default right hand)
python senz_v3_qt.py --port COM5 --hand right    # wired glove @ 921600 (--baud to override)
python senz_v3_qt.py --simulate --hand left       # left-hand layout
```
The base simulator on its own (sanity-check the synthetic stream):
```
python senz_v3_sim.py                 # prints a few synthetic frames
```
`--simulate` drives the whole pipeline — fingers curl, wrist rocks, force pads pulse
— so you see it move with nothing plugged in.

### In-window controls (right panel)
- **Theme: Dark/Light** — toggles both the control panel and the 3D viewbox
  (background + grid) between dark and light modes.
- **Sensor** dropdown — pick the wrist (forearm), any of the 8 finger IMUs, or the
  **hand-dorsum** to edit.
- **Enabled** — temporarily disable a sensor; its bone dims and freezes straight.
- **Axis remap** (X/Y/Z ← ±source) — invert/swap a sensor's axes to match its mount.
  Handedness is set by `--hand`; if a finger still points wrong, remap that sensor —
  the **left** hand typically needs **invert X** (defined as toward the thumb), the
  **right** hand usually needs none. Only valid signed permutations (real rotations)
  are accepted; duplicate axes are ignored.
- **Zero hand** — tare the current pose as neutral (hold flat, click). **Clear zero**
  undoes it. **Zero force** — re-baseline the velostat taxels.
- Telemetry shows the live **wrist-flex angle** (forearm vs hand frame).

Force channels (firmware order): thumb `force0–3`, index `force4–7`, middle `force8–11`
(2×2 each), **palm** `force12` center / `force13` thenar / `force14` hypothenar. `C15`
free. Palm taxels capture power/enclosing grasps the fingertips miss.

> **6-axis limitation:** the finger IMUs are accel+gyro (no magnetometer), so rotation
> about the gravity (yaw) axis is not directly observable — finger *curl* tracks well
> while *spread* can drift. The wrist BNO055 (with mag) anchors heading, and Zero-hand
> resets per-finger offsets. Inherent to 6-axis fusion, not a bug.

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
- **`senz_v3_qt.py`** — v3 **native GPU** hand viz (pyqtgraph/OpenGL): dorsum palm
  frame + flexing wrist + force overlay; host-side Madgwick fusion; `--hand right|left`.
  Schema-driven, so it serves both the **tactile** (1 IMU, fingers rest) and **proto**
  (8 IMU, fingers articulate) builds. Serial + `--simulate` (`--sim tactile|proto`).
- `senz_v3_tactile_sim.py` — Hardware Sprint v3 **tactile** simulator: 1 dorsum IMU
  (gross rock) + BNO wrist + 15-taxel grasp-cycle force. Default for `--simulate`.
- `senz_v3_sim.py` — v3 **proto** simulator (8 IMU): scripts a pose animation and emits
  the raw accel/gyro that reproduces it, so `--simulate --sim proto` exercises fusion.

### Calibration
- `senz_calibrate_pose.py` — v2 zero-pose capture; averages the flat-hand finger
  quaternions and writes `pose_offsets.json` for `senz_visualizer.py`.
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

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

## What changed since the v2 hardware sprint

The v2 sprint was a **5-finger flex-style glove** with a per-IMU stick-skeleton viz
(`senz_visualizer.py`, VPython) over serial/BLE. Everything below is what v3 added.

1. **v3 tactile-first build** (`firmware/senz_glove_v3_tactile`, `docs/PINOUT_v3_tactile.txt`):
   scope narrowed to **solid tactile data + gross hand movement** — BNO055 wrist + **1
   dorsum IMU** + a **15-taxel velostat** force array. Fine finger motion was deferred to
   the camera. New **native-GPU visualizer** `senz_v3_qt.py` (pyqtgraph/OpenGL, PyQt5) with
   a self-describing CSV stream (`senz_multi_io.py`) and host-side Madgwick fusion.
2. **Camera as ground-truth labeler** (`docs/senz_camera_hld.md`): `camera_tracker.py`
   (MediaPipe **Tasks** HandLandmarker — the legacy `solutions` API was removed on Py 3.13)
   and `camera_setup.py`, a framing/lighting UI with a skeleton overlay + quality warnings.
3. **Shared 21-landmark hand** (`hand_model.py`): one MediaPipe topology/geometry used by
   both the visualizer and the camera UI, so they draw the same hand. The 3D hand became the
   **21-landmark skeleton**, driven by **IMU orientation + finger-IMU articulation + optional
   live camera fusion** (`--camera`; IMU orients, camera articulates the fingers).
4. **Control hub** (`senz_hub.py` + double-click `senz_hub.bat`/`.vbs`): one launcher for the
   camera setup, visualizer, and synced recorder from shared settings.
5. **v3 pinch build** (`firmware/senz_glove_v3_pinch`, `docs/PINOUT_v3_pinch.txt`): the
   focused **index/middle/thumb** glove for pinching-gesture ML — 8 IMUs (thumb 3 incl. a
   9-axis base, index 2, middle 2, dorsum 1) + BNO wrist + **12 fingertip taxels**. Ring +
   pinky are **omitted everywhere**. Pinch features (`pinch.py`) turn the pads into
   distance/force/state. (Fixed a force-pipeline baseline startup transient along the way.)
6. **Bluetooth** (`host/senz_ble_io.py` + BLE in the pinch firmware): the glove streams the
   **same** self-describing CSV over **USB *and* BLE** (Nordic UART Service). USB-preferred at
   boot, BLE fallback on battery. `--ble` on the viz/recorder, plus a connection mode in the hub.
7. **Camera setup upgrades**: **Scan** enumerates all video devices (by name), open **several
   cameras at once** in a grid, and **optimization** controls (resolution, fps, MJPG format for
   USB bandwidth, detection downscale for CPU).
8. **Hand studio** (`senz_hand_studio.py`): the "final result" — the same fused pose skinned
   into a **mesh hand** (capsule *or* low-poly) with **tactile force glowing** on it, plus a
   **point-cloud** overlay.

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
banner). The hand is drawn as the **MediaPipe 21-landmark, 5-finger skeleton** —
the *same* per-finger colored hand the camera program shows (shared `hand_model.py`)
— so orientation reads at a glance. With the tactile build the dorsum IMU orients
that hand as a **canonical open pose** (gross movement); pass **`--camera`** to
**fuse** the live camera so the fingers actually articulate (**IMU = orientation,
camera = finger shape**). The **tactile overlay + force grids** remain the focus.

```
python senz_v3_qt.py --simulate --hand right     # no hardware (tactile sim, default)
python senz_v3_qt.py --port COM5 --hand right      # wired glove @ 921600
python senz_v3_qt.py --port COM5 --camera 0        # + fuse a webcam: fingers articulate
python senz_v3_tactile_sim.py                       # sim alone: print a few frames
```

### Control hub (`senz_hub.py`)

One small window to launch the pieces from a **shared set of settings** (glove port,
camera source, hand, sim build, label) — no remembering flags. Each tool opens in its
**own** window (spawned via `QProcess`); the hub stays open and shows running/stopped
status with a **Stop** per row.

```
python senz_hub.py
```

- **Camera setup** → `camera_setup.py` (frame/light the camera).
- **Hand visualizer** → `senz_v3_qt.py` (tick *Fuse camera* to add `--camera`).
- **Record (connect everything)** → `dataset_recorder.py` (glove + camera → synced CSV).
- Blank port = `--simulate`, blank camera = `--demo`; **Scan** probes camera indices.
- A single webcam feeds one program at a time, so the hub **warns** before starting a
  second camera consumer. Settings persist to `senz_hub.json`; light/dark toggle.

### Camera setup & alignment (`camera_setup.py`)

Since fingers come from the **camera** (as ground-truth labels, per
`docs/senz_camera_hld.md`), **`camera_setup.py`** helps you frame and light the phone
camera *before* recording. It's a native PyQt5 window: the live feed with a **MediaPipe
hand-skeleton overlay** (what the tracker sees), an **actionable warnings HUD** ("too
dark", "move closer", "blurry", "change angle", handedness-mismatch...), a **readiness**
banner, and light/dark themes with a **toggle button per overlay/panel**. When the
framing is good, **Copy recorder command** hands off to `dataset_recorder.py`.

```
python camera_setup.py                    # cameras from the saved config (default index 0)
python camera_setup.py --source 1          # a single USB camera index (or use Scan)
python camera_setup.py --sources 0,1       # TWO cameras at once (grid)
python camera_setup.py --source http://<phone-ip>:port/video   # phone IP camera
python camera_setup.py --demo               # no webcam/mediapipe (synthetic feed)
```

- **Scan + multi-camera**: **Scan USB / video devices** enumerates every connected camera
  (by **name** where possible — `pip install pygrabber` for names on Windows, else
  "Camera 0/1/..."). **Check one or several** and **Connect selected** to open them all at
  once in a **grid**; click a view to focus its detailed metrics/warnings. Add a phone URL
  with the **Add** field.
- **Optimization** (the capture-cost levers): **Resolution**, **FPS**, **Format**, and
  **Detect scale**. **Format = MJPG** is compressed, so a UVC camera sends far fewer bytes
  over USB — this is the lever that lets **several USB cameras share one bus** (raw YUY2
  saturates it). **Detect scale** feeds MediaPipe a downscaled frame to save CPU (landmarks
  are normalized, so the overlay still lands on the full-res image). **Apply (reconnect)**
  re-opens the cameras with the new settings.
- **USB / built-in cameras** are just an index (`0`, `1`, `2`...). (Windows uses the
  DirectShow backend for reliability.)
- **Phone cameras**: virtual-webcam apps (Iriun/Camo/EpocCam/DroidCam-connect) appear as an
  index — use Scan. HTTP-stream apps (IP Webcam `:8080/video`, DroidCam Wi-Fi `:4747/video`)
  use the URL; open it in a browser first to confirm it plays, same Wi-Fi as the PC.
- **Filters** (Auto-contrast / Brighten / Sharpen / **Green glove**) are toggleable
  **detection aids** — they help MediaPipe in poor light and are *not* saved to the dataset
  (the glove is the ML input, the camera only produces landmark labels), so they only improve
  label quality. **Green glove** recolors a green glove toward skin tone (keeping the finger
  shading MediaPipe reads as hand *shape*) so the hand detector fires — MediaPipe is trained on
  bare skin and otherwise ignores a solid-green glove. If detection is still flaky, add light
  and keep the whole hand in frame.
- **fps**: the readout is the real camera/inference throughput; MediaPipe on CPU is the
  ceiling (~20–30 fps). Requesting a lower resolution or fps via `--width/--height/--fps`,
  and closing other apps, is the main lever.

Uses the MediaPipe **Tasks HandLandmarker** (the legacy `solutions` API was removed in
mediapipe 0.10.x on Python 3.13); the model auto-downloads once to `host/models/`. Handedness
is mirror-corrected so a physical right hand reads as "Right". Needs `opencv-python` +
`mediapipe` for a real camera; `--demo` runs the UI without them.

---

## Hardware Sprint v3 — pinch build (index / middle / thumb)

A focused build for **pinching-gesture ML**: the three pinching digits fully
instrumented, everything else dropped. Firmware `firmware/senz_glove_v3_pinch/`,
wiring `docs/PINOUT_v3_pinch.txt`.

- **8 IMUs**: thumb 3 (base MCP **9-axis** + tip + base 6-axis), index 2, middle 2,
  and a **dorsum** 6-axis (back-of-hand / palm frame, the LAST IMU).
- **BNO055** wrist (9-axis, fused) — forearm frame.
- **12 velostat taxels** = three fingertip **2×2 pinch pads** (thumb C0-3, index
  C4-7, middle C8-11). **No palm pads** (that's the tactile build). `nforce=12`.

Because the visualizer is schema-driven it serves this build directly, and the
finger IMUs **articulate the 21-landmark fingers** (the tactile build's fingers
stay open; the proto/pinch build's fingers curl from their own IMUs). Camera fusion
still overrides the fingers when `--camera` is given. The **ring + pinky are omitted
everywhere** for this build — the 3D hand and the camera overlay draw only
thumb/index/middle (auto for `--sim pinch` or a `senz-v3pinch` banner; force it with
`--fingers thumb,index,middle`).

```
python senz_v3_qt.py --simulate --sim pinch --hand right   # pinch sim (no hardware)
python senz_v3_qt.py --port COM5 --hand right               # wired pinch glove (USB @ 921600)
python senz_v3_qt.py --ble senz-pinch --hand right          # same glove over Bluetooth LE
python senz_v3_pinch_sim.py                                  # sim alone: print a few frames
```

**Connection: USB + Bluetooth.** The firmware streams the *same* self-describing
CSV over both USB serial and **Bluetooth LE** (Nordic UART Service). It prefers USB
(if a host has the port open at boot it logs USB as primary) but always advertises
BLE, so on a battery a BLE central gets the identical stream — USB full-rate, BLE
decimated (`BLE_DECIM`, default 50 Hz) to fit the link. Host side: `--ble <name>`
on the visualizer / recorder, or pick a mode in the **hub** (USB / Bluetooth /
Simulate, with a **Scan BLE** button). Needs `bleak` (`pip install bleak`).

**Pinch features** (`pinch.py`, pure numpy) turn the two live signals into the
ML-relevant readout the visualizer shows and the recorder can log:

- **distance** — normalized thumb→index / thumb→middle fingertip gap (scale-free).
- **force** — pinch pressure per finger = `min(thumb pad, finger pad)`, so it is
  high only when the thumb **and** that specific finger press together (a mean
  would leak the shared thumb pad onto the idle finger).
- **state** — `open` / `index` / `middle` / `both`, force-driven (contact is the
  ground truth for a tactile pinch).

The pinch simulator scripts an alternating **index-thumb then middle-thumb pinch**
with matching fingertip contact, so `--sim pinch` exercises the whole path (fusion,
articulation, force pipeline, pinch classifier) with nothing plugged in.

> Force-pipeline note: `force_pipeline.py` now seeds each channel's baseline from
> its first sample, so an open pad reads ~0 grip immediately instead of decaying a
> false-grip transient over the first few seconds (fixed for all builds).

---

## Hand studio — the final mesh hand (all data, one view)

`senz_v3_qt.py` is the **diagnostic** view (sticks, spheres, per-sensor controls).
**`senz_hand_studio.py`** is the **beauty shot**: it takes the *same* fully-fused hand
and skins it into a solid **mesh hand** — tapered finger segments, rounded joints, a
domed palm + forearm — and paints the **tactile data onto it**: the thumb/index/middle
fingertips and the palm **glow with pressure** (base → hot orange → white-hot). One view
that shows everything coming in:

- **orientation** — the dorsum IMU orients the whole hand (BNO055 = forearm/wrist);
- **articulation** — the finger IMUs curl the fingers, or `--camera` lets the camera
  articulate them (the IMU still orients);
- **tactile** — the velostat force pads glow on the mesh.

It reuses `senz_v3_qt`'s pose + force + fusion pipeline and the 21-landmark `hand_model`,
so it's the exact same information rendered beautifully instead of diagnostically. Same
transports/flags, and it honors the pinch build's 3-finger set.

**View options** (in-panel, live): a **Mesh** selector — **Capsule** (smooth organic
hand), **Low-poly** (a chunky, flat-shaded *video-game hand* of boxes), or **None** — plus
a **Point cloud overlay** that shows the fused landmark positions as glowing points on top
of whichever mesh (or on their own with Mesh = None). All three are the same fused data.

```
python senz_hand_studio.py --simulate --sim pinch     # no hardware
python senz_hand_studio.py --port COM5 --camera 0      # wired glove + camera fusion
python senz_hand_studio.py --ble senz-pinch            # over Bluetooth
```

Launch it from the **hub** ("Hand studio") too. Rendering is real-time **OpenGL** (via
pyqtgraph's `GLViewWidget`) — a clean shaded hand with force glow, not a film-grade PBR
mesh. The pose is 21 world points per frame, so it can also drive an external engine
(Blender / three.js / a rigged glTF hand) later if you want higher fidelity.

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
- **Style: Noodle/Low-poly** — swap the smooth cylinder+sphere hand for a blocky
  low-poly one (~30× fewer triangles) when fps is tight.
- **Force test: On/Off** — hide the hand and show a **force-sensors-only** view for
  testing **and calibrating** the velostat array. One live row per channel shows the
  **raw ADC count** (0–4095), a bar, the min/max seen, and a touch dot. Press each
  pad in turn — a bar that moves (min≠max) is a wired, working channel; a flat bar is
  a dead/disconnected pad. It reports raw counts on purpose, since the normal
  `relative_grip` is auto-scaled and can look alive even when a pad isn't. Per-row
  calibration controls:
  - **on** checkbox — **deactivate a broken pad** (e.g. a dead thumb pad) so it no
    longer drives grip/contact; its raw bar still shows what it reads (dimmed).
  - **rev** checkbox — **reverse a channel** that reads backwards (high when open,
    low when pressed, from a flipped divider / mis-wired velostat leg). The ADC is
    inverted before processing so grip rises with pressure; the shown raw stays true.
  - **src** spinbox — **re-designate which physical channel feeds this pad**, for a
    finger whose array came back mis-wired/reversed, without re-soldering. Changing
    it re-zeros that channel.
  - **0** button — **zero/re-baseline just that channel**.
  - **Zero all (calibrate)** — hold the hand open, click: every channel takes its
    current reading as zero. **Reset min/max** clears the spread; **Reset order**
    restores identity routing, re-enables all pads, and clears reversals.
- **Accel: On/Off** — toggle the accelerometer in the finger-IMU fusion. Off = the
  Madgwick filter integrates **gyro only** (no gravity correction) — useful to see
  whether accel noise/vibration is driving orientation jitter.
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
- `senz_multi_io.py` — self-describing CSV I/O for the multi-IMU firmware over USB
  serial (learns the column schema + firmware id from the device banner at runtime).
- **`senz_ble_io.py`** — the Bluetooth LE counterpart: connects over the Nordic UART
  Service (`bleak`), reassembles notifications into the same self-describing stream,
  and hands out frames with the identical `.schema`/`read()`/`send()`/`close()`
  interface. `scan_ble()` lists nearby devices. Used by `--ble` on the visualizer /
  recorder / hub. Pure `StreamAssembler` core is headless-testable.
- `senz_parser.py` — v2 binary-frame reader (180-byte frame, daemon thread +
  freshest-frame queue). Serial + `--simulate`.
- `senz_visualizer.py` — v2 VPython hand skeleton, one bone per IMU (rendering +
  forward kinematics). Serial + `--simulate`. Fingers flat until fusion lands.
- **`senz_hub.py`** — control hub / launcher: one window, shared settings (port /
  camera / hand / sim / label), spawns each tool in its own window via `QProcess`,
  running/stopped status, single-camera guard, light/dark. Pure arg-building helpers
  are headless-testable.
- **`hand_model.py`** — the single source of the **21-landmark** hand topology,
  per-finger colors, canonical open-hand geometry, and `pose_from_world` (camera
  world-landmarks → orientation-free local shape). Pure numpy; imported by both the
  visualizer and the camera UI so they draw the same hand.
- **`senz_v3_qt.py`** — v3 **native GPU** hand viz (pyqtgraph/OpenGL): the MediaPipe
  **21-landmark 5-finger skeleton** (per `hand_model`) oriented by the dorsum IMU +
  flexing wrist + force overlay; host-side Madgwick fusion; `--hand right|left`.
  `--camera <src>` **fuses** a live camera (IMU orients, camera articulates fingers).
  Schema-driven, so it serves both the **tactile** (1 IMU) and **proto** (8 IMU)
  builds. Serial + `--simulate` (`--sim tactile|proto`).
- **`senz_hand_studio.py`** — the **"final result" mesh hand**: skins the same fused
  pose into a solid capsule-mesh hand (tapered finger segments, rounded joints, domed
  palm + forearm) with the **tactile force glowing** on the thumb/index/middle fingertips
  and palm. Same transports/flags as `senz_v3_qt.py` (`--port`/`--ble`/`--simulate`/
  `--sim`/`--camera`/`--hand`/`--fingers`); reuses its pose + force + fusion pipeline. The
  mesh-geometry helpers are pure numpy and headless-testable.
- `senz_v3_tactile_sim.py` — Hardware Sprint v3 **tactile** simulator: 1 dorsum IMU
  (gross rock) + BNO wrist + 15-taxel grasp-cycle force. Default for `--simulate`.
- `senz_v3_sim.py` — v3 **proto** simulator (8 IMU): scripts a pose animation and emits
  the raw accel/gyro that reproduces it, so `--simulate --sim proto` exercises fusion.
- `senz_v3_pinch_sim.py` — v3 **pinch** simulator (8 IMU, 12 fingertip taxels): scripts
  an alternating index-thumb / middle-thumb pinch with matching contact forces. Used by
  `--sim pinch`.
- **`pinch.py`** — pinch-gesture features (pure numpy): normalized thumb→finger tip
  distances, per-finger pinch force `min(thumb pad, finger pad)`, and an
  open/index/middle/both state. For the pinch build's ML + the viz readout.

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
- `camera_tracker.py` — MediaPipe Hands camera tracking (HLD objective 3). Accepts a
  camera index or a phone IP/RTSP URL; `keep_frame=True` exposes the frame + raw
  landmarks (+ capture-time stamp) via `get_frame()`; `get_world()` returns the 21
  metric world landmarks for 3D camera fusion.
- **`camera_setup.py`** — camera setup & alignment UI: live feed + per-finger colored
  skeleton overlay (shared `hand_model`) + quality warnings + readiness, light/dark
  themes, per-overlay toggles; `--demo` for no hardware. Pure-numpy metric/`assess`
  core is headless-testable.

## Notes
- The glove streams quaternions (no gimbal lock), so the hand won't flip when
  pointing up or down.
- The legacy stream tops out at 100 Hz (the BNO055 fusion ceiling); the viz
  drains to the latest frame, so it never lags behind the stream.

# senz host software

Live 3D hand visualization for the senz glove.

## Files
- **`live_hand_qt.py`** — **recommended.** GPU-accelerated 3D viz (pyqtgraph/OpenGL). Smooth 60-120 fps.
- `live_hand_viz.py` — legacy matplotlib viz (CPU, ~20-40 fps). Fallback if pyqtgraph won't install.
- `senz_io.py` — shared serial/BLE/sim sources + quaternion math (imported by `live_hand_qt.py`).

## Install
```
pip install -r requirements.txt
```
(Bluetooth needs `bleak`; the legacy viz needs `matplotlib`. Both are in the file.)

## Run (GPU viz)
```
python live_hand_qt.py --simulate          # no hardware
python live_hand_qt.py --port COM5         # USB serial
python live_hand_qt.py --ble               # Bluetooth (firmware USE_BLE true)
```

Find your COM port with:
```
python -m serial.tools.list_ports -v
```

## Controls
- **Drag** to orbit the 3D view, **scroll** to zoom.
- **Reset Level (tare)** — rest your hand flat, click to zero the baseline.
- **RotX/Y/Z < A?** — cycle which input axis (Axis 1/2/3 = quaternion x/y/z) drives each model rotation axis. Fixes mirroring/swaps.
- **RotX/Y/Z +/-** — invert an axis.
- **Invert finger** — flip finger open/closed direction.
- **Set Open / Set Fist** — calibrate the finger flexion range.

## Notes
- The glove streams quaternions (no gimbal lock), so the hand won't flip when pointing up/down.
- Sample rate is 100 Hz (the BNO055 fusion ceiling). The viz drains to the latest frame, so it never lags behind the stream.

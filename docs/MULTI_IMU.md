# Multi-Accelerometer Finger Tracking Glove

Branch **`multi-imu-finger-tracking`** — implementation of `docs/hld_new.md`.

This branch expands the single-BNO055 glove into a multi-modal data-collection
platform: a wrist reference IMU, a **SPI finger-IMU array**, **camera** hand
tracking, and **Velostat force** sensing, all recorded into one synchronized,
ML-ready dataset.

> The BNO055 drift-fix work lives on its own branch (`fix/bno055-drift`) and is
> intentionally **not** merged here — this branch stays independent so
> experimentation can't regress the stable firmware (HLD §2).

---

## Objective → file map

| HLD objective | Where it lives |
|---|---|
| 1. Multi-IMU firmware | `firmware/senz_glove_multi_imu/` (`.ino` + `icm42688.h` + `cs_expander.h`) |
| 2. Multi-IMU calibration | `host/imu_calibrate.py` |
| 3. Camera integration | `host/camera_tracker.py` |
| 4. Force-sensor pipeline | `host/force_pipeline.py` |
| 5. Dataset collection | `host/dataset_recorder.py` |
| 6. Sensor fusion (baseline) | `host/fusion/madgwick.py` |
| 7. ML dataset prep | `host/dataset_prep.py` |
| shared host I/O | `host/senz_multi_io.py` |

---

## Architecture

```text
Camera (MediaPipe Hands) ──► Host ◄── USB Serial ── ESP32-S3
                              │                         │
                       dataset_recorder.py        ┌─────┼───────────────┐
                              │                   ▼     ▼               ▼
                        data/<session>/        BNO055  ICM-42688-P x N  Velostat
                                               (I2C)   (SPI + CS expander)  (ADC)
                                               wrist   finger segments   contact
```

Everything sensor-side is sampled on **one MCU**, so all modalities share a
single `micros()` clock (`t_us` in every frame). The camera runs on the host and
is joined by host wall-clock time (sample-and-hold); precise alignment is done
offline in `dataset_prep.py`.

---

## Hardware

**Target MCU: ESP32-S3** (the `~10` finger IMUs + SPI + decoder + I2C + force
ADCs need more GPIO than an ESP32-C3 has). Pins below are defaults in the
firmware CONFIG block — change them to match your board.

### Finger IMU: ICM-42688-P (SPI)
6-axis (accel+gyro). A magnetometer on a finger segment is useless (ferrous
cross-talk), so heading comes from the wrist BNO055, not the finger IMUs.
Fixed full-scale: **accel ±8 g (4096 LSB/g)**, **gyro ±2000 dps (16.4 LSB/dps)**.

| Signal | ESP32-S3 pin | Firmware constant |
|---|---|---|
| SCLK | GPIO12 | `SPI_SCLK` |
| MISO | GPIO13 | `SPI_MISO` |
| MOSI | GPIO11 | `SPI_MOSI` |
| CS   | via expander | (see below) |

All IMUs share SCLK/MISO/MOSI; only chip-select is per-device.

### Chip-select expander (HLD §5)
~10 CS lines from a few GPIO via a binary decoder. With a decoder, exactly one
output is ever low → exactly one device selected → safe on a shared SPI bus.

| Decoder | Capacity | Address pins |
|---|---|---|
| 74HC138 | 8 IMUs | 3 (`CS_ADDR_PINS`) |
| 74HC4515 | 16 IMUs | 4 |

| Signal | ESP32-S3 pin | Firmware constant | Notes |
|---|---|---|---|
| A0 | GPIO4 | `CS_ADDR_PINS[0]` | address LSB |
| A1 | GPIO5 | `CS_ADDR_PINS[1]` | |
| A2 | GPIO6 | `CS_ADDR_PINS[2]` | |
| Enable | GPIO7 | `CS_ENABLE_PIN` | 74HC138 `G1`; tie `G2A/G2B`→GND |

Each decoder output `Yn` → the CSn pin of finger IMU `n`.

### Wrist IMU: BNO055 (I2C)
| Signal | ESP32-S3 pin | Firmware constant |
|---|---|---|
| SDA | GPIO8 | `I2C_SDA` |
| SCL | GPIO9 | `I2C_SCL` |
| ADD→GND | — | `BNO_ADDR = 0x28` |

### Force regions: Velostat (ADC)
Each region is a voltage divider: **Velostat → VREF**, **fixed series R (~10 kΩ)
→ GND**, ADC reads the junction. One ADC1 pin per region (`FORCE_PINS[]`,
default 5 fingertips). Velostat gives **relative** grip/contact, never Newtons.

---

## Serial stream (self-describing)

At boot (and on the `?` command) the firmware prints a banner + column header,
so the host learns the schema at runtime — change `NUM_FINGER_IMUS` / `NUM_FORCE`
in firmware and **no host edit is needed**:

```text
# senz-multi v1 nimu=3 nforce=5 rate=200
# columns: t_us,bno_cal,bno_qw,bno_qx,bno_qy,bno_qz,imu0_ok,imu0_ax,...,force0,...
<data lines>
```

Per finger IMU: `ok` flag + raw int16 `ax,ay,az,gx,gy,gz` (gyro is bias-corrected
unless in raw mode). Convert with the LSB constants in `senz_multi_io.py`.

### Firmware commands (host → device, newline-terminated)
| Cmd | Effect |
|---|---|
| `?` | re-emit banner + columns header |
| `XR` / `XN` | raw mode on/off (raw = uncorrected gyro, for calibration) |
| `C` | calibrate gyro bias of all IMUs (hold still), save to flash |
| `O,idx,gx,gy,gz,ax,ay,az` | set IMU `idx` offsets (LSB), apply live |
| `S` | save current offsets to NVS (flash) |
| `Z` | zero all offsets in RAM |

Calibration persists in NVS and survives re-flashing the main firmware.

---

## Host workflow

```bash
cd host
pip install -r requirements.txt          # pandas needed; mediapipe/opencv optional

# 1. Calibrate the finger IMUs (writes offsets to glove flash)
python imu_calibrate.py --port COM7

# 2. (optional) preview camera hand tracking
python camera_tracker.py --camera 0

# 3. Record a synchronized multi-modal session
python dataset_recorder.py --port COM7 --camera 0 --label grasp_cup
#    no hardware?  python dataset_recorder.py --simulate --duration 5

# 4. Prepare windowed train/val arrays for ML
python dataset_prep.py data/*grasp_cup --rate 100 --window 1.0 --hop 0.5

# offline fusion baseline (per-finger quaternion from accel+gyro)
python fusion/madgwick.py
```

Every host tool accepts `--simulate` (or runs without a port/camera) so the full
pipeline — record → prep — works with **no hardware** for development. Recorded
data and prepared `.npz` files go under `data/` (git-ignored).

---

## Milestone status (HLD §10)

| Phase | Status |
|---|---|
| 1. Select SPI IMU, bus + CS-expansion design | **done** — ICM-42688-P, 74HC138/4515 decoder |
| 2. Single-IMU bring-up, SPI + calibration | firmware + calibrator ready; needs hardware bring-up |
| 3. Scale to full array + BNO055 reference frame | firmware scales via `NUM_FINGER_IMUS`; needs hardware |
| 4. MediaPipe integration + sync | tracker + recorder integration ready |
| 5. Force regions + contact detection | firmware ADC + `force_pipeline.py` ready |
| 6. Build dataset framework, record pilots | `dataset_recorder.py` ready (sim-verified) |
| 7. Fusion experiments + ML dataset gen | Madgwick baseline + `dataset_prep.py` ready |

Software for every objective is in place and sim-verified end-to-end. Remaining
work is hardware bring-up (Phases 2–3) and fusion/ML research (Phases 6–7).

# HLD: Senz Glove — Hardware Sprint v2
**Branch:** `feature/multi-imu-expansion` off stable `main`
**Sprint Goal:** Swap board to ESP32-S3-DevKitC-1 and bring up 10x MPU-6500 per-finger IMU array on SPI at 200Hz per finger. Nothing else changes this sprint — no load cells, no camera. Get all accelerometers polling fast and streaming over serial, then BLE supervised.

---

## Pre-Sprint Checklist (do before 5PM)
- [ ] Pin headers soldered on ESP32-S3-DevKitC-1
- [ ] Arduino IDE has ESP32-S3 board package installed
- [ ] MPU-6500 library confirmed (recommend: `natanaeljr/esp32-MPU-driver` or `ElectronicCats/MPU6050`)
- [ ] BNO055 library installed (Adafruit BNO055)
- [ ] Madgwick library installed (MadgwickAHRS)

---

## What Is Changing This Sprint

| Item | Change |
|------|--------|
| Microcontroller | ESP32-C3 Super Mini → ESP32-S3-DevKitC-1 |
| Per-finger IMUs | None → 10x MPU-6500 on SPI |
| CS management | Direct GPIO per sensor — no decoder, no mux |
| Sample rate | 200Hz per finger IMU (SPI at 8MHz start, 20MHz verified) |
| Transport | Serial first (validated), BLE second (supervised) |

## What Is NOT Changing This Sprint

- BNO055 wrist IMU (stays on I2C, same calibration logic from v1)
- SSD1306 OLED (stays on I2C, same display logic)
- Calibration program for BNO055 (unchanged from v1)

---

## Why SPI Over I2C For The MPU-6500 Array

- I2C max is 400kHz (fast mode). Reading 14 bytes from one MPU-6500 takes ~300µs. 10 sensors sequentially = ~3ms minimum = only ~33Hz per sensor. Not acceptable.
- SPI on MPU-6500 runs at 8MHz (conservative start). At 8MHz, 14 bytes + CS toggle ≈ 16µs per sensor. 10 sensors = ~160µs total sweep.
- **Target: 200Hz per finger** — each of the 10 MPU-6500s polled every 5ms. Total SPI loop at ~160µs leaves ~4.84ms of remaining budget for Madgwick filter compute, BNO055 I2C poll, and BLE packing. This is tight but achievable — verify with a loop timer before declaring it done.
- After bring-up is stable, SPI clock can be increased toward 20MHz (MPU-6500 sensor register max), gaining additional headroom.

---

## Why No Decoder

Direct GPIO CS per sensor. No 74HC138 decoder. Reasons:
- ESP32-S3 has enough free GPIO once safe pins are chosen
- Removes a component, reduces wiring, removes decoder switching latency
- One less failure point during bring-up

---

## GPIO Budget — Honest Count

ESP32-S3-DevKitC-1 has 36 pins on headers. Subtract reserved:

| Reserved group | Count |
|----------------|-------|
| Strapping (GPIO0, 3, 45, 46) | 4 |
| USB D+/D− (GPIO19, 20) | 2 |
| Internal flash (GPIO26–32) | 7 |
| Internal PSRAM — N8R8 variant (GPIO33–37) | 5 |
| **Total reserved** | **18** |

36 total − 18 reserved = **18 genuinely usable GPIOs.**

This sprint uses 15 of those 18. That leaves **3 free pins**, not 11 as previously stated. Load cells (needing 10 pins for 5x HX711) do NOT fit in the remaining 3 pins — load cell wiring will require revisiting the GPIO map in the next sprint, likely using a dedicated ADC over I2C (ADS1115, already ordered) to reduce pin count. This is documented here so it doesn't surprise next sprint.

---

## Hardware

### Microcontroller: ESP32-S3-DevKitC-1
- Dual-core Xtensa LX7 @ 240MHz
- BLE 5.0 onboard, no external module
- **Power:** 5V via USB-C or 5V header pin → onboard regulator → 3.3V to chip and sensors
- **CRITICAL:** GPIO logic is 3.3V only — never connect 5V to any GPIO pin

### Power
| Rail | Source |
|------|--------|
| 5V input | 5V USB power bank (+) → 5V header pin |
| GND | Power bank (−) → GND pin |
| 3.3V sensors | ESP32-S3 3V3 pin → VCC on all sensors |

Use a 5V USB power bank (1000–2000mAh). Do not use raw LiPo (3.7V — too low) or 2S LiPo (7.4V — will damage regulator).

**Current draw this sprint:**
| Component | Draw |
|-----------|------|
| ESP32-S3 (BLE active) | ~80mA avg, 240mA peak |
| 10x MPU-6500 | ~5mA total |
| BNO055 | ~4mA |
| SSD1306 OLED | ~20mA |
| **Total** | **~110mA avg, ~270mA peak** |

1000mAh → ~3hr. 2000mAh → ~6hr.

---

## Sensor Architecture

### BNO055 — Wrist Reference IMU (unchanged from v1)
- I2C on GPIO8/9, address 0x28 (AD0 → GND)
- Absolute 9-axis orientation, loads saved calibration offsets from flash on boot
- Polled at 100Hz — provides absolute wrist reference frame for finger fusion

### 10x MPU-6500 — Per-Finger IMUs
- SPI on GPIO11/12/13, CS per sensor on dedicated GPIO (see pin table)
- 6-axis (accel + gyro only — no magnetometer)
- 14 bytes per read: accel XYZ (6), temp (2), gyro XYZ (6)
- Each sensor polled at **200Hz** — every 5ms

**⚠️ CS Pin Safety Note:** GPIO1 and GPIO2 are intentionally avoided as CS lines. These pins sit adjacent to strapping pins GPIO0 and GPIO3, and on some ESP32-S3 boards they exhibit spurious behavior during boot/reset that could assert a false CS and corrupt a sensor's internal state. CS lines use mid-range GPIOs only.

**Finger placement:**
| IMU # | CS GPIO | Finger | Segment |
|--------|---------|--------|---------|
| 1 | GPIO4 | Index | Proximal (base) |
| 2 | GPIO5 | Index | Distal (tip) |
| 3 | GPIO6 | Middle | Proximal |
| 4 | GPIO7 | Middle | Distal |
| 5 | GPIO14 | Ring | Proximal |
| 6 | GPIO15 | Ring | Distal |
| 7 | GPIO16 | Pinky | Proximal |
| 8 | GPIO17 | Pinky | Distal |
| 9 | GPIO18 | Thumb | Base |
| 10 | GPIO21 | Thumb | Tip |

### SSD1306 OLED (unchanged from v1)
- I2C, address 0x3C, shared bus with BNO055
- Shows: serial/BLE state, IMU count confirmed, calibration status, loop timing

---

## Pin Layout — ESP32-S3-DevKitC-1

### Reserved — Do Not Use
| GPIO | Reason |
|------|--------|
| GPIO0 | Strapping (boot mode) |
| GPIO1, GPIO2 | Boot-adjacent, avoid as CS |
| GPIO3 | Strapping |
| GPIO19 | USB D− |
| GPIO20 | USB D+ |
| GPIO26–32 | Internal SPI flash |
| GPIO33–37 | Internal PSRAM |
| GPIO45 | Strapping |
| GPIO46 | Input-only |

### Assigned Pins

| GPIO | Function | Connects To |
|------|----------|-------------|
| GPIO8 | I2C SDA | BNO055 SDA + OLED SDA |
| GPIO9 | I2C SCL | BNO055 SCL + OLED SCL |
| GPIO11 | SPI MOSI (SDI) | MPU-6500 SDI pin — all 10 sensors |
| GPIO12 | SPI SCLK | MPU-6500 SCLK pin — all 10 sensors |
| GPIO13 | SPI MISO (SDO) | MPU-6500 SDO pin — all 10 sensors |
| GPIO4 | CS — IMU #1 | Index proximal NCS |
| GPIO5 | CS — IMU #2 | Index distal NCS |
| GPIO6 | CS — IMU #3 | Middle proximal NCS |
| GPIO7 | CS — IMU #4 | Middle distal NCS |
| GPIO14 | CS — IMU #5 | Ring proximal NCS |
| GPIO15 | CS — IMU #6 | Ring distal NCS |
| GPIO16 | CS — IMU #7 | Pinky proximal NCS |
| GPIO17 | CS — IMU #8 | Pinky distal NCS |
| GPIO18 | CS — IMU #9 | Thumb base NCS |
| GPIO21 | CS — IMU #10 | Thumb tip NCS |
| 3V3 | Power out | VCC on all sensors |
| GND | Ground | GND on all sensors + power bank (−) |
| 5V | Power in | Power bank (+) |

**Free GPIOs remaining this sprint: GPIO38, GPIO39, GPIO40 — 3 pins only.**
**Load cells (next sprint) will require ADS1115 I2C ADC, not direct HX711 GPIO pairs, due to pin budget.**

---

## Software Architecture

### Sample Rate — Everything Is Designed Around This

**Target: 200Hz per finger IMU (each of 10 sensors, individually)**

Timing budget per 5ms loop:
| Task | Time estimate |
|------|--------------|
| 10x MPU-6500 SPI reads (at 8MHz) | ~160µs |
| 10x Madgwick filter updates | ~800µs (unverified — must profile) |
| BNO055 I2C read (at 400kHz) | ~300µs |
| BLE frame pack + notify trigger | ~50µs |
| **Total estimated** | **~1.31ms of 5ms budget** |

**The Madgwick estimate of ~80µs per filter is unverified.** If real compute is 200µs per filter, total becomes ~3.05ms — still fits. If it exceeds ~400µs per filter, the 200Hz target is at risk and the loop must be profiled and optimized before claiming the rate is met. Profile this early on day one.

### MPU-6500 SPI Mode
- CPOL = 1, CPHA = 1 (SPI Mode 3) — per MPU-6500 datasheet
- Start at 8MHz, verify stable reads on all 10, then step up toward 20MHz
- Read register address with bit7 set to 1 for read operations (MPU-6500 SPI protocol)
- CS (NCS) is active LOW — idle HIGH on all CS pins

### Boot Sequence
1. Load BNO055 calibration offsets from flash
2. Init I2C (GPIO8/9 at 400kHz) — bring up BNO055, confirm WHO_AM_I = 0x28, init OLED
3. Init SPI2 (GPIO11/12/13 at 8MHz, Mode 3) — cycle each CS pin, read WHO_AM_I from each MPU-6500 (expected: 0x70), display count on OLED
4. Flag any dead sensors on OLED, skip in main loop
5. Set SMPLRT_DIV = 4 on all confirmed MPU-6500s (200Hz at 1kHz internal clock)
6. Hold still prompt on OLED — average gyro bias over 3 seconds, save per-sensor offsets
7. Print "SERIAL READY" to USB serial
8. Start BLE advertising (OLED: `BLE: ADV`)

### Main Loop
1. Assert CS for IMU #1 → SPI read 14 bytes → deassert → store raw
2. Repeat for IMUs #2–10
3. Apply gyro bias offset per sensor
4. Run Madgwick update per sensor (accel + gyro, beta=0.1, dt=0.005)
5. Read BNO055 quaternion via I2C every 10th loop (~100Hz)
6. Express each finger quaternion relative to BNO055 wrist frame
7. Pack 180-byte frame (see format below)
8. Write frame to USB serial (always)
9. If BLE connected: push notify (supervised — see below)
10. Measure loop time with micros(), print warning to serial if >5ms

### Serial Frame Format (binary, always-on)
```
[4 bytes]   timestamp_ms (uint32)
[16 bytes]  bno055 quaternion (float32 w, x, y, z)
[160 bytes] mpu_quats[10] (float32 w, x, y, z per sensor)
Total: 180 bytes
```
Start byte: 0xAA. End byte: 0xFF. Parser validates these before accepting a frame.

---

## Software Deliverable — Real-Time 3D Hand Visualizer

### Goal
Visualize all 11 IMU sensors as a live 3D hand skeleton in a desktop window. No ML. No dataset recording. Just a digital hand that moves when the glove moves. This is the primary validation tool — if the hand looks right, the sensors are working right.

### Why VPython
VPython renders 3D objects from quaternions in real time, runs as a desktop window or browser tab, and has direct precedent for single-IMU quaternion visualization. Extending it to 11 sensors is a matter of building a skeleton model and rotating each bone segment independently. No game engine, no Unity, no MediaPipe needed.

### Architecture

```
ESP32-S3 (firmware)
    |
    |  USB-C serial  180-byte binary frame at 200Hz
    v
senz_parser.py
    |  Parses frame, pushes dict of 11 quaternions to thread-safe queue
    v
senz_visualizer.py  (VPython)
    |  Reads latest quaternions, rotates 3D bone objects at ~60Hz
    v
Browser / desktop window -- live 3D hand
```

### Hand Skeleton Model

The visualizer builds a simplified anatomically-correct hand skeleton from VPython cylinders (bones) and spheres (joints), driven purely by the 11 IMU quaternions.

**Bone hierarchy:**
```
Wrist (BNO055 -- absolute world reference)
|-- Index  proximal (IMU #1)  --> Index  distal (IMU #2)
|-- Middle proximal (IMU #3)  --> Middle distal (IMU #4)
|-- Ring   proximal (IMU #5)  --> Ring   distal (IMU #6)
|-- Pinky  proximal (IMU #7)  --> Pinky  distal (IMU #8)
+-- Thumb  base     (IMU #9)  --> Thumb  tip    (IMU #10)
```

Each bone is a VPython cylinder oriented by its IMU quaternion relative to the BNO055 wrist frame. The tip of each proximal bone is the origin of its distal bone — chained forward kinematics.

**Starting bone lengths (mm, adjustable per hand):**
| Segment | Length |
|---------|--------|
| Wrist block | 40 |
| Proximal phalanx (4 fingers) | 45 |
| Distal phalanx (4 fingers) | 25 |
| Thumb base | 35 |
| Thumb tip | 25 |

### File Structure

**`senz_parser.py`**
- Opens serial port at 921600 baud
- Finds 0xAA start byte, reads 180 bytes, verifies 0xFF end byte
- Unpacks: `struct.unpack('<I 4f 40f', payload)` - timestamp + 11 quaternions
- Pushes to `queue.Queue` (thread-safe, non-blocking)
- Runs in a daemon thread, never blocks the visualizer

**`senz_visualizer.py`**
- Builds hand skeleton on startup: 10 cylinders + 11 spheres
- Main loop at ~60Hz:
  1. Pull latest quaternion dict from queue (skip frame if empty)
  2. Convert each quaternion to rotation via `scipy.spatial.transform.Rotation`
  3. Apply BNO055 wrist quaternion to root frame
  4. Walk each finger chain, rotate cylinder `.axis` vector by its IMU quaternion relative to parent
  5. Update sphere positions at bone tips
- Finger color coding: index=red, middle=green, ring=blue, pinky=yellow, thumb=white

**`senz_calibrate_pose.py`**
- Run once per session before visualizing
- Prompts: hold hand flat, palm down, fingers straight
- Captures each sensor's current quaternion as the zero-pose reference
- Saves to `pose_offsets.json` — visualizer loads this on startup

### Dependencies
```
pip install vpython pyserial scipy numpy
```

### Known Limitation
Finger positions in space are approximated from fixed bone lengths — no position tracking, only orientation. The hand stays centered in the VPython scene and does not move through 3D space. This is correct behavior for an IMU-only system with no camera, not a bug.

### Done When
- Open visualizer, put on glove, wiggle all 5 fingers individually — each moves correctly in the digital hand
- Closing fist closes the digital hand
- Moving wrist rotates the entire hand in 3D
- No visible jitter or drift within a 30-second rest window

---

## **⚠️ CRITICAL OBJECTIVE — DO NOT ATTEMPT WITHOUT SUPERVISION**

### BLE High-Speed Transport

BLE is the target wireless transport but has non-trivial implementation requirements at 100Hz with 180-byte frames:

- **Default BLE MTU is 23 bytes.** The 180-byte frame will be silently fragmented or dropped without explicit MTU negotiation to ≥185 bytes. This is not automatic — it requires `esp_ble_gatt_set_local_mtu()` on the peripheral and a matching request from the central. If not done, the central receives garbage and there is no obvious error.
- **Default connection interval on most centrals is 30–100ms.** At 100ms, BLE physically cannot deliver 100Hz. Explicit connection parameter update request (via `esp_ble_gap_update_conn_params()`) to ≤10ms is required. Android/iOS may reject this — behavior is platform-dependent and not guaranteed.
- **2M PHY** must be negotiated explicitly to get maximum throughput margin.

**This combination of MTU negotiation + connection parameter update + PHY selection is the hardest BLE implementation task in this project. It requires testing against the actual central device (phone or laptop) used for the ML dataset.**

Do not start this until:
1. All 10 sensors are streaming cleanly over serial
2. Madgwick filters are running and loop timing is verified under 5ms
3. A supervisor is present to review the BLE implementation

**Fallback is always available:** USB-C serial at 921600 baud, same 180-byte binary frame, Python parser reads it directly. The Python dataset script should support both transports via a `--transport [serial|ble]` flag so either path produces identical CSV output.

---

## BLE Profile (implement after serial is verified, supervised)

**Device name:** `SENZ-GLOVE`
**Service UUID:** `4FAFC201-1FB5-459E-8FCC-C5C9C331914B`

| Characteristic | Properties | Description |
|----------------|------------|-------------|
| Sensor Data | Notify | 180-byte frame, target 100Hz |
| Calibration Control | Write | `0x01` = trigger cal |
| Device Status | Read + Notify | IMU count, loop time, cal status |

**BLE frame downsampling:** Sensors poll at 200Hz. BLE notifies at 100Hz. Firmware sends every other frame — no averaging, just skip. This is intentional: do not average quaternions naively, quaternion slerp is needed for correct averaging and is not worth the compute cost here.

---

## Madgwick Filter — Open Technical Problem

MPU-6500 has no magnetometer. A Madgwick or Mahony filter running at 200Hz on the ESP32-S3 is the only way to get a stable quaternion per finger. Key facts:

- Compute cost per filter update: **unverified — must profile on day one before assuming timing budget is safe**
- Beta parameter (filter gain): start at 0.1. Higher = faster convergence, more noise. Lower = smoother, slower to settle after motion.
- Without a filter, raw gyro integration drifts to garbage within seconds
- Even with Madgwick, 6-axis-only means heading (yaw) relative to the world will drift over long sessions since there is no magnetometer to anchor absolute heading. BNO055 wrist reference partially compensates for overall hand orientation but per-finger yaw drift is unavoidable. This is a known limitation of 6-axis IMUs and is not a bug to fix — it is a constraint to document for the ML dataset consumer.

---

## Today's Timeline (5PM start)

| Time | Task | Done When |
|------|------|-----------|
| 5:00–5:15 PM | Pre-sprint checklist. Confirm headers soldered, libraries installed, board shows up in Arduino IDE. | Board uploads blink sketch |
| 5:15–5:45 PM | Wire SPI bus (GPIO11/12/13) to ONE MPU-6500 only (IMU #1, GPIO4 CS). Wire I2C (GPIO8/9) to BNO055 and OLED. | Physical wiring done for 1 sensor |
| 5:45–6:15 PM | Flash sketch: init SPI Mode 3 at 8MHz, assert GPIO4, read WHO_AM_I, print to serial. Confirm 0x70. BNO055 WHO_AM_I on I2C. OLED "HELLO". | Serial prints 0x70 for IMU #1 |
| 6:15–7:00 PM | Wire remaining 9 CS pins (GPIO5,6,7,14,15,16,17,18,21). Ping WHO_AM_I all 10 in loop. Fix any dead sensors. | Serial confirms all 10 |
| 7:00–7:45 PM | Set SMPLRT_DIV=4 (200Hz). Write polling loop reading all 10. Print raw accel/gyro to serial. Verify no crosstalk between sensors. | Raw data from all 10 on serial |
| 7:45–8:00 PM | Break. Check solder joints. |  |
| 8:00–8:45 PM | Add Madgwick filter for IMU #1 only. Measure loop time with micros(). Profile: is filter under 100µs per call? | Loop time printed to serial |
| 8:45–9:15 PM | If timing is safe, extend Madgwick to all 10. Confirm full loop stays under 5ms. Integrate BNO055 quaternion read. | Full loop verified under 5ms |
| 9:15–9:30 PM | Commit everything that works. Push branch. Write a 3-line note of what needs doing tomorrow. | Clean commit |
| 9:30 PM+ | Stop. BLE is tomorrow, supervised. |  |

**If anything in the timeline slips, drop the Madgwick step and just get raw data stable. A clean raw serial stream is a better stopping point than a half-working filter.**

---

## Deliverables This Sprint

### Firmware
| # | Deliverable | Status |
|---|-------------|--------|
| 1 | This HLD document | ✅ Done |
| 2 | Pin layout PDF (print-ready) | ✅ Done |
| 3 | Git branch `feature/multi-imu-expansion` | ✅ Done |
| 4 | Arduino: SPI init + WHO_AM_I all 10 MPU-6500s | ✅ Done (`mpu6500.h` + `imuArrayInit()`) |
| 5 | Arduino: 200Hz polling loop, raw data all 10 | ✅ Done (`D` dumps raw; `SMPLRT_DIV=4`) |
| 6 | Arduino: BNO055 I2C read + cal load on boot | ✅ Done (NVS offsets, `K` to save) |
| 7 | Arduino: OLED status + loop timing display | ✅ Done (boot + ~1Hz heartbeat) |
| 8 | Arduino: Madgwick filter, all 10 sensors | 🔲 Supervised (stub: `orientationOf()` returns identity) |
| 9 | Arduino: 180-byte binary serial frame (0xAA header, 0xFF footer) | ✅ Done (carries identity finger quats until #8 lands) |
| 10 | Arduino: loop profiling — confirm under 5ms sustained | ⚙️ Code in place (max/overrun counters); sustained rate unverified — needs hardware |
| 11 | **Arduino: BLE MTU + conn param + notify — SUPERVISED ONLY** | 🔲 Supervised |

### Host Software — Visualizer
| # | Deliverable | Status |
|---|-------------|--------|
| 12 | `senz_parser.py` — serial frame reader, background thread, queue output | ✅ Done (sim-tested; `--simulate` CLI) |
| 13 | `senz_calibrate_pose.py` — zero-pose capture, saves `pose_offsets.json` | 🔲 Supervised (needs finger quaternions from #8) |
| 14 | `senz_visualizer.py` — VPython 3D hand skeleton, 11 bones, 60Hz render | ⚙️ Rendering + forward kinematics done (headless-tested, length-preserving); fingers render flat until fusion (#8) supplies real quaternions |
| 15 | Visualizer validated: all 5 fingers move independently, wrist rotates hand | ⚙️ Wrist + independent per-finger articulation verified in unit tests; end-to-end on hardware pending #8 |
| 16 | Python: BLE client (bleak) transport option — after BLE firmware verified | 🔲 Supervised |

---

## Deferred to Next Sprint

| Feature | Notes |
|---------|-------|
| Load cells (TAL221 + HX711) | Pin budget exhausted — will use ADS1115 I2C ADC instead of direct HX711 GPIO pairs |
| Camera / MediaPipe | Host-side only, no firmware needed, can be added anytime |
| Load cell fabric | Not planned |

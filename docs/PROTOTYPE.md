# senz — 8-Hour Prototype Build

A focused, time-boxed build to demonstrate:

1. **Digital flexion/extension of 4 fingers** (index, middle, ring, pinky — no thumb).
2. **General hand orientation in real time** from a 9-axis IMU.

This document is the build plan, wiring, feasibility notes, and calibration
steps. Code lives in:

- `firmware/senz_glove_prototype/senz_glove_prototype.ino`
- `host/live_hand_viz.py`

## Scope decision: orientation, not position (read this first)

The **GY-BNO055** is a 9-axis absolute-orientation sensor that runs Bosch's
sensor fusion *on the chip* and outputs orientation directly. It measures
**orientation** very well, but **cannot** measure absolute X/Y/Z position.

- **Roll / pitch — reliable.** Fused from the accelerometer + gyro; no drift.
- **Yaw / heading — reliable once calibrated.** The BNO055 fuses the
  magnetometer, so yaw does **not** drift like a bare gyro would. It just needs a
  short calibration dance (see below) to reach full accuracy.
- **Position (where the hand is in space) — not feasible.** It requires
  double-integrating acceleration, and the error grows quadratically — you drift
  meters within seconds. This is a fundamental limitation, not a tuning problem.
  Real position tracking needs an external reference (camera, UWB, lighthouse).

**So the prototype demonstrates hand _pose/attitude_, not hand _location_.** The
visualization tilts the whole hand by roll and reports pitch/yaw numerically.
Using the BNO055 upgrades yaw from "drifty" to "solid" versus a basic MPU6050.

## Bill of materials

| Item | Notes |
|------|-------|
| ESP32-C3 SuperMini | MCU + USB serial |
| Analog mux (CD4051 8-ch or CD4067 16-ch) | Only channels 0–3 used |
| 4x potentiometers (drone-controller pots) | DIY string pots, one per finger |
| GY-BNO055 9-axis IMU | I2C; on-chip fusion, absolute orientation incl. drift-free yaw |
| 0.91" 128x32 SSD1306 OLED | I2C; on-glove status dashboard (addr 0x3C) |
| Glove + fishing line + small spools | One spool per finger pot |
| 4x small return springs | Pot's internal centering spring often suffices |
| Hookup wire, USB-C cable | |

## Wiring

Pins are chosen to avoid the ESP32-C3 SuperMini strapping/special pins —
**GPIO2, GPIO8 (onboard LED), GPIO9 (BOOT button)** — and to keep the mux analog
output on an ADC1 pin (GPIO0–GPIO4).

### Multiplexer → ESP32-C3 SuperMini
| Mux pin | SuperMini | Firmware constant |
|---------|-----------|-------------------|
| S0 | GPIO1 | `MUX_S0` |
| S1 | GPIO3 | `MUX_S1` |
| S2 | GPIO4 | `MUX_S2` |
| S3 (4067 only) | GPIO10 | `MUX_S3` |
| SIG / common | GPIO0 (ADC1_CH0) | `MUX_SIG` |
| VCC | 3V3 | |
| GND + EN/INH | GND | enable tied active |

Each pot: outer legs to 3V3 and GND, wiper to mux channels C0..C3.

> Set `MUX_IS_16CH = true` in the firmware if you have the CD4067, otherwise
> leave it `false` for the CD4051. Only channels 0–3 are used either way.

### GY-BNO055 → ESP32-C3 SuperMini (I2C)
The GY-BNO055 board exposes: `VIN GND SCL/RX SDA/TX ADD INT BOOT RST`.
We use it in **I2C mode**, so `SCL/RX` is the clock and `SDA/TX` is the data line.

| BNO055 pin | SuperMini | Firmware constant | Notes |
|------------|-----------|-------------------|-------|
| VIN | 3V3 | | board has its own regulator; 3V3 is safe |
| GND | GND | | |
| SDA/TX | GPIO6 | `I2C_SDA` | I2C data |
| SCL/RX | GPIO7 | `I2C_SCL` | I2C clock |
| ADD | GND | `IMU_ADDR` = 0x28 | tie HIGH for 0x29 instead |
| INT | (leave unconnected) | | interrupt out, not used |
| BOOT | (leave unconnected) | | bootloader pin, not used |
| RST | (leave unconnected) | | optional; can pulse low to reset |

> `MUX_SIG` **must** stay on GPIO0–GPIO4 (ADC1). The other pins can move, but if
> you reuse GPIO2/8/9 you may see boot or upload glitches.

### SSD1306 OLED → ESP32-C3 SuperMini (I2C, shared bus)
The 0.91" OLED is I2C, so it sits on the **same two wires** as the BNO055. Its
address (0x3C) doesn't clash with the IMU (0x28), so just wire them in parallel.

| OLED pin | SuperMini | Notes |
|----------|-----------|-------|
| GND | GND | |
| VCC | 3V3 | module accepts 3.3–5V; use 3V3 |
| SCL | GPIO7 | same clock line as the BNO055 |
| SDA | GPIO6 | same data line as the BNO055 |

> The header pins on these modules ship **unsoldered** — solder them (or the
> wires) before use. If the screen stays blank, try address **0x3D** in the
> firmware (`OLED_ADDR`).

## Sensor choice: string-pots vs. flex sensors

The DIY VR glove community (e.g. **LucidGloves**) converged on **string +
spool + potentiometer** per finger because pots are cheap, durable, and accurate.
That is the build here, and you already own 4 drone-controller pots — so it costs
nothing extra.

**Resistive flex sensors** (the Spectra Symbol / Nintendo Power Glove part) are a
valid alternative. They are a variable resistor: ~30 kΩ straight → ~70 kΩ at a
90° bend. Wire each as a voltage divider (flex sensor + a ~47 kΩ fixed resistor)
and feed the divider junction into a mux channel — **the same signal path as a
pot wiper, so no firmware change is needed.**

| | String-pots (this build) | Flex sensors |
|---|---|---|
| Cost | Free (already owned) | ~$12–19 each |
| Assembly | Slower (spools/guides/springs) | Faster (tape/sew to finger) |
| Accuracy | High | Lower (drift, hysteresis) |
| Durability | High | Kink/fail near the base over time |
| Wires into the mux | Yes | Yes (drop-in) |

**Recommendation:** keep the string-pots. If the mechanical spool work threatens
the 8-hour budget, flex sensors are the fallback that requires zero code changes.

## On-glove dashboard (OLED)

The SSD1306 shows a 3-line status screen, refreshed at ~5 Hz:

```
senz glove   |      <- title + spinner (animates => firmware is running)
LINK: OK            <- host viz connected and reading
RUN  IMU c3 m3      <- running; BNO055 sys/mag calibration scores
```

- **Running:** the spinner in the corner animates every refresh — if it's
  moving, the main loop is alive.
- **LINK:** the host visualization sends a heartbeat byte each frame; the glove
  shows `OK` when it's seen one within `LINK_TIMEOUT_MS` (1.5 s), else `--`.
- **IMU:** shows the BNO055 system (`c`) and magnetometer (`m`) calibration
  scores (0–3) so you can watch them climb during the calibration dance.

**Libraries required** (Arduino Library Manager): `Adafruit SSD1306` and
`Adafruit GFX Library`. Set `OLED_ENABLED = false` in the firmware to build
without them.

## Mechanical (from the string-pot design notes)

- Mount a **grooved spool** on each pot shaft — never tie line to the bare shaft
  (side-load destroys the wiper).
- Add a small **guide eyelet** a few mm from the spool so the line pulls tangent;
  this is the single biggest noise-reducer.
- Use a **return spring** (the pot's own centering spring is usually enough) to
  keep the line taut on the release stroke.
- **Limit travel** to the pot's usable rotation (~270°) so you never slam the
  end-stop.
- Anchor the other end of each line to the corresponding fingertip of the glove.

## 8-hour plan

The trick to hitting 8 hours is running the **software and hardware tracks in
parallel** — the `--simulate` mode lets you finish and polish the visualization
while the mechanical build is still in progress.

| Time | Task |
|------|------|
| 0:00–0:30 | Flash firmware, confirm board + serial. Run `python live_hand_viz.py --simulate` to confirm the viz works. |
| 0:30–2:00 | Bench-test electronics on a breadboard: 1 pot through the mux + IMU on I2C. Confirm CSV stream in the serial monitor. |
| 2:00–4:30 | Mechanical: mount spools, guides, springs; anchor lines to glove fingertips. Verify each finger moves its pot smoothly. |
| 4:30–6:00 | Wire all 4 pots through the mux on the glove; mount the IMU on the back of the hand. |
| 6:00–7:00 | Connect glove to host. Calibrate fingers (open/fist). Tune the IMU. |
| 7:00–8:00 | Full demo run, fix noise/loose lines, record a short clip. |

## Running it

1. **Flash the firmware** (Arduino IDE / PlatformIO, ESP32 core installed).
   Edit the CONFIG block to match your wiring and mux type.
2. **Install host deps:** `pip install pyserial matplotlib numpy`
3. **Find the port:** Windows Device Manager (e.g. `COM5`) or Linux/macOS
   `ls /dev/tty*` (e.g. `/dev/ttyACM0`).
4. **Run the visualization:**
   ```
   python host/live_hand_viz.py --port COM5
   ```
   Or, with no hardware yet:
   ```
   python host/live_hand_viz.py --simulate
   ```

## Calibration

Finger range is calibrated on the host (firmware streams raw ADC counts):

1. Hold your hand **fully open** (fingers extended) and press **`o`**.
2. Make a **fist** (fingers fully flexed) and press **`c`**.

Each finger now maps to 0.0 (open) … 1.0 (fist).

### BNO055 orientation calibration
The BNO055 self-calibrates as it sees motion. The firmware streams a `cal` byte
and the visualization shows four 0–3 scores (sys / gyro / accel / mag); 3 = fully
calibrated. To bring them up after power-on:

- **Gyro:** set the glove down and keep it perfectly still for ~3 seconds.
- **Accel:** hold it in a few different steady orientations for a moment each.
- **Mag (for yaw):** wave it in slow figure-8 motions in the air.

You can demo before it's fully calibrated — roll/pitch are usable almost
immediately; yaw firms up once the mag score reaches 3.

**Persist the calibration so you don't repeat the dance every boot.** Flash the
standalone utility `firmware/senz_glove_calibrate` once, do the dance above
following its serial prompts, and it saves the 22-byte calibration profile to
the ESP32-C3's flash (NVS). Re-flash `senz_glove_prototype` and it loads that
profile on every boot — the OLED shows `cal: restored`. If the magnetometer
score drifts below 2 during a session the main firmware warns over serial and
flags `! MAG CAL LOW` on the OLED; wave another figure-8 to recover.

## Known limitations (prototype)

- No absolute hand position (IMU limitation — see scope note above).
- Yaw needs the BNO055 mag calibration dance before it's fully accurate.
- Single-turn pot range limits finger travel resolution.
- String pots need re-tensioning if lines stretch or slip.

## Next steps after the prototype

- ~~Read the BNO055 quaternion (reg 0x20) instead of Euler~~ — done; the
  firmware streams quaternions, which also dodges the SW_REV 0x0311 Euler bug.
- ~~Persist BNO055 calibration offsets so it starts calibrated~~ — done via the
  `senz_glove_calibrate` utility + NVS load on boot (see Calibration above).
- WiFi/BLE streaming off the SuperMini (code is behind `USE_BLE`, off by
  default per the drift-fix HLD; flip it to `true` to use it).
- 3D hand model in the visualization.
- Logging frames to `data/` for the eventual ML dataset.

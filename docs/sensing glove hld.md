# HLD: BNO055 Drift Fix — Sensing Glove

## Problem Statement
Heading drifts over time. Suspected cause: firmware/driver running in a mode that ignores the magnetometer (effectively 6DOF: accel+gyro only), so there's no absolute heading reference to correct gyro integration drift.

## Background
The BNO055 has onboard sensor fusion, so calibration isn't something built from scratch, it's a chip mode/register setting. Known behavior from the datasheet and field reports:

- Magnetometer calibration is mandatory immediately after every power-on reset, unlike the accelerometer and gyroscope.
- In NDOF mode, data should be discarded while system calibration status is 0, since that means the device hasn't found magnetic north yet. Heading will jump once it does.
- NDOF mode includes a fast magnetometer calibration (FMC) feature where even an incomplete figure-8 motion can fully calibrate it quickly.
- Some users report the opposite problem: heading can get worse at full 3/3 calibration, jittering and drifting up to 10 degrees, with the chip continuously overwriting any saved offsets.
- Magnetometer drift up to ~20 degrees has been reported in real-world moving conditions, with calibration level dropping fast during motion.
- There's a known firmware math bug (firmware version 0x0311) that distorts Euler angle output beyond ~20 degrees of pitch/roll. The documented workaround is to read quaternion output and convert to Euler in software instead of trusting the chip's Euler output directly.

## Objectives / To-Do List

1. **Diagnose before writing any new code**
   - Confirm which fusion mode is currently active (NDOF vs IMU-only/6DOF) by checking the operating mode register.
   - Log calibration status registers (sys/gyro/accel/mag, 0–3) over time during normal glove use.
   - Check the firmware version register for the known 0x0311 Euler angle bug.

2. **If it's a mode/config bug**
   - Set operating mode to NDOF explicitly in firmware init.
   - Switch from polling Euler angles directly to polling quaternion output and converting to Euler in software, to avoid the known math bug.

3. **Build the calibration program (separate from main runtime)**
   - Standalone script/mode, not bundled into the main hand-tracking firmware.
   - Prompts the user to perform a figure-8 motion to trigger fast magnetometer calibration.
   - Polls calibration status registers until magnetometer reaches 3/3 (and ideally system reaches 3).
   - Reads the 22-byte calibration offset profile once fully calibrated.
   - Writes offsets to flash/EEPROM for persistence across power cycles.
   - On every power-on, main firmware loads saved offsets instead of recalibrating from zero.
   - Adds a runtime check/flag (LED or serial message) if magnetometer calibration drops below 2 mid-session, since drift tends to creep back in during motion.

4. **Set expectations on hardware limitations**
   - Even with proper calibration, field reports show BNO055 mag fusion can still wander in motion, even at "3/3 fully calibrated."
   - If precision needs are tight, this may need a hardware conversation rather than a pure software fix.

5. **Data path (unchanged)**
   - ESP32 to BNO055 over I2C.
   - USB-C serial to host (BLE skipped, not worth the dev time).
   - CSV stream to MATLAB/Python for live plotting.

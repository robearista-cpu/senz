/*
 * senz_glove_calibrate
 * --------------------
 * STANDALONE BNO055 calibration utility for the senz glove (HLD objective 3).
 *
 * This is intentionally NOT part of the main hand-tracking firmware. Flash it
 * once, run the calibration dance, and it writes the 22-byte calibration
 * profile into the ESP32-C3's NVS (flash). The main firmware
 * (senz_glove_prototype) reads that profile on every boot via Preferences and
 * starts already calibrated, instead of drifting until the chip re-finds north.
 *
 * Procedure (follow the serial prompts at 115200 baud):
 *   1. Gyro:  set the glove down, hold perfectly still ~3 s.
 *   2. Accel: hold it in several steady orientations, a moment each.
 *   3. Mag:   wave slow figure-8s in the air (triggers fast mag calibration).
 * The tool polls the calibration-status registers and, once the magnetometer
 * reaches 3/3 (and system 3/3), reads the profile and saves it to flash.
 *
 * Re-run this whenever yaw accuracy degrades. NVS survives re-flashing the main
 * firmware, so you only calibrate when you actually need to.
 *
 * Wiring is identical to the main firmware: ESP32-C3 + BNO055 on I2C.
 */

#include <Wire.h>
#include <Preferences.h>

// --- Wiring (must match the main firmware) ---
static const int I2C_SDA = 6;
static const int I2C_SCL = 7;
static const uint8_t IMU_ADDR = 0x28;  // 0x29 if the ADD pin is tied HIGH

// --- BNO055 register map (subset; mirrors the main firmware) ---
static const uint8_t BNO_CHIP_ID = 0x00;  // should read 0xA0
static const uint8_t BNO_PAGE_ID = 0x07;
static const uint8_t BNO_CALIB_STAT = 0x35;
static const uint8_t BNO_OPR_MODE = 0x3D;
static const uint8_t BNO_PWR_MODE = 0x3E;
static const uint8_t BNO_SYS_TRIGGER = 0x3F;
static const uint8_t BNO_ACC_OFFSET_X_LSB = 0x55;  // start of the 22-byte profile
static const uint8_t BNO_CALIB_PROFILE_LEN = 22;   // 0x55..0x6A

static const uint8_t BNO_OPR_CONFIG = 0x00;
static const uint8_t BNO_OPR_NDOF = 0x0C;
static const uint8_t BNO_PWR_NORMAL = 0x00;

// NVS location shared with the main firmware.
static const char *NVS_NAMESPACE = "senz-bno";
static const char *NVS_CAL_KEY = "cal";

// ----------------------------------------------------------------------------
// I2C helpers
// ----------------------------------------------------------------------------
void imuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t imuRead8(uint8_t reg) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((int)IMU_ADDR, 1);
  return Wire.available() ? Wire.read() : 0;
}

void imuSetMode(uint8_t mode) {
  imuWrite(BNO_OPR_MODE, mode);
  delay(30);  // datasheet settle time for a mode switch
}

bool imuInit() {
  Wire.beginTransmission(IMU_ADDR);
  if (Wire.endTransmission() != 0) return false;
  if (imuRead8(BNO_CHIP_ID) != 0xA0) return false;

  imuSetMode(BNO_OPR_CONFIG);
  imuWrite(BNO_PAGE_ID, 0x00);
  imuWrite(BNO_PWR_MODE, BNO_PWR_NORMAL);
  delay(10);
  imuWrite(BNO_SYS_TRIGGER, 0x00);  // internal oscillator
  delay(10);
  imuSetMode(BNO_OPR_NDOF);  // fusion mode self-calibrates as it sees motion
  return true;
}

// Read the 22-byte calibration profile. Offset registers only hold valid data
// in CONFIG mode, so drop to CONFIG, burst-read, then return to NDOF.
bool imuReadCalProfile(uint8_t *buf) {
  imuSetMode(BNO_OPR_CONFIG);
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(BNO_ACC_OFFSET_X_LSB);
  if (Wire.endTransmission(false) != 0) {
    imuSetMode(BNO_OPR_NDOF);
    return false;
  }
  Wire.requestFrom((int)IMU_ADDR, (int)BNO_CALIB_PROFILE_LEN);
  if (Wire.available() < BNO_CALIB_PROFILE_LEN) {
    imuSetMode(BNO_OPR_NDOF);
    return false;
  }
  for (int i = 0; i < BNO_CALIB_PROFILE_LEN; i++) buf[i] = Wire.read();
  imuSetMode(BNO_OPR_NDOF);
  return true;
}

bool saveCalProfileToNVS(const uint8_t *buf) {
  Preferences prefs;
  if (!prefs.begin(NVS_NAMESPACE, false)) return false;  // read-write
  size_t n = prefs.putBytes(NVS_CAL_KEY, buf, BNO_CALIB_PROFILE_LEN);
  prefs.end();
  return n == BNO_CALIB_PROFILE_LEN;
}

// ----------------------------------------------------------------------------
// Calibration flow
// ----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("=== senz BNO055 calibration utility ===");

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  if (!imuInit()) {
    Serial.println("ERROR: BNO055 not found on I2C. Check wiring/address.");
    while (true) delay(1000);
  }
  Serial.println("BNO055 ready. Calibrate each subsystem to 3/3:");
  Serial.println("  GYRO  : set the glove down, hold still ~3 s");
  Serial.println("  ACCEL : hold in several steady orientations, a moment each");
  Serial.println("  MAG   : wave slow figure-8 motions in the air");
  Serial.println("Watching calibration status (sys/gyro/accel/mag)...");
}

void loop() {
  static uint32_t lastPrint = 0;

  uint8_t cal = imuRead8(BNO_CALIB_STAT);
  uint8_t sys = (cal >> 6) & 0x3;
  uint8_t gyr = (cal >> 4) & 0x3;
  uint8_t acc = (cal >> 2) & 0x3;
  uint8_t mag = cal & 0x3;

  if (millis() - lastPrint >= 500) {
    lastPrint = millis();
    Serial.printf("sys=%u gyro=%u accel=%u mag=%u\n", sys, gyr, acc, mag);
  }

  // Need a trustworthy heading reference: magnetometer fully calibrated, and
  // the fused system solution converged.
  if (mag == 3 && sys == 3) {
    Serial.println("\nFully calibrated. Reading calibration profile...");
    uint8_t profile[BNO_CALIB_PROFILE_LEN];
    if (!imuReadCalProfile(profile)) {
      Serial.println("ERROR: failed to read the profile; retrying...");
      delay(500);
      return;
    }

    Serial.print("Profile (22 bytes):");
    for (int i = 0; i < BNO_CALIB_PROFILE_LEN; i++)
      Serial.printf(" %02X", profile[i]);
    Serial.println();

    if (saveCalProfileToNVS(profile)) {
      Serial.println("Saved to flash (NVS). The main firmware will load it on "
                     "boot. You can re-flash senz_glove_prototype now.");
    } else {
      Serial.println("ERROR: failed to write the profile to flash.");
    }

    Serial.println("Done. Halting.");
    while (true) delay(1000);
  }

  delay(50);
}

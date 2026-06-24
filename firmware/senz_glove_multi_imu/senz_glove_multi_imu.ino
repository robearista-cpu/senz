/*
 * senz_glove_multi_imu
 * --------------------
 * Multi-modal sensing firmware for the senz finger-tracking glove
 * (HLD: docs/hld_new.md, branch multi-imu-finger-tracking, objective 1).
 *
 * Sensing modalities, all sampled on one MCU so they share a single clock:
 *   - BNO055      : wrist / global hand orientation (I2C, on-chip fusion)
 *   - ICM-42688-P : finger IMU array (SPI, addressed via a chip-select expander)
 *   - Velostat    : contact/grip force regions (analog -> ADC)
 *
 * Recommended MCU: ESP32-S3 (enough GPIO for SPI + decoder + I2C + ADC). The
 * pin map below is for an S3 dev board; adjust CONFIG to match your wiring.
 *
 * Transport: USB serial, self-describing CSV. At boot (and on the '?' command)
 * the firmware prints a banner + a "# columns:" header so the host
 * (senz_multi_io.py) can parse generically without hard-coding the layout:
 *
 *   # senz-multi v1 nimu=3 nforce=5 rate=200
 *   # columns: t_us,bno_cal,bno_qw,bno_qx,bno_qy,bno_qz,\
 *              imu0_ok,imu0_ax,imu0_ay,imu0_az,imu0_gx,imu0_gy,imu0_gz,...,force0,...
 *   <data lines at `rate` Hz>
 *
 *   t_us    : micros() timestamp at sample time (host de-jitters / aligns)
 *   bno_*   : wrist quaternion (float) + calibration byte
 *   imuK_*  : finger IMU K: ok flag, then raw int16 accel + gyro (gyro is
 *             bias-corrected with the stored calibration unless in raw mode)
 *   forceM  : raw 12-bit ADC for force region M
 *
 * Serial command protocol (host -> device, newline-terminated):
 *   ?                          re-emit the banner + columns header
 *   XR / XN                    raw mode on/off (raw = stream uncorrected gyro)
 *   C                          calibrate gyro bias of all IMUs (hold still), save
 *   O,idx,gx,gy,gz,ax,ay,az    set offsets for IMU idx (LSB), apply live
 *   S                          save current offsets to NVS (flash)
 *   Z                          zero all offsets in RAM
 */

#include <Preferences.h>
#include <SPI.h>
#include <Wire.h>

#include "cs_expander.h"
#include "icm42688.h"

// ============================================================================
// CONFIG -- edit to match your wiring (defaults: ESP32-S3 dev board)
// ============================================================================

// --- Finger IMU array (SPI) ---
static const uint8_t NUM_FINGER_IMUS = 3;  // Phase-2/3: scale up toward ~10
static const int SPI_SCLK = 12;
static const int SPI_MISO = 13;
static const int SPI_MOSI = 11;

// Chip-select expander address lines (LSB first) + active-high enable.
// 3 address lines -> up to 8 IMUs (74HC138); add a 4th for up to 16 (74HC4515).
static const uint8_t CS_ADDR_PINS[] = {4, 5, 6};
static const uint8_t CS_NUM_ADDR = sizeof(CS_ADDR_PINS) / sizeof(CS_ADDR_PINS[0]);
static const uint8_t CS_ENABLE_PIN = 7;

// --- Wrist IMU (BNO055, I2C) ---
static const int I2C_SDA = 8;
static const int I2C_SCL = 9;
static const uint8_t BNO_ADDR = 0x28;  // 0x29 if ADD pin HIGH

// --- Force regions (Velostat voltage dividers -> ADC) ---
// Use ADC1 pins on the S3 (GPIO1..10). One entry per sensing region.
static const int FORCE_PINS[] = {1, 2, 3, 10, 14};
static const uint8_t NUM_FORCE = sizeof(FORCE_PINS) / sizeof(FORCE_PINS[0]);

// --- Output rate ---
static const uint32_t SAMPLE_HZ = 200;

// ============================================================================
// BNO055 (minimal: quaternion + calibration in NDOF). The full drift-fix
// firmware lives on its own branch; here the BNO is just the wrist reference.
// ============================================================================
static const uint8_t BNO_CHIP_ID = 0x00;        // -> 0xA0
static const uint8_t BNO_CALIB_STAT = 0x35;
static const uint8_t BNO_OPR_MODE = 0x3D;
static const uint8_t BNO_PWR_MODE = 0x3E;
static const uint8_t BNO_SYS_TRIGGER = 0x3F;
static const uint8_t BNO_QUA_DATA_W_LSB = 0x20;  // 8 bytes
static const uint8_t BNO_OPR_CONFIG = 0x00;
static const uint8_t BNO_OPR_NDOF = 0x0C;
static const float BNO_QUA_LSB = 16384.0f;

// ============================================================================
// State
// ============================================================================
static const uint32_t FRAME_US = 1000000UL / SAMPLE_HZ;

SPIClass &spiBus = SPI;
CsExpander csExp(CS_ADDR_PINS, CS_NUM_ADDR, CS_ENABLE_PIN);

Icm42688 *fingerImu[16] = {nullptr};
bool imuPresent[16] = {false};
int16_t gyroBias[16][3] = {{0}};   // LSB, subtracted before streaming
int16_t accelOff[16][3] = {{0}};   // LSB, reserved (host 6-point tumble later)

bool bnoPresent = false;
float bqw = 1, bqx = 0, bqy = 0, bqz = 0;
uint8_t bnoCal = 0;

bool rawMode = false;  // when true, stream uncorrected gyro (for calibration)
uint32_t lastMicros = 0;

Preferences prefs;
static const char *NVS_NS = "senz-multi";
static const char *NVS_KEY = "imucal";  // blob: NUM_FINGER_IMUS * 6 int16

// ============================================================================
// BNO055 helpers
// ============================================================================
void bnoWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(BNO_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t bnoRead8(uint8_t reg) {
  Wire.beginTransmission(BNO_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((int)BNO_ADDR, 1);
  return Wire.available() ? Wire.read() : 0;
}

bool bnoInit() {
  Wire.beginTransmission(BNO_ADDR);
  if (Wire.endTransmission() != 0) return false;
  if (bnoRead8(BNO_CHIP_ID) != 0xA0) return false;
  bnoWrite(BNO_OPR_MODE, BNO_OPR_CONFIG);
  delay(25);
  bnoWrite(BNO_PWR_MODE, 0x00);
  delay(10);
  bnoWrite(BNO_SYS_TRIGGER, 0x00);
  delay(10);
  bnoWrite(BNO_OPR_MODE, BNO_OPR_NDOF);
  delay(25);
  return true;
}

void bnoReadQuat() {
  Wire.beginTransmission(BNO_ADDR);
  Wire.write(BNO_QUA_DATA_W_LSB);
  if (Wire.endTransmission(false) != 0) return;
  Wire.requestFrom((int)BNO_ADDR, 8);
  if (Wire.available() < 8) return;
  int16_t w = (int16_t)(Wire.read() | (Wire.read() << 8));
  int16_t x = (int16_t)(Wire.read() | (Wire.read() << 8));
  int16_t y = (int16_t)(Wire.read() | (Wire.read() << 8));
  int16_t z = (int16_t)(Wire.read() | (Wire.read() << 8));
  bqw = w / BNO_QUA_LSB;
  bqx = x / BNO_QUA_LSB;
  bqy = y / BNO_QUA_LSB;
  bqz = z / BNO_QUA_LSB;
}

// ============================================================================
// Finger IMU array
// ============================================================================
void imuArrayInit() {
  spiBus.begin(SPI_SCLK, SPI_MISO, SPI_MOSI, -1);  // CS handled by the expander
  csExp.begin();
  for (uint8_t i = 0; i < NUM_FINGER_IMUS; i++) {
    uint8_t idx = i;  // capture by value for the lambdas
    fingerImu[i] = new Icm42688(
        spiBus, [idx]() { csExp.select(idx); }, []() { csExp.deselect(); });
    imuPresent[i] = fingerImu[i]->begin();
  }
}

// ============================================================================
// NVS calibration persistence
// ============================================================================
void loadCalFromNVS() {
  prefs.begin(NVS_NS, true);
  size_t want = (size_t)NUM_FINGER_IMUS * 6 * sizeof(int16_t);
  if (prefs.getBytesLength(NVS_KEY) == want) {
    int16_t buf[16 * 6];
    prefs.getBytes(NVS_KEY, buf, want);
    for (uint8_t i = 0; i < NUM_FINGER_IMUS; i++) {
      gyroBias[i][0] = buf[i * 6 + 0];
      gyroBias[i][1] = buf[i * 6 + 1];
      gyroBias[i][2] = buf[i * 6 + 2];
      accelOff[i][0] = buf[i * 6 + 3];
      accelOff[i][1] = buf[i * 6 + 4];
      accelOff[i][2] = buf[i * 6 + 5];
    }
    Serial.println("# loaded IMU calibration from flash");
  } else {
    Serial.println("# no IMU calibration in flash; run imu_calibrate.py");
  }
  prefs.end();
}

void saveCalToNVS() {
  int16_t buf[16 * 6];
  for (uint8_t i = 0; i < NUM_FINGER_IMUS; i++) {
    buf[i * 6 + 0] = gyroBias[i][0];
    buf[i * 6 + 1] = gyroBias[i][1];
    buf[i * 6 + 2] = gyroBias[i][2];
    buf[i * 6 + 3] = accelOff[i][0];
    buf[i * 6 + 4] = accelOff[i][1];
    buf[i * 6 + 5] = accelOff[i][2];
  }
  prefs.begin(NVS_NS, false);
  prefs.putBytes(NVS_KEY, buf, (size_t)NUM_FINGER_IMUS * 6 * sizeof(int16_t));
  prefs.end();
  Serial.println("# saved IMU calibration to flash");
}

// Average gyro at rest -> bias. Keep the glove still during this.
void calibrateGyroBias() {
  const int N = 256;
  Serial.println("# calibrating gyro bias, hold still...");
  long sum[16][3] = {{0}};
  for (int s = 0; s < N; s++) {
    for (uint8_t i = 0; i < NUM_FINGER_IMUS; i++) {
      if (!imuPresent[i]) continue;
      int16_t ax, ay, az, gx, gy, gz;
      fingerImu[i]->readRaw(ax, ay, az, gx, gy, gz);
      sum[i][0] += gx;
      sum[i][1] += gy;
      sum[i][2] += gz;
    }
    delay(2);
  }
  for (uint8_t i = 0; i < NUM_FINGER_IMUS; i++) {
    if (!imuPresent[i]) continue;
    gyroBias[i][0] = (int16_t)(sum[i][0] / N);
    gyroBias[i][1] = (int16_t)(sum[i][1] / N);
    gyroBias[i][2] = (int16_t)(sum[i][2] / N);
  }
  saveCalToNVS();
}

// ============================================================================
// Self-describing header
// ============================================================================
void printHeader() {
  Serial.printf("# senz-multi v1 nimu=%u nforce=%u rate=%lu\n", NUM_FINGER_IMUS,
                NUM_FORCE, (unsigned long)SAMPLE_HZ);
  Serial.print("# columns: t_us,bno_cal,bno_qw,bno_qx,bno_qy,bno_qz");
  for (uint8_t i = 0; i < NUM_FINGER_IMUS; i++)
    Serial.printf(",imu%u_ok,imu%u_ax,imu%u_ay,imu%u_az,imu%u_gx,imu%u_gy,imu%u_gz",
                  i, i, i, i, i, i, i);
  for (uint8_t m = 0; m < NUM_FORCE; m++) Serial.printf(",force%u", m);
  Serial.println();
}

// ============================================================================
// Serial command handling
// ============================================================================
void handleCommands() {
  while (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) continue;
    char c = cmd.charAt(0);
    if (c == '?') {
      printHeader();
    } else if (cmd == "XR") {
      rawMode = true;
      Serial.println("# raw mode ON");
    } else if (cmd == "XN") {
      rawMode = false;
      Serial.println("# raw mode OFF");
    } else if (c == 'C') {
      calibrateGyroBias();
    } else if (c == 'S') {
      saveCalToNVS();
    } else if (c == 'Z') {
      memset(gyroBias, 0, sizeof(gyroBias));
      memset(accelOff, 0, sizeof(accelOff));
      Serial.println("# offsets zeroed (RAM)");
    } else if (c == 'O') {
      // O,idx,gx,gy,gz,ax,ay,az
      int idx, gx, gy, gz, ax, ay, az;
      if (sscanf(cmd.c_str(), "O,%d,%d,%d,%d,%d,%d,%d", &idx, &gx, &gy, &gz, &ax,
                 &ay, &az) == 7 &&
          idx >= 0 && idx < NUM_FINGER_IMUS) {
        gyroBias[idx][0] = gx;
        gyroBias[idx][1] = gy;
        gyroBias[idx][2] = gz;
        accelOff[idx][0] = ax;
        accelOff[idx][1] = ay;
        accelOff[idx][2] = az;
        Serial.printf("# offsets set for imu%d\n", idx);
      }
    }
  }
}

// ============================================================================
// Setup / loop
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(300);
  analogReadResolution(12);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);
  bnoPresent = bnoInit();

  imuArrayInit();
  loadCalFromNVS();

  Serial.printf("# bno=%d", bnoPresent ? 1 : 0);
  for (uint8_t i = 0; i < NUM_FINGER_IMUS; i++)
    Serial.printf(" imu%u=%d", i, imuPresent[i] ? 1 : 0);
  Serial.println();
  printHeader();

  lastMicros = micros();
}

void loop() {
  handleCommands();

  uint32_t now = micros();
  if (now - lastMicros < FRAME_US) return;
  lastMicros = now;

  uint32_t t_us = micros();

  if (bnoPresent) {
    bnoReadQuat();
    bnoCal = bnoRead8(BNO_CALIB_STAT);
  }

  String line = String(t_us) + "," + String(bnoCal) + "," + String(bqw, 4) +
                "," + String(bqx, 4) + "," + String(bqy, 4) + "," +
                String(bqz, 4);

  for (uint8_t i = 0; i < NUM_FINGER_IMUS; i++) {
    if (imuPresent[i]) {
      int16_t ax, ay, az, gx, gy, gz;
      fingerImu[i]->readRaw(ax, ay, az, gx, gy, gz);
      if (!rawMode) {
        gx -= gyroBias[i][0];
        gy -= gyroBias[i][1];
        gz -= gyroBias[i][2];
        ax -= accelOff[i][0];
        ay -= accelOff[i][1];
        az -= accelOff[i][2];
      }
      line += ",1," + String(ax) + "," + String(ay) + "," + String(az) + "," +
              String(gx) + "," + String(gy) + "," + String(gz);
    } else {
      line += ",0,0,0,0,0,0,0";
    }
  }

  for (uint8_t m = 0; m < NUM_FORCE; m++)
    line += "," + String(analogRead(FORCE_PINS[m]));

  Serial.println(line);
}

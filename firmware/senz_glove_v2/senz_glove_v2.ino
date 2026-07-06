/*
 * senz_glove_v2
 * -------------
 * Hardware Sprint v2 firmware for the senz finger-tracking glove.
 * HLD: senz_glove_hld_v2  |  branch: feature/multi-imu-expansion
 *
 * Board : ESP32-S3-DevKitC-1 (dual-core LX7 @ 240 MHz, BLE 5.0 onboard)
 * Sensors:
 *   - BNO055        wrist reference IMU  (I2C, on-chip 9-axis fusion)
 *   - 10x MPU-6500  per-finger IMUs      (SPI, ONE dedicated CS GPIO each)
 *   - SSD1306 OLED  status display       (I2C, shared bus with BNO055)
 *
 * Transport: USB serial, 180-byte binary frame, always on (see FRAME below).
 *   [0xAA][u32 t_ms][4f bno_quat][10 * 4f mpu_quat][0xFF]  = 182 bytes on wire
 *   Payload (between the markers) is 180 bytes: 4 + 16 + 160.
 *
 * SCOPE OF THIS FILE:
 *   SPI/I2C bring-up, 10x WHO_AM_I sweep, 200 Hz raw polling, gyro-bias
 *   calibration, BNO055 read + NVS cal load, OLED status, loop-timing profiler,
 *   binary frame packer, AND per-finger Madgwick fusion + finger-relative-to-
 *   wrist quaternion composition (see orientationOf() and madgwick.h).
 *
 *   NOT here (supervised) -- serial is the validated fallback transport:
 *     - BLE MTU/conn-param/PHY transport (not started)
 *
 * Serial command protocol (host -> device, newline-terminated):
 *   ?   re-emit the human-readable boot banner + sensor status
 *   C   calibrate gyro bias of all present IMUs (hold still), save to flash
 *   K   capture current BNO055 calibration offsets to flash (do when cal=3/3/3/3)
 *   D   print ONE human-readable raw accel/gyro line for every IMU (bring-up)
 *   Z   zero all gyro-bias offsets in RAM
 */

#include <Preferences.h>
#include <SPI.h>
#include <Wire.h>

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "madgwick.h"
#include "mpu6500.h"

// ============================================================================
// CONFIG -- ESP32-S3-DevKitC-1 pin map (HLD "Pin Layout" table). CS lines use
// mid-range GPIOs only: GPIO1/2 are intentionally avoided (boot-adjacent).
// ============================================================================

// --- I2C: BNO055 wrist IMU + SSD1306 OLED (shared bus) ---
static const int I2C_SDA = 8;
static const int I2C_SCL = 9;
static const uint8_t BNO_ADDR = 0x28;   // AD0 -> GND
static const uint8_t OLED_ADDR = 0x3C;
static const int OLED_W = 128, OLED_H = 64;

// --- SPI: shared bus to all 10 MPU-6500s ---
static const int SPI_MOSI = 11;  // SDI on every sensor
static const int SPI_SCLK = 12;  // SCLK
static const int SPI_MISO = 13;  // SDO
static const uint32_t SPI_HZ = 8000000;  // 8 MHz start; step toward 20 MHz later

// --- Per-finger MPU-6500 chip selects (HLD finger-placement table) ---
static const uint8_t NUM_IMU = 10;
static const int CS_PIN[NUM_IMU] = {4, 5, 6, 7, 14, 15, 16, 17, 18, 21};
static const char *IMU_LABEL[NUM_IMU] = {
    "idx-prox", "idx-dist", "mid-prox", "mid-dist", "ring-prox",
    "ring-dist", "pnk-prox", "pnk-dist", "thmb-base", "thmb-tip"};

// --- Rates ---
static const uint32_t SAMPLE_HZ = 200;                     // per-finger poll
static const uint32_t FRAME_US = 1000000UL / SAMPLE_HZ;    // 5000 us
static const uint8_t SMPLRT_DIV = 4;                       // 1kHz/(1+4)=200Hz
static const uint32_t BNO_EVERY = 2;  // read wrist every 2nd loop -> ~100 Hz
static const uint32_t LOOP_WARN_US = 5000;  // warn if a loop exceeds the budget

// ============================================================================
// BNO055 registers (minimal: quaternion + calibration, NDOF mode). Matches the
// raw-I2C approach from the v1 multi-IMU sketch -- no Adafruit_BNO055 needed.
// ============================================================================
static const uint8_t BNO_CHIP_ID = 0x00;   // reads 0xA0
static const uint8_t BNO_CALIB_STAT = 0x35;
static const uint8_t BNO_OPR_MODE = 0x3D;
static const uint8_t BNO_SYS_TRIGGER = 0x3F;
static const uint8_t BNO_PWR_MODE = 0x3E;
static const uint8_t BNO_QUA_DATA_W_LSB = 0x20;  // 8 bytes
static const uint8_t BNO_CAL_DATA = 0x55;        // 22 bytes of offsets/radii
static const uint8_t BNO_OPR_CONFIG = 0x00;
static const uint8_t BNO_OPR_NDOF = 0x0C;
static const float BNO_QUA_LSB = 16384.0f;
static const uint8_t BNO_CAL_LEN = 22;

// ============================================================================
// State
// ============================================================================
SPIClass &spiBus = SPI;
Adafruit_SSD1306 oled(OLED_W, OLED_H, &Wire, -1);

Mpu6500 *imu[NUM_IMU] = {nullptr};
bool imuPresent[NUM_IMU] = {false};
int16_t gyroBias[NUM_IMU][3] = {{0}};   // LSB, subtracted before fusion
uint8_t imuCount = 0;

// Per-finger Madgwick filters (HLD beta=0.1). One orientation state each.
MadgwickImu fusion[NUM_IMU];
static const float MADGWICK_BETA = 0.1f;
static const float DEG2RAD = 0.01745329252f;
// raw gyro LSB -> rad/s: /16.4 LSB-per-dps then *deg2rad
static const float GYRO_RAW_TO_RAD = DEG2RAD / Mpu6500::GYR_LSB_PER_DPS;

bool bnoPresent = false, oledPresent = false;
float bqw = 1, bqx = 0, bqy = 0, bqz = 0;
uint8_t bnoCal = 0;

uint32_t loopCounter = 0;
uint32_t loopMaxUs = 0, loopOverruns = 0;

Preferences prefs;
static const char *NVS_NS = "senz-v2";
static const char *NVS_GYRO = "gyrobias";  // NUM_IMU * 3 int16
static const char *NVS_BNOCAL = "bnocal";   // 22 bytes

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

void bnoReadBytes(uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire.beginTransmission(BNO_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((int)BNO_ADDR, (int)n);
  for (uint8_t i = 0; i < n && Wire.available(); i++) buf[i] = Wire.read();
}

// Write saved calibration offsets back to the BNO055. Must be in CONFIG mode.
void bnoApplyCal(const uint8_t *cal) {
  bnoWrite(BNO_OPR_MODE, BNO_OPR_CONFIG);
  delay(25);
  for (uint8_t i = 0; i < BNO_CAL_LEN; i++) bnoWrite(BNO_CAL_DATA + i, cal[i]);
  bnoWrite(BNO_OPR_MODE, BNO_OPR_NDOF);
  delay(25);
}

bool bnoInit() {
  Wire.beginTransmission(BNO_ADDR);
  if (Wire.endTransmission() != 0) return false;
  if (bnoRead8(BNO_CHIP_ID) != 0xA0) return false;  // 0xA0 = BNO055 chip id
  bnoWrite(BNO_OPR_MODE, BNO_OPR_CONFIG);
  delay(25);
  bnoWrite(BNO_PWR_MODE, 0x00);  // normal power
  delay(10);
  bnoWrite(BNO_SYS_TRIGGER, 0x00);
  delay(10);

  // Load saved calibration offsets from flash, if present (HLD boot step 1).
  uint8_t cal[BNO_CAL_LEN];
  prefs.begin(NVS_NS, true);
  bool haveCal = prefs.getBytesLength(NVS_BNOCAL) == BNO_CAL_LEN &&
                 prefs.getBytes(NVS_BNOCAL, cal, BNO_CAL_LEN) == BNO_CAL_LEN;
  prefs.end();

  bnoWrite(BNO_OPR_MODE, BNO_OPR_NDOF);
  delay(25);
  if (haveCal) {
    bnoApplyCal(cal);
    Serial.println("# BNO055 calibration loaded from flash");
  } else {
    Serial.println("# no BNO055 calibration in flash; will auto-cal (move it)");
  }
  return true;
}

void bnoReadQuat() {
  uint8_t b[8];
  bnoReadBytes(BNO_QUA_DATA_W_LSB, b, 8);
  int16_t w = (int16_t)(b[0] | (b[1] << 8));
  int16_t x = (int16_t)(b[2] | (b[3] << 8));
  int16_t y = (int16_t)(b[4] | (b[5] << 8));
  int16_t z = (int16_t)(b[6] | (b[7] << 8));
  bqw = w / BNO_QUA_LSB;
  bqx = x / BNO_QUA_LSB;
  bqy = y / BNO_QUA_LSB;
  bqz = z / BNO_QUA_LSB;
}

// Capture the BNO055's current fusion calibration to flash. Do this once the
// OLED reports cal 3/3/3/3 (system/gyro/accel/mag all fully calibrated).
void bnoSaveCal() {
  if (!bnoPresent) return;
  uint8_t cal[BNO_CAL_LEN];
  bnoWrite(BNO_OPR_MODE, BNO_OPR_CONFIG);
  delay(25);
  bnoReadBytes(BNO_CAL_DATA, cal, BNO_CAL_LEN);
  bnoWrite(BNO_OPR_MODE, BNO_OPR_NDOF);
  delay(25);
  prefs.begin(NVS_NS, false);
  prefs.putBytes(NVS_BNOCAL, cal, BNO_CAL_LEN);
  prefs.end();
  Serial.println("# BNO055 calibration saved to flash");
}

// ============================================================================
// MPU-6500 array (SPI, direct CS per sensor)
// ============================================================================
void imuArrayInit() {
  spiBus.begin(SPI_SCLK, SPI_MISO, SPI_MOSI, -1);  // CS handled per-driver
  imuCount = 0;
  for (uint8_t i = 0; i < NUM_IMU; i++) {
    imu[i] = new Mpu6500(spiBus, CS_PIN[i], SPI_HZ);
    imu[i]->initCs();
  }
  // Deselect-then-probe each sensor: read WHO_AM_I (expect 0x70), flag dead.
  for (uint8_t i = 0; i < NUM_IMU; i++) {
    imuPresent[i] = imu[i]->begin();
    if (imuPresent[i]) {
      imu[i]->setSampleRateDiv(SMPLRT_DIV);  // 200 Hz
      imuCount++;
    }
    Serial.printf("# imu%u (%s, CS=GPIO%d) who=0x%02X -> %s\n", i, IMU_LABEL[i],
                  CS_PIN[i], imu[i]->whoAmI(), imuPresent[i] ? "OK" : "DEAD");
  }
}

// ============================================================================
// Gyro-bias calibration (average at rest). Keep the glove still.
// ============================================================================
void loadGyroBias() {
  prefs.begin(NVS_NS, true);
  size_t want = sizeof(gyroBias);
  if (prefs.getBytesLength(NVS_GYRO) == want) {
    prefs.getBytes(NVS_GYRO, gyroBias, want);
    Serial.println("# gyro bias loaded from flash");
  } else {
    Serial.println("# no gyro bias in flash; run 'C' to calibrate");
  }
  prefs.end();
}

void calibrateGyroBias() {
  const int N = 600;  // ~3 s at 200 Hz (HLD boot step 6)
  Serial.println("# gyro bias: hold still ~3s...");
  showStatus("CAL GYRO", "hold still");
  long sum[NUM_IMU][3] = {{0}};
  for (int s = 0; s < N; s++) {
    for (uint8_t i = 0; i < NUM_IMU; i++) {
      if (!imuPresent[i]) continue;
      int16_t ax, ay, az, gx, gy, gz;
      imu[i]->readRaw(ax, ay, az, gx, gy, gz);
      sum[i][0] += gx;
      sum[i][1] += gy;
      sum[i][2] += gz;
    }
    delay(5);
  }
  for (uint8_t i = 0; i < NUM_IMU; i++) {
    if (!imuPresent[i]) continue;
    gyroBias[i][0] = (int16_t)(sum[i][0] / N);
    gyroBias[i][1] = (int16_t)(sum[i][1] / N);
    gyroBias[i][2] = (int16_t)(sum[i][2] / N);
  }
  prefs.begin(NVS_NS, false);
  prefs.putBytes(NVS_GYRO, gyroBias, sizeof(gyroBias));
  prefs.end();
  Serial.println("# gyro bias saved to flash");
}

// ============================================================================
// Orientation -- per-finger Madgwick fusion + wrist-relative composition.
// ----------------------------------------------------------------------------
// Runs this finger's 6-axis Madgwick filter on the latest sample, then expresses
// the orientation RELATIVE to the BNO055 wrist frame so that rotating the whole
// hand rigidly does NOT move the fingers:
//     q_rel = conj(q_wrist) * q_finger      (host recovers q_wrist * q_rel)
// gx/gy/gz are bias-corrected raw gyro LSB (converted to rad/s here); ax/ay/az
// are raw accel LSB (Madgwick normalizes, so scale is irrelevant). dt in seconds.
// ============================================================================
void orientationOf(uint8_t i, int16_t gx, int16_t gy, int16_t gz, int16_t ax,
                   int16_t ay, int16_t az, float dt, float qRel[4]) {
  fusion[i].update(gx * GYRO_RAW_TO_RAD, gy * GYRO_RAW_TO_RAD,
                   gz * GYRO_RAW_TO_RAD, (float)ax, (float)ay, (float)az, dt);
  float qFinger[4];
  fusion[i].quat(qFinger);
  const float qWrist[4] = {bqw, bqx, bqy, bqz};
  float qWristConj[4];
  quatConj(qWrist, qWristConj);
  quatMul(qWristConj, qFinger, qRel);
}

// ============================================================================
// 180-byte binary frame: [0xAA][u32 t_ms][bno 4f][mpu 10*4f][0xFF]
// ============================================================================
void writeFrame(uint32_t t_ms, const float mpuQuat[NUM_IMU][4]) {
  uint8_t frame[182];
  size_t p = 0;
  frame[p++] = 0xAA;
  memcpy(&frame[p], &t_ms, 4);            p += 4;
  float bno[4] = {bqw, bqx, bqy, bqz};
  memcpy(&frame[p], bno, 16);             p += 16;
  memcpy(&frame[p], mpuQuat, 4 * 4 * NUM_IMU);  // 160 bytes, contiguous
  p += 4 * 4 * NUM_IMU;
  frame[p++] = 0xFF;
  Serial.write(frame, p);  // p == 182
}

// ============================================================================
// OLED status. Refreshing the SSD1306 costs a few ms of I2C, so this is called
// only outside the tight streaming path (boot, commands, ~1 Hz heartbeat).
// ============================================================================
void showStatus(const char *l1, const char *l2) {
  if (!oledPresent) return;
  oled.clearDisplay();
  oled.setCursor(0, 0);
  oled.println("SENZ v2");
  oled.println(l1);
  if (l2) oled.println(l2);
  oled.display();
}

void showHeartbeat() {
  if (!oledPresent) return;
  char l1[24], l2[24];
  snprintf(l1, sizeof(l1), "IMU %u/10  cal%u", imuCount, bnoCal & 0x03);
  snprintf(l2, sizeof(l2), "max %luus ov%lu", (unsigned long)loopMaxUs,
           (unsigned long)loopOverruns);
  showStatus(l1, l2);
}

// ============================================================================
// Bring-up helpers
// ============================================================================
void printBanner() {
  Serial.printf("# senz-v2 imu=%u/%u bno=%d oled=%d rate=%lu frame=180B\n",
                imuCount, NUM_IMU, bnoPresent ? 1 : 0, oledPresent ? 1 : 0,
                (unsigned long)SAMPLE_HZ);
}

// Human-readable one-shot dump of every IMU (bring-up validation, 'D' command).
void debugDump() {
  Serial.println("# --- raw dump (bias-corrected gyro) ---");
  for (uint8_t i = 0; i < NUM_IMU; i++) {
    if (!imuPresent[i]) {
      Serial.printf("imu%u %-9s DEAD\n", i, IMU_LABEL[i]);
      continue;
    }
    int16_t ax, ay, az, gx, gy, gz;
    imu[i]->readRaw(ax, ay, az, gx, gy, gz);
    gx -= gyroBias[i][0];
    gy -= gyroBias[i][1];
    gz -= gyroBias[i][2];
    Serial.printf("imu%u %-9s a=%6d,%6d,%6d g=%6d,%6d,%6d\n", i, IMU_LABEL[i],
                  ax, ay, az, gx, gy, gz);
  }
}

void handleCommands() {
  while (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) continue;
    char c = cmd.charAt(0);
    if (c == '?') {
      printBanner();
    } else if (c == 'C') {
      calibrateGyroBias();
    } else if (c == 'K') {
      bnoSaveCal();
    } else if (c == 'D') {
      debugDump();
    } else if (c == 'Z') {
      memset(gyroBias, 0, sizeof(gyroBias));
      Serial.println("# gyro bias zeroed (RAM)");
    }
  }
}

// ============================================================================
// Setup / loop
// ============================================================================
void setup() {
  Serial.begin(921600);  // matches host senz_parser.py
  delay(300);

  // I2C first: BNO055 + OLED share the bus (HLD boot steps 1-2).
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  oledPresent = oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (oledPresent) {
    oled.setTextSize(1);
    oled.setTextColor(SSD1306_WHITE);
    showStatus("booting...", nullptr);
  }

  bnoPresent = bnoInit();

  // SPI: bring up the finger array, WHO_AM_I sweep, set 200 Hz (boot steps 3-5).
  imuArrayInit();
  loadGyroBias();
  for (uint8_t i = 0; i < NUM_IMU; i++) fusion[i].setBeta(MADGWICK_BETA);

  printBanner();
  showHeartbeat();

  // HLD boot step 7. BLE advertising (step 8) is the supervised max-effort task
  // and is intentionally NOT started here -- serial is the validated transport.
  Serial.println("SERIAL READY");
  Serial.println("# BLE deferred (supervised); streaming 180-byte frames now");
}

void loop() {
  handleCommands();

  uint32_t now = micros();
  static uint32_t lastFrame = 0;
  if (now - lastFrame < FRAME_US) return;
  lastFrame = now;

  // Actual elapsed time for the Madgwick integration (robust to loop slip).
  static uint32_t lastFusionUs = 0;
  float dt = (lastFusionUs == 0) ? (1.0f / SAMPLE_HZ)
                                 : (now - lastFusionUs) * 1e-6f;
  lastFusionUs = now;
  if (dt <= 0.0f || dt > 0.05f) dt = 1.0f / SAMPLE_HZ;  // clamp glitches

  uint32_t t0 = micros();
  uint32_t t_ms = millis();

  // 1-4. Poll all 10 finger IMUs, bias-correct gyro, run Madgwick fusion, and
  // express each finger orientation relative to the wrist (see orientationOf).
  float mpuQuat[NUM_IMU][4];
  for (uint8_t i = 0; i < NUM_IMU; i++) {
    if (!imuPresent[i]) {
      mpuQuat[i][0] = 1; mpuQuat[i][1] = 0;
      mpuQuat[i][2] = 0; mpuQuat[i][3] = 0;
      continue;
    }
    int16_t ax, ay, az, gx, gy, gz;
    imu[i]->readRaw(ax, ay, az, gx, gy, gz);
    gx -= gyroBias[i][0];
    gy -= gyroBias[i][1];
    gz -= gyroBias[i][2];
    orientationOf(i, gx, gy, gz, ax, ay, az, dt, mpuQuat[i]);
  }

  // 5. Wrist reference at ~100 Hz (every 2nd loop). NOTE: the HLD text says
  // "every 10th loop" but the loop runs at 200 Hz, so every 2nd loop is the
  // rate that actually yields ~100 Hz. Flagged in the doc review.
  if (bnoPresent && (loopCounter % BNO_EVERY) == 0) {
    bnoReadQuat();
    bnoCal = bnoRead8(BNO_CALIB_STAT);
  }

  // 6. Fingers already expressed relative to the wrist frame in orientationOf().
  // 7-8. Pack and stream the binary frame (always on).
  writeFrame(t_ms, mpuQuat);

  // 9. (deferred) BLE notify -- supervised.
  // 10. Loop-timing profiler.
  uint32_t dt = micros() - t0;
  if (dt > loopMaxUs) loopMaxUs = dt;
  if (dt > LOOP_WARN_US) {
    loopOverruns++;
    Serial.printf("# WARN loop %luus > %luus budget\n", (unsigned long)dt,
                  (unsigned long)LOOP_WARN_US);
  }

  // ~1 Hz OLED heartbeat (kept off the hot path; costs a few ms when it fires).
  if ((loopCounter % SAMPLE_HZ) == 0) showHeartbeat();
  loopCounter++;
}

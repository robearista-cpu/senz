/*
 * senz_glove_v3_pinch
 * -------------------
 * v3 PINCH build firmware: the index/middle/thumb glove focused on PINCHING
 * gestures for ML. Same IMU set as senz_glove_v3_proto, but the force array is
 * just the THREE fingertip 2x2 pads (no palm taxels) -> nforce=12.
 * Board: ESP32-S3-DevKitC-1.  Pin layout: docs/PINOUT_v3_pinch.txt.
 *
 * SELF-CONTAINED: needs only the ESP32 Arduino core (SPI.h / Wire.h /
 * Preferences.h). No Adafruit libraries, no project headers -- drop this folder
 * into your sketchbook and flash. The MPU + BNO055 drivers are inlined here,
 * ported from the validated firmware/senz_glove_v2 code.
 *
 * Sensors:
 *   - 8x IMU (SPI, one CS each):
 *        0 index-prox  GPIO4    1 index-dist  GPIO5
 *        2 mid-prox    GPIO14   3 mid-dist    GPIO17
 *        4 thumb-base  GPIO18   5 thumb-tip   GPIO21   (MPU-6500, WHO=0x70)
 *        6 thumb-meta  GPIO39   base MCP (MPU-9250, WHO=0x71, 9-axis; mag not read)
 *        7 hand-dorsum GPIO38   back-of-hand / palm frame (6-axis)
 *      thumb = 3 IMUs (2x 6-axis + 1x 9-axis base), index = 2, middle = 2, dorsum = 1.
 *      Driver accepts BOTH 0x70 and 0x71, so 6500 + 9250 mix on one bus.
 *   - 1x BNO055 wrist IMU (I2C 0x28, on-chip NDOF fusion -> quaternion)
 *   - 12x Velostat force taxels via CD74HC4067 (SIG=GPIO10 ADC1, S0..S3=6,7,15,16):
 *        thumb 2x2 -> C0..C3, index 2x2 -> C4..C7, middle 2x2 -> C8..C11
 *      (three fingertip pinch pads; NO palm pads -- that is the tactile build).
 *
 * Output: the SAME self-describing CSV over BOTH USB serial (@921600) and
 * Bluetooth LE (Nordic UART Service). Consumable by host/senz_multi_io.py (USB,
 * learns the schema from the banner), host/senz_ble_io.py (BLE, same parser), and
 * host/force_pipeline.py (turns forceN into relative grip); host/pinch.py turns
 * the fingertip pads + posed hand into pinch features.
 *
 *   # senz-v3pinch nimu=8 nforce=12 rate=200
 *   # columns: t_us,bno_qw,bno_qx,bno_qy,bno_qz,imu0_ax,...,imu7_gz,force0..force11
 *   <data lines>   (accel = raw int16, gyro = bias-corrected int16, force = 0..4095)
 *
 * TRANSPORT: USB is preferred. At boot the firmware waits briefly for a USB host
 * to open the CDC port; if one is connected it logs "USB" as primary. BLE is
 * ALWAYS advertised too, so if USB is absent (running on a battery) a BLE central
 * can connect and gets the identical stream. Frames go to USB at the full rate and
 * to BLE at a decimated rate (BLE_DECIM) to fit BLE bandwidth. Both stay live, so
 * it degrades gracefully rather than hard-switching.
 *
 * Commands (host -> device, newline-terminated) work over EITHER USB or the BLE
 * RX characteristic:
 *   ?   re-emit the banner + column header (+ USB-only per-sensor status)
 *   C   calibrate gyro bias of all present IMUs (HOLD STILL ~1s), save to flash
 *   Z   zero all gyro-bias offsets in RAM
 *   D   print ONE human-readable raw line per IMU + all taxels (bring-up debug)
 */

#include <Preferences.h>
#include <SPI.h>
#include <Wire.h>

// BLE (Nordic UART Service). Built-in ESP32 Bluedroid stack -- no external lib.
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

// ============================================================================
// CONFIG -- see docs/PINOUT_v3_pinch.txt (single source of truth for pins)
// ============================================================================

// --- I2C: BNO055 wrist ---
static const int I2C_SDA = 8;
static const int I2C_SCL = 9;
static const uint8_t BNO_ADDR = 0x28;   // AD0 -> GND

// --- SPI: shared bus to all finger IMUs ---
static const int SPI_MOSI = 11;   // SDA/SDI on every sensor
static const int SPI_SCLK = 12;   // SCL/SCLK
static const int SPI_MISO = 13;   // AD0/SDO
static const uint32_t SPI_HZ = 8000000;   // 8 MHz start; rated to 20 MHz later

// --- Finger IMUs (SPI, one CS each) ---
static const uint8_t NUM_IMU = 8;
static const int CS_PIN[NUM_IMU]       = {4, 5, 14, 17, 18, 21, 39, 38};
static const char *IMU_LABEL[NUM_IMU]  = {"idx-prox", "idx-dist", "mid-prox",
                                          "mid-dist", "thmb-base", "thmb-tip",
                                          "thmb-meta", "hand-dorsum"};
static const uint8_t IMU_EXPECT[NUM_IMU] = {0x70, 0x70, 0x70, 0x70, 0x70,
                                            0x70, 0x71, 0x70};

// --- Velostat force array (CD74HC4067): three fingertip 2x2 pads only ---
static const uint8_t NUM_FORCE = 12;        // C0..C3 thumb, C4..C7 index, C8..C11 middle
static const int MUX_SEL[4] = {6, 7, 15, 16};   // S0,S1,S2,S3
static const int MUX_SIG = 10;                  // COM/SIG -> ADC1_CH9
static const uint8_t FORCE_AVG = 4;             // ADC samples averaged per taxel

// --- Rates ---
static const uint32_t SAMPLE_HZ = 200;
static const uint32_t FRAME_US = 1000000UL / SAMPLE_HZ;   // 5000 us
static const uint8_t SMPLRT_DIV = 4;                      // 1kHz/(1+4)=200Hz
static const uint32_t BNO_EVERY = 2;   // read wrist every 2nd loop -> ~100 Hz

// --- BLE (Nordic UART Service) — UUIDs MUST match host/senz_io.py ---
static const char *BLE_NAME = "senz-pinch";
#define NUS_SERVICE_UUID "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_RX_UUID      "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  // host -> device (write)
#define NUS_TX_UUID      "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  // device -> host (notify)
// USB streams every frame; BLE streams every BLE_DECIM-th frame (200Hz/4 = 50Hz)
// so the row rate fits BLE bandwidth. Raise to 1 for full rate if your link allows.
static const uint8_t BLE_DECIM = 4;
static const uint16_t BLE_CHUNK = 180;   // notify payload chunk (needs MTU >= ~185)
static const uint32_t USB_WAIT_MS = 1200;   // wait this long at boot for a USB host

// ============================================================================
// Inline MPU-6500 / MPU-9250 SPI driver (ported from senz_glove_v2/mpu6500.h,
// WHO_AM_I check widened to accept 0x70 OR 0x71).
// ============================================================================
class MpuSpi {
 public:
  static constexpr float ACC_LSB_PER_G = 4096.0f;    // +/-8 g
  static constexpr float GYR_LSB_PER_DPS = 16.4f;    // +/-2000 dps

  MpuSpi(SPIClass &spi, int csPin, uint32_t hz = SPI_HZ)
      : spi_(spi), cs_(csPin), settings_(hz, MSBFIRST, SPI_MODE3) {}

  void initCs() {
    pinMode(cs_, OUTPUT);
    digitalWrite(cs_, HIGH);
  }

  // Reset, force SPI-only, verify WHO_AM_I (0x70 or 0x71), set ranges + DLPF.
  bool begin() {
    writeReg(REG_PWR_MGMT_1, 0x80);   // device reset
    delay(100);
    writeReg(REG_PWR_MGMT_1, 0x01);   // wake, auto clock (PLL)
    delay(10);
    writeReg(REG_USER_CTRL, 0x10);    // I2C_IF_DIS: SPI only
    delay(1);

    uint8_t who = whoAmI();
    if (who != 0x70 && who != 0x71) return false;   // 6500=0x70, 9250=0x71

    writeReg(REG_GYRO_CONFIG, 0x18);    // FS_SEL=3  -> +/-2000 dps
    writeReg(REG_ACCEL_CONFIG, 0x10);   // AFS_SEL=2 -> +/-8 g
    writeReg(REG_CONFIG, 0x03);         // gyro DLPF 41 Hz -> 1 kHz internal
    writeReg(REG_ACCEL_CONFIG2, 0x03);  // accel DLPF 41 Hz
    writeReg(REG_SMPLRT_DIV, SMPLRT_DIV);
    delayMicroseconds(250);
    return true;
  }

  uint8_t whoAmI() { return readReg(REG_WHO_AM_I); }

  void readRaw(int16_t &ax, int16_t &ay, int16_t &az, int16_t &gx, int16_t &gy,
               int16_t &gz) {
    uint8_t b[14];
    readRegs(REG_ACCEL_XOUT_H, b, 14);
    ax = (int16_t)((b[0] << 8) | b[1]);
    ay = (int16_t)((b[2] << 8) | b[3]);
    az = (int16_t)((b[4] << 8) | b[5]);
    // b[6..7] = temperature, skipped
    gx = (int16_t)((b[8] << 8) | b[9]);
    gy = (int16_t)((b[10] << 8) | b[11]);
    gz = (int16_t)((b[12] << 8) | b[13]);
  }

  void writeReg(uint8_t reg, uint8_t val) {
    spi_.beginTransaction(settings_);
    digitalWrite(cs_, LOW);
    spi_.transfer(reg & 0x7F);
    spi_.transfer(val);
    digitalWrite(cs_, HIGH);
    spi_.endTransaction();
  }

  uint8_t readReg(uint8_t reg) {
    spi_.beginTransaction(settings_);
    digitalWrite(cs_, LOW);
    spi_.transfer(reg | 0x80);
    uint8_t v = spi_.transfer(0x00);
    digitalWrite(cs_, HIGH);
    spi_.endTransaction();
    return v;
  }

  void readRegs(uint8_t reg, uint8_t *buf, size_t n) {
    spi_.beginTransaction(settings_);
    digitalWrite(cs_, LOW);
    spi_.transfer(reg | 0x80);
    for (size_t i = 0; i < n; i++) buf[i] = spi_.transfer(0x00);
    digitalWrite(cs_, HIGH);
    spi_.endTransaction();
  }

 private:
  static constexpr uint8_t REG_SMPLRT_DIV = 0x19;
  static constexpr uint8_t REG_CONFIG = 0x1A;
  static constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
  static constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
  static constexpr uint8_t REG_ACCEL_CONFIG2 = 0x1D;
  static constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
  static constexpr uint8_t REG_USER_CTRL = 0x6A;
  static constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
  static constexpr uint8_t REG_WHO_AM_I = 0x75;

  SPIClass &spi_;
  int cs_;
  SPISettings settings_;
};

// ============================================================================
// BNO055 (minimal raw-I2C, NDOF quaternion) -- ported from senz_glove_v2
// ============================================================================
static const uint8_t BNO_CHIP_ID = 0x00;       // reads 0xA0
static const uint8_t BNO_OPR_MODE = 0x3D;
static const uint8_t BNO_SYS_TRIGGER = 0x3F;
static const uint8_t BNO_PWR_MODE = 0x3E;
static const uint8_t BNO_QUA_DATA_W_LSB = 0x20;   // 8 bytes
static const uint8_t BNO_OPR_CONFIG = 0x00;
static const uint8_t BNO_OPR_NDOF = 0x0C;
static const float BNO_QUA_LSB = 16384.0f;

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

bool bnoInit() {
  Wire.beginTransmission(BNO_ADDR);
  if (Wire.endTransmission() != 0) return false;
  if (bnoRead8(BNO_CHIP_ID) != 0xA0) return false;
  bnoWrite(BNO_OPR_MODE, BNO_OPR_CONFIG);
  delay(25);
  bnoWrite(BNO_PWR_MODE, 0x00);       // normal power
  delay(10);
  bnoWrite(BNO_SYS_TRIGGER, 0x00);
  delay(10);
  bnoWrite(BNO_OPR_MODE, BNO_OPR_NDOF);
  delay(25);
  return true;
}

// ============================================================================
// State
// ============================================================================
SPIClass &spiBus = SPI;

MpuSpi *imu[NUM_IMU] = {nullptr};
bool imuPresent[NUM_IMU] = {false};
int16_t gyroBias[NUM_IMU][3] = {{0}};

bool bnoPresent = false;
float bqw = 1, bqx = 0, bqy = 0, bqz = 0;

uint16_t force[NUM_FORCE] = {0};
uint32_t frameCount = 0;

Preferences prefs;
static const char *NVS_NS = "senz-v3pk";
static const char *NVS_GYRO = "gyrobias";     // NUM_IMU * 3 int16

// --- BLE state ---
BLEServer *bleServer = nullptr;
BLECharacteristic *bleTx = nullptr;      // notify (device -> host)
volatile bool bleConnected = false;      // a central is connected
volatile bool usbPrimary = false;        // a USB host opened the CDC port at boot

void handleChar(char c);                 // fwd decl (shared USB + BLE command handler)

// ============================================================================
// Velostat scan (CD74HC4067)
// ============================================================================
uint16_t readTaxel(uint8_t ch) {
  for (uint8_t b = 0; b < 4; b++) digitalWrite(MUX_SEL[b], (ch >> b) & 1);
  delayMicroseconds(5);                 // let mux + divider node settle
  uint32_t acc = 0;
  for (uint8_t i = 0; i < FORCE_AVG; i++) acc += analogRead(MUX_SIG);
  return (uint16_t)(acc / FORCE_AVG);
}

void scanForce() {
  for (uint8_t ch = 0; ch < NUM_FORCE; ch++) force[ch] = readTaxel(ch);
}

// ============================================================================
// Calibration (gyro bias -> NVS)
// ============================================================================
void loadGyroBias() {
  prefs.begin(NVS_NS, true);
  size_t need = sizeof(int16_t) * NUM_IMU * 3;
  if (prefs.getBytesLength(NVS_GYRO) == need)
    prefs.getBytes(NVS_GYRO, gyroBias, need);
  prefs.end();
}

void saveGyroBias() {
  prefs.begin(NVS_NS, false);
  prefs.putBytes(NVS_GYRO, gyroBias, sizeof(int16_t) * NUM_IMU * 3);
  prefs.end();
}

void calibrateGyro() {
  const int N = 200;
  long acc[NUM_IMU][3] = {{0}};
  Serial.println("# calibrating gyro bias -- HOLD STILL");
  for (int s = 0; s < N; s++) {
    for (uint8_t i = 0; i < NUM_IMU; i++) {
      if (!imuPresent[i]) continue;
      int16_t ax, ay, az, gx, gy, gz;
      imu[i]->readRaw(ax, ay, az, gx, gy, gz);
      acc[i][0] += gx; acc[i][1] += gy; acc[i][2] += gz;
    }
    delay(5);
  }
  for (uint8_t i = 0; i < NUM_IMU; i++)
    for (uint8_t k = 0; k < 3; k++) gyroBias[i][k] = (int16_t)(acc[i][k] / N);
  saveGyroBias();
  Serial.println("# gyro bias saved to flash");
}

// ============================================================================
// BLE (Nordic UART Service) — advertise, notify, receive commands
// ============================================================================
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *s) override { bleConnected = true; }
  void onDisconnect(BLEServer *s) override {
    bleConnected = false;
    s->getAdvertising()->start();       // re-advertise so a client can reconnect
  }
};

// Host -> device writes land here; feed them into the shared command handler,
// line-buffered so a "?\n" over BLE behaves like it does over USB.
class RxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    // getData()/getLength() are stable across ESP32 core 2.x/3.x (unlike
    // getValue(), whose return type changed std::string -> String).
    uint8_t *d = c->getData();
    size_t len = c->getLength();
    for (size_t i = 0; i < len; i++) {
      char ch = (char)d[i];
      if (ch == '\n' || ch == '\r') continue;
      handleChar(ch);
    }
  }
};

void bleInit() {
  BLEDevice::init(BLE_NAME);
  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new ServerCallbacks());
  BLEService *svc = bleServer->createService(NUS_SERVICE_UUID);

  bleTx = svc->createCharacteristic(NUS_TX_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  bleTx->addDescriptor(new BLE2902());   // CCCD so centrals can subscribe

  BLECharacteristic *rx = svc->createCharacteristic(
      NUS_RX_UUID,
      BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  rx->setCallbacks(new RxCallbacks());

  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE_UUID);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
}

// Notify a buffer over BLE in <=BLE_CHUNK pieces (the host reassembles by '\n').
void bleNotify(const char *buf, size_t len) {
  if (!bleConnected || bleTx == nullptr) return;
  size_t off = 0;
  while (off < len) {
    size_t n = len - off;
    if (n > BLE_CHUNK) n = BLE_CHUNK;
    bleTx->setValue((uint8_t *)(buf + off), n);
    bleTx->notify();
    off += n;
    delay(0);   // yield so the BLE stack can flush
  }
}

// Meta lines (banner / column header) go to USB always and BLE always.
void outMeta(const char *buf, size_t len) {
  Serial.write((const uint8_t *)buf, len);
  bleNotify(buf, len);
}

// Data rows: USB every frame, BLE decimated (BLE_DECIM) to fit bandwidth.
void outData(const char *buf, size_t len) {
  Serial.write((const uint8_t *)buf, len);
  if ((frameCount % BLE_DECIM) == 0) bleNotify(buf, len);
}

// ============================================================================
// Banner / output
// ============================================================================
void printBanner() {
  char buf[640];
  int n;

  // Machine-readable banner + column header -> USB AND BLE (host parses these).
  n = snprintf(buf, sizeof(buf), "# senz-v3pinch nimu=%u nforce=%u rate=%lu\n",
               NUM_IMU, NUM_FORCE, (unsigned long)SAMPLE_HZ);
  outMeta(buf, n);

  n = snprintf(buf, sizeof(buf), "# columns: t_us,bno_qw,bno_qx,bno_qy,bno_qz");
  for (uint8_t i = 0; i < NUM_IMU; i++)
    n += snprintf(buf + n, sizeof(buf) - n,
                  ",imu%u_ax,imu%u_ay,imu%u_az,imu%u_gx,imu%u_gy,imu%u_gz",
                  i, i, i, i, i, i);
  for (uint8_t m = 0; m < NUM_FORCE; m++)
    n += snprintf(buf + n, sizeof(buf) - n, ",force%u", m);
  n += snprintf(buf + n, sizeof(buf) - n, "\n");
  outMeta(buf, n);

  // Human-readable status (comment lines, ignored by the parser) -- USB only.
  Serial.printf("# transport: primary=%s  BLE=%s (name '%s')\n",
                usbPrimary ? "USB" : "BLE",
                bleConnected ? "connected" : "advertising", BLE_NAME);
  Serial.printf("# BNO055 wrist: %s\n", bnoPresent ? "OK" : "MISSING");
  for (uint8_t i = 0; i < NUM_IMU; i++)
    Serial.printf("# IMU %u %-9s CS=GPIO%d expect=0x%02X : %s\n", i, IMU_LABEL[i],
                  CS_PIN[i], IMU_EXPECT[i], imuPresent[i] ? "OK" : "DEAD/miswired");
}

void debugDump() {
  for (uint8_t i = 0; i < NUM_IMU; i++) {
    if (!imuPresent[i]) { Serial.printf("# IMU %u %s DEAD\n", i, IMU_LABEL[i]); continue; }
    int16_t ax, ay, az, gx, gy, gz;
    imu[i]->readRaw(ax, ay, az, gx, gy, gz);
    Serial.printf("# IMU %u %-9s a=%6d %6d %6d  g=%6d %6d %6d\n", i, IMU_LABEL[i],
                  ax, ay, az, gx, gy, gz);
  }
  Serial.print("# force:");
  scanForce();
  for (uint8_t m = 0; m < NUM_FORCE; m++) Serial.printf(" %4u", force[m]);
  Serial.println();
}

// Shared command handler: one char, from EITHER USB serial or the BLE RX char.
void handleChar(char c) {
  if (c == '?') printBanner();
  else if (c == 'C' || c == 'c') calibrateGyro();
  else if (c == 'Z' || c == 'z') { memset(gyroBias, 0, sizeof(gyroBias));
                                   Serial.println("# gyro bias zeroed (RAM)"); }
  else if (c == 'D' || c == 'd') debugDump();
}

void handleCommand() {   // poll USB serial (BLE RX is delivered via RxCallbacks)
  while (Serial.available()) handleChar(Serial.read());
}

// ============================================================================
// Setup / loop
// ============================================================================
void setup() {
  Serial.begin(921600);

  // USB FIRST: wait briefly for a host to open the CDC port. On the ESP32-S3
  // native USB, `Serial` is truthy once the host asserts DTR. If a USB host is
  // present we mark it primary; either way BLE is brought up as a fallback.
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < USB_WAIT_MS) delay(10);
  usbPrimary = (bool)Serial;
  delay(300);

  bleInit();   // always advertise BLE so battery/untethered use just works

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);
  spiBus.begin(SPI_SCLK, SPI_MISO, SPI_MOSI, -1);

  // Mux control pins + ADC
  for (uint8_t b = 0; b < 4; b++) { pinMode(MUX_SEL[b], OUTPUT); digitalWrite(MUX_SEL[b], LOW); }
  analogReadResolution(12);
  analogSetPinAttenuation(MUX_SIG, ADC_11db);   // full ~0-3.3V range

  // Finger IMUs: CS idle-high first, then bring up
  for (uint8_t i = 0; i < NUM_IMU; i++) {
    imu[i] = new MpuSpi(spiBus, CS_PIN[i], SPI_HZ);
    imu[i]->initCs();
  }
  for (uint8_t i = 0; i < NUM_IMU; i++) imuPresent[i] = imu[i]->begin();

  bnoPresent = bnoInit();
  loadGyroBias();

  printBanner();
}

void loop() {
  handleCommand();

  static uint32_t lastFrame = 0;
  uint32_t now = micros();
  if ((uint32_t)(now - lastFrame) < FRAME_US) return;
  lastFrame += FRAME_US;
  if ((uint32_t)(micros() - lastFrame) > FRAME_US) lastFrame = micros();  // catch up

  // 1) finger IMUs (bias-corrected gyro)
  int16_t a[NUM_IMU][3], g[NUM_IMU][3];
  for (uint8_t i = 0; i < NUM_IMU; i++) {
    if (!imuPresent[i]) { a[i][0]=a[i][1]=a[i][2]=g[i][0]=g[i][1]=g[i][2]=0; continue; }
    int16_t ax, ay, az, gx, gy, gz;
    imu[i]->readRaw(ax, ay, az, gx, gy, gz);
    a[i][0] = ax; a[i][1] = ay; a[i][2] = az;
    g[i][0] = gx - gyroBias[i][0];
    g[i][1] = gy - gyroBias[i][1];
    g[i][2] = gz - gyroBias[i][2];
  }

  // 2) wrist BNO055 quaternion (~100 Hz)
  if (bnoPresent && (frameCount % BNO_EVERY) == 0) {
    uint8_t b[8];
    bnoReadBytes(BNO_QUA_DATA_W_LSB, b, 8);
    bqw = (int16_t)(b[0] | (b[1] << 8)) / BNO_QUA_LSB;
    bqx = (int16_t)(b[2] | (b[3] << 8)) / BNO_QUA_LSB;
    bqy = (int16_t)(b[4] | (b[5] << 8)) / BNO_QUA_LSB;
    bqz = (int16_t)(b[6] | (b[7] << 8)) / BNO_QUA_LSB;
  }

  // 3) velostat taxels
  scanForce();

  // 4) build the CSV row into one buffer, then emit to USB (every frame) + BLE
  //    (decimated). Building once avoids many small writes and lets BLE reuse it.
  char row[512];
  int n = snprintf(row, sizeof(row), "%lu,%.4f,%.4f,%.4f,%.4f",
                   (unsigned long)now, bqw, bqx, bqy, bqz);
  for (uint8_t i = 0; i < NUM_IMU; i++)
    n += snprintf(row + n, sizeof(row) - n, ",%d,%d,%d,%d,%d,%d",
                  a[i][0], a[i][1], a[i][2], g[i][0], g[i][1], g[i][2]);
  for (uint8_t m = 0; m < NUM_FORCE; m++)
    n += snprintf(row + n, sizeof(row) - n, ",%u", force[m]);
  n += snprintf(row + n, sizeof(row) - n, "\n");
  outData(row, n);

  frameCount++;
}

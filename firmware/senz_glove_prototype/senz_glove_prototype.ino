/*
 * senz_glove_prototype
 * ---------------------
 * Prototype firmware for the senz training glove.
 *
 * CURRENT SETUP: 1 potentiometer, read directly on an ADC pin (no multiplexer).
 *
 * Hardware:
 *   - ESP32-C3 SuperMini
 *   - 1x potentiometer wired as a DIY string pot (one finger)
 *     wiper -> GPIO0 (ADC1_CH0), outer legs -> 3V3 and GND
 *   - 1x GY-BNO055 9-axis absolute-orientation IMU on I2C
 *   - 1x 0.91" 128x32 SSD1306 OLED on I2C (shares the bus with the BNO055)
 *
 * Adding more fingers later: just add their ADC pins to FINGER_PINS[] below.
 * The ESP32-C3 has ADC1 on GPIO0,1,3,4 (skip GPIO2 - it's a strapping pin).
 *
 * Streams one CSV line per frame at ~50 Hz, over USB serial OR BLE (see USE_BLE):
 *
 *       f0[,f1...],qw,qx,qy,qz,cal
 *
 *   f0..  : raw 12-bit ADC (0..4095), one per finger
 *   qw..qz: orientation QUATERNION (float, 4 decimals) from the BNO055.
 *           Quaternions avoid gimbal lock, so the model never flips when the
 *           hand points straight up/down (unlike Euler roll/pitch/yaw).
 *   cal   : BNO055 calibration status byte (sys<<6|gyro<<4|accel<<2|mag), 0..3 each
 *
 * Over BLE it uses the Nordic UART Service (NUS); the host connects by the
 * advertised name "senz-glove". The ESP32-C3 supports BLE only (no classic SPP).
 */

// ============================================================================
// Set this to choose how data leaves the board:
//   false -> USB serial (plug in USB-C)  <- default per the HLD data-path decision
//   true  -> BLE wireless (Nordic UART Service, host connects by name)
// The HLD ("sensing glove hld.md") deliberately skips BLE for now; the code is
// kept here behind the flag in case it's wanted later.
// ============================================================================
#define USE_BLE false

#include <Wire.h>
#include <Preferences.h>       // NVS-backed flash storage for the BNO055 cal profile
#include <Adafruit_GFX.h>      // install "Adafruit GFX Library"
#include <Adafruit_SSD1306.h>  // install "Adafruit SSD1306"

#if USE_BLE
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#endif

// ----------------------------------------------------------------------------
// CONFIG -- edit these to match your wiring
// ----------------------------------------------------------------------------

// Finger ADC pins (one entry per potentiometer). Add more to scale up:
//   {0}          -> 1 finger  on GPIO0
//   {0, 1, 3, 4} -> 4 fingers on GPIO0/1/3/4
// Use only ADC1 pins (GPIO0,1,3,4); avoid GPIO2 (strapping pin).
static const int FINGER_PINS[] = {0};
static const int NUM_FINGERS = sizeof(FINGER_PINS) / sizeof(FINGER_PINS[0]);

// I2C pins for the BNO055 + OLED (shared bus).
static const int I2C_SDA = 6;
static const int I2C_SCL = 7;

// BNO055 I2C address: 0x28 with ADD/ADR pin LOW (default), 0x29 if HIGH.
static const uint8_t IMU_ADDR = 0x28;

// --- OLED dashboard (0.91" 128x32 SSD1306) ---
static const bool OLED_ENABLED = true;
static const uint8_t OLED_ADDR = 0x3C;  // most 0.91" modules; some are 0x3D
static const int OLED_W = 128;
static const int OLED_H = 32;
static const uint32_t OLED_UPDATE_HZ = 5;

// --- Host link / connection status ---
static const uint32_t LINK_TIMEOUT_MS = 1500;

// --- BLE ---
static const char *BLE_NAME = "senz-glove";

// Output rate (BNO055 fusion tops out at 100 Hz)
static const uint32_t SAMPLE_HZ = 100;

// ----------------------------------------------------------------------------
// BNO055 register map
// ----------------------------------------------------------------------------
static const uint8_t BNO_CHIP_ID = 0x00;       // should read 0xA0
static const uint8_t BNO_SW_REV_LSB = 0x04;    // firmware version (LSB)
static const uint8_t BNO_SW_REV_MSB = 0x05;    // firmware version (MSB)
static const uint8_t BNO_PAGE_ID = 0x07;
static const uint8_t BNO_CALIB_STAT = 0x35;
static const uint8_t BNO_OPR_MODE = 0x3D;
static const uint8_t BNO_PWR_MODE = 0x3E;
static const uint8_t BNO_SYS_TRIGGER = 0x3F;
static const uint8_t BNO_QUA_DATA_W_LSB = 0x20;     // 8 bytes: w, x, y, z (int16)
static const uint8_t BNO_ACC_OFFSET_X_LSB = 0x55;   // start of the 22-byte cal profile
static const uint8_t BNO_CALIB_PROFILE_LEN = 22;    // 0x55..0x6A: offsets + radii

static const uint8_t BNO_OPR_CONFIG = 0x00;
static const uint8_t BNO_OPR_NDOF = 0x0C;  // full 9-axis fusion
static const uint8_t BNO_PWR_NORMAL = 0x00;

// Known buggy firmware: SW_REV 0x0311 distorts Euler output beyond ~20 deg of
// pitch/roll. We stream quaternions (not Euler), so we only warn on it.
static const uint16_t BNO_SW_REV_BUGGY = 0x0311;

static const float QUA_LSB = 16384.0f;  // 1 quaternion unit = 2^14 LSB

// NVS (flash) location of the saved calibration profile, written by the
// standalone calibrator sketch (firmware/senz_glove_calibrate).
static const char *NVS_NAMESPACE = "senz-bno";
static const char *NVS_CAL_KEY = "cal";

// ----------------------------------------------------------------------------
// State
// ----------------------------------------------------------------------------
static const uint32_t FRAME_US = 1000000UL / SAMPLE_HZ;

float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
uint8_t calStat = 0;
uint32_t lastMicros = 0;
bool imuPresent = false;
bool calRestored = false;  // did we load a saved calibration profile at boot?
bool magCalLow = false;    // mag calibration has dropped below 2 mid-session

Adafruit_SSD1306 display(OLED_W, OLED_H, &Wire, -1);
bool oledPresent = false;
uint32_t lastOledMs = 0;
uint32_t lastLinkMs = 0;
uint8_t spinIdx = 0;

// ----------------------------------------------------------------------------
// Transport (USB serial or BLE)
// ----------------------------------------------------------------------------
#if USE_BLE
// Nordic UART Service UUIDs.
#define NUS_SERVICE "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_RX_UUID "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  // host -> device
#define NUS_TX_UUID "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  // device -> host

BLECharacteristic *pTxChar = nullptr;
volatile bool bleConnected = false;

class ServerCB : public BLEServerCallbacks {
  // Called with connection info -> request a FAST connection interval so
  // notifications flow quickly (this is the main lever for BLE "fps").
  void onConnect(BLEServer *s, esp_ble_gatts_connect_evt_param *param) override {
    bleConnected = true;
    // Units of 1.25 ms: 0x06 = 7.5 ms, 0x0C = 15 ms. latency 0, timeout 2 s.
    s->updateConnParams(param->connect.remote_bda, 0x06, 0x0C, 0, 200);
  }
  void onConnect(BLEServer *s) override { bleConnected = true; }
  void onDisconnect(BLEServer *s) override {
    bleConnected = false;
    BLEDevice::startAdvertising();  // allow reconnect
  }
};

class RxCB : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override { lastLinkMs = millis(); }
};

void bleInit() {
  BLEDevice::init(BLE_NAME);
  BLEDevice::setMTU(185);  // room for our CSV lines
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCB());
  BLEService *svc = server->createService(NUS_SERVICE);
  pTxChar = svc->createCharacteristic(NUS_TX_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  pTxChar->addDescriptor(new BLE2902());
  BLECharacteristic *pRx = svc->createCharacteristic(
      NUS_RX_UUID,
      BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  pRx->setCallbacks(new RxCB());
  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
}
#endif

// Send one CSV line over whichever transport is active.
void sendLine(const String &line) {
#if USE_BLE
  if (bleConnected && pTxChar) {
    String out = line + "\n";
    pTxChar->setValue((uint8_t *)out.c_str(), out.length());
    pTxChar->notify();
  }
#else
  Serial.println(line);
#endif
}

// ----------------------------------------------------------------------------
// Finger reads
// ----------------------------------------------------------------------------
int readFinger(int idx) {
  int pin = FINGER_PINS[idx];
  long acc = 0;
  const int N = 4;  // average to knock down noise
  for (int i = 0; i < N; i++) acc += analogRead(pin);
  return (int)(acc / N);
}

// ----------------------------------------------------------------------------
// OLED dashboard
// ----------------------------------------------------------------------------
void drawDashboard() {
  if (!oledPresent) return;
  const char spin[] = {'|', '/', '-', '\\'};
  bool linkOk = (millis() - lastLinkMs) < LINK_TIMEOUT_MS;

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  display.print("senz glove   ");
  display.print(spin[spinIdx & 0x3]);

  display.setCursor(0, 8);
  display.print("LINK: ");
  display.print(linkOk ? "OK" : "--");

  display.setCursor(0, 16);
  display.print("RUN ");
  if (imuPresent) {
    display.print("IMU c");
    display.print((calStat >> 6) & 0x3);
    display.print(" m");
    display.print(calStat & 0x3);
  } else {
    display.print("IMU --");
  }

  display.setCursor(0, 24);
  if (imuPresent && magCalLow) {
    display.print("! MAG CAL LOW");
  } else if (calRestored) {
    display.print("cal: restored");
  }

  display.display();
  spinIdx++;
}

// ----------------------------------------------------------------------------
// BNO055 helpers
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

bool imuReadQuat(float &w, float &x, float &y, float &z) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(BNO_QUA_DATA_W_LSB);
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom((int)IMU_ADDR, 8);
  if (Wire.available() < 8) return false;

  int16_t rw = (int16_t)(Wire.read() | (Wire.read() << 8));
  int16_t rx = (int16_t)(Wire.read() | (Wire.read() << 8));
  int16_t ry = (int16_t)(Wire.read() | (Wire.read() << 8));
  int16_t rz = (int16_t)(Wire.read() | (Wire.read() << 8));

  w = rw / QUA_LSB;
  x = rx / QUA_LSB;
  y = ry / QUA_LSB;
  z = rz / QUA_LSB;
  return true;
}

// Switch operating mode. CONFIG<->fusion transitions need a settle delay
// (datasheet: ~7 ms into a fusion mode, ~19 ms back to CONFIG). 30 ms is safe.
void imuSetMode(uint8_t mode) {
  imuWrite(BNO_OPR_MODE, mode);
  delay(30);
}

// Pull a saved 22-byte calibration profile out of NVS (written by the
// standalone calibrator sketch). Returns false if none is stored yet.
bool loadCalProfileFromNVS(uint8_t *buf) {
  Preferences prefs;
  if (!prefs.begin(NVS_NAMESPACE, true)) return false;  // read-only
  size_t n = prefs.getBytesLength(NVS_CAL_KEY);
  bool ok = (n == BNO_CALIB_PROFILE_LEN) &&
            (prefs.getBytes(NVS_CAL_KEY, buf, BNO_CALIB_PROFILE_LEN) ==
             BNO_CALIB_PROFILE_LEN);
  prefs.end();
  return ok;
}

bool imuInit() {
  Wire.beginTransmission(IMU_ADDR);
  if (Wire.endTransmission() != 0) return false;
  if (imuRead8(BNO_CHIP_ID) != 0xA0) return false;

  // --- Diagnostics (HLD objective 1): firmware version + buggy-Euler check ---
  uint16_t swRev =
      ((uint16_t)imuRead8(BNO_SW_REV_MSB) << 8) | imuRead8(BNO_SW_REV_LSB);
  Serial.printf("BNO055 chip OK, SW_REV=0x%04X\n", swRev);
  if (swRev == BNO_SW_REV_BUGGY)
    Serial.println("WARN: SW_REV 0x0311 has the Euler-angle math bug >~20 deg; "
                   "we stream quaternions, so it's avoided.");

  // --- CONFIG mode: set clock/power, then (re)load the calibration profile ---
  imuSetMode(BNO_OPR_CONFIG);
  imuWrite(BNO_PAGE_ID, 0x00);
  imuWrite(BNO_PWR_MODE, BNO_PWR_NORMAL);
  delay(10);
  imuWrite(BNO_SYS_TRIGGER, 0x00);  // use internal oscillator
  delay(10);

  // Restore saved offsets so we start calibrated instead of from zero
  // (HLD objective 3). Offset registers are only writable in CONFIG mode.
  uint8_t profile[BNO_CALIB_PROFILE_LEN];
  if (loadCalProfileFromNVS(profile)) {
    for (int i = 0; i < BNO_CALIB_PROFILE_LEN; i++)
      imuWrite(BNO_ACC_OFFSET_X_LSB + i, profile[i]);
    calRestored = true;
    Serial.println("BNO055 calibration profile restored from flash.");
  } else {
    Serial.println(
        "No saved BNO055 calibration in flash; run the calibrator sketch.");
  }

  // --- Full 9-axis fusion, then verify the mode actually took (objective 1) ---
  imuSetMode(BNO_OPR_NDOF);
  uint8_t mode = imuRead8(BNO_OPR_MODE) & 0x0F;
  Serial.printf("BNO055 OPR_MODE readback=0x%02X (%s)\n", mode,
                mode == BNO_OPR_NDOF ? "NDOF" : "NOT NDOF!");
  return true;
}

// ----------------------------------------------------------------------------
// Setup / loop
// ----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(200);

  analogReadResolution(12);  // 0..4095 on ESP32-C3

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  if (OLED_ENABLED) {
    oledPresent = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
    if (oledPresent) {
      display.clearDisplay();
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.println("senz glove");
      display.println("booting...");
      display.display();
    }
  }

  imuPresent = imuInit();

#if USE_BLE
  bleInit();
#endif

  lastMicros = micros();
}

void loop() {
  // --- Host link status ---
#if USE_BLE
  if (bleConnected) lastLinkMs = millis();
#else
  while (Serial.available()) {  // heartbeat byte from the host viz
    Serial.read();
    lastLinkMs = millis();
  }
#endif

  // --- OLED dashboard refresh (slow, independent of the sample rate) ---
  uint32_t ms = millis();
  if (oledPresent && (ms - lastOledMs) >= (1000UL / OLED_UPDATE_HZ)) {
    lastOledMs = ms;
    drawDashboard();
  }

  uint32_t now = micros();
  if (now - lastMicros < FRAME_US) return;
  lastMicros = now;

  // --- Fingers ---
  int fingers[NUM_FINGERS];
  for (int i = 0; i < NUM_FINGERS; i++) fingers[i] = readFinger(i);

  // --- IMU orientation as a quaternion (no gimbal lock) ---
  if (imuPresent) {
    imuReadQuat(qw, qx, qy, qz);
    calStat = imuRead8(BNO_CALIB_STAT);

    // Drift tends to creep back in during motion (HLD objective 3): flag it
    // when the magnetometer calibration falls below 2, clear it on recovery.
    uint8_t magCal = calStat & 0x3;
    if (magCal < 2 && !magCalLow) {
      magCalLow = true;
      Serial.println("WARN: BNO055 mag calibration dropped below 2; yaw may "
                     "drift. Wave a slow figure-8 to recover.");
    } else if (magCal >= 2 && magCalLow) {
      magCalLow = false;
      Serial.println("INFO: BNO055 mag calibration recovered.");
    }
  }

  // --- Build + stream the CSV line ---
  String line = String(fingers[0]);
  for (int i = 1; i < NUM_FINGERS; i++) line += "," + String(fingers[i]);
  line += "," + String(qw, 4) + "," + String(qx, 4) + "," + String(qy, 4) +
          "," + String(qz, 4) + "," + String(calStat);
  sendLine(line);
}

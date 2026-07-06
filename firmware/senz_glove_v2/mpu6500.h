/*
 * mpu6500.h
 * ---------
 * Minimal SPI driver for the InvenSense MPU-6500 6-axis IMU, used for the senz
 * per-finger IMU array (HLD: senz_glove_hld_v2, branch feature/multi-imu-expansion).
 *
 * Unlike the ICM-42688 array (which sat behind a chip-select expander), the v2
 * design gives every MPU-6500 its OWN chip-select GPIO -- no decoder, no mux.
 * So this driver owns its CS pin directly and toggles it around each transfer.
 * NCS is active LOW and idles HIGH.
 *
 * Fixed full-scale config (see begin()), kept identical to the ICM-42688 driver
 * so the host scale helpers are unchanged:
 *   accel = +/-8 g     -> 4096 LSB/g
 *   gyro  = +/-2000 dps -> 16.4 LSB/dps
 *
 * SPI: Mode 3 (CPOL=1, CPHA=1) per the MPU-6500 datasheet. 8 MHz to start;
 * the register interface is rated to 20 MHz once bring-up is stable. Read
 * transfers set bit7 of the register address (MPU-6500 SPI protocol).
 */
#pragma once

#include <Arduino.h>
#include <SPI.h>

class Mpu6500 {
 public:
  static constexpr float ACC_LSB_PER_G = 4096.0f;   // +/-8 g
  static constexpr float GYR_LSB_PER_DPS = 16.4f;    // +/-2000 dps
  static constexpr uint8_t WHO_AM_I_VALUE = 0x70;    // MPU-6500 (not 0x71/0x7C)

  Mpu6500(SPIClass &spi, int csPin, uint32_t hz = 8000000)
      : spi_(spi), cs_(csPin), settings_(hz, MSBFIRST, SPI_MODE3) {}

  // Drive CS idle-high before any bus activity. Call once, before begin().
  void initCs() {
    pinMode(cs_, OUTPUT);
    digitalWrite(cs_, HIGH);
  }

  // Reset, force SPI-only, verify WHO_AM_I, set ranges + DLPF, wake accel+gyro.
  // Returns false if the WHO_AM_I does not read back 0x70 (dead / miswired).
  bool begin() {
    writeReg(REG_PWR_MGMT_1, 0x80);  // device reset
    delay(100);
    writeReg(REG_PWR_MGMT_1, 0x01);  // wake, auto-select best clock (PLL)
    delay(10);
    writeReg(REG_USER_CTRL, 0x10);   // I2C_IF_DIS: disable I2C slave, SPI only
    delay(1);

    if (whoAmI() != WHO_AM_I_VALUE) return false;

    writeReg(REG_GYRO_CONFIG, 0x18);   // FS_SEL=3  -> +/-2000 dps
    writeReg(REG_ACCEL_CONFIG, 0x10);  // AFS_SEL=2 -> +/-8 g
    writeReg(REG_CONFIG, 0x03);        // gyro DLPF 41 Hz -> 1 kHz internal rate
    writeReg(REG_ACCEL_CONFIG2, 0x03); // accel DLPF 41 Hz
    delayMicroseconds(250);
    return true;
  }

  uint8_t whoAmI() { return readReg(REG_WHO_AM_I); }

  // Output data rate = 1 kHz / (1 + div). div=4 -> 200 Hz (per-finger target).
  // Only takes effect while a gyro DLPF is enabled (CONFIG != 0/7), which
  // begin() sets. See HLD "Boot Sequence" step 5.
  void setSampleRateDiv(uint8_t div) { writeReg(REG_SMPLRT_DIV, div); }

  // Burst-read accel XYZ, temp, gyro XYZ as raw int16 counts (big-endian wire).
  // The 14-byte read matches the HLD's "14 bytes per read" timing budget.
  void readRaw(int16_t &ax, int16_t &ay, int16_t &az, int16_t &gx, int16_t &gy,
               int16_t &gz) {
    uint8_t b[14];
    readRegs(REG_ACCEL_XOUT_H, b, 14);
    ax = (int16_t)((b[0] << 8) | b[1]);
    ay = (int16_t)((b[2] << 8) | b[3]);
    az = (int16_t)((b[4] << 8) | b[5]);
    // b[6..7] = temperature, skipped (not streamed)
    gx = (int16_t)((b[8] << 8) | b[9]);
    gy = (int16_t)((b[10] << 8) | b[11]);
    gz = (int16_t)((b[12] << 8) | b[13]);
  }

  void writeReg(uint8_t reg, uint8_t val) {
    spi_.beginTransaction(settings_);
    digitalWrite(cs_, LOW);
    spi_.transfer(reg & 0x7F);  // bit7 clear = write
    spi_.transfer(val);
    digitalWrite(cs_, HIGH);
    spi_.endTransaction();
  }

  uint8_t readReg(uint8_t reg) {
    spi_.beginTransaction(settings_);
    digitalWrite(cs_, LOW);
    spi_.transfer(reg | 0x80);  // bit7 set = read
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
  // Register map (MPU-6500 datasheet)
  static constexpr uint8_t REG_SMPLRT_DIV = 0x19;
  static constexpr uint8_t REG_CONFIG = 0x1A;
  static constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
  static constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
  static constexpr uint8_t REG_ACCEL_CONFIG2 = 0x1D;
  static constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;  // 14 bytes: accel/temp/gyro
  static constexpr uint8_t REG_USER_CTRL = 0x6A;
  static constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
  static constexpr uint8_t REG_WHO_AM_I = 0x75;

  SPIClass &spi_;
  int cs_;
  SPISettings settings_;
};

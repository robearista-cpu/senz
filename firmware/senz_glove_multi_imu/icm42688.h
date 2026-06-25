/*
 * icm42688.h
 * ----------
 * Minimal SPI driver for the TDK InvenSense ICM-42688-P 6-axis IMU, used for
 * the senz finger-IMU array (HLD objective 1, "multi-imu-finger-tracking").
 *
 * The chip-select line is NOT owned here: finger IMUs sit behind a chip-select
 * expander (see cs_expander.h), so each instance is handed select()/deselect()
 * callbacks instead of a GPIO. This lets one driver class serve every IMU on
 * the shared SPI bus.
 *
 * Fixed full-scale config (see begin()):
 *   accel = +/-8 g     -> 4096 LSB/g
 *   gyro  = +/-2000 dps -> 16.4 LSB/dps
 * Raw int16 counts are streamed to the host, which converts using the LSB
 * constants below. Keep these in sync with the host (senz_multi_io.py).
 */
#pragma once

#include <Arduino.h>
#include <SPI.h>

#include <functional>

class Icm42688 {
 public:
  static constexpr float ACC_LSB_PER_G = 4096.0f;    // +/-8 g
  static constexpr float GYR_LSB_PER_DPS = 16.4f;     // +/-2000 dps
  static constexpr uint8_t WHO_AM_I_VALUE = 0x47;

  Icm42688(SPIClass &spi, std::function<void()> sel, std::function<void()> desel,
           uint32_t hz = 8000000)
      : spi_(spi), sel_(sel), desel_(desel), settings_(hz, MSBFIRST, SPI_MODE3) {}

  // Reset, verify WHO_AM_I, configure ranges/ODR, power up accel+gyro.
  bool begin() {
    writeReg(REG_DEVICE_CONFIG, 0x01);  // soft reset
    delay(2);
    writeReg(REG_BANK_SEL, 0x00);       // select user bank 0
    if (whoAmI() != WHO_AM_I_VALUE) return false;

    writeReg(REG_GYRO_CONFIG0, 0x06);   // FS=+/-2000dps, ODR=1 kHz
    writeReg(REG_ACCEL_CONFIG0, 0x26);  // FS=+/-8g,    ODR=1 kHz
    writeReg(REG_PWR_MGMT0, 0x0F);      // gyro + accel in Low-Noise mode
    delayMicroseconds(250);             // gyro needs a moment to spin up
    return true;
  }

  uint8_t whoAmI() { return readReg(REG_WHO_AM_I); }

  // Burst-read accel XYZ + gyro XYZ as raw int16 counts (big-endian on the wire).
  void readRaw(int16_t &ax, int16_t &ay, int16_t &az, int16_t &gx, int16_t &gy,
               int16_t &gz) {
    uint8_t b[12];
    readRegs(REG_ACCEL_DATA_X1, b, 12);
    ax = (int16_t)((b[0] << 8) | b[1]);
    ay = (int16_t)((b[2] << 8) | b[3]);
    az = (int16_t)((b[4] << 8) | b[5]);
    gx = (int16_t)((b[6] << 8) | b[7]);
    gy = (int16_t)((b[8] << 8) | b[9]);
    gz = (int16_t)((b[10] << 8) | b[11]);
  }

  void writeReg(uint8_t reg, uint8_t val) {
    spi_.beginTransaction(settings_);
    sel_();
    spi_.transfer(reg & 0x7F);  // MSB clear = write
    spi_.transfer(val);
    desel_();
    spi_.endTransaction();
  }

  uint8_t readReg(uint8_t reg) {
    spi_.beginTransaction(settings_);
    sel_();
    spi_.transfer(reg | 0x80);  // MSB set = read
    uint8_t v = spi_.transfer(0x00);
    desel_();
    spi_.endTransaction();
    return v;
  }

  void readRegs(uint8_t reg, uint8_t *buf, size_t n) {
    spi_.beginTransaction(settings_);
    sel_();
    spi_.transfer(reg | 0x80);
    for (size_t i = 0; i < n; i++) buf[i] = spi_.transfer(0x00);
    desel_();
    spi_.endTransaction();
  }

 private:
  // Register map (user bank 0)
  static constexpr uint8_t REG_DEVICE_CONFIG = 0x11;
  static constexpr uint8_t REG_ACCEL_DATA_X1 = 0x1F;  // 12 bytes: accel+gyro
  static constexpr uint8_t REG_PWR_MGMT0 = 0x4E;
  static constexpr uint8_t REG_GYRO_CONFIG0 = 0x4F;
  static constexpr uint8_t REG_ACCEL_CONFIG0 = 0x50;
  static constexpr uint8_t REG_WHO_AM_I = 0x75;
  static constexpr uint8_t REG_BANK_SEL = 0x76;

  SPIClass &spi_;
  std::function<void()> sel_;
  std::function<void()> desel_;
  SPISettings settings_;
};

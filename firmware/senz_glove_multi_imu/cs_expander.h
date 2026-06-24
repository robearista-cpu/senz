/*
 * cs_expander.h
 * -------------
 * Chip-select expander for the SPI finger-IMU array (HLD section 5).
 *
 * ~10 SPI IMUs would burn ~10 GPIO if each chip-select had its own pin. Instead
 * we drive a binary decoder (e.g. 74HC138 3->8, or 74HC4515 4->16) so a few
 * address lines + an enable address every device. A decoder guarantees that at
 * most ONE output is ever asserted, which is exactly what a shared SPI bus
 * needs: one active-low CS at a time, all others released.
 *
 *   addrPins  : decoder address lines, LSB first (A0, A1, A2[, A3]).
 *   enablePin : active-high enable (tie to 74HC138 G1; G2A/G2B -> GND).
 *               LOW  => decoder disabled, every output HIGH => all deselected.
 *               HIGH => the addressed output goes LOW => that device selected.
 *
 * Sequence per transfer: select(i) sets the address then asserts enable;
 * deselect() drops enable so the bus is fully released before the next select.
 * Because we always deselect() between transfers, the address lines only ever
 * change while the decoder is disabled -> no chance of momentarily selecting
 * the wrong device.
 */
#pragma once

#include <Arduino.h>

class CsExpander {
 public:
  CsExpander(const uint8_t *addrPins, uint8_t numAddr, uint8_t enablePin)
      : addr_(addrPins), n_(numAddr), en_(enablePin) {}

  void begin() {
    for (uint8_t i = 0; i < n_; i++) {
      pinMode(addr_[i], OUTPUT);
      digitalWrite(addr_[i], LOW);
    }
    pinMode(en_, OUTPUT);
    digitalWrite(en_, LOW);  // start fully deselected
  }

  void select(uint8_t idx) {
    for (uint8_t i = 0; i < n_; i++) digitalWrite(addr_[i], (idx >> i) & 0x1);
    digitalWrite(en_, HIGH);
  }

  void deselect() { digitalWrite(en_, LOW); }

  uint8_t capacity() const { return (uint8_t)(1u << n_); }

 private:
  const uint8_t *addr_;
  uint8_t n_;
  uint8_t en_;
};

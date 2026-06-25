#!/usr/bin/env python3
"""
senz_multi_io.py
================
Host-side I/O for the multi-modal glove firmware (senz_glove_multi_imu).

Unlike the original single-IMU `senz_io.py`, the multi-IMU stream is
*self-describing*: the firmware prints a banner and a ``# columns:`` header at
boot (and on the ``?`` command), so this module learns the schema at runtime
instead of hard-coding column positions. That means changing NUM_FINGER_IMUS or
NUM_FORCE in the firmware needs no host edit.

Provides:
  - MultiSerialSource(port)   live frames from the glove, with schema discovery
  - SimMultiSource(...)       synthetic frames so the rest of the pipeline runs
                              with no hardware
  - Frame: a parsed line as an ordered dict {column_name: value}
  - scale helpers: imu_accel_g(), imu_gyro_dps()

Stream contract (must match icm42688.h / the firmware):
    accel: +/-8 g     -> 4096 LSB/g
    gyro : +/-2000 dps -> 16.4 LSB/dps
"""

import math
import re
import time

# Keep in sync with icm42688.h
ACC_LSB_PER_G = 4096.0
GYR_LSB_PER_DPS = 16.4


# ----------------------------------------------------------------------------
# Schema + parsing
# ----------------------------------------------------------------------------
class Schema:
    """Column layout discovered from the firmware's ``# columns:`` header."""

    def __init__(self, columns, nimu, nforce, rate):
        self.columns = columns
        self.nimu = nimu
        self.nforce = nforce
        self.rate = rate
        # Columns that should stay integers (everything but the BNO quaternion).
        self._float_cols = {"bno_qw", "bno_qx", "bno_qy", "bno_qz"}

    @classmethod
    def from_header(cls, banner, columns_line):
        cols = [c.strip() for c in columns_line.split("columns:", 1)[1].split(",")]
        nimu = nforce = 0
        rate = 0
        m = re.search(r"nimu=(\d+)", banner)
        if m:
            nimu = int(m.group(1))
        m = re.search(r"nforce=(\d+)", banner)
        if m:
            nforce = int(m.group(1))
        m = re.search(r"rate=(\d+)", banner)
        if m:
            rate = int(m.group(1))
        return cls(cols, nimu, nforce, rate)

    def parse(self, line):
        """CSV data line -> dict keyed by column name, or None if malformed."""
        parts = line.split(",")
        if len(parts) != len(self.columns):
            return None
        out = {}
        try:
            for name, raw in zip(self.columns, parts):
                out[name] = float(raw) if name in self._float_cols else int(raw)
        except ValueError:
            return None
        return out

    def imu_indices(self):
        return list(range(self.nimu))

    def force_indices(self):
        return list(range(self.nforce))


def imu_accel_g(frame, k):
    """Finger IMU k acceleration in g as (x, y, z)."""
    return tuple(frame[f"imu{k}_a{ax}"] / ACC_LSB_PER_G for ax in "xyz")


def imu_gyro_dps(frame, k):
    """Finger IMU k angular rate in deg/s as (x, y, z)."""
    return tuple(frame[f"imu{k}_g{ax}"] / GYR_LSB_PER_DPS for ax in "xyz")


# ----------------------------------------------------------------------------
# Live serial source
# ----------------------------------------------------------------------------
class MultiSerialSource:
    """Reads the self-describing multi-IMU stream from the glove over serial."""

    def __init__(self, port, baud=115200, discover_timeout=5.0):
        import serial  # pyserial

        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)  # let the board reset/boot
        self.ser.reset_input_buffer()
        self.schema = self._discover(discover_timeout)

    def _discover(self, timeout):
        """Ask for the header and parse the banner + columns lines."""
        self.ser.write(b"?\n")
        banner = None
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("# senz-multi"):
                banner = line
            elif line.startswith("# columns:") and banner:
                return Schema.from_header(banner, line)
            elif line.startswith("# columns:"):
                # columns arrived before a banner we saw; use a bare banner
                return Schema.from_header("", line)
        raise TimeoutError("no '# columns:' header from the glove; wrong port?")

    def read(self):
        """Return the next parsed frame (dict), or None on a bad/blank line."""
        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        if not line or line.startswith("#"):
            return None
        return self.schema.parse(line)

    def send(self, cmd):
        """Send a firmware command (e.g. 'XR', 'C', 'O,0,..')."""
        if not cmd.endswith("\n"):
            cmd += "\n"
        self.ser.write(cmd.encode("utf-8"))

    def close(self):
        self.ser.close()


# ----------------------------------------------------------------------------
# Simulated source (no hardware)
# ----------------------------------------------------------------------------
class SimMultiSource:
    """Synthetic frames matching the firmware schema, for offline development."""

    def __init__(self, nimu=3, nforce=5, rate=200):
        cols = ["t_us", "bno_cal", "bno_qw", "bno_qx", "bno_qy", "bno_qz"]
        for i in range(nimu):
            cols += [f"imu{i}_ok"] + [f"imu{i}_{a}{x}" for a in "ag" for x in "xyz"]
        cols += [f"force{m}" for m in range(nforce)]
        self.schema = Schema(cols, nimu, nforce, rate)
        self._rate = rate
        self.t0 = time.time()

    def read(self):
        t = time.time() - self.t0
        f = {"t_us": int(t * 1e6), "bno_cal": 0xFF}
        # Gently rocking wrist quaternion.
        half = math.radians(20 * math.sin(t * 0.5)) / 2
        f["bno_qw"], f["bno_qx"] = math.cos(half), math.sin(half)
        f["bno_qy"], f["bno_qz"] = 0.0, 0.0
        for i in range(self.schema.nimu):
            # ~1g gravity on Z, a little oscillating gyro per finger.
            f[f"imu{i}_ok"] = 1
            f[f"imu{i}_ax"] = int(0.05 * ACC_LSB_PER_G * math.sin(t + i))
            f[f"imu{i}_ay"] = int(0.05 * ACC_LSB_PER_G * math.cos(t + i))
            f[f"imu{i}_az"] = int(ACC_LSB_PER_G)
            f[f"imu{i}_gx"] = int(40 * GYR_LSB_PER_DPS * math.sin(t * 1.5 + i))
            f[f"imu{i}_gy"] = int(20 * GYR_LSB_PER_DPS * math.cos(t * 1.2 + i))
            f[f"imu{i}_gz"] = 0
        for m in range(self.schema.nforce):
            f[f"force{m}"] = int(200 + 1500 * max(0.0, math.sin(t * 0.8 - m)))
        time.sleep(1.0 / self._rate)
        return f

    def send(self, cmd):
        pass

    def close(self):
        pass


def open_multi_source(port=None, simulate=False, nimu=3, nforce=5, baud=115200):
    """Pick a source: serial if `port`, else a simulator."""
    if simulate or not port:
        return SimMultiSource(nimu=nimu, nforce=nforce)
    return MultiSerialSource(port, baud=baud)

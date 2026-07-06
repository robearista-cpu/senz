#!/usr/bin/env python3
"""
senz_parser.py
==============
Host-side reader for the senz_glove_v2 binary serial stream (HLD v2, deliverable
#12). The v2 firmware streams a fixed 180-byte binary frame per sample instead
of the self-describing CSV used by the multi-IMU build -- binary keeps the
200 Hz / 11-quaternion payload small enough for the serial (and later BLE) link.

Wire format (182 bytes total), little-endian to match the ESP32-S3:
    0xAA                      start marker
    uint32   t_ms             millis() timestamp
    float32  bno[4]           wrist quaternion  (w, x, y, z)
    float32  mpu[10][4]       10 finger quaternions (w, x, y, z each)
    0xFF                      end marker
The 180-byte payload is `struct.unpack('<I 4f 40f', ...)`.

This module ONLY parses -- it does no quaternion math. Turning these quaternions
into a moving 3D hand (finger-relative-to-wrist composition) is the visualizer's
job and is deferred to the supervised session. Until the firmware's Madgwick
filter lands, the 10 finger quaternions arrive as identity (flat hand).

Provides:
  - SerialFrameSource(port)  live frames, synced + validated, daemon-threaded
  - SimFrameSource(...)       synthetic frames so the pipeline runs hardware-free
  - Frame                     a parsed sample (t_ms, bno, mpu[10])
  - open_frame_source(...)    pick serial vs sim
"""

import math
import queue
import struct
import threading
import time

# ---- Wire contract (keep in sync with senz_glove_v2.ino) -------------------
START_BYTE = 0xAA
END_BYTE = 0xFF
NUM_IMU = 10
PAYLOAD_FMT = "<I 4f 40f"           # t_ms + bno quat + 10 finger quats
PAYLOAD_LEN = struct.calcsize(PAYLOAD_FMT)  # 180
assert PAYLOAD_LEN == 180, PAYLOAD_LEN
BAUD = 921600


class Frame:
    """One parsed sample: timestamp + wrist quaternion + 10 finger quaternions."""

    __slots__ = ("t_ms", "bno", "mpu")

    def __init__(self, t_ms, bno, mpu):
        self.t_ms = t_ms          # int, device millis()
        self.bno = bno            # (w, x, y, z)
        self.mpu = mpu            # list of 10 (w, x, y, z) tuples

    @classmethod
    def from_payload(cls, payload):
        """Unpack a 180-byte payload (no markers) into a Frame."""
        vals = struct.unpack(PAYLOAD_FMT, payload)
        t_ms = vals[0]
        bno = vals[1:5]
        flat = vals[5:]
        mpu = [tuple(flat[i * 4:i * 4 + 4]) for i in range(NUM_IMU)]
        return cls(t_ms, bno, mpu)

    def __repr__(self):
        w, x, y, z = self.bno
        return f"Frame(t={self.t_ms} bno=({w:.3f},{x:.3f},{y:.3f},{z:.3f}) x{NUM_IMU})"


# ----------------------------------------------------------------------------
# Live serial source
# ----------------------------------------------------------------------------
class SerialFrameSource:
    """Reads and validates the v2 binary frame stream over USB serial.

    A background daemon thread owns the port, resynchronises on the 0xAA/0xFF
    markers, and pushes the latest good Frame into a size-1 queue so the caller
    (e.g. a 60 Hz visualizer) always gets the freshest sample without blocking
    or falling behind the 200 Hz producer.
    """

    def __init__(self, port, baud=BAUD, start_timeout=5.0):
        import serial  # pyserial

        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)  # let the board reset/boot
        self.ser.reset_input_buffer()
        self._q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Fail fast if nothing valid arrives (wrong port / wrong baud / no glove).
        if not self._wait_first(start_timeout):
            raise TimeoutError("no valid 180-byte frame; wrong port or baud?")

    def _wait_first(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._q.empty():
                return True
            time.sleep(0.02)
        return False

    def _read_exact(self, n):
        """Read exactly n bytes or return None if the port stalls/closes."""
        buf = bytearray()
        while len(buf) < n and not self._stop.is_set():
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _run(self):
        while not self._stop.is_set():
            # Sync to a start marker. Firmware also emits '#' comment lines and
            # "SERIAL READY"; those never contain 0xAA so they're skipped here.
            b = self.ser.read(1)
            if not b or b[0] != START_BYTE:
                continue
            payload = self._read_exact(PAYLOAD_LEN)
            if payload is None:
                continue
            end = self.ser.read(1)
            if not end or end[0] != END_BYTE:
                # Framing slipped; drop and resync on the next 0xAA.
                self._dropped += 1
                continue
            try:
                frame = Frame.from_payload(payload)
            except struct.error:
                self._dropped += 1
                continue
            self._push(frame)

    def _push(self, frame):
        # Keep only the newest frame: clear the stale one if the reader is slow.
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put(frame)

    def read(self, block=True, timeout=1.0):
        """Return the latest Frame, or None if none arrived within `timeout`."""
        try:
            return self._q.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def send(self, cmd):
        """Send a firmware command (e.g. '?', 'C', 'D', 'K')."""
        if not cmd.endswith("\n"):
            cmd += "\n"
        self.ser.write(cmd.encode("ascii"))

    @property
    def dropped(self):
        """Count of frames dropped to marker desync (link-health diagnostic)."""
        return self._dropped

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.ser.close()


# ----------------------------------------------------------------------------
# Simulated source (no hardware)
# ----------------------------------------------------------------------------
class SimFrameSource:
    """Synthetic frames matching the v2 wire contract, for offline development.

    A gently rocking wrist quaternion plus per-finger curl, matching the
    firmware contract: finger quaternions are expressed RELATIVE to the wrist
    frame (q_rel), so the visualizer's Rw @ R(q_rel) recovers world orientation.
    """

    def __init__(self, rate=200):
        self._rate = rate
        self.t0 = time.time()

    def read(self, block=True, timeout=None):
        t = time.time() - self.t0
        half = math.radians(20 * math.sin(t * 0.5)) / 2  # +/-20 deg wrist roll
        bno = (math.cos(half), math.sin(half), 0.0, 0.0)
        # Each finger curls about its x-axis, phase-staggered so they move
        # independently (relative-to-wrist quaternions, like the firmware sends).
        mpu = []
        for i in range(NUM_IMU):
            ang = math.radians(35.0 * (0.5 + 0.5 * math.sin(t * 1.2 + i * 0.6)))
            mpu.append((math.cos(ang / 2), math.sin(ang / 2), 0.0, 0.0))
        time.sleep(1.0 / self._rate)
        return Frame(int(t * 1000), bno, mpu)

    def send(self, cmd):
        pass

    dropped = 0

    def close(self):
        pass


def open_frame_source(port=None, simulate=False, baud=BAUD):
    """Pick a source: serial if `port` given and not simulating, else the sim."""
    if simulate or not port:
        return SimFrameSource()
    return SerialFrameSource(port, baud=baud)


# ----------------------------------------------------------------------------
# CLI: sanity-check the stream and print live frame rate (bring-up validation)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="senz v2 binary frame reader")
    ap.add_argument("--port", help="serial port, e.g. COM7 or /dev/ttyACM0")
    ap.add_argument("--simulate", action="store_true", help="no hardware")
    ap.add_argument("--baud", type=int, default=BAUD)
    args = ap.parse_args()

    src = open_frame_source(args.port, simulate=args.simulate, baud=args.baud)
    print("reading frames (Ctrl-C to stop)...")
    n, t_last, rate_t0 = 0, time.time(), time.time()
    try:
        while True:
            f = src.read()
            if f is None:
                continue
            n += 1
            now = time.time()
            if now - t_last >= 1.0:
                hz = n / (now - rate_t0)
                print(f"{hz:6.1f} Hz  dropped={src.dropped}  {f}")
                t_last = now
    except KeyboardInterrupt:
        pass
    finally:
        src.close()

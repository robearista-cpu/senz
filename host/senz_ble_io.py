#!/usr/bin/env python3
"""
senz_ble_io.py  --  Bluetooth LE transport for the self-describing v3 stream
============================================================================
The v3 glove firmware streams the SAME self-describing CSV over USB serial AND
over Bluetooth LE (Nordic UART Service). This module is the BLE counterpart to
``senz_multi_io.MultiSerialSource``: it connects to the glove over BLE, learns the
schema from the banner, and hands out parsed frames with the exact same interface
(``.schema``, ``read()``, ``send()``, ``close()``) so the visualizer / recorder
don't care which transport is used.

BLE runs on a background asyncio thread (via ``bleak``); notifications are
reassembled into lines and fed through the ``senz_multi_io`` line/schema parser.
The constructor BLOCKS until the schema is discovered (like the serial source),
so callers can read ``.schema`` immediately.

``bleak`` is only imported when BLE is actually used, and a clear message is
raised if it isn't installed (``pip install bleak``).

    from senz_ble_io import open_ble_source, scan_ble
    src = open_ble_source("senz-pinch")     # by advertised name (or an address)
    print(src.schema.nimu, src.schema.nforce)
    frame = src.read()
"""

import threading
import time

from senz_multi_io import Schema
from senz_io import NUS_RX, NUS_TX     # Nordic UART Service UUIDs (match firmware)

DEFAULT_BLE_NAME = "senz-pinch"


def _require_bleak():
    try:
        import bleak  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Bluetooth needs the 'bleak' package -- install it with "
            "`pip install bleak` (already listed in host/requirements.txt).") from e


def _win_init_mta():
    """On Windows, put the CURRENT thread's COM apartment into MTA so bleak's WinRT
    backend can deliver BLE callbacks.

    A Qt GUI thread is a Single-Threaded Apartment (STA); on it bleak raises
    "Thread is configured for Windows GUI but callbacks are not working" and no
    scan/advertisement callbacks fire, because WinRT needs either a pumped STA
    (which a blocking asyncio ``run_until_complete`` starves) or an MTA. A fresh
    worker thread starts uninitialized, so we can claim it as MTA here. Best-effort
    and a silent no-op off Windows / if the apartment is already set."""
    import sys
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        # CoInitializeEx(NULL, COINIT_MULTITHREADED = 0x0). S_OK(0)/S_FALSE(1) fine;
        # RPC_E_CHANGED_MODE means it's already STA and can't be changed -- ignore.
        ctypes.windll.ole32.CoInitializeEx(None, 0x0)
    except Exception:
        pass


def _run_bleak(coro_factory, wait):
    """Run a bleak coroutine on a DEDICATED worker thread (MTA on Windows) and
    return its result, re-raising any error. Using our own thread -- never the
    caller's, which for a GUI app is the STA main thread bleak refuses to run on --
    is what makes BLE work from inside Qt. ``coro_factory`` is a 0-arg function that
    builds the coroutine (so bleak is imported on the worker thread)."""
    import asyncio

    out = {"val": None, "err": None}

    def _worker():
        _win_init_mta()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            out["val"] = loop.run_until_complete(coro_factory())
        except Exception as e:                      # surfaced to the caller below
            out["err"] = e
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(wait)
    if t.is_alive():
        raise TimeoutError(f"BLE operation did not finish within {wait:.0f}s")
    if out["err"] is not None:
        raise out["err"]
    return out["val"]


# ----------------------------------------------------------------------------
# Line + schema assembler (pure, no BLE -- unit-testable)
# ----------------------------------------------------------------------------
class StreamAssembler:
    """Turns a byte stream of the self-describing CSV into a schema + frames.

    Feed raw notification bytes with ``feed()``. It reassembles ``\\n``-delimited
    lines and, exactly like ``MultiSerialSource._discover`` + ``Schema.parse``:
      - remembers the ``# senz-...`` banner,
      - builds ``self.schema`` when the ``# columns:`` line arrives,
      - parses subsequent data lines into ``self.latest`` (the freshest frame).
    Thread-safe; the BLE thread feeds while the GUI thread reads.
    """

    def __init__(self):
        self.schema = None
        self.latest = None
        self._banner = None
        self._buf = b""
        self._lock = threading.Lock()

    def feed(self, data):
        with self._lock:
            self._buf += data
            while b"\n" in self._buf:
                raw, self._buf = self._buf.split(b"\n", 1)
                self._line(raw.decode("utf-8", errors="ignore").strip())

    def _line(self, line):
        if not line:
            return
        if line.startswith("# senz-"):
            self._banner = line
        elif line.startswith("# columns:"):
            # A columns header (with or without a preceding banner) defines schema.
            self.schema = Schema.from_header(self._banner or "", line)
        elif line.startswith("#"):
            return                                  # other comment/status line
        elif self.schema is not None:
            frame = self.schema.parse(line)
            if frame is not None:
                self.latest = frame

    def take(self):
        """Return the freshest frame and clear it (so read() reflects new data)."""
        with self._lock:
            f, self.latest = self.latest, None
            return f


# ----------------------------------------------------------------------------
# BLE source
# ----------------------------------------------------------------------------
class BleMultiSource:
    """BLE (Nordic UART Service) source with the ``open_multi_source`` interface.

    ``target`` is an advertised device name (e.g. "senz-pinch") or a BLE address.
    The constructor blocks up to ``discover_timeout`` for the schema banner.
    """

    def __init__(self, target=DEFAULT_BLE_NAME, discover_timeout=20.0):
        _require_bleak()
        import asyncio

        self.target = target
        self._asm = StreamAssembler()
        self._ready = threading.Event()
        self._err = None
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._client = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # Block until the schema banner arrives (matches the serial source).
        deadline = time.time() + discover_timeout
        while time.time() < deadline:
            if self._asm.schema is not None:
                self.schema = self._asm.schema
                return
            if self._err is not None:
                raise RuntimeError(f"BLE connect failed: {self._err}")
            time.sleep(0.05)
        self.close()
        raise TimeoutError(
            f"no '# columns:' banner over BLE from {target!r} within "
            f"{discover_timeout:.0f}s (powered / advertising / in range?)")

    # --- background asyncio loop ---
    def _run(self):
        import asyncio

        _win_init_mta()              # MTA so WinRT BLE callbacks fire (see helper)
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:       # surfaced to the constructor / ignored later
            self._err = str(e)

    async def _main(self):
        import asyncio

        from bleak import BleakClient, BleakScanner

        # Resolve by address first, else by advertised name.
        device = None
        if ":" in self.target or "-" in self.target and len(self.target) >= 17:
            device = await BleakScanner.find_device_by_address(self.target, timeout=15.0)
        if device is None:
            device = await BleakScanner.find_device_by_name(self.target, timeout=15.0)
        if device is None:
            self._err = f"device {self.target!r} not found"
            return

        async with BleakClient(device) as client:
            self._client = client

            def on_notify(_char, data):
                self._asm.feed(bytes(data))

            await client.start_notify(NUS_TX, on_notify)
            await client.write_gatt_char(NUS_RX, b"?\n", response=False)  # request banner
            while self._running and client.is_connected:
                await asyncio.sleep(0.2)
        self._client = None

    # --- pull interface (matches the serial source) ---
    def read(self):
        return self._asm.take()

    def send(self, cmd):
        if not cmd.endswith("\n"):
            cmd += "\n"
        if self._client is None:
            return
        import asyncio

        try:
            asyncio.run_coroutine_threadsafe(
                self._client.write_gatt_char(NUS_RX, cmd.encode("utf-8"), response=False),
                self._loop)
        except Exception:
            pass

    def close(self):
        self._running = False


def open_ble_source(target=DEFAULT_BLE_NAME, discover_timeout=20.0):
    """Connect to the glove over BLE; returns a source with .schema/read/send/close."""
    return BleMultiSource(target, discover_timeout=discover_timeout)


def scan_ble(timeout=6.0):
    """Return a list of (name, address) for nearby BLE devices (for a UI picker).

    senz gloves advertise a 'senz-...' name; unnamed devices are still listed so
    the user can pick by address. Empty list if bleak is missing / nothing found.
    """
    try:
        _require_bleak()
    except RuntimeError:
        return []

    async def _scan():
        from bleak import BleakScanner
        devs = await BleakScanner.discover(timeout=timeout)
        return [(d.name or "(unknown)", d.address) for d in devs]

    # Run on a dedicated MTA worker thread -- calling bleak directly from the Qt GUI
    # (STA) thread is what triggers "callbacks are not working" on Windows.
    found = _run_bleak(_scan, wait=timeout + 10.0)
    # senz devices first, then the rest.
    found.sort(key=lambda na: (not str(na[0]).lower().startswith("senz"), str(na[0])))
    return found


def main():
    """Standalone: scan for BLE devices and print them."""
    print("scanning for BLE devices (6s) ...")
    for name, addr in scan_ble():
        print(f"  {name:24s} {addr}")


if __name__ == "__main__":
    main()

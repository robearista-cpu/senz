#!/usr/bin/env python3
"""
camera_tracker.py  --  MediaPipe Hands camera tracking (HLD objective 3)
========================================================================
Provides the camera modality: absolute hand position + 21 hand landmarks that
the IMUs cannot give. Used two ways:

  - Standalone:  python camera_tracker.py --record landmarks.csv
                 (live preview window; press q to stop)
  - As a module: the dataset recorder constructs a HandTracker, runs it on a
                 background thread, and pulls get_latest() each frame to fuse
                 camera landmarks with the serial sensor stream.

MediaPipe Hands returns, per detected hand:
  - 21 landmarks, each (x, y, z): x/y normalized to the image [0,1], z a
    relative depth (smaller = closer to the camera), wrist as the origin.
  - handedness ("Left"/"Right") with a confidence score.

mediapipe + opencv are optional heavy deps; they're imported lazily so the rest
of the host tooling runs without them. See host/requirements.txt.
"""

import threading
import time

# 21 MediaPipe hand landmark names, in index order.
LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def _coerce_source(src):
    """Camera source: an int index, or a URL string (phone IP/RTSP camera).

    ``"0"`` -> 0 (index); ``"http://..."`` stays a string for cv2.VideoCapture.
    """
    if isinstance(src, int):
        return src
    try:
        return int(src)
    except (TypeError, ValueError):
        return src


def _raw_landmarks(result):
    """Extract (landmarks_norm, score, present, handedness) from a Tasks
    HandLandmarkerResult.

    landmarks_norm is a list of 21 (x, y, z) normalized to the image ([0,1] x/y,
    relative z), or None if no hand. score is the handedness confidence;
    handedness is "Left"/"Right"/"Unknown".
    """
    if not result.hand_landmarks:
        return None, 0.0, 0, "Unknown"
    lms = result.hand_landmarks[0]
    coords = [(p.x, p.y, p.z) for p in lms]
    score, handed = 0.0, "Unknown"
    if result.handedness:
        cat = result.handedness[0][0]
        score, handed = float(cat.score), cat.category_name
    return coords, score, 1, handed


def _world_landmarks(result):
    """21 metric (x,y,z) world landmarks (meters, hand-centered) or None -- the
    3D hand shape, used by the visualizer's camera fusion."""
    w = getattr(result, "hand_world_landmarks", None)
    if not w:
        return None
    return [(p.x, p.y, p.z) for p in w[0]]


def _preflight_url(url, timeout=3.0):
    """Quick TCP reachability check for a phone-cam URL. Returns an error string if
    the host:port can't be reached fast, else None -- so a wrong IP / not-running
    app fails in ~3s instead of blocking OpenCV for the ~21s OS connect timeout."""
    import socket
    from urllib.parse import urlparse

    try:
        u = urlparse(url)
        host = u.hostname
        if not host:
            return None                      # unparseable -> let OpenCV try
        port = u.port or (554 if u.scheme == "rtsp" else 80)
        socket.create_connection((host, port), timeout=timeout).close()
        return None
    except Exception as e:
        return (f"cannot reach {url} ({e}) -- is the phone-cam app running and on the "
                "same Wi-Fi? check the IP:port and the /video path")


# MediaPipe Tasks HandLandmarker model (fetched once, cached next to this file).
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/1/hand_landmarker.task")


def ensure_hand_model(path=None):
    """Locate hand_landmarker.task, downloading it once if missing.

    Search order: $SENZ_HAND_MODEL, an explicit `path`, ./hand_landmarker.task,
    ./models/hand_landmarker.task (next to this file). If none exist, download to
    ./models/. Returns the path, or None if unavailable (offline + not cached).
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.environ.get("SENZ_HAND_MODEL"), path,
              os.path.join(here, "hand_landmarker.task"),
              os.path.join(here, "models", "hand_landmarker.task")):
        if c and os.path.exists(c):
            return c
    dest = os.path.join(here, "models", "hand_landmarker.task")
    try:
        import urllib.request
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"[camera_tracker] downloading hand model -> {dest} ...")
        urllib.request.urlretrieve(_MODEL_URL, dest)
        print("[camera_tracker] hand model ready.")
        return dest
    except Exception as e:
        print(f"[camera_tracker] hand model download failed: {e}")
        print(f'  download it manually:  curl -L -o "{dest}" "{_MODEL_URL}"')
        return None


def _swap_handed(h):
    """Left<->Right. MediaPipe reports handedness for the image it sees; when we
    mirror the frame (selfie view) that inverts it, so swap to match the real hand."""
    return {"Left": "Right", "Right": "Left"}.get(h, h)


# --- Detection-aid video filters (help MediaPipe see the hand in poor light) ---
# They affect the frame fed to the model + shown in the UI, NOT the ML dataset (the
# glove is the model input; the camera only produces landmark labels), so they are
# safe to use and purely improve label quality.
def _filter_clahe(rgb):
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def _filter_gamma(rgb, gamma=1.6):
    import cv2
    import numpy as np
    inv = 1.0 / max(0.01, gamma)
    table = (((np.arange(256) / 255.0) ** inv) * 255).astype("uint8")
    return cv2.LUT(rgb, table)


def _filter_sharpen(rgb):
    import cv2
    blur = cv2.GaussianBlur(rgb, (0, 0), 3)
    return cv2.addWeighted(rgb, 1.5, blur, -0.5, 0)


FILTER_FUNCS = {"auto_contrast": _filter_clahe, "brighten": _filter_gamma,
                "sharpen": _filter_sharpen}


def list_cameras(max_index=6):
    """Probe indices [0, max_index) and return those that open + yield a frame.

    For picking a USB / built-in camera. NOTE: this briefly opens each camera
    (webcam LED may blink). Skips busy/absent indices.
    """
    import sys
    import cv2
    win = sys.platform.startswith("win")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if win else cv2.VideoCapture(i)
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append(i)
        finally:
            cap.release()
    return found


def list_video_devices(max_index=8):
    """Probe cameras and return ``[(index, name)]`` for the ones that work.

    Friendly device names come from pygrabber's DirectShow enumeration on Windows
    (``pip install pygrabber``); without it (or on other OSes) the name falls back
    to ``"Camera <index>"``. For populating a device picker so you can plug in a
    camera, Scan, and select it by name.
    """
    idxs = list_cameras(max_index)
    names = {}
    try:                                   # best-effort friendly names (Windows)
        from pygrabber.dshow_graph import FilterGraph
        for i, name in enumerate(FilterGraph().get_input_devices()):
            names[i] = name
    except Exception:
        pass
    return [(i, names.get(i, f"Camera {i}")) for i in idxs]


class HandTracker:
    """Webcam hand tracker. Runs the MediaPipe Tasks HandLandmarker on a thread.

    (The legacy ``mp.solutions.hands`` API was dropped in mediapipe 0.10.x on
    newer Pythons, so this uses the Tasks HandLandmarker + a downloaded model.)
    Works with a built-in/USB camera index or a phone IP/RTSP URL.
    """

    def __init__(self, camera=0, max_hands=1, det_conf=0.5, track_conf=0.5,
                 show=False, keep_frame=False, mirror=True, width=None,
                 height=None, req_fps=None, filters=None, fourcc=None, det_scale=1.0):
        self.camera = _coerce_source(camera)   # int index OR URL string (phone cam)
        self.max_hands = max_hands
        self.det_conf = det_conf
        self.track_conf = track_conf
        self.show = show
        self.keep_frame = keep_frame            # also retain the RGB frame + raw landmarks
        self.mirror = mirror                    # selfie flip (natural view); handedness compensated
        self.width = width                      # requested capture resolution / fps
        self.height = height
        self.req_fps = req_fps
        self.fourcc = fourcc                    # e.g. "MJPG": compressed = far less USB bandwidth
        self.det_scale = float(det_scale or 1.0)  # downscale the frame fed to MediaPipe (CPU)
        self._filters = dict(filters or {})     # live detection-aid filters {name: bool}

        self._latest = None
        self._frame = None    # (frame_rgb, lms, score, present, t_cap, handed) or None
        self._world = None    # 21 metric (x,y,z) world landmarks or None (fusion)
        self._error = None    # last fatal thread error (surfaced to the UI)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._mp = None
        self._t0 = 0.0
        self._last_ts = -1

    # --- lifecycle ---
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        import os
        import cv2          # opencv-python

        landmarker = self._make_landmarker()
        if landmarker is None:
            return            # error recorded in self._error; thread exits cleanly
        # For URL (phone) sources: pre-flight the host:port so a wrong IP / not-running
        # app fails fast + clearly, and bound the FFMPEG read timeout.
        if isinstance(self.camera, str):
            err = _preflight_url(self.camera)
            if err:
                with self._lock:
                    self._error = err
                return
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000000")
        cap = self._open_capture(cv2)
        if not cap.isOpened():
            with self._lock:
                self._error = (f"could not open camera source {self.camera!r} -- check "
                               "the index, or that the phone-cam app / URL is live")
            cap.release()
            return
        try:
            while self._running:
                ok, frame = cap.read()
                t_cap = time.time()          # stamp CAPTURE time, before inference
                if not ok:
                    time.sleep(0.02)
                    continue
                if self.mirror:
                    frame = cv2.flip(frame, 1)               # selfie view
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = self._apply_filters(rgb)               # detection-aid filters
                result = self._detect(landmarker, rgb)
                parsed = self._parse(result, t_cap)
                with self._lock:
                    self._latest = parsed
                    self._error = None
                    self._world = _world_landmarks(result)
                    if self.keep_frame:
                        lms, score, present, handed = _raw_landmarks(result)
                        if self.mirror:
                            handed = _swap_handed(handed)
                        self._frame = (rgb, lms, score, present, t_cap, handed)
                if self.show:
                    self._preview(cv2, rgb[:, :, ::-1].copy(), result)
        except Exception as e:                # don't die silently -- report to the UI
            with self._lock:
                self._error = f"camera thread error: {e}"
        finally:
            cap.release()
            try:
                landmarker.close()
            except Exception:
                pass
            if self.show:
                cv2.destroyAllWindows()

    def _make_landmarker(self):
        """Build the Tasks HandLandmarker (VIDEO mode). Records self._error + returns
        None on any failure (missing model, bad mediapipe) so _run exits cleanly."""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker, HandLandmarkerOptions, RunningMode)
        except Exception as e:
            with self._lock:
                self._error = f"mediapipe Tasks API import failed: {e}"
            return None
        model = ensure_hand_model()
        if not model:
            with self._lock:
                self._error = ("hand_landmarker.task model missing and auto-download "
                               "failed -- see the console for a manual command")
            return None
        try:
            opts = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model),
                running_mode=RunningMode.VIDEO,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.det_conf,
                min_hand_presence_confidence=self.det_conf,
                min_tracking_confidence=self.track_conf)
            lm = HandLandmarker.create_from_options(opts)
            self._mp = mp
            self._t0 = time.time()
            self._last_ts = -1
            return lm
        except Exception as e:
            with self._lock:
                self._error = f"HandLandmarker init failed: {e}"
            return None

    def _detect(self, landmarker, rgb):
        """Run VIDEO-mode detection with monotonically increasing ms timestamps.

        When ``det_scale < 1`` the frame is downscaled JUST for detection to save
        CPU (important with several cameras); landmarks are normalized [0,1] so
        they still map onto the full-resolution display frame unchanged."""
        mp = self._mp
        det_rgb = rgb
        if self.det_scale < 0.999:
            import cv2
            h, w = rgb.shape[:2]
            det_rgb = cv2.resize(rgb, (max(1, int(w * self.det_scale)),
                                       max(1, int(h * self.det_scale))))
            det_rgb = det_rgb if det_rgb.flags["C_CONTIGUOUS"] else det_rgb.copy()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=det_rgb)
        ts = int((time.time() - self._t0) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return landmarker.detect_for_video(image, ts)

    def _preview(self, cv2, frame_bgr, result):
        """Standalone preview window: draw landmark dots (Tasks API has no drawing_utils)."""
        h, w = frame_bgr.shape[:2]
        for lms in (result.hand_landmarks or []):
            for p in lms:
                cv2.circle(frame_bgr, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)
        cv2.imshow("senz camera tracker (q to quit)", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._running = False

    def _open_capture(self, cv2):
        """Open the source. On Windows, int (USB/built-in) cameras use the
        DirectShow backend (faster/more reliable). Applies requested res/fps."""
        import sys
        if isinstance(self.camera, int) and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.camera)
        # Pixel format FIRST: MJPG is compressed, so a UVC camera sends far fewer
        # bytes over USB -- the key lever for running several USB cameras at once
        # (raw YUY2 saturates the bus). Then request resolution + fps.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.req_fps:
            cap.set(cv2.CAP_PROP_FPS, float(self.req_fps))
        return cap

    def _apply_filters(self, rgb):
        with self._lock:
            flt = dict(self._filters)
        for key, on in flt.items():
            if on and key in FILTER_FUNCS:
                try:
                    rgb = FILTER_FUNCS[key](rgb)
                except Exception:
                    pass
        return rgb

    def set_filters(self, filters):
        """Live-update the detection-aid filters, e.g. {'auto_contrast': True}."""
        with self._lock:
            self._filters = dict(filters or {})

    def _parse(self, result, t_cap=None):
        """MediaPipe result -> a flat, recorder-friendly dict (or None).

        ``t_host`` is stamped now (post-inference); ``t_cap`` is the capture time
        passed in from _run (pre-inference) -- their difference is the pipeline
        latency, and t_cap is the timestamp to align on (see camera HLD C2).
        """
        stamp = time.time()
        if t_cap is None:
            t_cap = stamp
        if not result.hand_landmarks:
            return {"t_host": stamp, "t_cap": t_cap, "hand_present": 0}
        lms = result.hand_landmarks[0]
        handed = "Unknown"
        if result.handedness:
            handed = result.handedness[0][0].category_name
            if self.mirror:
                handed = _swap_handed(handed)
        out = {"t_host": stamp, "t_cap": t_cap, "hand_present": 1, "handedness": handed}
        for i, name in enumerate(LANDMARK_NAMES):
            out[f"{name}_x"] = lms[i].x
            out[f"{name}_y"] = lms[i].y
            out[f"{name}_z"] = lms[i].z
        # Palm center: mean of wrist + the five MCP knuckles, handy as a position ref.
        mcp = ["wrist", "thumb_cmc", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"]
        for ax in "xyz":
            out[f"palm_{ax}"] = sum(out[f"{m}_{ax}"] for m in mcp) / len(mcp)
        return out

    def get_latest(self):
        with self._lock:
            return self._latest

    def get_frame(self):
        """Latest (frame_rgb, landmarks_norm, score, present, t_cap, handedness),
        or None.

        Requires ``keep_frame=True``. frame_rgb is the flipped RGB ndarray;
        landmarks_norm is a list of 21 (x,y,z) normalized to the image (or None).
        For an overlay UI that needs the image + raw landmarks together.
        """
        with self._lock:
            return self._frame

    def get_error(self):
        """Last fatal thread error string (or None) -- for the UI to display."""
        with self._lock:
            return self._error

    def get_world(self):
        """Latest 21 metric world landmarks (x,y,z) or None -- for 3D camera fusion."""
        with self._lock:
            return self._world

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    @staticmethod
    def landmark_columns():
        cols = ["hand_present", "handedness"]
        for name in LANDMARK_NAMES:
            cols += [f"{name}_x", f"{name}_y", f"{name}_z"]
        cols += ["palm_x", "palm_y", "palm_z"]
        return cols


def _record(path, camera, duration):
    """Standalone landmark recorder -> CSV."""
    import csv

    tracker = HandTracker(camera=camera, show=True).start()
    cols = ["t_host"] + HandTracker.landmark_columns()
    t_end = time.time() + duration if duration else None
    seen = 0
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        last_stamp = None
        try:
            while tracker._running:
                frame = tracker.get_latest()
                if frame and frame["t_host"] != last_stamp:
                    last_stamp = frame["t_host"]
                    w.writerow(frame)
                    seen += 1
                if t_end and time.time() > t_end:
                    break
                time.sleep(0.005)
        except KeyboardInterrupt:
            pass
    tracker.stop()
    print(f"Recorded {seen} frames -> {path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="senz MediaPipe hand tracker")
    ap.add_argument("--camera", default="0",
                    help="camera index (0,1,...) or a URL (phone IP/RTSP camera)")
    ap.add_argument("--record", metavar="CSV", help="record landmarks to CSV")
    ap.add_argument("--duration", type=float, default=0, help="seconds (0 = until q)")
    ap.add_argument("--width", type=int, help="requested capture width")
    ap.add_argument("--height", type=int, help="requested capture height")
    ap.add_argument("--fps", type=int, help="requested capture fps")
    ap.add_argument("--list", action="store_true", help="scan + print working camera indices")
    args = ap.parse_args()

    if args.list:
        print("working camera indices:", list_cameras())
    elif args.record:
        _record(args.record, args.camera, args.duration)
    else:
        t = HandTracker(camera=args.camera, show=True, width=args.width,
                        height=args.height, req_fps=args.fps).start()
        try:
            while t._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            t.stop()

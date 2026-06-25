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


class HandTracker:
    """Webcam hand tracker. Runs MediaPipe Hands on a background thread."""

    def __init__(self, camera=0, max_hands=1, det_conf=0.5, track_conf=0.5,
                 show=False):
        self.camera = camera
        self.max_hands = max_hands
        self.det_conf = det_conf
        self.track_conf = track_conf
        self.show = show

        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    # --- lifecycle ---
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        import cv2          # opencv-python
        import mediapipe as mp

        hands = mp.solutions.hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.det_conf,
            min_tracking_confidence=self.track_conf,
        )
        draw = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(self.camera)
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                parsed = self._parse(result)
                with self._lock:
                    self._latest = parsed

                if self.show:
                    if result.multi_hand_landmarks:
                        for lm in result.multi_hand_landmarks:
                            draw.draw_landmarks(
                                frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
                    cv2.imshow("senz camera tracker (q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._running = False
        finally:
            cap.release()
            hands.close()
            if self.show:
                cv2.destroyAllWindows()

    def _parse(self, result):
        """MediaPipe result -> a flat, recorder-friendly dict (or None)."""
        stamp = time.time()
        if not result.multi_hand_landmarks:
            return {"t_host": stamp, "hand_present": 0}
        lms = result.multi_hand_landmarks[0].landmark
        handed = "Unknown"
        if result.multi_handedness:
            handed = result.multi_handedness[0].classification[0].label
        out = {"t_host": stamp, "hand_present": 1, "handedness": handed}
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
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--record", metavar="CSV", help="record landmarks to CSV")
    ap.add_argument("--duration", type=float, default=0, help="seconds (0 = until q)")
    args = ap.parse_args()

    if args.record:
        _record(args.record, args.camera, args.duration)
    else:
        t = HandTracker(camera=args.camera, show=True).start()
        try:
            while t._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            t.stop()

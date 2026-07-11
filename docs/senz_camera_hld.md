# HLD: Senz Camera — Ground-Truth Labeling Side

**Branch (suggested):** `feature/camera-groundtruth` off `main`
**Goal:** Use a **phone camera** (monocular RGB, fixed and facing the hand) as a
**ground-truth labeler**: record MediaPipe hand pose **tightly time-synced** to the
glove's tactile + gross-IMU stream, so we can train models that predict finger
pose / grip **from the glove alone**. The camera is a *teacher at training time*,
**not** required at inference.

This complements Hardware Sprint v3 (tactile-first): the glove deliberately dropped
per-finger IMUs and leans on the camera for fine finger motion. The camera supplies
exactly what the glove no longer measures — **per-finger articulation** and
**absolute hand position** — as labels.

---

## Locked decisions (from review)

| Question | Decision | Consequence |
|----------|----------|-------------|
| Camera role | **Ground-truth labeler** | Camera not in the inference loop; optimize for *label fidelity*, not latency. |
| Hardware | **Phone camera** (monocular RGB, used as a webcam) | 2D + rough relative depth only; metric 3D is unreliable → labels must be view-stable, not raw metric xyz. |
| Placement | **Fixed, facing the hand** | Constant camera pose → consistent image mapping, simpler (optional) calibration; you work inside its view. |
| Top priority | **Time-sync + dataset recording** | The core engineering effort is aligning camera frames to the serial clock; everything else is secondary. |

## What is changing / not changing

**Changing (this sprint):**
- Phone-as-webcam ingestion path + **capture-time** stamping in `camera_tracker.py`.
- Camera↔sensor **clock alignment** and a **sync-validation** tool.
- Recorder + prep upgrades to carry camera timestamps/confidence and to **interpolate**
  (not just hold) landmarks, and to split **glove features (X)** vs **camera labels (Y)**.
- A **capture protocol** doc (rig, lighting, sync gesture, session labels).

**NOT changing:** the glove firmware/stream (serial stays the timing master), the
`force_pipeline`, the fusion, the `senz_v3_qt.py` viz. **No live fusion, no depth
camera, no multi-cam, no metric 3D reconstruction** this sprint (see Roadmap).

---

## Architecture / data flow

```
  PHONE CAM (RGB, ~30 fps)                    GLOVE (ESP32-S3, 200 Hz)
   |  webcam bridge (USB app or IP/RTSP URL)   |  USB serial CSV, firmware t_us
   v                                           v
  camera_tracker.HandTracker                  senz_multi_io.MultiSerialSource
   - stamp t_cap AT cap.read()  <-- key        - t_us (master clock)
   - MediaPipe Hands -> 21 landmarks           - tactile (force) + dorsum IMU + wrist
   - det_conf, hand_present                    - host tags t_host at read
        \                                     /
         \___________ dataset_recorder.py ___/
                 serial = timing master; every serial row carries
                 (t_us, t_host) + processed force; camera joined by
                 t_cap (nearest, later interpolated). writes frames.csv + meta.json
                                   |
                                   v
                 dataset_prep.py  (+ clock fit + interpolation + label extraction)
                   - fit t_us <-> t_host (per session)
                   - map camera t_cap -> t_us timeline
                   - interpolate landmarks onto the 100 Hz grid
                   - derive VIEW-STABLE labels (finger angles, hand-local landmarks)
                   - split X = glove features, Y = camera labels
                                   |
                                   v
                 data/prepared/<name>.npz   (X glove, Y camera-truth, per window)
```

---

## Objectives

Numbered like the existing HLD objectives; each maps to concrete files.

### C1 — Phone-as-webcam ingestion
Two supported paths (document both, default to whichever the phone supports best):
- **USB/Wi-Fi webcam app** (DroidCam / Iriun / Camo): phone appears as a normal
  camera index → `cv2.VideoCapture(index)` works unchanged.
- **IP camera** (MJPEG/RTSP URL from an app): `cv2.VideoCapture("http://<phone-ip>:port/video")`.
`camera_tracker.py`: add a `--url` / `source=` option so `VideoCapture` accepts an int
**or** a URL string. Request the highest stable fps; log the achieved fps into `meta.json`.
Honest note: IP/RTSP adds variable buffering latency → makes C2/C3 mandatory.

### C2 — Capture-time timestamping (sync core, part 1)
**Problem:** today `HandTracker._parse` stamps `t_host = time.time()` **after** MediaPipe
inference (`camera_tracker.py:100`). On CPU that lags the actual photon capture by the
grab + inference time (~30–80 ms), a *systematic* bias that smears labels during motion.
**Fix:** stamp **`t_cap` immediately after `cap.read()` returns**, carry it through
`_parse` into the landmark dict, and use `t_cap` (not the post-inference stamp) for all
alignment. Keep the post-inference stamp too, as `t_proc`, to *measure* pipeline latency.

### C3 — Camera↔sensor clock alignment (sync core, part 2)
Two clocks exist: firmware **`t_us`** (precise, evenly spaced, the master) and host
**wall clock** (what the camera has). Per session:
1. **Clock fit:** least-squares fit `t_us ≈ a·t_host + b` over all serial rows (the
   recorder already logs both per row) → a linear map between the timelines.
2. **Map camera in:** convert each camera `t_cap` into the `t_us` timeline via that fit.
3. **Residual offset:** a per-session **sync gesture** (a sharp finger tap / clap) makes a
   spike in *both* the velostat/IMU signal and the camera landmark velocity. Cross-correlate
   the two to estimate a constant residual offset (exposure + transport latency) and subtract
   it. Target alignment error **< ~1 camera frame (~33 ms)**; report the estimate.

### C4 — Recorder upgrades
`dataset_recorder.py`: serial stays the master (unchanged). Add camera columns
`cam_t_cap`, `cam_t_proc`, `cam_det_conf` alongside the existing `cam_*` landmarks; the
sample-and-hold join stays for the raw file (interpolation happens in prep). Record the
sync-gesture window (a keypress marker or just "tap in the first 3 s") into `meta.json`.

### C5 — Label representation (what the camera actually teaches)
Monocular MediaPipe gives image-normalized `x,y ∈ [0,1]` + **rough** relative `z`; raw
metric 3D is not trustworthy. So labels are **view-stable derived quantities**, computed
in `dataset_prep.py`:
- **Per-finger flexion angles** — angles across each finger's MCP→PIP→DIP→TIP landmarks
  (the curl the glove no longer senses). Primary target.
- **Hand-local normalized landmarks** — translate to the wrist origin, scale by a hand-size
  reference (e.g. wrist→middle_mcp), optionally rotate to a canonical palm frame from the
  MCP line. Removes camera translation/scale (and most rotation) → labels don't depend on
  where the hand sits in frame.
- **Absolute hand position** — the raw palm center (`palm_x/y`) kept as a *coarse* position
  label (the one thing IMUs can't give), flagged as low-precision.
- **Grasp/contact state** — optional derived label (closed-hand / open) for classification.

### C6 — Occlusion & quality gating
The velostat/faraday over the fingertips and the wrist/dorsum module hide fingers; MediaPipe
will still *hallucinate* hidden joints at low confidence. Add:
- `cam_det_conf` + `hand_present` gating → a per-frame **`label_valid`** flag; frames below
  threshold (or with no hand) produce **no target** (masked in training), never a bad label.
- A **max-gap** rule: if the camera dropped out longer than N frames, don't interpolate
  across it — mask instead.
- Rig guidance: aim the camera at the side where fingers are most visible; expect
  faraday-wrapped fingertips to be the worst case.

### C7 — Sync validation tool (new `host/sync_check.py`)
Given a recorded session, run C3's clock fit + cross-correlation on the sync gesture and
**report**: estimated offset, residual jitter, % frames with valid labels, camera fps,
mean pipeline latency (`t_proc − t_cap`). This is the go/no-go gate on a session's usability.

### C8 — Dataset-prep integration
`dataset_prep.py`: (a) apply the C3 clock map so camera `t_cap` lands on the `t_us` grid;
(b) **linear-interpolate** landmark/angle columns between valid camera samples instead of
pure forward-fill (`clean()` currently ffills `cam_*`); (c) tag columns into **feature
group X** (glove: `imu*`, `bno_q*`, `force*`) vs **label group Y** (camera-derived), and
export both in the `.npz` so training uses glove→camera. Keep masked/invalid targets out.

### C9 — Capture protocol (new `docs/CAMERA_CAPTURE.md`)
Fixed phone mount + framing of the working volume; lighting; per-session **sync gesture**;
label naming (`grasp_cup`, `pinch`, `open_close`…); session length; "keep the hand in frame"
rules; a checklist. Deterministic, repeatable sessions = clean labels.

---

## Deep dive: the time-sync budget (the priority)

Why this is the whole game for a labeler: if a label is 60 ms early during a fast finger
curl, the model is taught the wrong glove→pose mapping. Contributions to misalignment and
how each is handled:

| Source | Typical | Handling |
|--------|---------|----------|
| Post-inference stamping bias | 30–80 ms | **Eliminated** by capture-time stamp `t_cap` (C2). |
| Two independent clocks | drift/offset | **Linear `t_us↔t_host` fit** per session (C3.1). |
| Exposure + USB/IP transport latency | 10–50 ms, ~constant | **Estimated + subtracted** via sync-gesture cross-correlation (C3.3). |
| Camera 30 fps vs glove 200 Hz | ≤ 33 ms quantization | **Interpolate** landmarks onto the grid (C8); mask fast-motion frames if needed. |
| Camera dropouts / occlusion | variable | **Max-gap mask + confidence gate** (C6). |

**Acceptance:** `sync_check.py` reports residual alignment **< ~1 camera frame** and a
labeled-frame yield the session can be judged on. Anything worse → recapture.

---

## Data schema additions (frames.csv)

Existing glove columns unchanged. New camera columns:
`cam_t_cap`, `cam_t_proc`, `cam_det_conf`, `cam_hand_present`, `cam_handedness`,
`cam_<landmark>_{x,y,z}` (21), `cam_palm_{x,y,z}`. Prep adds derived target columns:
`lbl_<finger>_flex`, `lbl_<landmark>_{x,y,z}` (hand-local normalized), `lbl_valid`.

---

## Verification

- **No-hardware:** a `SimCameraSource` (scripted 21-landmark motion) lets `dataset_recorder`
  and `sync_check` run headless; assert clock fit recovers a known injected offset.
- **Sync:** record a real tap; `sync_check.py` must recover the offset to < 1 frame and the
  interpolation must reduce label-vs-sensor lag vs the current ffill.
- **Occlusion:** frames with `det_conf` below threshold produce `lbl_valid=0` and no target.
- **Prep:** unit-test the clock map, interpolation, and X/Y split on a synthetic session;
  confirm `.npz` carries glove X and camera Y with masks.
- **End-to-end:** `dataset_recorder --simulate` (glove sim) + `SimCameraSource` → prep →
  `.npz`; shapes and feature/label groups sane.

## Risks / honest limitations

- **Monocular depth is weak** → we label with view-stable angles + normalized landmarks, not
  raw metric xyz; absolute hand position is coarse.
- **Phone-webcam latency varies** (esp. IP/RTSP) → we do not trust wall-clock naively; C2+C3
  (capture stamping + per-session offset estimation) are mandatory, not optional.
- **MediaPipe hallucinates occluded fingers** → confidence gating + masking; faraday-wrapped
  fingertips are the worst case, expect gaps.
- **30 fps aliases fast finger motion** → interpolation + optional fast-motion masking; keep
  demonstrations at moderate speed for cleanest labels.
- Legacy `mp.solutions.hands` gives only a handedness score, not per-landmark visibility;
  if finer gating is needed, migrate to the **MediaPipe Tasks HandLandmarker** (presence +
  visibility per landmark) — noted as a possible C6 upgrade.

## Roadmap (explicitly out of scope now)

Live camera fusion into `senz_v3_qt.py`; depth camera (RealSense) for metric 3D + easier
registration; multi-cam triangulation; egocentric rig; full camera↔glove hand-eye
calibration. This sprint deliberately stops at **synced, quality-gated ground-truth
labels feeding the ML dataset**.

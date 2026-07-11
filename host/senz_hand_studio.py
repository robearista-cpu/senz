#!/usr/bin/env python3
"""
senz_hand_studio.py  --  the "final result" mesh hand (all data, one view)
==========================================================================
Where ``senz_v3_qt.py`` is the diagnostic view (sticks + spheres + per-sensor
controls), THIS is the beauty shot: it takes the SAME fully-fused hand and skins
it into a solid **mesh hand** -- tapered finger segments, rounded joints, a domed
palm + forearm -- and paints the **tactile data onto it** (the thumb/index/middle
fingertips and the palm glow with pressure). It reads as a robot / mannequin hand
that shows everything at once:

  - orientation  : the dorsum IMU orients the whole hand (BNO055 = forearm/wrist).
  - articulation : finger IMUs curl the fingers (proto/pinch build), or the CAMERA
                   articulates them when ``--camera`` is given (IMU still orients).
  - tactile      : the velostat force pads glow on the fingertips + palm.

It reuses the pure pose + force pipeline from senz_v3_qt (``compute_skeleton``,
``SensorConfigV3``, Madgwick fusion, the force pipeline, camera fusion) and the
21-landmark ``hand_model`` -- so it is exactly the same information, rendered
beautifully instead of diagnostically.

    python senz_hand_studio.py --simulate --sim pinch     # no hardware
    python senz_hand_studio.py --port COM5 --camera 0      # wired glove + camera fusion
    python senz_hand_studio.py --ble senz-pinch            # over Bluetooth

Same transports/flags as senz_v3_qt (--port/--ble/--simulate/--sim/--hand/--camera/
--fingers). The mesh-geometry helpers are pure numpy and headless-testable.
"""

import argparse
import math
import sys

import numpy as np

import hand_model as hmod
from senz_v3_qt import (
    THEMES, SensorConfigV3, compute_skeleton, finger_imu_map, make_palm,
    build_wrist_palm, align_z_to, ThreadedMultiReader, _wrist_quat, FORCE_BASE,
    FORCE_FINGERTIP, FOREARM_LEN, WRIST_RADIUS, _EY)

# ----------------------------------------------------------------------------
# Mesh geometry helpers (pure numpy -- headless-testable)
# ----------------------------------------------------------------------------
# Per-tier segment radii along a finger (mcp -> pip -> dip -> tip): fingers taper
# to the tip. The wrist is the thickest. Thumb is beefier, pinky slimmer.
_TIER_R = [0.30, 0.26, 0.225, 0.185]
_FINGER_R_SCALE = {"thumb": 1.15, "index": 1.0, "middle": 1.05,
                   "ring": 0.95, "pinky": 0.85}
WRIST_R = 0.46


def _build_radii():
    r = {0: WRIST_R}
    for finger, lms in hmod.FINGER_LANDMARKS.items():
        s = _FINGER_R_SCALE.get(finger, 1.0)
        for tier, lm in enumerate(lms):
            r[lm] = _TIER_R[tier] * s
    return r


LANDMARK_RADIUS = _build_radii()


def landmark_radius(i):
    return LANDMARK_RADIUS.get(i, 0.2)


def mesh_bones(fingers):
    """Bones to skin as tapered tubes: the phalanges WITHIN each finger (mcp-pip-
    dip-tip). The knuckle cross-links are left to the palm mesh; the thumb also
    gets a root tube to the wrist so it attaches. Returns [(a, b), ...]."""
    bones = []
    for f in fingers:
        lms = hmod.FINGER_LANDMARKS[f]
        for j in range(len(lms) - 1):
            bones.append((lms[j], lms[j + 1]))
        if f == "thumb":
            bones.append((0, lms[0]))     # wrist -> thumb CMC (ground the thumb)
    return bones


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def glow_color(base, grip):
    """Blend a base color toward hot orange then white-hot as grip 0..1 rises --
    a fingertip 'pressure glow' that reads at a glance."""
    g = _clamp01(grip)
    hot = np.array([1.0, 0.5, 0.15])
    white = np.array([1.0, 0.95, 0.82])
    base = np.array(base, dtype=float)
    if g < 0.6:
        c = base + (hot - base) * (g / 0.6)
    else:
        c = hot + (white - hot) * ((g - 0.6) / 0.4)
    return tuple(float(v) for v in c)


def finger_base_color(landmark, metallic):
    """Muted per-finger tint over a metallic base, so orientation still reads but
    the hand looks like one material rather than a rainbow."""
    finger = hmod.FINGER_OF[landmark]
    tint = np.array(hmod.FINGER_COLORS.get(finger, hmod.PALM_COLOR))
    base = np.array(metallic, dtype=float)
    c = base + (tint - base) * 0.28
    return tuple(float(v) for v in c)


def _grip_of(finger, fp):
    """Mean relative grip of a finger's 2x2 force pad (0..1)."""
    if not fp or finger not in FORCE_BASE:
        return 0.0
    base = FORCE_BASE[finger]
    vals = [fp[base + c]["relative_grip"] for c in range(4) if base + c < len(fp)]
    return float(np.mean(vals)) if vals else 0.0


# ----------------------------------------------------------------------------
# Mesh renderer (pyqtgraph OpenGL) -- imported lazily so headless import works
# ----------------------------------------------------------------------------
class MeshHand:
    """Skins the fused 21-landmark pose into a solid mesh hand with force glow."""

    def __init__(self, view, fingers, palm_taxels, theme="dark"):
        import pyqtgraph.opengl as gl

        self.gl = gl
        self.fingers = list(fingers)
        self.palm_taxels = palm_taxels or []
        self._metallic = (0.60, 0.64, 0.72) if theme == "dark" else (0.80, 0.82, 0.88)

        self._bones = mesh_bones(self.fingers)
        self._lms = hmod.active_landmarks(self.fingers)
        # Force glows at each active thumb/index/middle fingertip + its last bone.
        self._force_fingers = [f for f in FORCE_FINGERTIP if f in self.fingers]

        mesh_kw = dict(smooth=True, shader="shaded", drawEdges=False, glOptions="opaque")
        sph = gl.MeshData.sphere(rows=14, cols=22, radius=1.0)
        self._sv, self._sf = sph.vertexes(), sph.faces()

        # Forearm stub: a unit tapered cylinder we stretch behind the wrist.
        facyl = gl.MeshData.cylinder(rows=1, cols=18,
                                     radius=[WRIST_RADIUS, WRIST_RADIUS * 0.82], length=1.0)
        self._fa_v, self._fa_f = facyl.vertexes(), facyl.faces()
        self.forearm = gl.GLMeshItem(vertexes=self._fa_v, faces=self._fa_f,
                                     color=self._metallic + (1.0,), **mesh_kw)
        view.addItem(self.forearm)

        # Domed palm slab (flexes with the wrist).
        v, f = build_wrist_palm(np.zeros(3), np.eye(3), np.eye(3),
                                half_w=1.85, half_t=0.46, arch=0.5)
        self.palm = gl.GLMeshItem(vertexes=v, faces=f,
                                  color=self._shade(self._metallic, 1.05) + (1.0,), **mesh_kw)
        view.addItem(self.palm)

        # One tapered tube per phalange (unit-length cylinder, radius per endpoint).
        self._bone_v, self._bone_f, self.bone_items, self.bone_base = [], [], [], []
        for a, b in self._bones:
            cyl = gl.MeshData.cylinder(rows=1, cols=16,
                                       radius=[landmark_radius(a), landmark_radius(b)],
                                       length=1.0)
            self._bone_v.append(cyl.vertexes())
            self._bone_f.append(cyl.faces())
            col = finger_base_color(b, self._metallic)
            it = gl.GLMeshItem(vertexes=cyl.vertexes(), faces=cyl.faces(),
                               color=col + (1.0,), **mesh_kw)
            view.addItem(it)
            self.bone_items.append(it)
            self.bone_base.append(col)
        # (a,b) -> finger, for the last phalange of each force finger (glow target).
        self._tip_bone = {}
        for fg in self._force_fingers:
            lms = hmod.FINGER_LANDMARKS[fg]
            self._tip_bone[(lms[-2], lms[-1])] = fg

        # One sphere per active joint (smooths the segment junctions).
        self.joint_items, self.joint_base = {}, {}
        for lm in self._lms:
            col = finger_base_color(lm, self._metallic)
            jt = gl.GLMeshItem(vertexes=self._sv, faces=self._sf, color=col + (1.0,), **mesh_kw)
            view.addItem(jt)
            self.joint_items[lm] = jt
            self.joint_base[lm] = col
        self._tip_joint = {FORCE_FINGERTIP[fg]: fg for fg in self._force_fingers}

        # Palm taxel glow (additive points on the palm surface).
        self.palm_scatter = None
        if self.palm_taxels:
            self.palm_scatter = gl.GLScatterPlotItem(
                pos=np.zeros((len(self.palm_taxels), 3)), size=0.001, pxMode=False)
            self.palm_scatter.setGLOptions("additive")
            view.addItem(self.palm_scatter)

    @staticmethod
    def _shade(c, k):
        return tuple(min(1.0, v * k) for v in c)

    def _sphere_at(self, center, radius):
        return self._sv * radius + np.asarray(center)

    def _tube(self, verts, base, tip):
        d = np.asarray(tip) - np.asarray(base)
        L = max(1e-6, float(np.linalg.norm(d)))
        v = verts.copy()
        v[:, 2] *= L
        return (align_z_to(d) @ v.T).T + np.asarray(base)

    def update(self, skel, fp):
        wp, Rf, Rh = skel["wrist_pos"], skel["R_forearm"], skel["R_hand"]
        pts = skel["points"]
        grips = {fg: _grip_of(fg, fp) for fg in self._force_fingers}

        v, f = build_wrist_palm(wp, Rf, Rh, half_w=1.85, half_t=0.46, arch=0.5)
        self.palm.setMeshData(vertexes=v, faces=f)
        fa_dir = Rf @ (-_EY)                       # forearm points back from the wrist
        self.forearm.setMeshData(vertexes=self._tube(self._fa_v, wp, wp + fa_dir * FOREARM_LEN),
                                 faces=self._fa_f)

        for (a, b), verts, faces, it, base in zip(self._bones, self._bone_v,
                                                  self._bone_f, self.bone_items, self.bone_base):
            it.setMeshData(vertexes=self._tube(verts, pts[a], pts[b]), faces=faces)
            fg = self._tip_bone.get((a, b))
            if fg is not None:
                it.setColor(glow_color(base, grips[fg]) + (1.0,))

        for lm, jt in self.joint_items.items():
            r = landmark_radius(lm)
            fg = self._tip_joint.get(lm)
            if fg is not None:                     # fingertip: swell + glow with force
                g = grips[fg]
                jt.setMeshData(vertexes=self._sphere_at(pts[lm], r * (1.0 + 0.35 * g)),
                               faces=self._sf)
                jt.setColor(glow_color(self.joint_base[lm], g) + (1.0,))
            else:
                jt.setMeshData(vertexes=self._sphere_at(pts[lm], r), faces=self._sf)

        self._palm_patch(wp, Rh, fp)

    def _palm_patch(self, wp, Rh, fp):
        if self.palm_scatter is None or not fp:
            return
        pts, cols, sizes = [], [], []
        for _lbl, ch, local in self.palm_taxels:
            rel = fp[ch]["relative_grip"] if ch < len(fp) else 0.0
            pts.append(np.asarray(wp) + Rh @ np.array(local, dtype=float))
            cols.append(glow_color((0.35, 0.4, 0.5), rel) + (min(1.0, 0.35 + rel),))
            sizes.append(0.22 + 0.4 * rel)
        self.palm_scatter.setData(pos=np.array(pts), color=np.array(cols),
                                  size=np.array(sizes))


# ----------------------------------------------------------------------------
# Compact "studio" control panel
# ----------------------------------------------------------------------------
class StudioPanel:
    def __init__(self, on_zero_hand, on_zero_force):
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(280)
        lay = QtWidgets.QVBoxLayout(panel)
        self._theme = "dark"
        self._theme_cb = None
        self.theme_btn = QtWidgets.QPushButton("Theme: Dark")
        self.theme_btn.clicked.connect(self._toggle)
        lay.addWidget(self.theme_btn)
        lay.addWidget(QtWidgets.QLabel("<b>senz hand studio</b>"))
        lay.addWidget(QtWidgets.QLabel(
            "<i>the fully-fused hand: orientation + articulation + tactile glow</i>"))
        for text, cb in (("Zero hand (tare pose)", on_zero_hand),
                         ("Zero force", on_zero_force)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(cb)
            lay.addWidget(b)
        self.legend = QtWidgets.QLabel()
        self.legend.setFixedHeight(18)
        self.legend.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 rgb(38,44,58), "
            "stop:0.6 rgb(255,128,38), stop:1 rgb(255,242,209)); border-radius:3px;")
        lay.addWidget(QtWidgets.QLabel("force glow  (low → high)"))
        lay.addWidget(self.legend)
        self.info = QtWidgets.QLabel()
        self.info.setFont(QtGui.QFont("Courier New", 9))
        self.info.setAlignment(QtCore.Qt.AlignTop)
        lay.addWidget(self.info, stretch=1)
        self.widget = panel

    def set_theme_callback(self, cb):
        self._theme_cb = cb

    def _toggle(self):
        self._theme = "light" if self._theme == "dark" else "dark"
        self.theme_btn.setText(f"Theme: {self._theme.capitalize()}")
        if self._theme_cb:
            self._theme_cb(self._theme)

    def set_info(self, text):
        self.info.setText(text)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    import time

    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtWidgets

    import pinch as pinchmod
    import senz_multi_io as mio
    from fusion.madgwick import MadgwickAHRS
    from force_pipeline import ForceArray, process_frame

    ap = argparse.ArgumentParser(description="senz hand studio -- fused mesh hand")
    ap.add_argument("--port", help="serial port, e.g. COM5")
    ap.add_argument("--ble", metavar="NAME", help="connect over Bluetooth LE")
    ap.add_argument("--simulate", action="store_true", help="no hardware")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--beta", type=float, default=0.1, help="Madgwick gain")
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--sim", choices=["tactile", "proto", "pinch"], default="pinch")
    ap.add_argument("--camera", metavar="SRC", help="camera source to FUSE (articulates fingers)")
    ap.add_argument("--fingers", metavar="LIST", help="fingers to draw (default: build-based)")
    args = ap.parse_args()

    if args.ble:
        from senz_ble_io import open_ble_source
        source = open_ble_source(args.ble)
    elif args.simulate or not args.port:
        if args.sim == "proto":
            from senz_v3_sim import SimV3Source
            source = SimV3Source()
        elif args.sim == "pinch":
            from senz_v3_pinch_sim import SimV3PinchSource
            source = SimV3PinchSource()
        else:
            from senz_v3_tactile_sim import SimV3TactileSource
            source = SimV3TactileSource()
    else:
        source = mio.open_multi_source(port=args.port, baud=args.baud)
    reader = ThreadedMultiReader(source)
    schema = reader.schema

    n_imu = max(0, schema.nimu)
    dorsum_sensor = n_imu if n_imu >= 1 else 0
    imu_map = finger_imu_map(n_imu)
    is_pinch = args.sim == "pinch" or "pinch" in (schema.build or "")
    if args.fingers:
        active_fingers = hmod.parse_fingers(args.fingers)
    elif is_pinch:
        active_fingers = list(hmod.PINCH_FINGERS)
    else:
        active_fingers = list(hmod.FINGERS)

    cfg = SensorConfigV3(1 + n_imu)
    filters = [MadgwickAHRS(beta=args.beta) for _ in range(n_imu)]
    forces = ForceArray(schema.nforce, rate=schema.rate or 200)
    state = {"prev_t": None, "raw_quats": None, "fps": 0.0, "tlast": None}
    has_palm = schema.nforce >= 15
    palm = make_palm(args.hand) if has_palm else None

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.setObjectName("studio")
    win.setWindowTitle(f"senz hand studio ({args.hand})")
    win.resize(1120, 720)
    root = QtWidgets.QHBoxLayout(win)

    view = gl.GLViewWidget()
    view.setCameraPosition(distance=12, elevation=20, azimuth=-70)
    grid = gl.GLGridItem()
    grid.setSize(16, 16)
    grid.setSpacing(1, 1)
    grid.translate(0, 0, -3.2)
    view.addItem(grid)
    root.addWidget(view, stretch=4)

    hand = MeshHand(view, active_fingers, palm, theme="dark")

    tracker = None
    if args.camera:
        try:
            from camera_tracker import HandTracker
            src = int(args.camera) if str(args.camera).isdigit() else args.camera
            tracker = HandTracker(camera=src, keep_frame=False).start()
            print(f"camera fusion ON (source {args.camera!r})")
        except Exception as e:
            print(f"camera fusion unavailable ({e}); IMU-only")

    def zero_hand():
        if state["raw_quats"]:
            cfg.capture_zero(state["raw_quats"])

    def zero_force():
        forces.__init__(schema.nforce, rate=schema.rate or 200)

    panel = StudioPanel(lambda: zero_hand(), lambda: zero_force())
    root.addWidget(panel.widget, stretch=1)

    def apply_theme(mode):
        th = THEMES[mode]
        view.setBackgroundColor(th["gl_bg"])
        try:
            grid.setColor(th["grid"])
        except Exception:
            pass
        panel.widget.setStyleSheet(th["qss"])
        win.setStyleSheet(f'QWidget#studio {{ background-color: {th["win_bg"]}; }}')

    panel.set_theme_callback(apply_theme)
    apply_theme("dark")

    def tick():
        frame = reader.get()
        if frame is None:
            return
        t_us = frame.get("t_us", 0)
        if state["prev_t"] is None:
            dt = 1.0 / (schema.rate or 200)
        else:
            dt = max(1e-4, min(0.05, (t_us - state["prev_t"]) * 1e-6))
        state["prev_t"] = t_us
        raw_quats = [_wrist_quat(frame)]
        d2r = math.pi / 180.0
        for k in range(n_imu):
            ax, ay, az = mio.imu_accel_g(frame, k)
            gx, gy, gz = mio.imu_gyro_dps(frame, k)
            raw_quats.append(filters[k].update(gx * d2r, gy * d2r, gz * d2r, ax, ay, az, dt))
        state["raw_quats"] = raw_quats
        world = tracker.get_world() if tracker is not None else None
        skel = compute_skeleton(raw_quats, cfg, dorsum_sensor, args.hand,
                                world_lms=world, imu_map=imu_map)
        fp = process_frame(frame, forces)
        hand.update(skel, fp)

        now = time.time()
        if state["tlast"] is not None:
            state["fps"] += 0.1 * (1.0 / max(1e-6, now - state["tlast"]) - state["fps"])
        state["tlast"] = now
        pf = pinchmod.pinch_features(skel["points"], fp)
        lines = [f"fps ~ {state['fps']:4.1f}    hand={args.hand}",
                 f"drive: {skel['driven']}",
                 f"wrist flex {skel['flex_deg']:5.1f} deg", ""]
        for c in ("thumb", "index", "middle"):
            if c in active_fingers:
                lines.append(f"{c:6s} grip {_grip_of(c, fp):.2f}")
        lines += ["", f"pinch: {pf['state'].upper()}"]
        panel.set_info("\n".join(lines))

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(10)
    win.show()
    try:
        (app.exec_ if hasattr(app, "exec_") else app.exec)()
    finally:
        reader.close()
        if tracker is not None:
            tracker.stop()


if __name__ == "__main__":
    main()

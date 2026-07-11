#!/usr/bin/env python3
"""
senz_hub.py  --  control hub / launcher for the senz tools
==========================================================
One small window to launch the pieces from a shared set of settings (glove port,
camera source, hand, sim kind, label). Each tool opens in its **own** window
(spawned via QProcess); the hub stays open and shows running/stopped status.

  - Camera setup ...... camera_setup.py   (frame/light the phone or USB camera)
  - Hand visualizer ... senz_v3_qt.py      (3D IMU hand; optional camera fusion)
  - Record ............ dataset_recorder.py (glove + camera -> synced dataset)
                        = "connect everything"

Blank glove port -> the tools run in --simulate. Blank camera -> camera setup
runs in --demo. One webcam feeds one program at a time, so the hub warns before
starting a second camera consumer.

    python senz_hub.py

The arg-building + config helpers are pure (Qt imported lazily), so they are
unit-testable headless.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PROGRAMS = [
    ("Camera setup", "camera_setup"),
    ("Hand visualizer", "viz"),
    ("Record  (connect everything)", "record"),
]

DEFAULT_HUB_CONFIG = {
    "port": "",          # glove serial port; blank -> --simulate
    "source": "0",       # camera index or URL; blank -> --demo (camera setup)
    "hand": "right",
    "sim": "tactile",    # senz_v3_qt --sim
    "label": "",
    "fuse": False,       # feed the camera into the hand visualizer
    "theme": "dark",
}


# ----------------------------------------------------------------------------
# Pure helpers (headless-testable)
# ----------------------------------------------------------------------------
def load_hub_config(path):
    cfg = dict(DEFAULT_HUB_CONFIG)
    if path and os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        for k in DEFAULT_HUB_CONFIG:
            if k in data:
                cfg[k] = data[k]
    return cfg


def save_hub_config(path, cfg):
    with open(path, "w") as fh:
        json.dump({k: cfg.get(k) for k in DEFAULT_HUB_CONFIG}, fh, indent=2)


def build_args(program, cfg):
    """Return the script + argv for a program given the shared settings."""
    cam = str(cfg.get("source", "")).strip()
    port = str(cfg.get("port", "")).strip()
    hand = cfg.get("hand", "right")
    sim = cfg.get("sim", "tactile")
    label = str(cfg.get("label", "")).strip()
    fuse = bool(cfg.get("fuse", False))

    if program == "camera_setup":
        return ["camera_setup.py"] + (["--source", cam] if cam else ["--demo"])
    if program == "viz":
        a = ["senz_v3_qt.py", "--hand", hand, "--sim", sim]
        a += (["--port", port] if port else ["--simulate"])
        if fuse and cam:
            a += ["--camera", cam]
        return a
    if program == "record":
        a = ["dataset_recorder.py"]
        a += (["--port", port] if port else ["--simulate"])
        if cam.isdigit():                 # recorder takes an int camera index only
            a += ["--camera", cam]
        if label:
            a += ["--label", label]
        return a
    raise ValueError(f"unknown program {program!r}")


def uses_camera(program, cfg):
    """Whether launching `program` will open the physical camera."""
    cam = str(cfg.get("source", "")).strip()
    if program == "camera_setup":
        return bool(cam)                  # blank -> --demo (no real camera)
    if program == "viz":
        return bool(cfg.get("fuse")) and bool(cam)
    if program == "record":
        return cam.isdigit()
    return False


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="senz control hub / launcher")
    ap.add_argument("--config", default=os.path.join(HERE, "senz_hub.json"))
    args = ap.parse_args()

    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    from senz_v3_qt import THEMES

    cfg = load_hub_config(args.config)
    procs = {}   # program key -> QProcess

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.setObjectName("senzhub")
    win.setWindowTitle("senz control hub")
    win.resize(560, 460)
    lay = QtWidgets.QVBoxLayout(win)

    theme_state = {"mode": cfg["theme"]}
    theme_btn = QtWidgets.QPushButton(f"Theme: {cfg['theme'].capitalize()}")
    lay.addWidget(theme_btn)

    lay.addWidget(QtWidgets.QLabel("<b>Shared settings</b>"))
    form = QtWidgets.QFormLayout()
    port_edit = QtWidgets.QLineEdit(cfg["port"])
    port_edit.setPlaceholderText("blank = simulate (no glove)")
    src_row = QtWidgets.QHBoxLayout()
    src_edit = QtWidgets.QLineEdit(str(cfg["source"]))
    src_edit.setPlaceholderText("index (0,1) or http URL; blank = demo")
    scan_btn = QtWidgets.QPushButton("Scan")
    src_row.addWidget(src_edit)
    src_row.addWidget(scan_btn)
    hand_box = QtWidgets.QComboBox()
    hand_box.addItems(["right", "left"])
    hand_box.setCurrentText(cfg["hand"])
    sim_box = QtWidgets.QComboBox()
    sim_box.addItems(["tactile", "proto", "pinch"])
    sim_box.setCurrentText(cfg["sim"])
    label_edit = QtWidgets.QLineEdit(cfg["label"])
    label_edit.setPlaceholderText("e.g. grasp_cup")
    fuse_cb = QtWidgets.QCheckBox("Fuse camera into the hand visualizer")
    fuse_cb.setChecked(bool(cfg["fuse"]))
    form.addRow("Glove port:", port_edit)
    form.addRow("Camera:", src_row)
    form.addRow("Hand:", hand_box)
    form.addRow("Sim build:", sim_box)
    form.addRow("Label:", label_edit)
    form.addRow("", fuse_cb)
    lay.addLayout(form)

    def sync_cfg():
        cfg.update(port=port_edit.text().strip(), source=src_edit.text().strip(),
                   hand=hand_box.currentText(), sim=sim_box.currentText(),
                   label=label_edit.text().strip(), fuse=fuse_cb.isChecked(),
                   theme=theme_state["mode"])

    lay.addWidget(QtWidgets.QLabel("<b>Launch</b>"))
    rows = {}
    for title, key in PROGRAMS:
        row = QtWidgets.QHBoxLayout()
        launch = QtWidgets.QPushButton(title)
        launch.setMinimumWidth(230)
        status = QtWidgets.QLabel("stopped")
        stop = QtWidgets.QPushButton("Stop")
        stop.setEnabled(False)
        row.addWidget(launch)
        row.addWidget(status, stretch=1)
        row.addWidget(stop)
        lay.addLayout(row)
        rows[key] = (launch, status, stop)

    lay.addStretch(1)
    hint = QtWidgets.QLabel(
        "Blank port = simulate · blank camera = demo · one camera feeds one "
        "program at a time.")
    hint.setWordWrap(True)
    lay.addWidget(hint)
    save_btn = QtWidgets.QPushButton("Save settings")
    lay.addWidget(save_btn)
    status_bar = QtWidgets.QLabel("")
    lay.addWidget(status_bar)

    # --- behavior ---
    def set_row_running(key, running):
        launch, status, stop = rows[key]
        status.setText("running" if running else "stopped")
        launch.setEnabled(not running)
        stop.setEnabled(running)

    def on_finished(key):
        def _cb(code, _status):
            set_row_running(key, False)
            status_bar.setText(f"{key} exited (code {code})")
        return _cb

    def launch(key):
        sync_cfg()
        if uses_camera(key, cfg):
            busy = [k for k, p in procs.items()
                    if k != key and p.state() != QtCore.QProcess.NotRunning
                    and uses_camera(k, cfg)]
            if busy:
                r = QtWidgets.QMessageBox.question(
                    win, "Camera in use",
                    f"'{busy[0]}' is already using the camera. A single webcam can "
                    "only feed one program at a time. Launch anyway?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                if r != QtWidgets.QMessageBox.Yes:
                    return
        argv = build_args(key, cfg)
        proc = QtCore.QProcess(win)
        proc.setProgram(sys.executable)
        proc.setArguments(argv)
        proc.setWorkingDirectory(HERE)
        proc.finished.connect(on_finished(key))
        proc.start()
        procs[key] = proc
        set_row_running(key, True)
        status_bar.setText("launched: " + " ".join(argv))

    def stop(key):
        p = procs.get(key)
        if p is not None and p.state() != QtCore.QProcess.NotRunning:
            p.kill()

    for key, (launch_btn, _s, stop_btn) in rows.items():
        launch_btn.clicked.connect(lambda _=False, k=key: launch(k))
        stop_btn.clicked.connect(lambda _=False, k=key: stop(k))

    def on_scan():
        status_bar.setText("scanning cameras...")
        QtWidgets.QApplication.processEvents()
        from camera_tracker import list_cameras
        cams = list_cameras()
        status_bar.setText("cameras: " + (", ".join(map(str, cams)) or "none found"))
        if cams:
            src_edit.setText(str(cams[0]))
    scan_btn.clicked.connect(on_scan)

    def apply_theme(mode):
        th = THEMES[mode]
        win.setStyleSheet(th["qss"] +
                          f'QWidget#senzhub {{ background-color: {th["win_bg"]}; }}')

    def toggle_theme():
        theme_state["mode"] = "light" if theme_state["mode"] == "dark" else "dark"
        theme_btn.setText(f"Theme: {theme_state['mode'].capitalize()}")
        apply_theme(theme_state["mode"])
    theme_btn.clicked.connect(toggle_theme)

    def on_save():
        sync_cfg()
        save_hub_config(args.config, cfg)
        status_bar.setText(f"saved -> {args.config}")
    save_btn.clicked.connect(on_save)

    apply_theme(theme_state["mode"])
    win.show()

    def cleanup():
        for p in procs.values():
            if p.state() != QtCore.QProcess.NotRunning:
                p.kill()
    app.aboutToQuit.connect(cleanup)
    (app.exec_ if hasattr(app, "exec_") else app.exec)()


if __name__ == "__main__":
    main()

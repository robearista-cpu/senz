#!/usr/bin/env python3
"""
dataset.py
==========
Load and preprocess recorded glove sessions (from record.py) into arrays ready
for machine learning. NumPy only -- no pandas required.

Quick use:
    import dataset
    X, y, names = dataset.load_dataset("data")      # per-frame features + labels
    Xw, yw = dataset.windows(X, y, win=50, hop=10)   # sequences for RNN/CNN

Feature vector per frame (columns of X):
    [ f0..fN (normalized 0..1) , qw, qx, qy, qz ]
"""

import csv
import glob
import os

import numpy as np
import senz_io

NF = senz_io.NUM_FINGERS
ADC_MAX = senz_io.ADC_MAX


def load_session(path):
    """Load one CSV. Returns dict with t, label, fingers (N,NF), quat (N,4)."""
    t, fingers, quat, labels = [], [], [], []
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            t.append(float(row["t_s"]))
            labels.append(row["label"])
            fingers.append([float(row[f"f{i}"]) for i in range(NF)])
            quat.append([float(row[k]) for k in ("qw", "qx", "qy", "qz")])
    return {
        "t": np.array(t),
        "label": labels[0] if labels else "",
        "fingers": np.array(fingers).reshape(-1, NF),
        "quat": np.array(quat).reshape(-1, 4),
    }


def features(session, finger_min=0.0, finger_max=ADC_MAX):
    """Build the per-frame feature matrix: normalized fingers + quaternion."""
    f = (session["fingers"] - finger_min) / (finger_max - finger_min)
    f = np.clip(f, 0.0, 1.0)
    return np.hstack([f, session["quat"]])


def load_dataset(directory="data", pattern="*.csv"):
    """Load every session in a folder -> (X, y, label_names).

    X : (total_frames, NF+4) features
    y : (total_frames,) integer class ids
    label_names : list mapping id -> string label
    """
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    Xs, ys = [], []
    label_names = []
    for p in paths:
        s = load_session(p)
        if s["label"] not in label_names:
            label_names.append(s["label"])
        cid = label_names.index(s["label"])
        X = features(s)
        Xs.append(X)
        ys.append(np.full(len(X), cid))
    if not Xs:
        return np.empty((0, NF + 4)), np.empty(0, int), []
    return np.vstack(Xs), np.concatenate(ys), label_names


def windows(X, y, win=50, hop=10):
    """Slice per-frame data into overlapping windows for sequence models.

    Returns Xw (n_windows, win, n_features) and yw (n_windows,) using the
    majority label in each window.
    """
    Xw, yw = [], []
    for start in range(0, len(X) - win + 1, hop):
        seg = X[start : start + win]
        lab = np.bincount(y[start : start + win]).argmax()
        Xw.append(seg)
        yw.append(lab)
    return np.array(Xw), np.array(yw)


if __name__ == "__main__":
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "data"
    X, y, names = load_dataset(d)
    print(f"Loaded {len(X)} frames, {X.shape[1]} features, classes: {names}")
    if len(X):
        for i, n in enumerate(names):
            print(f"  {n}: {(y == i).sum()} frames")

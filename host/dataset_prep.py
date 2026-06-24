#!/usr/bin/env python3
"""
dataset_prep.py  --  ML dataset preparation (HLD objective 7)
=============================================================
Turns one or more raw recorder sessions (data/<session>/frames.csv) into clean,
fixed-rate, windowed arrays ready for model training.

Pipeline:
  1. Load     -- read frames.csv (+ meta.json) for each session.
  2. Clean    -- drop duplicate timestamps, coerce numerics, forward-fill the
                 held camera columns, drop rows with no sensor data.
  3. Align    -- resample onto a uniform grid (default 100 Hz) on the firmware
                 t_us clock, so every row is evenly spaced.
  4. Segment  -- slide a fixed window (default 1.0 s, 50% hop) over the session.
  5. Split    -- train/val split over windows.
  6. Export   -- data/prepared/<name>.npz with X (windows, time, features),
                 plus the feature names and per-window session ids.

Pure pandas/numpy. Validation utilities check for gaps and NaNs.

Usage:
    python dataset_prep.py data/20260624_152326_grasp_cup
    python dataset_prep.py data/*           --rate 100 --window 1.0 --hop 0.5
    python dataset_prep.py data/sessionA --validate-only
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


# Columns that are identifiers/labels, not model features.
NON_FEATURE = {"t_host", "t_us", "handedness", "cam_handedness", "cam_hand_present"}


def load_session(path):
    """Load one session dir -> (DataFrame, meta dict)."""
    csv_path = os.path.join(path, "frames.csv")
    df = pd.read_csv(csv_path)
    meta = {}
    meta_path = os.path.join(path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    return df, meta


def clean(df):
    """Drop dup timestamps, coerce numerics, hold camera columns forward."""
    df = df.drop_duplicates(subset="t_us").sort_values("t_us").reset_index(drop=True)
    # Camera columns are sample-and-held; forward-fill gaps between cam updates.
    cam_cols = [c for c in df.columns if c.startswith("cam_")]
    if cam_cols:
        df[cam_cols] = df[cam_cols].ffill()
    # Coerce everything numeric-looking; non-numeric (handedness) stays object.
    for c in df.columns:
        if c not in ("handedness", "cam_handedness"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows with no IMU/orientation data at all.
    sensor_cols = [c for c in df.columns
                   if c.startswith(("imu", "bno_q", "force")) and c in df]
    df = df.dropna(subset=[c for c in sensor_cols if c in df.columns], how="all")
    return df.reset_index(drop=True)


def resample(df, rate_hz):
    """Resample onto a uniform grid at rate_hz using the t_us clock."""
    t0 = df["t_us"].iloc[0]
    t = (df["t_us"] - t0) / 1e6  # seconds from start
    df = df.assign(_t=t).set_index("_t")
    feat = df.select_dtypes(include=[np.number])
    dur = float(t.iloc[-1])
    grid = np.arange(0.0, dur, 1.0 / rate_hz)
    # Interpolate each numeric column onto the grid.
    out = {c: np.interp(grid, feat.index.values, feat[c].values) for c in feat.columns}
    res = pd.DataFrame(out, index=pd.Index(grid, name="t_s"))
    return res


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE]


def window(df, rate_hz, win_s, hop_s):
    """Slide a fixed window -> (X[n_win, n_t, n_feat], feature_names)."""
    feats = feature_columns(df)
    arr = df[feats].to_numpy(dtype=np.float32)
    win = int(round(win_s * rate_hz))
    hop = max(1, int(round(hop_s * rate_hz)))
    if len(arr) < win:
        return np.empty((0, win, len(feats)), np.float32), feats
    idx = range(0, len(arr) - win + 1, hop)
    X = np.stack([arr[i:i + win] for i in idx]) if idx else \
        np.empty((0, win, len(feats)), np.float32)
    return X, feats


def validate(df, rate_hz):
    """Report timing gaps and NaN counts; returns a summary dict."""
    t = df["t_us"].to_numpy(dtype=float) / 1e6
    dt = np.diff(t)
    nominal = 1.0 / rate_hz
    gaps = int(np.sum(dt > 3 * nominal)) if len(dt) else 0
    nans = int(df.select_dtypes(include=[np.number]).isna().to_numpy().sum())
    return {
        "rows": int(len(df)),
        "duration_s": round(float(t[-1] - t[0]), 2) if len(t) else 0.0,
        "median_dt_ms": round(float(np.median(dt) * 1e3), 3) if len(dt) else 0.0,
        "timing_gaps": gaps,
        "nan_cells": nans,
    }


def prepare(paths, rate_hz=100.0, win_s=1.0, hop_s=0.5, val_frac=0.2,
            out_dir="data/prepared", name="dataset", validate_only=False):
    Xs, sess_ids, feat_names = [], [], None
    for sid, path in enumerate(paths):
        df, _ = load_session(path)
        df = clean(df)
        print(f"[{os.path.basename(path)}] {validate(df, rate_hz)}")
        if validate_only:
            continue
        res = resample(df, rate_hz)
        X, feats = window(res, rate_hz, win_s, hop_s)
        if feat_names is None:
            feat_names = feats
        elif feats != feat_names:
            print(f"  WARNING: feature columns differ; skipping {path}")
            continue
        Xs.append(X)
        sess_ids.append(np.full(len(X), sid, dtype=np.int32))
        print(f"  -> {len(X)} windows of {X.shape[1]}x{X.shape[2]}")

    if validate_only:
        return None

    if not Xs or sum(len(x) for x in Xs) == 0:
        print("No windows produced (sessions too short for the window size?).")
        return None

    X = np.concatenate(Xs)
    sess = np.concatenate(sess_ids)

    # Deterministic split over windows (no RNG so reruns are reproducible).
    n = len(X)
    n_val = int(round(n * val_frac))
    val_mask = np.zeros(n, dtype=bool)
    if n_val:
        val_mask[np.linspace(0, n - 1, n_val).round().astype(int)] = True

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(
        out_path,
        X_train=X[~val_mask], X_val=X[val_mask],
        sess_train=sess[~val_mask], sess_val=sess[val_mask],
        features=np.array(feat_names), rate_hz=rate_hz,
        window=win_s, hop=hop_s,
    )
    print(f"Saved {n} windows ({(~val_mask).sum()} train / {val_mask.sum()} val), "
          f"{len(feat_names)} features -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="senz ML dataset preparation")
    ap.add_argument("sessions", nargs="+", help="session dirs (globs ok)")
    ap.add_argument("--rate", type=float, default=100.0, help="resample rate Hz")
    ap.add_argument("--window", type=float, default=1.0, help="window seconds")
    ap.add_argument("--hop", type=float, default=0.5, help="hop seconds")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--name", default="dataset", help="output basename")
    ap.add_argument("--out", default="data/prepared", help="output dir")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    paths = []
    for pat in args.sessions:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    paths = [p for p in paths if os.path.isdir(p)]
    if not paths:
        print("No session directories found.")
        return

    prepare(paths, rate_hz=args.rate, win_s=args.window, hop_s=args.hop,
            val_frac=args.val_frac, out_dir=args.out, name=args.name,
            validate_only=args.validate_only)


if __name__ == "__main__":
    main()

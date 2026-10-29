"""Generate the dataset with fixed, run-level splits.

Every run = one terrain layout (own random seed) + one constant target speed.
Splits are assigned per run, never per window, so overlapping windows from the
same drive can't leak between train and test.

    train / val / test_id   speeds 0.5-0.9 and 1.2-1.6 m/s, severity 0-0.65
    test_speed_gap          speeds 0.95-1.15 m/s (inside the training range, never seen)
    test_speed_fast         speeds 2.0-2.6 m/s (faster than anything in training)
    test_rougher            severity 0.8-1.0 (rougher than anything in training)

Usage:  python generate.py --out data --runs-per-class 24 [--procs 2]
        (--runs-per-class sets the train size; the other splits scale with it)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from rover.parallel import pmap
from rover.sim import SimCfg
from rover.terrain import CLASSES

TRAIN_SPEEDS = [(0.5, 0.9), (1.2, 1.6)]
TRAIN_SEV = (0.0, 0.65)
SPLITS = {  # name: (fraction of train size, speed ranges, severity range)
    "train":           (1.0,  TRAIN_SPEEDS,  TRAIN_SEV),
    "val":             (0.25, TRAIN_SPEEDS,  TRAIN_SEV),
    "test_id":         (0.25, TRAIN_SPEEDS,  TRAIN_SEV),
    "test_speed_gap":  (0.25, [(0.95, 1.15)], TRAIN_SEV),
    "test_speed_fast": (0.25, [(2.0, 2.6)],  TRAIN_SEV),
    "test_rougher":    (0.25, TRAIN_SPEEDS,  (0.8, 1.0)),
}
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "wheel_speed"]


def _uniform_union(rng, ranges):
    lens = np.array([b - a for a, b in ranges])
    i = rng.choice(len(ranges), p=lens / lens.sum())
    return rng.uniform(*ranges[i])


def design(runs_per_class: int, master_seed: int = 2026):
    rows = []
    for si, (split, (frac, speeds, sev)) in enumerate(SPLITS.items()):
        n = max(1, int(round(frac * runs_per_class)))
        for ci, kind in enumerate(CLASSES):
            for i in range(n):
                seed = master_seed * 1_000_000 + si * 10_000 + ci * 1_000 + i
                rng = np.random.default_rng(seed + 7)
                rows.append(dict(run_id=f"{split}-{kind}-{i:03d}", split=split, terrain=kind,
                                 severity=round(float(rng.uniform(*sev)), 4),
                                 speed_cmd=round(float(_uniform_union(rng, speeds)), 4), seed=seed))
    return rows


def _job(sim, row):
    t0 = time.time()
    sig, win, meta, _ = sim.run(row["terrain"], row["severity"], row["speed_cmd"], row["seed"], SimCfg())
    return row, sig, win, meta, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--runs-per-class", type=int, default=24)
    ap.add_argument("--procs", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(f"{args.out}/runs", exist_ok=True)
    plan = design(args.runs_per_class)
    print(f"{len(plan)} runs planned")

    run_rows, win_rows = [], []
    X = {s: [] for s in SPLITS}
    t_start = time.time()
    for k, (row, sig, win, meta, dt) in enumerate(pmap(_job, plan, args.procs)):
        np.savez_compressed(f"{args.out}/runs/{row['run_id']}.npz",
                            t=sig["t"].astype(np.float32), acc=sig["acc"].astype(np.float32),
                            gyro=sig["gyro"].astype(np.float32), wheel_speed=sig["wheel_speed"].astype(np.float32),
                            speed_true=sig["speed_true"].astype(np.float32), pos=sig["pos"].astype(np.float32),
                            rpy=sig["rpy"].astype(np.float32))
        run_rows.append(dict(**row, steady_speed=meta["steady_speed"], settle_time_s=meta["settle_time_s"],
                             n_windows=len(win), flipped=meta["flipped"], offtrack=meta["offtrack"],
                             terrain_params=json.dumps({k2: (round(v, 5) if isinstance(v, float) else v)
                                                        for k2, v in meta["terrain"].items()})))
        full = np.concatenate([sig["acc"], sig["gyro"], sig["wheel_speed"][:, None]], 1).astype(np.float32)
        for wi, w in enumerate(win):
            X[row["split"]].append(full[w["start"]:w["end"]])
            win_rows.append(dict(run_id=row["run_id"], split=row["split"], terrain=row["terrain"],
                                 severity=row["severity"], speed_cmd=row["speed_cmd"], window=wi,
                                 start=w["start"], end=w["end"], roughness_mm=round(w["roughness_mm"], 4),
                                 speed_true=round(w["speed_true"], 4), wheel_speed=round(w["wheel_speed"], 4)))
        if (k + 1) % 10 == 0 or k + 1 == len(plan):
            el = time.time() - t_start
            print(f"{k+1}/{len(plan)} runs  {el/60:.1f} min elapsed, ~{el/(k+1)*(len(plan)-k-1)/60:.1f} min left", flush=True)

    _write_csv(f"{args.out}/runs.csv", run_rows)
    _write_csv(f"{args.out}/windows.csv", win_rows)
    for s in SPLITS:
        rows = [r for r in win_rows if r["split"] == s]
        np.savez_compressed(f"{args.out}/windows_{s}.npz", X=np.stack(X[s]) if X[s] else np.zeros((0, 100, 7), np.float32),
                            terrain=np.array([CLASSES.index(r["terrain"]) for r in rows]),
                            roughness_mm=np.array([r["roughness_mm"] for r in rows], np.float32),
                            run_id=np.array([r["run_id"] for r in rows]), channels=np.array(CHANNELS),
                            classes=np.array(CLASSES))
    print("done in %.1f min" % ((time.time() - t_start) / 60))


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()

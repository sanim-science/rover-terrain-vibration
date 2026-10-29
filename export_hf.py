"""Convert the generated dataset to Parquet for the Hugging Face Hub.

One row per 1-second window; each signal channel is a list of 100 floats (100 Hz).
Needs only numpy, pandas and pyarrow (not PyBullet); works on Windows, macOS, Linux.

    pip install numpy pandas pyarrow
    python export_hf.py --data data_full --out hf_dataset ^
        --repo-id YOUR_HF_NAME/rover-terrain-vibration ^
        --github https://github.com/YOUR_GITHUB_NAME/rover-terrain-vibration

It writes hf_dataset/data/<split>.parquet, hf_dataset/runs/ (raw drives) and
hf_dataset/README.md (the dataset card with your links filled in). Upload with:
    hf upload YOUR_HF_NAME/rover-terrain-vibration hf_dataset --repo-type=dataset

Load it back with:
    from datasets import load_dataset
    ds = load_dataset("YOUR_HF_NAME/rover-terrain-vibration")
    ds["test_speed_fast"][0]["acc_z"]      # 100 floats
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test_id", "test_speed_gap", "test_speed_fast", "test_rougher")
HERE = os.path.dirname(os.path.abspath(__file__))


def build_frame(data: str, split: str) -> pd.DataFrame:
    z = np.load(os.path.join(data, f"windows_{split}.npz"))
    channels = [str(c) for c in z["channels"]]
    meta = pd.read_csv(os.path.join(data, "windows.csv"))
    meta = meta[meta["split"] == split].reset_index(drop=True)
    assert len(meta) == len(z["X"]), (split, len(meta), len(z["X"]))
    assert (meta["run_id"].values == z["run_id"]).all(), f"{split}: window order mismatch"
    runs = pd.read_csv(os.path.join(data, "runs.csv")).set_index("run_id")
    df = pd.DataFrame({
        "run_id": meta["run_id"], "window": meta["window"],
        "terrain": meta["terrain"], "terrain_id": z["terrain"].astype(int),
        "severity": meta["severity"], "roughness_mm": meta["roughness_mm"],
        "speed_cmd": meta["speed_cmd"], "speed_true": meta["speed_true"],
        "wheel_speed_mean": meta["wheel_speed"],
        "terrain_params": runs.loc[meta["run_id"], "terrain_params"].values,
    })
    for i, ch in enumerate(channels):
        df[ch] = [row.astype(np.float32).tolist() for row in z["X"][:, :, i]]
    return df


def write_card(out: str, repo_id: str | None, github: str | None, report: str | None):
    card = open(os.path.join(HERE, "hf_dataset_card.md"), encoding="utf-8").read()
    if repo_id:
        card = card.replace("<user>/<dataset-name>", repo_id)
    if github:
        card = card.replace("<GitHub link>", f"[{github}]({github})")
    card = card.replace("<report / arXiv link once available>", report or "Report: coming soon.")
    left = [p for p in ("<user>/<dataset-name>", "<GitHub link>") if p in card]
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    if left:
        print("NOTE: still unfilled in README.md:", left, "- pass --repo-id / --github")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_full")
    ap.add_argument("--out", default="hf_dataset")
    ap.add_argument("--repo-id", help="your Hugging Face dataset id, e.g. alice/rover-terrain-vibration")
    ap.add_argument("--github", help="URL of the GitHub repository")
    ap.add_argument("--report", help="link to the report, if you have one")
    args = ap.parse_args()
    if args.repo_id and "/" not in args.repo_id:
        ap.error("--repo-id must look like NAME/DATASET")
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    total = 0
    for split in SPLITS:
        df = build_frame(args.data, split)
        df.to_parquet(os.path.join(args.out, "data", f"{split}.parquet"), index=False)
        total += len(df)
        print(f"{split:16s} {len(df):6d} windows")
    # raw per-drive 100 Hz recordings, for people who want other window lengths
    shutil.copy2(os.path.join(args.data, "runs.csv"), os.path.join(args.out, "runs.csv"))
    shutil.copytree(os.path.join(args.data, "runs"), os.path.join(args.out, "runs"), dirs_exist_ok=True)
    write_card(args.out, args.repo_id, args.github, args.report)
    n_runs = len(os.listdir(os.path.join(args.out, "runs")))
    print(f"done: {total} windows, {n_runs} drive files, card written to {os.path.join(args.out, 'README.md')}")


if __name__ == "__main__":
    main()

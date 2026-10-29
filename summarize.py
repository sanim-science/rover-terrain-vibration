"""Build results tables (Markdown) and the headline figure from saved results.

Reads  results/baselines.json, results/nn.json, results/checks.json, <data>/runs.csv
Writes results/tables.md and figures/summary_all_models.png

Usage:  python summarize.py --data data_full
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from rover import plotstyle as ps

plt = ps.plt
TESTS = ["test_id", "test_speed_gap", "test_speed_fast", "test_rougher"]
COND = {"test_id": "seen speeds, new layouts", "test_speed_gap": "speed gap 0.95-1.15 m/s",
        "test_speed_fast": "faster 2.0-2.6 m/s", "test_rougher": "rougher, severity 0.8-1"}
MODELS = [  # key prefix in the JSONs, display name
    ("A_vibration|RF", "Random forest, vibration features"),
    ("B_vibration+speed|RF", "Random forest, + wheel speed"),
    ("C_per_metre|RF", "Random forest, per-metre features"),
    ("A_vibration|linear", "Linear, vibration features"),
    ("B_vibration+speed|linear", "Linear, + wheel speed"),
    ("C_per_metre|linear", "Linear, per-metre features"),
    ("MLP_spectrum|NN", "MLP on spectra + speed"),
    ("MLP_raw|NN", "MLP on raw signal + speed"),
    ("CNN_raw|NN", "1-D CNN on raw signal + speed"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_full")
    ap.add_argument("--results", default="results")
    ap.add_argument("--fig", default="figures")
    args = ap.parse_args()
    R = json.load(open(f"{args.results}/baselines.json"))
    try:
        N = json.load(open(f"{args.results}/nn.json"))
    except FileNotFoundError:
        N = {"classification": {}, "roughness": {}}
    C = {**R["classification"], **N["classification"]}
    G = {**R["roughness"], **N["roughness"]}
    models = [(k, n) for k, n in MODELS if f"{k}|test_id" in C]
    runs = pd.read_csv(f"{args.data}/runs.csv")

    md = []
    md.append(f"Dataset: {len(runs):,} drives, {R['n_windows']['train']:,} training windows "
              f"({R['n_runs']['train']} drives); test splits {min(R['n_windows'][t] for t in TESTS):,}-"
              f"{max(R['n_windows'][t] for t in TESTS):,} windows ({R['n_runs']['test_id']} drives each).\n")
    md.append("**Naming the ground type** - macro-F1 (chance 0.20), 95 % bootstrap interval over drives:\n")
    md.append("| model | " + " | ".join(COND[t] for t in TESTS) + " |")
    md.append("|---|" + "---|" * len(TESTS))
    for k, n in models:
        md.append(f"| {n} | " + " | ".join(
            f"{C[f'{k}|{t}']['macro_f1']:.2f} ({C[f'{k}|{t}']['ci'][0]:.2f}-{C[f'{k}|{t}']['ci'][1]:.2f})" for t in TESTS) + " |")
    md.append("\n**Roughness** - mean absolute error (median relative error):\n")
    md.append("| model | " + " | ".join(COND[t] for t in TESTS) + " |")
    md.append("|---|" + "---|" * len(TESTS))
    for k, n in models:
        md.append(f"| {n} | " + " | ".join(
            f"{G[f'{k}|{t}']['mae_mm']:.2f} mm ({100*G[f'{k}|{t}']['med_rel']:.0f}%)" for t in TESTS) + " |")
    md.append("\n**Roughness on a ground type never seen in training** (random forest, vibration features; "
              "trained on the other four types, tested on new layouts of the held-out type):\n")
    md.append("| held-out type | true mean | error, type seen | error, type never seen |")
    md.append("|---|---|---|---|")
    for kind in ["smooth", "gravel", "cobble", "washboard", "rocky"]:
        v = R["roughness_unseen_type"][f"A_vibration|RF|{kind}"]
        md.append(f"| {kind} | {v['mean_true_mm']:.2f} mm | {v['mae_seen']:.2f} mm ({100*v['med_rel_seen']:.0f}%) | "
                  f"{v['mae_unseen']:.2f} mm ({100*v['med_rel_unseen']:.0f}%) |")
    open(f"{args.results}/tables.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # headline figure: every model, every condition
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(models) + 1.3))
    ys = np.arange(len(models))[::-1]
    for ti, t in enumerate(TESTS):
        v = [C[f"{k}|{t}"]["macro_f1"] for k, _ in models]
        lo = [C[f"{k}|{t}"]["ci"][0] for k, _ in models]
        hi = [C[f"{k}|{t}"]["ci"][1] for k, _ in models]
        off = (1.5 - ti) * 0.14
        ax.errorbar(v, ys + off, xerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt=ps.MARKERS[ti], ms=6,
                    color=ps.SERIES[ti], elinewidth=1, capsize=0, label=COND[t])
    for y in ys[:-1]:
        ax.axhline(y - 0.5, color=ps.GRID, lw=0.6)
    ax.axvline(0.2, color=ps.MUTED, lw=1, ls=":")
    ax.text(0.205, ys[0] + 0.45, "chance", fontsize=7, color=ps.INK2)
    ax.set_yticks(ys, [n for _, n in models], fontsize=8)
    ax.tick_params(axis="y", labelcolor=ps.INK2, length=0)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel("macro-F1, naming the ground type")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.09), ncol=4, fontsize=7.5)
    ax.set_title("How much does terrain recognition drop on conditions never seen in training?", loc="left")
    fig.savefig(f"{args.fig}/summary_all_models.png"); plt.close(fig)


if __name__ == "__main__":
    main()

"""Baselines: how well can 1-s IMU windows name the ground and measure its roughness,
and how much does that drop on unseen speeds / rougher ground / unseen terrain types?

Models  : random forest (RF) and a linear model (logistic / ridge) on hand-crafted
          features, three feature sets (see rover/features.py).
Metrics : classification - macro-F1 (5 classes, chance = 0.20) and accuracy
          roughness      - mean absolute error in mm, and median relative error
CIs     : 95 % bootstrap over *runs* (windows from one run move together).

Usage:  python experiments.py --data data
"""
from __future__ import annotations

import argparse
import gc
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rover import plotstyle as ps
from rover.features import FEATURE_SETS, featurize
from rover.terrain import CLASSES

plt = ps.plt
TESTS = ["test_id", "test_speed_gap", "test_speed_fast", "test_rougher"]
TEST_LABEL = {"test_id": "new layouts\nseen speeds", "test_speed_gap": "speed gap\n0.95-1.15 m/s",
              "test_speed_fast": "faster\n2.0-2.6 m/s", "test_rougher": "rougher\nseverity 0.8-1"}
N_BOOT = 1000


def load(data):
    D = {}
    for s in ["train", "val"] + TESTS:
        z = np.load(f"{data}/windows_{s}.npz", allow_pickle=False)
        D[s] = dict(X=z["X"], y=z["terrain"], r=z["roughness_mm"], run=z["run_id"])
    return D


def boot_ci(metric, y, p, runs, rng):
    """95 % CI of metric(y, p), resampling whole runs."""
    u = np.unique(runs)
    idx = {r: np.where(runs == r)[0] for r in u}
    vals = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[r] for r in rng.choice(u, len(u))])
        vals.append(metric(y[pick], p[pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=range(len(CLASSES)), zero_division=0)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def med_rel(y, p):
    return float(np.median(np.abs(p - y) / y))


def models_cls(seed):
    return {"RF": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed),
            "linear": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=4000))}


def models_reg(seed):
    return {"RF": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed),
            "linear": make_pipeline(StandardScaler(), Ridge(alpha=1.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures-only", action="store_true", help="redraw from saved results")
    ap.add_argument("--resume", action="store_true", help="reuse saved main results, redo only what is missing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs("figures", exist_ok=True)
    rng = np.random.default_rng(0)
    D = load(args.data)
    if args.figures_only:
        res = json.load(open(f"{args.out}/baselines.json"))
        z = np.load(f"{args.out}/predictions.npz")
        preds = {}
        for k in z.files:
            fs, m, s, kind = k.split("|")
            preds.setdefault((fs, m, s), [None, None])[0 if kind == "cls" else 1] = z[k]
        figures(res, preds, D)
        return
    print({s: len(d["y"]) for s, d in D.items()})

    F = feature_cache(args.data, D)

    res = {"classification": {}, "roughness": {}, "roughness_unseen_type": {}, "n_windows": {s: int(len(d['y'])) for s, d in D.items()},
           "n_runs": {s: int(len(np.unique(d["run"]))) for s, d in D.items()}, "stage": "start"}
    preds = {}
    if args.resume and os.path.exists(f"{args.out}/baselines.json"):
        res = json.load(open(f"{args.out}/baselines.json"))
        z = np.load(f"{args.out}/predictions.npz")
        for k in z.files:
            fs_, m_, s_, kind = k.split("|")
            preds.setdefault((fs_, m_, s_), [None, None])[0 if kind == "cls" else 1] = z[k]
        print("resumed at stage", res.get("stage"))
    for fs in (FEATURE_SETS if res.get("stage") == "start" else []):
        Xtr, ytr, rtr = F[(fs, "train")], D["train"]["y"], np.log(D["train"]["r"])
        for mname in ("RF", "linear"):
            clf = models_cls(0)[mname].fit(Xtr, ytr)
            reg = models_reg(0)[mname].fit(Xtr, rtr)
            for s in TESTS:
                y, r, runs = D[s]["y"], D[s]["r"], D[s]["run"]
                p = clf.predict(F[(fs, s)])
                pr = np.exp(reg.predict(F[(fs, s)]))
                preds[(fs, mname, s)] = (p, pr)
                key = f"{fs}|{mname}|{s}"
                res["classification"][key] = dict(macro_f1=macro_f1(y, p), acc=float((y == p).mean()),
                                                  ci=boot_ci(macro_f1, y, p, runs, rng))
                res["roughness"][key] = dict(mae_mm=mae(r, pr), med_rel=med_rel(r, pr),
                                             ci=boot_ci(mae, r, pr, runs, rng), mean_true_mm=float(r.mean()))
                print(f"{fs:18s} {mname:6s} {s:16s} F1={res['classification'][key]['macro_f1']:.3f} "
                      f"{np.round(res['classification'][key]['ci'], 3)}  rough MAE={res['roughness'][key]['mae_mm']:.3f} mm "
                      f"(median rel err {100*res['roughness'][key]['med_rel']:.0f}%)", flush=True)
            del clf, reg; gc.collect()
    if res.get("stage") == "start":
        res["stage"] = "main_done"
        save(res, preds, args.out)

    # roughness on a ground TYPE never seen in training (leave one terrain out)
    for fs in FEATURE_SETS:
        for mname in ("RF", "linear"):
            if f"{fs}|{mname}|{CLASSES[-1]}" in res["roughness_unseen_type"]:
                continue
            # reference: every type seen in training (predict, then free the model)
            reg_all = models_reg(0)[mname].fit(F[(fs, "train")], np.log(D["train"]["r"]))
            pr_all_full = np.exp(reg_all.predict(F[(fs, "test_id")]))
            del reg_all; gc.collect()
            for ci, kind in enumerate(CLASSES):
                m_tr = D["train"]["y"] != ci
                reg = models_reg(0)[mname].fit(F[(fs, "train")][m_tr], np.log(D["train"]["r"][m_tr]))
                m_te = D["test_id"]["y"] == ci
                r = D["test_id"]["r"][m_te]
                pr = np.exp(reg.predict(F[(fs, "test_id")][m_te]))
                pr_all = pr_all_full[m_te]
                del reg; gc.collect()
                res["roughness_unseen_type"][f"{fs}|{mname}|{kind}"] = dict(
                    mae_unseen=mae(r, pr), mae_seen=mae(r, pr_all), med_rel_unseen=med_rel(r, pr),
                    med_rel_seen=med_rel(r, pr_all), mean_true_mm=float(r.mean()))
            print(fs, mname, "leave-one-type-out:",
                  {k.split('|')[2]: round(v["mae_unseen"], 2) for k, v in res["roughness_unseen_type"].items() if k.startswith(f"{fs}|{mname}|")},
                  flush=True)
            save(res, preds, args.out)

    res["stage"] = "done"
    save(res, preds, args.out)
    figures(res, preds, D)


def save(res, preds, out):
    json.dump(res, open(f"{out}/baselines.json", "w"), indent=1)
    np.savez_compressed(f"{out}/predictions.npz", **{"|".join(k) + "|cls": v[0] for k, v in preds.items()},
                        **{"|".join(k) + "|rough": v[1] for k, v in preds.items()})


def feature_cache(data, D):
    """Features for every set and split, cached in <data>/features_cache.npz."""
    path = f"{data}/features_cache.npz"
    if os.path.exists(path):
        z = np.load(path)
        F = {tuple(k.split("|")): z[k] for k in z.files}
        if all((fs, s) in F for fs in FEATURE_SETS for s in D):
            print("features loaded from cache")
            return F
    F = {}
    for fs, fn in FEATURE_SETS.items():
        for s, d in D.items():
            F[(fs, s)], names = featurize(d["X"], fn)
        print(f"features {fs}: {len(names)}", flush=True)
    np.savez_compressed(path, **{f"{fs}|{s}": v for (fs, s), v in F.items()})
    return F


def figures(res, preds, D):
    fsets = list(FEATURE_SETS)
    labels = {"A_vibration": "A: vibration only", "B_vibration+speed": "B: + wheel speed", "C_per_metre": "C: per-metre (speed-normalised)"}
    for task, metric, ylab, fname in [("classification", "macro_f1", "macro-F1 (chance 0.20)", "baseline_classification.png"),
                                      ("roughness", "mae_mm", "roughness error, MAE (mm)", "baseline_roughness.png")]:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
        for ax, mname in zip(axes, ("RF", "linear")):
            x = np.arange(len(TESTS))
            for i, fs in enumerate(fsets):
                v = [res[task][f"{fs}|{mname}|{s}"][metric] for s in TESTS]
                lo = [res[task][f"{fs}|{mname}|{s}"]["ci"][0] for s in TESTS]
                hi = [res[task][f"{fs}|{mname}|{s}"]["ci"][1] for s in TESTS]
                off = (i - 1) * 0.22
                ax.errorbar(x + off, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt=ps.MARKERS[i], ms=6,
                            color=ps.SERIES[i], ecolor=ps.SERIES[i], elinewidth=1.2, capsize=0, label=labels[fs])
            ax.set_xticks(x, [TEST_LABEL[s] for s in TESTS], fontsize=7)
            ax.set_title("random forest" if mname == "RF" else "linear model", loc="left")
            if task == "classification":
                ax.axhline(0.2, color=ps.MUTED, lw=1, ls=":"); ax.set_ylim(0, 1.02)
            else:
                ax.set_ylim(bottom=0)
        axes[0].set_ylabel(ylab)
        axes[1].legend(loc="lower left" if task == "classification" else "upper left")
        fig.savefig(f"figures/{fname}"); plt.close(fig)

    # confusion matrices (random forest): seen speeds, then faster speeds with raw vs per-metre features
    panels = [("C_per_metre", "test_id", "C per-metre, seen speeds"),
              ("A_vibration", "test_speed_fast", "A vibration only, FASTER"),
              ("C_per_metre", "test_speed_fast", "C per-metre, FASTER")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), gridspec_kw=dict(wspace=0.08))
    for ax, (fs, s, title) in zip(axes, panels):
        p, _ = preds[(fs, "RF", s)]
        cm = confusion_matrix(D[s]["y"], p, labels=range(len(CLASSES)), normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if cm[i, j] > 0.55 else ps.INK)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, fontsize=7)
        ax.set_yticks(range(len(CLASSES)), CLASSES if ax is axes[0] else [], fontsize=7)
        ax.set_xlabel("predicted"); ax.grid(False)
        ax.set_title(title, loc="left", fontsize=9)
    axes[0].set_ylabel("true")
    fig.suptitle("Random forest confusion matrices (each row sums to 1)", fontsize=9, x=0.02, ha="left")
    fig.savefig("figures/baseline_confusion.png"); plt.close(fig)

    # roughness: predicted vs true, seen speeds vs faster speeds
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    for ax, fs in zip(axes, fsets):
        for s, c, mk in (("test_id", ps.SERIES[0], "o"), ("test_speed_fast", ps.SERIES[1], "s")):
            _, pr = preds[(fs, "RF", s)]
            ax.scatter(D[s]["r"], pr, s=6, alpha=0.45, color=c, marker=mk, lw=0, label=TEST_LABEL[s].replace("\n", " "))
        ax.plot([0, 4], [0, 4], color=ps.INK2, lw=1)
        ax.set_title(labels[fs], loc="left", fontsize=9)
        ax.set_xlabel("true roughness (mm RMS)")
    axes[0].set_ylabel("predicted (random forest)")
    axes[0].legend(loc="upper left", markerscale=2.5)
    axes[0].set_xlim(0, 4); axes[0].set_ylim(0, 4)
    fig.savefig("figures/baseline_roughness_scatter.png"); plt.close(fig)


if __name__ == "__main__":
    main()

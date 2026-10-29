"""Neural-network baselines that learn from (nearly) raw signals.

Question: does a learned model fix the speed problem by itself, without hand-made
speed normalisation?

Inputs (no hand-crafted features):
  raw       the 1-s window, each channel minus its window mean (removes gravity and
            bias), standardised per channel; plus log wheel speed      -> 601 inputs
  spectrum  log power spectrum 1-50 Hz of each channel; plus log speed  -> 301 inputs
  cnn       the 1-s window as a 6-channel sequence (same normalisation as raw)
            + log wheel speed, into a small 1-D CNN (rover/cnn.py)
Models  : multilayer perceptron (scikit-learn, 256-128 ReLU units) and the 1-D CNN,
          Adam, early stopping on the *val* split (best epoch by macro-F1 / MAE).
Seeds   : 3; metrics are averaged over seeds, CIs bootstrap over drives.

Usage:  python experiments_nn.py --data data_full
Writes results/nn.json and figures/nn_vs_features.png (with results/baselines.json).
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier, MLPRegressor

from experiments import TESTS, load, macro_f1, mae, med_rel
from rover.cnn import CNN1D

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEEDS = (0, 1, 2)
EPOCHS = 40
N_BOOT = 1000


def rep_raw(X, stats=None):
    sig = X[:, :, :6] - X[:, :, :6].mean(1, keepdims=True)
    if stats is None:
        stats = sig.reshape(-1, 6).std(0) + 1e-9
    sig = sig / stats
    v = np.log(np.maximum(X[:, :, 6].mean(1), 0.05))[:, None]
    return np.concatenate([sig.reshape(len(X), -1), v], 1), stats


def rep_spec(X, stats=None):
    sig = X[:, :, :6] - X[:, :, :6].mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(sig * np.hanning(100)[None, :, None], axis=1)) ** 2
    P = np.log10(P[:, 1:51] + 1e-10).reshape(len(X), -1)          # 1..50 Hz, 6 channels
    v = np.log(np.maximum(X[:, :, 6].mean(1), 0.05))[:, None]
    F = np.concatenate([P, v], 1)
    if stats is None:
        stats = (F.mean(0), F.std(0) + 1e-9)
    return (F - stats[0]) / stats[1], stats


REPS = {"raw": rep_raw, "spectrum": rep_spec}


def standardise_raw_tail(F, stats_v=None):
    # standardise the trailing log-speed column of the raw representation
    if stats_v is None:
        stats_v = (F[:, -1].mean(), F[:, -1].std() + 1e-9)
    F = F.copy(); F[:, -1] = (F[:, -1] - stats_v[0]) / stats_v[1]
    return F, stats_v


def fit_best(model, Xtr, ytr, Xva, yva, score, higher_better, seed):
    rng = np.random.default_rng(seed)
    best, best_state, classes = None, None, np.unique(ytr) if higher_better else None
    for ep in range(EPOCHS):
        idx = rng.permutation(len(Xtr))
        for b in range(0, len(idx), 256):
            j = idx[b:b + 256]
            if classes is not None:
                model.partial_fit(Xtr[j], ytr[j], classes=classes)
            else:
                model.partial_fit(Xtr[j], ytr[j])
        s = score(yva, model.predict(Xva))
        if best is None or (s > best if higher_better else s < best):
            best, best_ep = s, ep
            best_state = ([w.copy() for w in model.coefs_], [b.copy() for b in model.intercepts_])
    model.coefs_, model.intercepts_ = best_state
    return model, best_ep, best


def boot_ci_multi(metric, y, preds, runs, rng):
    """CI of the seed-averaged metric, resampling whole drives."""
    u = np.unique(runs); idx = {r: np.where(runs == r)[0] for r in u}
    vals = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[r] for r in rng.choice(u, len(u))])
        vals.append(np.mean([metric(y[pick], p[pick]) for p in preds]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


CNN_EPOCHS = 30


def cnn_inputs(X, stats=None):
    sig = (X[:, :, :6] - X[:, :, :6].mean(1, keepdims=True)).transpose(0, 2, 1)   # (n, 6, 100)
    v = np.log(np.maximum(X[:, :, 6].mean(1), 0.05))
    if stats is None:
        stats = (sig.std(axis=(0, 2)) + 1e-9, v.mean(), v.std() + 1e-9)
    return (sig / stats[0][None, :, None]).astype(np.float32), ((v - stats[1]) / stats[2]).astype(np.float32), stats


def train_cnn(task, x, s, y, xv, sv, yv, seed, n_out):
    net = CNN1D(n_out=n_out, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_p, best_ep = None, None, -1
    for ep in range(CNN_EPOCHS):
        idx = rng.permutation(len(x))
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]
            _, gr = net.loss_grad(x[j], s[j], y[j], task)
            net.step(gr, lr=1e-3 if ep < 20 else 3e-4)
        out = net.predict_raw(xv, sv)
        score = macro_f1(yv, out.argmax(1)) if task == "cls" else -mae(yv, out[:, 0])
        if best is None or score > best:
            best, best_p, best_ep = score, net.get(), ep
    net.set(best_p)
    return net, best_ep


def run_cnn(D, res, rng):
    x, s, st = cnn_inputs(D["train"]["X"])
    lr_ = np.log(D["train"]["r"]); mu, sd = lr_.mean(), lr_.std()
    xv, sv, _ = cnn_inputs(D["val"]["X"], st)
    T = {t: cnn_inputs(D[t]["X"], st)[:2] for t in TESTS}
    pc, pr = {t: [] for t in TESTS}, {t: [] for t in TESTS}
    for seed in SEEDS:
        clf, ep_c = train_cnn("cls", x, s, D["train"]["y"], xv, sv, D["val"]["y"], seed, 5)
        reg, ep_r = train_cnn("reg", x, s, ((lr_ - mu) / sd).astype(np.float32), xv, sv,
                              ((np.log(D["val"]["r"]) - mu) / sd).astype(np.float32), seed, 1)
        res["best_epoch"][f"cnn|seed{seed}"] = dict(cls=ep_c, reg=ep_r)
        print(f"cnn seed {seed}: best epochs cls {ep_c}, reg {ep_r}", flush=True)
        for t in TESTS:
            pc[t].append(clf.predict_raw(*T[t]).argmax(1))
            pr[t].append(np.exp(reg.predict_raw(*T[t])[:, 0] * sd + mu))
    summarise("CNN_raw", pc, pr, D, res, rng)


def summarise(name, pc, pr, D, res, rng=None):
    rng = np.random.default_rng(zlib.crc32(name.encode()))   # per-model, so resuming gives identical CIs
    for s in TESTS:
        y, r, runs = D[s]["y"], D[s]["r"], D[s]["run"]
        f1s = [macro_f1(y, p) for p in pc[s]]
        key = f"{name}|NN|{s}"
        res["classification"][key] = dict(macro_f1=float(np.mean(f1s)), seed_range=[float(min(f1s)), float(max(f1s))],
                                          ci=boot_ci_multi(macro_f1, y, pc[s], runs, rng))
        res["roughness"][key] = dict(mae_mm=float(np.mean([mae(r, p) for p in pr[s]])),
                                     med_rel=float(np.mean([med_rel(r, p) for p in pr[s]])),
                                     ci=boot_ci_multi(mae, r, pr[s], runs, rng))
        print(f"{name:12s} {s:16s} F1={res['classification'][key]['macro_f1']:.3f} "
              f"seeds {np.round(res['classification'][key]['seed_range'], 3)} CI {np.round(res['classification'][key]['ci'], 3)}  "
              f"rough MAE={res['roughness'][key]['mae_mm']:.3f} mm ({100*res['roughness'][key]['med_rel']:.0f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_full")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", choices=["mlp", "cnn"], default=None)
    ap.add_argument("--resume", action="store_true", help="skip models already in nn.json")
    args = ap.parse_args()
    D = load(args.data)
    rng = None
    path = f"{args.out}/nn.json"
    res = {"classification": {}, "roughness": {}, "best_epoch": {}}
    if args.resume and os.path.exists(path):
        res = json.load(open(path))
    done = lambda name: f"{name}|NN|test_id" in res["classification"]
    for rname, rep in (REPS.items() if args.only != "cnn" else []):
        if done(f"MLP_{rname}"):
            print("skip", rname); continue
        Ftr, st = rep(D["train"]["X"])
        if rname == "raw":
            Ftr, sv = standardise_raw_tail(Ftr)
        F = {"train": Ftr}
        for s in ["val"] + TESTS:
            F[s], _ = rep(D[s]["X"], st)
            if rname == "raw":
                F[s], _ = standardise_raw_tail(F[s], sv)
        pc, pr = {s: [] for s in TESTS}, {s: [] for s in TESTS}
        for seed in SEEDS:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, learning_rate_init=1e-3, random_state=seed)
            clf, ep_c, _ = fit_best(clf, F["train"], D["train"]["y"], F["val"], D["val"]["y"], macro_f1, True, seed)
            reg = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, learning_rate_init=1e-3, random_state=seed)
            ytr_r = np.log(D["train"]["r"])
            reg, ep_r, _ = fit_best(reg, F["train"], ytr_r, F["val"], np.log(D["val"]["r"]), mae, False, seed)
            res["best_epoch"][f"{rname}|seed{seed}"] = dict(cls=ep_c, reg=ep_r)
            for s in TESTS:
                pc[s].append(clf.predict(F[s]))
                pr[s].append(np.exp(reg.predict(F[s])))
            print(f"{rname} seed {seed}: best epochs cls {ep_c}, reg {ep_r}", flush=True)
        summarise(f"MLP_{rname}", pc, pr, D, res, rng)
        json.dump(res, open(path, "w"), indent=1)
    if args.only != "mlp" and not done("CNN_raw"):
        run_cnn(D, res, rng)
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()

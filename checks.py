"""Simulator sanity checks.  Writes figures/ and results/checks.json.

1. static     parked car: accelerometer must read (0, 0, +9.81), gyro ~0
2. noise      perfectly flat ground: any vibration is simulator artefact.
              Sweep wheel collision shape x contact model x physics rate.
3. washboard  periodic ridges of wavelength L at speed v must give a peak at v / L Hz
4. converge   same terrains at 400 / 1000 / 2000 Hz physics: do features agree
              better than the spread between different random terrains?

Usage:  python checks.py [static noise washboard converge]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from rover import plotstyle as ps
from rover.features import _band, _psd
from rover.parallel import pmap
from rover.sim import NoiseCfg, SimCfg

plt = ps.plt
FIG, RES = "figures", "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def _run(sim, a):
    kind, sev, v, seed, cfgkw = a
    sig, win, meta, _ = sim.run(kind, sev, v, seed, SimCfg(**cfgkw))
    return sig, win, meta


def _steady(sig, meta, skip=2.0):
    i = int(max(skip, meta["settle_time_s"] or skip) * 100)
    return slice(i, None)


# ---------------------------------------------------------------- 1. static
def check_static():
    sig, _, meta = next(pmap(_run, [("flat", 0, 0.0, 0, dict(duration=4.0, noise=None))], procs=1))
    s = slice(100, None)
    out = {"acc_mean": sig["acc"][s].mean(0).tolist(), "acc_std": sig["acc"][s].std(0).tolist(),
           "gyro_mean": sig["gyro"][s].mean(0).tolist()}
    print("static:", {k: np.round(v, 4).tolist() for k, v in out.items()})
    assert abs(out["acc_mean"][2] - 9.81) < 0.02 and max(map(abs, out["acc_mean"][:2])) < 0.02
    return out


# ---------------------------------------------------------------- 2. noise floor
def check_noise():
    rates = [100, 200, 400, 1000, 2000]
    combos = [(g, w, t) for g in ("heightfield", "plane") for w in ("cylinder", "sphere") for t in (None, 1e4)]
    jobs = [("flat", 0, 1.5, 1, dict(duration=6.0, noise=None, phys_hz=r, ground=g, wheel_shape=w,
                                     tire_stiffness=t)) for g, w, t in combos for r in rates]
    rows = []
    for (kind, _, _, _, kw), (sig, _, meta) in zip(jobs, pmap(_run, jobs)):
        s = slice(200, None)
        rows.append(dict(ground=kw["ground"], wheel=kw["wheel_shape"],
                         contact="rigid" if kw["tire_stiffness"] is None else "soft",
                         phys_hz=kw["phys_hz"], az_std=float(sig["acc"][s, 2].std()),
                         ax_std=float(sig["acc"][s, 0].std()), gy_std=float(sig["gyro"][s, 1].std())))
        print("noise:", rows[-1])

    plot_noise(rows)
    return rows


def plot_noise(rows):
    rates = sorted({x["phys_hz"] for x in rows})
    combos = [("cylinder", "rigid"), ("cylinder", "soft"), ("sphere", "rigid"), ("sphere", "soft")]
    floor = 1e-3
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, ground in zip(axes, ("plane", "heightfield")):
        for i, (w, c) in enumerate(combos):
            r = sorted([x for x in rows if x["ground"] == ground and x["wheel"] == w and x["contact"] == c],
                       key=lambda x: x["phys_hz"])
            y = np.maximum([x["az_std"] for x in r], floor)
            ax.plot(rates, y, marker=ps.MARKERS[i], ms=6, color=ps.SERIES[i], clip_on=False,
                    ls="-" if w == "sphere" else "--", label=f"{w} wheels, {c} contact")
        ax.axhline(NoiseCfg().acc_white, color=ps.MUTED, lw=1, ls=":")
        ax.text(rates[0], NoiseCfg().acc_white * 1.3, "added sensor noise", color=ps.INK2, fontsize=7)
        ax.axvspan(880, 1130, color=ps.GRID, alpha=0.7, lw=0, zorder=0)
        ax.text(1000, 12, "used", ha="center", fontsize=7, color=ps.INK2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(rates, [str(r) for r in rates]); ax.minorticks_off()
        ax.set_xlabel("physics rate (Hz)")
        ax.set_title("analytic flat plane" if ground == "plane" else "flat heightfield (the dataset's ground)", loc="left")
    axes[0].text(110, 1.4e-3, "sphere wheels: exactly 0", fontsize=7, color=ps.INK2)
    axes[0].set_ylabel("vertical accel. std on flat ground (m/s²)")
    axes[0].set_ylim(floor, 20)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Vibration the simulator invents on perfectly flat ground (1.5 m/s, no sensor noise)",
                 fontsize=10, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(f"{FIG}/check_noise_floor.png"); plt.close(fig)


# ---------------------------------------------------------------- 3. washboard
def check_washboard():
    speeds = [0.6, 1.2, 1.8, 2.4]
    jobs = [("washboard", 0.5, v, 11, dict(duration=14.0, noise=None)) for v in speeds]
    res = list(pmap(_run, jobs))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    out = []
    for i, (v, (sig, _, meta)) in enumerate(zip(speeds, res)):
        lam = meta["terrain"]["wavelength_m"]
        s = _steady(sig, meta)
        az = sig["acc"][s, 2]
        # Welch average over 2 s segments
        seg = 200
        P = np.mean([_psd_long(az[k:k + seg]) for k in range(0, len(az) - seg, seg // 2)], axis=0)
        f = np.fft.rfftfreq(2048, 1 / 100)
        v_true = float(np.mean(sig["speed_true"][s]))
        expected = v_true / lam
        m = (f > 1) & (f < 45)
        peak = float(f[m][np.argmax(P[m])])
        out.append(dict(speed=v, v_true=v_true, wavelength=lam, expected_hz=expected, peak_hz=peak))
        ax.semilogy(f[m], P[m], color=ps.SERIES[i], lw=1.6, label=f"{v_true:.1f} m/s")
        ax.axvline(expected, color=ps.SERIES[i], lw=1, ls=":")
        print("washboard:", out[-1])
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("vertical accel. PSD ((m/s²)²/Hz)")
    ax.set_title(f"Washboard ridges every {out[0]['wavelength']*100:.0f} cm: peak should sit at speed ÷ spacing (dotted)", loc="left")
    ax.legend(title="speed", loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(f"{FIG}/check_washboard_peak.png"); plt.close(fig)
    return out


def _psd_long(x):
    x = x - x.mean()
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w, n=2048)
    return np.abs(X) ** 2 / (100 * (w ** 2).sum())


# ---------------------------------------------------------------- 4. convergence
BANDS = [(1, 5), (5, 10), (10, 20), (20, 40)]


def _feat(sig, meta):
    s = _steady(sig, meta)
    f, P = _psd(sig["acc"][s])
    fg, Pg = _psd(sig["gyro"][s])
    return [np.log10(_band(f, P[:, 2], a, b)) for a, b in BANDS] + \
           [np.log10(_band(fg, Pg[:, 1], a, b)) for a, b in BANDS]


def check_converge():
    kinds = ["smooth", "gravel", "cobble", "washboard", "rocky"]
    rates = [400, 1000, 2000]
    seeds = range(100, 105)
    jobs = [(k, 0.5, 1.5, sd, dict(duration=8.0, noise=None, phys_hz=r)) for k in kinds for sd in seeds for r in rates]
    F = {}
    for (k, _, _, sd, kw), (sig, _, meta) in zip(jobs, pmap(_run, jobs)):
        F[(k, sd, kw["phys_hz"])] = np.array(_feat(sig, meta))
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharex=True, sharey=True)
    for ax, r in zip(axes, (400, 1000)):
        for i, k in enumerate(kinds):
            a = np.array([F[(k, sd, r)] for sd in seeds]); b = np.array([F[(k, sd, 2000)] for sd in seeds])
            ax.scatter(b.ravel(), a.ravel(), s=14, color=ps.CLASS_COLOR[k], marker=ps.MARKERS[i],
                       edgecolor=ps.SURFACE, linewidth=0.5, label=k)
        lo, hi = -5, 3.5
        ax.plot([lo, hi], [lo, hi], color=ps.INK2, lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("feature at 2000 Hz physics (log10 band power)")
        ax.set_title(f"{r} Hz vs 2000 Hz physics", loc="left")
    axes[0].set_ylabel("same feature, same terrain")
    axes[1].legend(loc="lower right")
    fig.savefig(f"{FIG}/check_convergence.png"); plt.close(fig)
    # numbers: mean |difference| between rates vs between-terrain spread (same class)
    for r in (400, 1000):
        d = np.mean([np.abs(F[(k, sd, r)] - F[(k, sd, 2000)]).mean() for k in kinds for sd in seeds])
        out[f"mean_abs_diff_{r}_vs_2000"] = float(d)
    spread = np.mean([np.array([F[(k, sd, 2000)] for sd in seeds]).std(0).mean() for k in kinds])
    out["within_class_seed_spread_2000"] = float(spread)
    per = {}
    for k in kinds:
        per[k] = {r: float(np.mean([np.abs(F[(k, sd, r)] - F[(k, sd, 2000)]).mean() for sd in seeds])) for r in (400, 1000)}
        per[k]["seed_spread"] = float(np.array([F[(k, sd, 2000)] for sd in seeds]).std(0).mean())
    out["per_class"] = per
    print("converge:", json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["static", "noise", "washboard", "converge"]
    path = f"{RES}/checks.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    if which == ["replot"]:
        plot_noise(res["noise"]); sys.exit()
    for w in which:
        res[w] = globals()[f"check_{w}"]()
        json.dump(res, open(path, "w"), indent=1)
    print("done:", which)

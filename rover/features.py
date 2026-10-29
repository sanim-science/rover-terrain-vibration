"""Hand-crafted window features.

Set A  "vibration"  : per-channel time statistics + log band power in fixed Hz bands
Set B  "+speed"     : A + wheel-encoder speed
Set C  "per-metre"  : speed-normalised spectra.  Using the encoder speed v, each
                      channel's spectrum is converted from temporal frequency f [Hz]
                      to spatial frequency k = f / v [cycles per metre], and from
                      acceleration / angular rate to displacement / angle
                      (divide by (2 pi f)^4 or (2 pi f)^2).  The band power in a fixed
                      k-band is then (ideally) a property of the ground, not the speed.
                      Plus shape statistics that don't scale with speed, plus log v.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

FS = 100
CH = ["ax", "ay", "az", "gx", "gy", "gz"]
HZ_BANDS = [(1, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 50)]
K_BANDS = [(2.5, 4.0), (4.0, 6.3), (6.3, 10.0), (10.0, 15.0)]  # cycles / m
NFFT = 1024


def _psd(x):
    """One-sided PSD of each column of x (n, c) with a Hann window, zero-padded."""
    x = x - x.mean(0)
    w = np.hanning(len(x))[:, None]
    X = np.fft.rfft(x * w, n=NFFT, axis=0)
    psd = (np.abs(X) ** 2) / (FS * (w ** 2).sum())
    psd[1:-1] *= 2
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    return f, psd


def _band(f, psd, lo, hi):
    m = (f >= lo) & (f < hi)
    df = f[1] - f[0]
    return psd[m].sum(0) * df


def feats_A(win):
    """win: (100, 7) = acc(3), gyro(3), wheel_speed."""
    x = win[:, :6]
    d = x - x.mean(0)
    out = {}
    for i, c in enumerate(CH):
        out[f"{c}_std"] = d[:, i].std()
        out[f"{c}_p2p"] = np.ptp(d[:, i])
        out[f"{c}_kurt"] = kurtosis(d[:, i])
        out[f"{c}_skew"] = skew(d[:, i])
        out[f"{c}_jerk"] = np.abs(np.diff(x[:, i])).mean()
    f, P = _psd(x)
    for i, c in enumerate(CH):
        for lo, hi in HZ_BANDS:
            out[f"{c}_hz{lo}-{hi}"] = np.log10(_band(f, P[:, i], lo, hi) + 1e-12)
        tot = P[f >= 1, i]
        out[f"{c}_centroid"] = (f[f >= 1] * tot).sum() / (tot.sum() + 1e-12)
        out[f"{c}_peakhz"] = f[f >= 1][np.argmax(tot)]
    return out


def feats_B(win):
    out = feats_A(win)
    out["wheel_speed"] = win[:, 6].mean()
    return out


def feats_C(win):
    x = win[:, :6]
    v = max(win[:, 6].mean(), 0.05)
    f, P = _psd(x)
    fz = np.where(f > 0, f, np.inf)
    out = {}
    for i, c in enumerate(CH):
        power = 4 if c.startswith("a") else 2       # acc -> displacement, rate -> angle
        Pd = P[:, i] / (2 * np.pi * fz) ** power
        for lo, hi in K_BANDS:
            band = _band(f, Pd, lo * v, hi * v)     # variance in that spatial band
            out[f"{c}_k{lo}-{hi}"] = np.log10(band + 1e-16)
        d = x[:, i] - x[:, i].mean()
        out[f"{c}_kurt"] = kurtosis(d)
        out[f"{c}_skew"] = skew(d)
        # peak spatial frequency (bumps per metre) of the displacement-equivalent spectrum
        m = (f >= 1) & (f <= 40)
        out[f"{c}_peak_k"] = f[m][np.argmax(P[m, i])] / v
    out["log_speed"] = np.log(v)
    return out


FEATURE_SETS = {"A_vibration": feats_A, "B_vibration+speed": feats_B, "C_per_metre": feats_C}


def featurize(X, fn):
    rows = [fn(w) for w in X]
    names = list(rows[0])
    return np.array([[r[k] for k in names] for r in rows], dtype=float), names

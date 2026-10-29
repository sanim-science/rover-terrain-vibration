"""Parametric terrain generator.

Every terrain is a height grid h[ix, iy] in metres on a regular grid with spacing
DX along the driving direction (x) and across it (y).  Each terrain class has a
severity knob s in [0, 1] that scales its roughness, so the dataset can hold out
"rougher than anything seen in training" as a test condition.

Classes
-------
smooth     near-flat ground: long gentle undulations plus sub-millimetre texture
gravel     isotropic random texture, correlation length 1.5-3 cm
cobble     densely packed rounded bumps (paving / cobblestones)
washboard  periodic ridges across the track (corrugated dirt road)
rocky      sparse large rocks on a lightly textured base
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfiltfilt

DX = 0.01  # grid spacing [m]
CLASSES = ("smooth", "gravel", "cobble", "washboard", "rocky")
FLAT_LEAD_IN = 1.5  # first metres of every track are blended to flat for a clean spawn


def _random_field(rng, shape, corr_len, sigma):
    """Gaussian random field with given correlation length [m] and std [m]."""
    f = gaussian_filter(rng.standard_normal(shape), sigma=corr_len / DX, mode="wrap")
    f /= f.std() + 1e-12
    return f * sigma


def _stamp_bumps(h, rng, centers, radii, heights):
    """Add cosine-dome bumps: z = H * 0.5 * (1 + cos(pi r / R)) for r < R."""
    nx, ny = h.shape
    for (cx, cy), R, H in zip(centers, radii, heights):
        i0, i1 = int((cx - R) / DX), int((cx + R) / DX) + 2
        j0, j1 = int((cy - R) / DX), int((cy + R) / DX) + 2
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, nx), min(j1, ny)
        if i0 >= i1 or j0 >= j1:
            continue
        xs = np.arange(i0, i1) * DX - cx
        ys = np.arange(j0, j1) * DX - cy
        r = np.sqrt(xs[:, None] ** 2 + ys[None, :] ** 2)
        dome = np.where(r < R, H * 0.5 * (1 + np.cos(np.pi * np.minimum(r / R, 1))), 0.0)
        # max() rather than + so overlapping stones don't stack into towers
        h[i0:i1, j0:j1] = np.maximum(h[i0:i1, j0:j1], dome)


def make_terrain(kind: str, severity: float, length: float, width: float, rng):
    """Return (h, params) with h of shape (nx, ny) in metres, x along the track."""
    s = float(severity)
    nx, ny = int(round(length / DX)), int(round(width / DX))
    X = np.arange(nx) * DX
    params: dict = {"kind": kind, "severity": s}

    if kind == "flat":  # perfectly flat, used only for the simulator noise-floor check
        h = np.zeros((nx, ny))

    elif kind == "smooth":
        lam = rng.uniform(2.0, 5.0)
        amp = rng.uniform(0.003, 0.008)
        tex = 0.0002 + 0.0004 * s
        h = amp * np.sin(2 * np.pi * X / lam + rng.uniform(0, 2 * np.pi))[:, None] * np.ones((1, ny))
        h += _random_field(rng, (nx, ny), 0.02, tex)
        params.update(undulation_wavelength_m=lam, undulation_amp_m=amp, texture_std_m=tex)

    elif kind == "gravel":
        corr = rng.uniform(0.015, 0.03)
        sigma = 0.0008 + 0.0024 * s
        h = _random_field(rng, (nx, ny), corr, sigma)
        params.update(corr_len_m=corr, height_std_m=sigma)

    elif kind == "cobble":
        spacing = rng.uniform(0.08, 0.12)
        height = 0.0025 + 0.006 * s
        gx = np.arange(spacing / 2, length, spacing)
        gy = np.arange(spacing / 2, width, spacing)
        cx, cy = np.meshgrid(gx, gy, indexing="ij")
        jit = 0.15 * spacing
        centers = np.stack([cx.ravel() + rng.uniform(-jit, jit, cx.size),
                            cy.ravel() + rng.uniform(-jit, jit, cy.size)], 1)
        radii = rng.uniform(0.40, 0.55, len(centers)) * spacing
        heights = height * rng.uniform(0.7, 1.3, len(centers))
        h = np.zeros((nx, ny))
        _stamp_bumps(h, rng, centers, radii, heights)
        h += _random_field(rng, (nx, ny), 0.01, 0.0002)
        params.update(spacing_m=spacing, stone_height_m=height)

    elif kind == "washboard":
        lam = rng.uniform(0.15, 0.45)
        amp = 0.0012 + 0.0036 * s
        # slow wavelength wander (+-5%) so the ridge spacing isn't a perfect tone
        phase = 2 * np.pi * np.cumsum(1.0 / (lam * (1 + 0.05 * np.sin(2 * np.pi * X / rng.uniform(8, 15))))) * DX
        h = amp * np.sin(phase + rng.uniform(0, 2 * np.pi))[:, None] * np.ones((1, ny))
        h += _random_field(rng, (nx, ny), 0.02, 0.0003)
        params.update(wavelength_m=lam, amplitude_m=amp)

    elif kind == "rocky":
        density = rng.uniform(3.0, 5.0)  # rocks per m^2
        height = 0.005 + 0.015 * s
        n = rng.poisson(density * length * width)
        centers = np.stack([rng.uniform(0, length, n), rng.uniform(0, width, n)], 1)
        radii = rng.uniform(0.06, 0.15, n)
        heights = height * rng.uniform(0.6, 1.4, n)
        h = _random_field(rng, (nx, ny), 0.02, 0.0005)
        _stamp_bumps(h, rng, centers, radii, heights)
        params.update(rock_density_per_m2=density, rock_height_m=height)

    else:
        raise ValueError(f"unknown terrain kind {kind!r}")

    # blend the lead-in to flat so the car always spawns on level ground
    ramp = np.clip((X - FLAT_LEAD_IN) / 0.5, 0, 1)[:, None]
    h = h * ramp
    return h.astype(np.float64), params


def highpass_along_x(h, cutoff_len=1.0):
    """Remove wavelengths longer than cutoff_len [m] along x (zero-phase 4th-order
    Butterworth).  What remains is what we call 'roughness'; slopes and long gentle
    undulations are terrain shape, not roughness."""
    sos = butter(4, 1.0 / cutoff_len, btype="high", fs=1.0 / DX, output="sos")
    return sosfiltfilt(sos, h, axis=0)

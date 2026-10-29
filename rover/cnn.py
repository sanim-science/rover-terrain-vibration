"""A small 1-D convolutional network in plain NumPy (no deep-learning library needed).

    input  (B, 6, 100)  acc xyz + gyro xyz, window mean removed, standardised
    conv 6->16, k=7  -> ReLU -> max-pool 2
    conv 16->32, k=5 -> ReLU -> max-pool 2
    conv 32->32, k=5 -> ReLU -> global average pool
    concat log wheel speed (optional)
    dense 33->32 -> ReLU -> dense 32->n_out
Loss: softmax cross-entropy (classification) or squared error (regression).
Optimiser: Adam with decoupled weight decay.  Gradients are verified against finite
differences in tests/test_cnn_grad.py.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------- layers
def conv_fwd(x, W, b):
    """x (B,C,L), W (C*k, F), b (F,) -> out (B,F,Lo), cache"""
    B, C, L = x.shape
    k = W.shape[0] // C
    Lo = L - k + 1
    cols = sliding_window_view(x, k, axis=2)               # (B, C, Lo, k)
    cols = cols.transpose(0, 2, 1, 3).reshape(B * Lo, C * k)
    out = (cols @ W + b).reshape(B, Lo, -1).transpose(0, 2, 1)
    return out, (cols, x.shape, k)


def conv_bwd(dout, W, cache):
    cols, (B, C, L), k = cache
    Lo = L - k + 1
    d = dout.transpose(0, 2, 1).reshape(B * Lo, -1)
    dW = cols.T @ d
    db = d.sum(0)
    dcols = (d @ W.T).reshape(B, Lo, C, k)
    dx = np.zeros((B, C, L), dtype=dout.dtype)
    for j in range(k):
        dx[:, :, j:j + Lo] += dcols[:, :, :, j].transpose(0, 2, 1)
    return dx, dW, db


def pool_fwd(x):
    B, F, L = x.shape
    L2 = L // 2
    xr = x[:, :, :2 * L2].reshape(B, F, L2, 2)
    idx = xr.argmax(3)
    out = np.take_along_axis(xr, idx[..., None], 3)[..., 0]
    return out, (idx, x.shape)


def pool_bwd(dout, cache):
    idx, (B, F, L) = cache
    L2 = L // 2
    dxr = np.zeros((B, F, L2, 2), dtype=dout.dtype)
    np.put_along_axis(dxr, idx[..., None], dout[..., None], 3)
    dx = np.zeros((B, F, L), dtype=dout.dtype)
    dx[:, :, :2 * L2] = dxr.reshape(B, F, 2 * L2)
    return dx


# ---------------------------------------------------------------- network
class CNN1D:
    def __init__(self, n_out, use_speed=True, seed=0, channels=(16, 32, 32), kernels=(7, 5, 5),
                 hidden=32, in_ch=6, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.use_speed, self.n_out, self.dtype = use_speed, n_out, dtype
        self.p = {}
        c_in = in_ch
        for i, (c, k) in enumerate(zip(channels, kernels)):
            self.p[f"W{i}"] = (rng.standard_normal((c_in * k, c)) * np.sqrt(2 / (c_in * k))).astype(dtype)
            self.p[f"b{i}"] = np.zeros(c, dtype)
            c_in = c
        d_in = c_in + (1 if use_speed else 0)
        self.p["Wd"] = (rng.standard_normal((d_in, hidden)) * np.sqrt(2 / d_in)).astype(dtype)
        self.p["bd"] = np.zeros(hidden, dtype)
        self.p["Wo"] = (rng.standard_normal((hidden, n_out)) * np.sqrt(1 / hidden)).astype(dtype)
        self.p["bo"] = np.zeros(n_out, dtype)
        self.n_conv = len(channels)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x, s):
        caches = []
        h = x
        for i in range(self.n_conv):
            h, cc = conv_fwd(h, self.p[f"W{i}"], self.p[f"b{i}"])
            mask = h > 0
            h = h * mask
            if i < self.n_conv - 1:
                h, pc = pool_fwd(h)
            else:
                pc = None
            caches.append((cc, mask, pc))
        Lg = h.shape[2]
        g = h.mean(2)                                        # global average pool
        z = np.concatenate([g, s[:, None]], 1) if self.use_speed else g
        a = z @ self.p["Wd"] + self.p["bd"]
        am = a > 0
        hd = a * am
        out = hd @ self.p["Wo"] + self.p["bo"]
        self._cache = (caches, Lg, z, am, hd)
        return out

    def backward(self, dout):
        caches, Lg, z, am, hd = self._cache
        gr = {}
        gr["Wo"] = hd.T @ dout; gr["bo"] = dout.sum(0)
        dh = (dout @ self.p["Wo"].T) * am
        gr["Wd"] = z.T @ dh; gr["bd"] = dh.sum(0)
        dz = dh @ self.p["Wd"].T
        dg = dz[:, :-1] if self.use_speed else dz
        d = np.repeat(dg[:, :, None] / Lg, Lg, axis=2)
        for i in reversed(range(self.n_conv)):
            cc, mask, pc = caches[i]
            if pc is not None:
                d = pool_bwd(d, pc)
            d = d * mask
            d, gr[f"W{i}"], gr[f"b{i}"] = conv_bwd(d, self.p[f"W{i}"], cc)
        return gr

    def loss_grad(self, x, s, y, task):
        out = self.forward(x, s)
        n = len(x)
        if task == "cls":
            z = out - out.max(1, keepdims=True)
            pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
            loss = -np.log(pr[np.arange(n), y] + 1e-12).mean()
            d = pr.copy(); d[np.arange(n), y] -= 1; d /= n
        else:
            e = out[:, 0] - y
            loss = 0.5 * np.mean(e ** 2)
            d = (e / n)[:, None]
        return loss, self.backward(d.astype(out.dtype))

    def step(self, gr, lr=1e-3, wd=1e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k, g in gr.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= (lr * (mh / (np.sqrt(vh) + eps) + (wd * self.p[k] if k.startswith("W") else 0))).astype(self.dtype)

    def predict_raw(self, x, s, batch=1024):
        return np.concatenate([self.forward(x[i:i + batch], s[i:i + batch]) for i in range(0, len(x), batch)])

    def get(self):
        return {k: v.copy() for k, v in self.p.items()}

    def set(self, p):
        self.p = {k: v.copy() for k, v in p.items()}

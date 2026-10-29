"""Finite-difference check of every gradient in rover/cnn.py (float64)."""
import numpy as np

from rover.cnn import CNN1D


def _check(task, use_speed):
    rng = np.random.default_rng(1)
    net = CNN1D(n_out=5 if task == "cls" else 1, use_speed=use_speed, seed=3, dtype=np.float64,
                channels=(4, 5, 3), hidden=6)
    x = rng.standard_normal((4, 6, 40)); s = rng.standard_normal(4)
    y = rng.integers(0, 5, 4) if task == "cls" else rng.standard_normal(4)
    _, gr = net.loss_grad(x, s, y, task)
    worst = 0.0
    for k, P in net.p.items():
        flat = P.ravel()
        for i in rng.choice(flat.size, min(12, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + 1e-6; lp, _ = net.loss_grad(x, s, y, task)
            flat[i] = old - 1e-6; lm, _ = net.loss_grad(x, s, y, task)
            flat[i] = old
            num = (lp - lm) / 2e-6
            ana = gr[k].ravel()[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
            assert rel < 1e-5 or abs(num - ana) < 1e-9, (task, k, i, num, ana)
    return worst


def test_grads():
    for task in ("cls", "reg"):
        for sp in (True, False):
            w = _check(task, sp)
            print(f"{task} speed={sp}: worst relative error {w:.2e}")


if __name__ == "__main__":
    test_grads(); print("OK")

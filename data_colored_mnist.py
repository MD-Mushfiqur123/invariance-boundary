"""ColoredMNIST-style data (EXP6).

Images: 2-channel (red, green) MNIST. Label y = digit parity (0=even, 1=odd).
Color correlation with label per environment:
  e0 train: corr +0.9   e1 train: corr +0.3   test_id: +0.9   test_ood: -0.9 (flipped)
Color tint strength a = corr * sign(y) : image = img * (tint applied multiplicatively
per channel), so color is a strong shortcut feature in train envs.
"""
import gzip
import struct
import numpy as np

import os

MNIST_DIR = "/tmp"


def _load_idx(path):
    with gzip.open(path, "rb") as f:
        zero, dtype, ndim = struct.unpack(">HBB", f.read(4))
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist():
    xtr = _load_idx(os.path.join(MNIST_DIR, "train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    ytr = _load_idx(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte.gz"))
    xte = _load_idx(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    yte = _load_idx(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte.gz"))
    return xtr, ytr, xte, yte


def colorize(x, y, rho, rng):
    """x: (n,28,28) grayscale; returns (n,2,28,28) RGB-ish 2-channel tensor."""
    n = len(x)
    par = 2 * (y % 2) - 1
    flip = rng.random(n) < 0.25
    y_noisy = np.where(flip, 1 - (y % 2), (y % 2)).astype(np.int64)
    t = 2 * y_noisy - 1
    color = ((rng.random(n) < (1 + rho * t) / 2) * 2 - 1).astype(np.float32)
    g = rng.normal(0, 0.15, n).astype(np.float32) + color
    red = np.clip(0.5 + 0.5 * g, 0, 1)[:, None, None]
    green = np.clip(0.5 - 0.5 * g, 0, 1)[:, None, None]
    out = np.stack([x * red, x * green], 1)
    return out.astype(np.float32), y_noisy


def make_envs(n_per_env=6000, seed=0):
    xtr, ytr, xte, yte = load_mnist()
    rng = np.random.default_rng(seed)
    data = {}
    specs = {"e0": 0.9, "e1": 0.3, "e2": 0.0, "test_id": 0.9, "test_ood": -0.9}
    for k, rho in specs.items():
        xs, ys = (xtr, ytr) if k in ("e0", "e1") else (xte, yte)
        idx = rng.choice(len(xs), size=min(n_per_env if k in ("e0", "e1") else len(xs), len(xs)), replace=False)
        imgs, labels = colorize(xs[idx], ys[idx], rho, rng)
        data[k] = (imgs, labels)
    return data


if __name__ == "__main__":
    d = make_envs(n_per_env=2000, seed=0)
    for k, (imgs, labels) in d.items():
        flat = imgs.mean(axis=(2, 3))
        color_stat = (flat[:, 0] - flat[:, 1]).mean()
        acc_by_color = np.mean((flat[:, 0] - flat[:, 1] > 0) == (labels == 1))
        print(f"{k:9s} imgs {imgs.shape} labels {labels.mean():.2f} "
              f"red-green mean {color_stat:+.3f} color-predicts-label {acc_by_color:.3f}")

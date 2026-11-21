"""Colab master run — self-contained, GPU-ready.

Runs the full ColoredMNIST config battery (EXP10 + EXP13 families) with >=10 seeds
on a free Colab GPU, so the headline numbers get seed counts that meet Nature-MI
review standards (the sandbox/CPU runs used 3-5 seeds).

Usage in Colab (one code cell):

    !wget -q https://raw.githubusercontent.com/MD-Mushfiqur123/invariance-boundary/<BRANCH>/colab_master_run.py
    !python colab_master_run.py --seeds 10

Outputs: prints a summary table and writes colab_results.csv (one row per
seed x config x split). Download it via the Colab file browser and share it back.
"""
__version__ = "v2-gpu-fix"
import argparse
import csv
import gzip
import os
import struct
import urllib.request
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}
MNIST_DIR = os.environ.get("MNIST_DIR", "/tmp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_mnist():
    os.makedirs(MNIST_DIR, exist_ok=True)
    for name, url in MNIST_URLS.items():
        path = os.path.join(MNIST_DIR, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)


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
    par = 2 * (y % 2) - 1
    flip = rng.random(len(x)) < 0.25
    y_noisy = np.where(flip, 1 - (y % 2), (y % 2)).astype(np.int64)
    t = 2 * y_noisy - 1
    color = ((rng.random(len(x)) < (1 + rho * t) / 2) * 2 - 1).astype(np.float32)
    g = rng.normal(0, 0.15, len(x)).astype(np.float32) + color
    red = np.clip(0.5 + 0.5 * g, 0, 1)[:, None, None]
    green = np.clip(0.5 - 0.5 * g, 0, 1)[:, None, None]
    return np.stack([x * red, x * green], 1).astype(np.float32), y_noisy


def make_envs(n_per_env=6000, seed=0):
    xtr, ytr, xte, yte = load_mnist()
    rng = np.random.default_rng(seed)
    data = {}
    specs = {"e0": 0.9, "e1": 0.3, "e2": 0.0, "test_id": 0.9, "test_ood": -0.9}
    for k, rho in specs.items():
        xs, ys = (xtr, ytr) if k in ("e0", "e1") else (xte, yte)
        size = min(n_per_env if k in ("e0", "e1") else len(xs), len(xs))
        idx = rng.choice(len(xs), size=size, replace=False)
        data[k] = colorize(xs[idx], ys[idx], rho, rng)
    return data


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(1152, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.backbone(x))


def rex_penalty(losses):
    losses = torch.stack(losses)
    m = losses.mean()
    return ((losses - m) ** 2).mean()


def irm_penalty(model, envs, idx):
    penalty = 0.0
    for k in envs:
        xb, yb = envs[k][0][idx], envs[k][1][idx]
        scale = torch.ones(1, requires_grad=True, device=DEVICE)
        loss_e = F.cross_entropy(scale * model(xb), yb)
        g = torch.autograd.grad(loss_e, scale, create_graph=True)[0]
        penalty = penalty + g ** 2
    return penalty


def make_env_tensors(data, env_keys):
    return {k: (torch.tensor(data[k][0], device=DEVICE), torch.tensor(data[k][1], device=DEVICE))
            for k in env_keys}


def train_loop(model, data, env_keys, epochs, bs, lr, seed, rex=None, irm=None, dro=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    envs = make_env_tensors(data, env_keys)
    keys = list(env_keys)
    n = min(len(v[0]) for v in envs.values())
    w = torch.ones(len(keys), device=DEVICE) / len(keys)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        env_loss_sum = torch.zeros(len(keys), device=DEVICE)
        n_batches = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            losses = [F.cross_entropy(model(envs[k][0][idx]), envs[k][1][idx]) for k in keys]
            loss = sum(losses) / len(losses)
            if rex is not None:
                loss = loss + rex * rex_penalty(losses)
            if irm is not None:
                loss = loss + irm * irm_penalty(model, envs, idx)
            if dro is not None:
                loss = (w * torch.stack(losses)).sum()
            loss.backward()
            opt.step()
            if dro is not None:
                env_loss_sum = env_loss_sum + torch.stack(losses).detach()
                n_batches += 1
        if dro is not None:
            lm = env_loss_sum / n_batches
            w = F.softmax(dro * len(keys) * lm, dim=0)
    return model


@torch.no_grad()
def acc(model, imgs, labels, bs=2048):
    model.eval()
    x = torch.tensor(imgs, device=DEVICE)
    y = torch.tensor(labels, device=DEVICE)
    correct = 0
    for i in range(0, len(x), bs):
        correct += model(x[i:i + bs]).argmax(1).eq(y[i:i + bs]).sum().item()
    return correct / len(x)


def retrain_head_dfr(model, e1, epochs, bs, lr, seed):
    m = CNN().to(DEVICE)
    m.backbone.load_state_dict(model.backbone.state_dict())
    for p in m.backbone.parameters():
        p.requires_grad = False
    torch.manual_seed(seed)
    m.head = nn.Sequential(nn.Flatten(), nn.Linear(1152, 64), nn.ReLU(), nn.Linear(64, 2)).to(DEVICE)
    m = m.to(DEVICE)
    opt = torch.optim.Adam(m.head.parameters(), lr=lr)
    x = torch.tensor(e1[0], device=DEVICE)
    y = torch.tensor(e1[1], device=DEVICE)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(x), device=DEVICE)
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(m(x[idx]), y[idx]).backward()
            opt.step()
    return m


CONFIGS = {
    "ERM-2env":      dict(env_keys=("e0", "e1")),
    "V-REx-2env":    dict(env_keys=("e0", "e1"), rex=30.0),
    "ERM-3env":      dict(env_keys=("e0", "e1", "e2")),
    "V-REx-3env":    dict(env_keys=("e0", "e1", "e2"), rex=30.0),
    "IRMv1-2env":    dict(env_keys=("e0", "e1"), irm=25.0),
    "IRMv1-3env":    dict(env_keys=("e0", "e1", "e2"), irm=25.0),
    "GroupDRO-2env": dict(env_keys=("e0", "e1"), dro=0.1),
    "GroupDRO-3env": dict(env_keys=("e0", "e1", "e2"), dro=0.1),
    "ERM+DFR":       dict(env_keys=("e0", "e1"), dfr=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-per-env", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--out", default="colab_results.csv")
    args = ap.parse_args()

    print(f"device={DEVICE} seeds={args.seeds} n_per_env={args.n_per_env} "
          f"epochs={args.epochs} harness={__version__} configs={args.configs}", flush=True)
    get_mnist()

    rows = []
    res = {c: {"id": [], "ood": []} for c in args.configs}
    for seed in range(args.seeds):
        data = make_envs(n_per_env=args.n_per_env, seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        for c in args.configs:
            cfg = CONFIGS[c]
            kw = dict(epochs=args.epochs, bs=args.bs, lr=args.lr, seed=seed,
                      rex=cfg.get("rex"), irm=cfg.get("irm"), dro=cfg.get("dro"))
            m = train_loop(CNN().to(DEVICE), data, cfg["env_keys"], **kw)
            if cfg.get("dfr"):
                m = retrain_head_dfr(m, data["e1"], args.epochs // 2, args.bs, args.lr, seed)
            aid = acc(m, *data["test_id"])
            aood = acc(m, *data["test_ood"])
            res[c]["id"].append(aid)
            res[c]["ood"].append(aood)
            rows.append([seed, c, f"{aid:.4f}", f"{aood:.4f}"])
        print(f"seed {seed} done", flush=True)

    with open(args.out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["seed", "config", "acc_id", "acc_ood"])
        wtr.writerows(rows)

    print(f"\n=== ColoredMNIST master run ({args.seeds} seeds, device={DEVICE}) ===")
    ref = {"ERM-2env": 0.121, "V-REx-2env": 0.620, "ERM-3env": 0.425,
           "V-REx-3env": 0.680, "IRMv1-2env": 0.633, "IRMv1-3env": 0.638,
           "GroupDRO-2env": 0.130, "GroupDRO-3env": 0.448, "ERM+DFR": 0.478}
    for c in args.configs:
        i, o = np.array(res[c]["id"]), np.array(res[c]["ood"])
        r = ref.get(c)
        print(f"{c:14s} ID {i.mean():.4f}+-{i.std():.4f}   OOD {o.mean():.4f}+-{o.std():.4f}"
              + (f"   (5-seed CPU ref {r:.3f})" if r else ""))
    print(f"\nwrote {args.out} — download it from the Colab file browser and share it back.")


if __name__ == "__main__":
    main()

"""Colab synthetic master run — self-contained, GPU-ready.

Re-runs the synthetic SCM battery (EXP1 + component ablations + the 3-env
decorrelation test) with 10 seeds so every synthetic headline number in the
paper carries publication-grade seed counts.

Usage in Colab (T4 GPU):

    !wget -qO colab_synthetic_master.py https://raw.githubusercontent.com/MD-Mushfiqur123/invariance-boundary/<SHA>/colab_synthetic_master.py
    !python colab_synthetic_master.py --seeds 10

Writes colab_synthetic_results.csv (per seed x config) and prints mean +- sd
with Cohen's d vs ERM and the causal-mask recall of the SCIN variants.
"""
import argparse
import csv
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

__version__ = "v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CAUSAL, N_SPURIOUS, N_NUISANCE = 4, 2, 16
N_LATENT = N_CAUSAL + N_SPURIOUS + N_NUISANCE
OBS_DIM = 64
ENV_RHO = {"e0": 0.9, "e1": 0.3, "e2": 0.0, "test_id": 0.9, "test_ood": -0.9}
ENV_SEED = {"e0": 100, "e1": 200, "e2": 300, "test_id": 400, "test_ood": 500}


class MixingNetwork:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0 / np.sqrt(N_LATENT), size=(N_LATENT, 32)).astype(np.float32)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(32), size=(32, OBS_DIM)).astype(np.float32)
        self.W2b = rng.normal(0, 1.0 / np.sqrt(32), size=(32, OBS_DIM)).astype(np.float32)

    def __call__(self, z):
        t = np.tanh(z @ self.W1)
        return t @ self.W2 + 0.05 * (t * t) @ self.W2b


def sample_latents(rho, n, rng):
    z_c = rng.normal(0, 1, size=(n, N_CAUSAL))
    logits = z_c[:, 0] * z_c[:, 1] + 0.8 * z_c[:, 2] - 1.0 * z_c[:, 3] + 0.3 * rng.normal(0, 1, n)
    y = (logits > 0).astype(np.float64)
    t = 2 * y - 1
    z_s1 = rho * 1.2 * t + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)
    z_s2 = 0.5 * rho * t + 0.3 * rng.normal(0, 1, n)
    z_n = rng.normal(0, 1, size=(n, N_NUISANCE))
    return np.concatenate([z_c, np.stack([z_s1, z_s2], 1), z_n], 1), y


def make_data(n_train=8192, n_test=4096):
    mix = MixingNetwork()
    data = {}
    for k, rho in ENV_RHO.items():
        n = n_train if k in ("e0", "e1", "e2") else n_test
        rng = np.random.default_rng(ENV_SEED[k] + 777)
        z, y = sample_latents(rho, n, rng)
        data[k] = (mix(z).astype(np.float32), y.astype(np.int64))
    return data


class SCINBase(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, latent=22):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, latent))
        self.latent = latent
        self.dag_scale = 0.5
        self.W = nn.Parameter(0.1 * torch.randn(latent, latent) / latent ** 0.5)
        self.logit_mask = nn.Parameter(torch.zeros(latent))
        self.head = nn.Linear(latent, 2)

    def adjacency(self):
        return self.W * self.W

    def mask(self):
        return torch.sigmoid(self.logit_mask).detach().cpu().numpy()


class SCINFull(SCINBase):
    def forward(self, x):
        h = self.enc(x)
        h = h + self.dag_scale * torch.tanh(h @ self.adjacency())
        return self.head(h * torch.sigmoid(self.logit_mask))


class SCINNoDAG(SCINBase):
    def forward(self, x):
        return self.head(self.enc(x) * torch.sigmoid(self.logit_mask))


class SCINNoMask(SCINBase):
    def forward(self, x):
        h = self.enc(x)
        return self.head(h + self.dag_scale * torch.tanh(h @ self.adjacency()))


class ERMBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Linear(128, 2)

    def forward(self, x):
        return self.head(self.enc(x))


def rex_penalty(losses):
    losses = torch.stack(losses)
    return ((losses - losses.mean()) ** 2).mean()


def train(model, data, env_keys, epochs, bs, lr, seed, rex=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    envs = {k: (torch.tensor(data[k][0], device=DEVICE), torch.tensor(data[k][1], device=DEVICE))
            for k in env_keys}
    n = min(len(v[0]) for v in envs.values())
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            losses = [F.cross_entropy(model(envs[k][0][idx]), envs[k][1][idx])
                      for k in env_keys]
            loss = sum(losses) / len(losses)
            if rex is not None:
                loss = loss + rex * rex_penalty(losses)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, data, key, bs=4096):
    model.eval()
    x = torch.tensor(data[key][0], device=DEVICE)
    y = torch.tensor(data[key][1], device=DEVICE)
    correct = 0
    for i in range(0, len(x), bs):
        correct += model(x[i:i + bs]).argmax(1).eq(y[i:i + bs]).sum().item()
    return correct / len(x)


CONFIGS = {
    "ERM-2env":    dict(cls=ERMBaseline, env_keys=("e0", "e1")),
    "V-REx-2env":  dict(cls=ERMBaseline, env_keys=("e0", "e1"), rex=30.0),
    "ERM-3env":    dict(cls=ERMBaseline, env_keys=("e0", "e1", "e2")),
    "V-REx-3env":  dict(cls=ERMBaseline, env_keys=("e0", "e1", "e2"), rex=30.0),
    "SCIN-full":   dict(cls=SCINFull, env_keys=("e0", "e1"), rex=30.0, mask_metric=True),
    "SCIN-noDAG":  dict(cls=SCINNoDAG, env_keys=("e0", "e1"), rex=30.0, mask_metric=True),
    "SCIN-noMask": dict(cls=SCINNoMask, env_keys=("e0", "e1"), rex=30.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-test", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--out", default="colab_synthetic_results.csv")
    args = ap.parse_args()

    print(f"device={DEVICE} harness={__version__} seeds={args.seeds} "
          f"n_train={args.n_train} epochs={args.epochs}", flush=True)

    rows, recalls = [], {c: [] for c in args.configs}
    res = {c: {"id": [], "ood": []} for c in args.configs}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        data = make_data(args.n_train, args.n_test)
        for c in args.configs:
            cfg = CONFIGS[c]
            m = train(cfg["cls"]().to(DEVICE), data, cfg["env_keys"],
                      args.epochs, args.bs, args.lr, seed, rex=cfg.get("rex"))
            aid = evaluate(m, data, "test_id")
            aood = evaluate(m, data, "test_ood")
            res[c]["id"].append(aid)
            res[c]["ood"].append(aood)
            rows.append([seed, c, f"{aid:.4f}", f"{aood:.4f}"])
            if cfg.get("mask_metric"):
                mk = m.mask()
                recalls[c].append(float(np.mean(mk[:N_CAUSAL] > 0.5)))
        print(f"seed {seed} done", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "config", "acc_id", "acc_ood"])
        w.writerows(rows)

    print(f"\n=== Synthetic SCM battery ({args.seeds} seeds, device={DEVICE}) ===")
    base = np.array(res.get("ERM-2env", res[args.configs[0]])["ood"])
    for c in args.configs:
        i, o = np.array(res[c]["id"]), np.array(res[c]["ood"])
        delta = o - base
        d = delta.mean() / delta.std() if delta.std() > 1e-6 else float("nan")
        line = (f"{c:12s} ID {i.mean():.4f}+-{i.std():.4f}   OOD {o.mean():.4f}+-{o.std():.4f}   "
                f"Cohen d vs ERM {d:+.2f}")
        if recalls[c]:
            line += f"   causal-mask recall {np.mean(recalls[c]):.3f}"
        print(line)
    print(f"\nwrote {args.out} — download it from the Colab file browser and share it back.")


if __name__ == "__main__":
    main()

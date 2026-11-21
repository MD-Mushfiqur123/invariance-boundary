"""Colab WILDS EXP19: does the cheap probe transfer matrix PREDICT where a deep
model collapses? Prospective test on cached Camelyon17-WILDS (no download).

EXP18b showed the 2-minute color-probe matrix splits hospitals into mapping
blocks: {0,1,3} mutually compatible, {2,4} a separate block (~chance transfer
between blocks). EXP17 showed the CNN collapses to 0.51 on hospital 2. If probe
transfer from the training hospitals predicts per-hospital deep-model OOD
accuracy, we get a practical, minutes-cheap deployment diagnostic: measure the
matrix, know which target hospital fails — and the law says why (mapping
heterogeneity = the rho-flip regime, where no gain from recombination is
predicted; EXP17's null delta).

Design:
  1. Patch pools per hospital: probe pool (2500), CNN train pool {0,3}
     (4000 each), CNN eval pool (2000 per hospital 0..4) — all mutually disjoint.
  2. Transfer matrix from probes trained on h0 and h3 (8-dim color/staining
     features, train-time normalization carried over); per-target predictor =
     mean raw transfer accuracy from {h0, h3}, majority-adjusted.
  3. Train TorsoCNN on {0,3} (N seeds); evaluate on every hospital's eval pool.
  4. Correlate CNN accuracy vs predicted transfer across the 5 hospitals
     (all-5 and unseen-targets-only h1/h2/h4). Verdict: targets with predicted
     transfer-excess ~0 should sit at the majority baseline; the correlation
     is the diagnostic's score.

Usage in Colab:

    !pip install -q wilds
    !wget -qO colab_wilds_predict.py https://raw.githubusercontent.com/MD-Mushfiqur123/invariance-boundary/<SHA>/colab_wilds_predict.py
    !python colab_wilds_predict.py

--smoke runs on random tensors without wilds.
"""
import argparse
import csv
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

__version__ = "v1-predict"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 96
HOSPITALS = (0, 1, 2, 3, 4)
TRAIN_HOSPITALS = (0, 3)
ROLES = {0: "train-1", 1: "ID-val", 2: "OOD-test", 3: "train-2", 4: "third"}


class TorsoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))


def proxy_features(imgs):
    m = imgs.mean(dim=(2, 3))
    s = imgs.std(dim=(2, 3))
    return torch.cat([m, s, m[:, :1] - m[:, 1:2], m[:, 1:2] - m[:, 2:3]], dim=1)


def train_probe(f, y, steps=300, seed=0):
    torch.manual_seed(seed)
    n = len(y)
    perm = torch.randperm(n)
    tr, te = perm[:n // 2], perm[n // 2:]
    mu, sd = f[tr].mean(0, keepdim=True), f[tr].std(0, keepdim=True).clamp_min(1e-6)
    lin = nn.Linear(f.shape[1], 2)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin((f[tr] - mu) / sd), y[tr]).backward()
        opt.step()
    return lin, mu, sd


def train_cnn(model, xs, ys, epochs, bs, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xs), device=xs.device)
        for i in range(0, len(xs), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(xs[idx]), ys[idx]).backward()
            opt.step()
    return model


@torch.no_grad()
def acc(model, xs, ys, bs=512):
    model.eval()
    correct = 0
    for i in range(0, len(xs), bs):
        correct += model(xs[i:i + bs]).argmax(1).eq(ys[i:i + bs]).sum().item()
    return correct / len(xs)


def pearson_spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    def r(u, v):
        if u.std() == 0 or v.std() == 0:
            return float("nan")
        return float(np.corrcoef(u, v)[0, 1])
    return r(a, b), r(ra, rb)


def load_smoke(args, seed=0):
    print("SMOKE MODE: random tensors instead of wilds data", flush=True)
    n_per = max(args.n_eval, args.n_train)
    rng = np.random.default_rng(seed)
    pools = {}
    for h in HOSPITALS:
        labels = torch.tensor((rng.random(n_per) < 0.5).astype(np.int64))
        imgs = torch.tensor(rng.random((n_per, 3, IMG, IMG)), dtype=torch.float32)
        pools[h] = (imgs, labels)
    tmat = {0: {j: 0.7 - 0.05 * abs(j) for j in HOSPITALS}, 3: {j: 0.7 - 0.05 * abs(4 - j) for j in HOSPITALS}}
    return pools, {h: pools[h] for h in TRAIN_HOSPITALS}, tmat, "smoke"


def to_tensor_img(a):
    t = torch.tensor(a)
    return t.permute(2, 0, 1) if t.ndim == 3 else t


def load_wilds(args, seed=0):
    print("loading camelyon17-wilds (reuses cached data, no re-download)...", flush=True)
    from wilds import get_dataset
    ds = get_dataset("camelyon17", download=True)
    fields = list(ds.metadata_fields)
    print(f"metadata fields: {fields}", flush=True)

    def field(*names):
        for nm in names:
            if nm in fields:
                return fields.index(nm)
        raise ValueError(f"none of {names} found in metadata fields {fields}")

    H = field("hospital", "center", "site")
    Y = field("y", "tumor")
    md = ds.metadata_array.numpy()
    y = ds.y_array.numpy().flatten().astype(np.int64)
    rng = np.random.default_rng(seed)
    used = np.zeros(len(md), dtype=bool)

    def take(h, n, key):
        mask = (md[:, H] == h) & (y >= 0) & ~used
        idxs = rng.choice(np.where(mask)[0], size=min(n, int(mask.sum())), replace=False)
        used[idxs] = True
        imgs = torch.stack([to_tensor_img(np.asarray(ds[int(i)][0])) for i in idxs]).float()[:, :, :IMG, :IMG]
        lab = torch.tensor(y[idxs]).long()
        print(f"  {key}: {len(idxs)} patches (tumor rate {lab.float().mean():.3f})", flush=True)
        return imgs, lab

    probe, train_pools, eval_pools, labels = {}, {}, {}, {}
    for h in HOSPITALS:
        probe[h] = take(h, args.probe_n, f"probe-pool h{h}")
    for h in TRAIN_HOSPITALS:
        train_pools[h] = take(h, args.n_train, f"cnn-train h{h}")
    for h in HOSPITALS:
        eval_pools[h] = take(h, args.n_eval, f"cnn-eval h{h}")
        labels[h] = eval_pools[h][1]

    print("\nprobe transfer matrix (raw acc, train-time normalization from source):", flush=True)
    tmat = {}
    for src in TRAIN_HOSPITALS:
        lin, mu, sd = train_probe(proxy_features(probe[src][0]), probe[src][1])
        tmat[src] = {}
        for j in HOSPITALS:
            tmat[src][j] = float(lin((proxy_features(eval_pools[j][0]) - mu) / sd).argmax(1).eq(labels[j]).float().mean())
        print(f"  from h{src}: " + "  ".join(f"h{j} {tmat[src][j]:.3f}" for j in HOSPITALS), flush=True)
    return eval_pools, train_pools, tmat, "wilds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--probe-n", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="wilds_predict_results.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"device={DEVICE} harness={__version__} seeds={args.seeds} smoke={args.smoke}", flush=True)
    loader = load_smoke if args.smoke else load_wilds
    pools, train_pools, tmat, fmt = loader(args)

    labels = {h: pools[h][1] for h in HOSPITALS}
    tumor_rate = {h: labels[h].float().mean().item() for h in HOSPITALS}
    pred = {j: float(np.mean([tmat[s][j] for s in TRAIN_HOSPITALS])) - max(tumor_rate[j], 1 - tumor_rate[j]) for j in HOSPITALS}
    print("\npredicted transfer excess per target (mean of {h0,h3} probes, majority-adjusted):", flush=True)
    print("  " + "  ".join(f"h{j} {pred[j]:+.3f}" for j in HOSPITALS), flush=True)

    xs = torch.cat([train_pools[h][0] for h in TRAIN_HOSPITALS]).to(DEVICE)
    ys = torch.cat([train_pools[h][1] for h in TRAIN_HOSPITALS]).to(DEVICE)
    rows, cnn = [], {h: [] for h in HOSPITALS}
    for seed in range(args.seeds):
        m = train_cnn(TorsoCNN().to(DEVICE), xs, ys, args.epochs, args.bs, args.lr, seed)
        for h in HOSPITALS:
            a = acc(m, pools[h][0].to(DEVICE), labels[h].to(DEVICE))
            cnn[h].append(a)
            rows.append([seed, h, ROLES[h], f"{a:.4f}", f"{pred[h]:+.4f}"])
        print(f"seed {seed} done", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "target_hospital", "role", "cnn_acc", "pred_transfer_excess"])
        w.writerows(rows)

    cnn_mean = {h: float(np.mean(cnn[h])) for h in HOSPITALS}
    excess = {h: cnn_mean[h] - max(tumor_rate[h], 1 - tumor_rate[h]) for h in HOSPITALS}
    print(f"\n=== EXP19: probe matrix vs deep-model OOD ({args.seeds} seeds, data={fmt}) ===", flush=True)
    print("target  role      CNN acc            excess   predicted transfer excess", flush=True)
    for h in HOSPITALS:
        sd = float(np.std(cnn[h]))
        print(f"  h{h}    {ROLES[h]:9s} {cnn_mean[h]:.4f}+-{sd:.4f}   {excess[h]:+.3f}   {pred[h]:+.3f}", flush=True)
    r_all, rho_all = pearson_spearman([pred[h] for h in HOSPITALS], [cnn_mean[h] for h in HOSPITALS])
    unseen = [h for h in HOSPITALS if h not in TRAIN_HOSPITALS]
    r_u, rho_u = pearson_spearman([pred[h] for h in unseen], [cnn_mean[h] for h in unseen])
    print(f"\nPearson r (all 5 targets):  {r_all:+.3f}   Spearman rho: {rho_all:+.3f}", flush=True)
    print(f"Pearson r (unseen {unseen}): {r_u:+.3f}   Spearman rho: {rho_u:+.3f}", flush=True)
    print("\nDiagnostic claim if r is high: a ~2-minute probe matrix predicts which target", flush=True)
    print("hospital a deep model collapses on — the mapping-shift regime made measurable.", flush=True)
    print(f"wrote {args.out} — share it back.", flush=True)


if __name__ == "__main__":
    main()

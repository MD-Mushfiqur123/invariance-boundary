"""Colab RxRx1 second-dataset pilot (EXP20) — replicates the EXP19 transfer
diagnostic on a second public dataset (fluorescence microscopy, siRNA
perturbation classification). Downloads ~7.4 GB once (data/rxrx1_v1.0).

Verified wilds API facts (wilds/datasets/rxrx1_dataset.py):
  - metadata_fields = ['cell_type', 'experiment', 'plate', 'well', 'site', 'y']
  - y = sirna_id, 1139 classes; original images are large PNGs -> we downscale
    to 96x96 on load and keep only what we sample.
  - 'official' split: train = 33 experiments (site 1), val = 4 experiments,
    test = 14 experiments (held-out lab/experiment shift targets).
  - Images arrive as 0-255 PNGs; WILDS default transform rescales to [0,1].

Replication design (mirrors EXP19):
  1. Training experiments: 3 experiments, one per cell type (HEPG2/HUVEC/RPE),
     drawn from the official 'train' split.
  2. Targets: up to 3 held-out train-split experiments (mapping aligned) + up
     to 4 test-split experiments (lab shift, regime unknown) = up to 7 points.
  3. Shortcut proxy: 8-dim per-channel statistics; multiclass linear probe
     (-> siRNA classes) trained per training experiment on its own probe pool;
     the train-time normalization is carried over to the target eval pools;
     per-target predictor = mean probe accuracy from the 3 training
     experiments, majority-adjusted.
  4. TorsoCNN (head -> 1139 classes) trained on disjoint 3-training-experiment
     pools (N seeds); evaluated on every target's disjoint eval pool.
  5. Verdict (pre-registered): does predicted probe transfer correlate with
     per-target CNN accuracy (Pearson/Spearman over all targets and over the
     unseen test-split targets only), i.e. does the transfer diagnostic
     replicate outside histopathology?

Kill question (pre-registered): if the correlation here is ~0 while Camelyon17
gave +0.88, the diagnostic was pathology-specific, not general.

Usage in Colab (T4 GPU):

    !pip install -q wilds
    !wget -qO colab_rxrx1_pilot.py https://raw.githubusercontent.com/MD-Mushfiqur123/invariance-boundary/<SHA>/colab_rxrx1_pilot.py
    !python colab_rxrx1_pilot.py

--smoke runs on planted synthetic tensors without wilds or any download.
"""
import argparse
import csv
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

__version__ = "v1-rxrx1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 96
N_SOURCES = 3
N_HELDOUT = 3
N_TEST = 4


class TorsoCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))


def proxy_features(imgs):
    m = imgs.mean(dim=(2, 3))
    s = imgs.std(dim=(2, 3))
    return torch.cat([m, s, m[:, :1] - m[:, 1:2], m[:, 1:2] - m[:, 2:3]], dim=1)


def train_probe_multiclass(f, y, n_classes, steps=300, seed=0):
    torch.manual_seed(seed)
    n = len(y)
    if len(torch.unique(y)) < 2:
        return None
    mu, sd = f.mean(0, keepdim=True), f.std(0, keepdim=True).clamp_min(1e-6)
    lin = nn.Linear(f.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    idx = torch.arange(n)
    for _ in range(steps):
        j = idx[torch.randperm(n)[:512]] if n > 512 else idx
        opt.zero_grad()
        F.cross_entropy(lin((f[j] - mu) / sd), y[j]).backward()
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
        if len(u) < 2 or u.std() == 0 or v.std() == 0:
            return float("nan")
        return float(np.corrcoef(u, v)[0, 1])
    return r(a, b), r(ra, rb)


def synth_pool(rng, n, n_classes, perm=None, noise=0.12):
    labels = torch.tensor(rng.integers(0, n_classes, n).astype(np.int64))
    cols = torch.zeros(n_classes, 3)
    cols[:, 0] = torch.linspace(0.15, 0.85, n_classes)
    cols[:, 1] = 1.0 - cols[:, 0]
    cols[:, 2] = 0.5
    if perm is not None:
        cols = cols[torch.tensor(perm)]
    c = cols[labels].view(-1, 3, 1, 1)
    imgs = (c + noise * torch.randn(n, 3, IMG, IMG)).clamp(0, 1)
    return imgs, labels


def load_smoke(args, seed=0):
    print("SMOKE MODE: planted synthetic tensors instead of wilds data", flush=True)
    n_classes = 6
    rng = np.random.default_rng(seed)
    per = max(args.n_eval, args.n_train, args.probe_n, 100)
    probe_pools, train_pools = [], []
    for _ in range(N_SOURCES):
        probe_pools.append(synth_pool(rng, per, n_classes))
        train_pools.append(synth_pool(rng, per, n_classes))
    pools, targets = {}, {}
    for i in range(N_HELDOUT):
        pools[f"t{i}"] = synth_pool(rng, per, n_classes)
        targets[f"t{i}"] = (f"heldout-exp{i}", "heldout-train")
    perms = [
        None,
        list(range(n_classes))[::-1],
        [(i + 1) % n_classes for i in range(n_classes)],
        [(i + 3) % n_classes for i in range(n_classes)],
    ][:N_TEST]
    for i, pm in enumerate(perms):
        t = f"t{N_HELDOUT + i}"
        pools[t] = synth_pool(rng, per, n_classes, perm=pm)
        targets[t] = (f"test-exp{i}", "test-split")
    return probe_pools, train_pools, pools, targets, n_classes, "smoke"


def to_small_img(a):
    if isinstance(a, np.ndarray):
        arr = a
    else:
        arr = np.asarray(a.resize((IMG, IMG)))
    return torch.tensor(arr).permute(2, 0, 1).float() / 255.0


def load_wilds(args, seed=0):
    print("loading rxrx1-wilds (~7.4 GB one-time download; cached afterwards)...", flush=True)
    from wilds import get_dataset
    ds = get_dataset("rxrx1", download=True)
    fields = list(ds.metadata_fields)
    print(f"metadata fields: {fields}", flush=True)

    def field(*names):
        for nm in names:
            if nm in fields:
                return fields.index(nm)
        raise ValueError(f"none of {names} found in metadata fields {fields}")

    E = field("experiment", "plate")
    C = field("cell_type")
    md = ds.metadata_array.numpy()
    y = ds.y_array.numpy().flatten().astype(np.int64)
    split = ds.split_array.numpy() if hasattr(ds.split_array, "numpy") else np.asarray(ds.split_array)
    n_classes = int(y.max()) + 1
    rng = np.random.default_rng(seed)
    used = np.zeros(len(md), dtype=bool)

    exp_ids = sorted(set(md[:, E].tolist()))
    exp_cell = {e: md[np.where(md[:, E] == e)[0][0], C] for e in exp_ids}
    exp_split = {e: split[np.where(md[:, E] == e)[0][0]] for e in exp_ids}
    train_exps = [e for e in exp_ids if exp_split[e] == 0]
    test_exps = [e for e in exp_ids if exp_split[e] == 2]

    def first_per_cell(pool, cap):
        cell_seen, chosen = set(), []
        for e in pool:
            c = exp_cell[e]
            if c not in cell_seen:
                cell_seen.add(c)
                chosen.append(e)
                if len(chosen) >= cap:
                    break
        return chosen

    train_pick = first_per_cell(train_exps, N_SOURCES)
    if len(train_pick) < N_SOURCES:
        raise RuntimeError(f"only {len(train_pick)} train-split cell types found (need {N_SOURCES})")
    heldout_pick = first_per_cell([e for e in train_exps if e not in train_pick], N_HELDOUT)
    test_pick = first_per_cell(test_exps, N_TEST + 1)
    for e in test_exps:
        if len(test_pick) >= N_TEST:
            break
        if e not in test_pick:
            test_pick.append(e)
    targets = {}
    for i, e in enumerate(heldout_pick[:N_HELDOUT]):
        targets[f"t{i}"] = (e, "heldout-train")
    for i, e in enumerate(test_pick[:N_TEST]):
        targets[f"t{N_HELDOUT + i}"] = (e, "test-split")
    print(f"source experiments {train_pick} (one per cell type, train split); "
          f"targets: " + ", ".join(f"{t}=exp{e}({role})" for t, (e, role) in targets.items()), flush=True)

    def take(exp, n, key):
        mask = (md[:, E] == exp) & ~used
        n_av = int(mask.sum())
        idxs = rng.choice(np.where(mask)[0], size=min(n, n_av), replace=False)
        used[idxs] = True
        imgs = torch.stack([to_small_img(ds[int(i)][0]) for i in idxs])
        lab = torch.tensor(y[idxs]).long()
        maj = lab.bincount(minlength=n_classes).max().item() / len(lab)
        print(f"  {key} (exp {exp}): {len(idxs)} images, {len(lab.unique())} classes, majority {maj:.4f}", flush=True)
        return imgs, lab

    probe_pools, train_pools, pools = [], [], {}
    for i, e in enumerate(train_pick):
        probe_pools.append(take(e, args.probe_n, f"probe-pool src{i}"))
    for i, e in enumerate(train_pick):
        train_pools.append(take(e, args.n_train, f"cnn-train src{i}"))
    for t, (e, role) in targets.items():
        pools[t] = take(e, args.n_eval, f"cnn-eval {t} ({role})")
    return probe_pools, train_pools, pools, targets, n_classes, "wilds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--probe-n", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="rxrx1_pilot_results.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"device={DEVICE} harness={__version__} seeds={args.seeds} smoke={args.smoke}", flush=True)
    loader = load_smoke if args.smoke else load_wilds
    probe_pools, train_pools, pools, targets, n_classes, fmt = loader(args)

    labels = {t: pools[t][1] for t in targets}

    print("\nprobe transfer matrix (multiclass siRNA acc; train-time normalization from source):", flush=True)
    tmat = {i: {} for i in range(N_SOURCES)}
    for i in range(N_SOURCES):
        probe = train_probe_multiclass(proxy_features(probe_pools[i][0]), probe_pools[i][1], n_classes, steps=args.steps)
        if probe is None:
            raise RuntimeError(f"probe {i} degenerate (<2 classes)")
        lin, mu, sd = probe
        for t in targets:
            tmat[i][t] = float(lin((proxy_features(pools[t][0]) - mu) / sd).argmax(1).eq(labels[t]).float().mean())
        print(f"  from src{i}: " + "  ".join(f"{t} {tmat[i][t]:.3f}" for t in targets), flush=True)

    pred = {t: float(np.mean([tmat[i][t] for i in range(N_SOURCES)])) for t in targets}
    print("\npredicted probe transfer per target (mean of 3 source probes):", flush=True)
    print("  " + "  ".join(f"{t} {pred[t]:.3f}" for t in targets), flush=True)

    xs = torch.cat([p[0] for p in train_pools]).to(DEVICE)
    ys = torch.cat([p[1] for p in train_pools]).to(DEVICE)
    rows, cnn = [], {t: [] for t in targets}
    for seed in range(args.seeds):
        m = train_cnn(TorsoCNN(n_classes).to(DEVICE), xs, ys, args.epochs, args.bs, args.lr, seed)
        for t in targets:
            a = acc(m, pools[t][0].to(DEVICE), labels[t].to(DEVICE))
            cnn[t].append(a)
            rows.append([seed, t, targets[t][1], f"{a:.4f}", f"{pred[t]:.4f}", f"{targets[t][0]}"])
        print(f"seed {seed} done", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "target", "role", "cnn_acc", "pred_transfer", "experiment_id"])
        w.writerows(rows)

    cnn_mean = {t: float(np.mean(cnn[t])) for t in targets}
    print(f"\n=== EXP20 RxRx1 transfer diagnostic ({args.seeds} seeds, data={fmt}) ===", flush=True)
    print("target  role            CNN acc            predicted probe transfer", flush=True)
    for t in targets:
        sdv = float(np.std(cnn[t]))
        print(f"  {t}   {targets[t][1]:13s} {cnn_mean[t]:.4f}+-{sdv:.4f}   {pred[t]:.3f}", flush=True)
    unseen = [t for t in targets if targets[t][1] == "test-split"]
    r_all, rho_all = pearson_spearman([pred[t] for t in targets], [cnn_mean[t] for t in targets])
    r_u, rho_u = pearson_spearman([pred[t] for t in unseen], [cnn_mean[t] for t in unseen])
    print(f"\nPearson r (all {len(targets)} targets): {r_all:+.3f}   Spearman: {rho_all:+.3f}", flush=True)
    print(f"Pearson r (unseen {unseen}): {r_u:+.3f}   Spearman: {rho_u:+.3f}", flush=True)
    print("\nKill question (pre-registered): vs Camelyon17 EXP19 r=+0.88 — high r here =", flush=True)
    print("the transfer diagnostic replicates outside histopathology; ~0 = pathology-specific.", flush=True)
    print(f"wrote {args.out} — share it back.", flush=True)


if __name__ == "__main__":
    main()

"""Colab WILDS real-data pilot — self-contained, GPU-ready, SOLO (no partner needed).

Tests the paper's decorrelation law on real hospital data (Camelyon17-wilds):
does adding a third training hospital whose shortcut-to-label association is
measured WEAK help OOD, in the paired design of COLLABORATION_PROPOSAL.md?

Steps:
  1. Download Camelyon17-wilds (public, ~15 GB; no credentials needed).
  2. Measure per-hospital shortcut strength: accuracy of a linear probe that
     predicts the tumor label from cheap color/staining proxy features,
     subsampled per hospital.
  3. Paired pilot (3 seeds each): train a torso CNN on hospitals {1,2}; then
     {1,2,3}; evaluate on the wilds OOD split (hospital 4) and ID split.
     Decision rule: if hospital 3's shortcut strength is below the {1,2} mean,
     adding it should HELP OOD; above, the law predicts little/no help —
     the sign comparison is the pilot verdict.

Usage in Colab (T4 GPU):

    !pip install -q wilds
    !wget -qO colab_wilds_pilot.py https://raw.githubusercontent.com/MD-Mushfiqur123/invariance-boundary/<SHA>/colab_wilds_pilot.py
    !python colab_wilds_pilot.py --seeds 3

Sandbox smoke mode (--smoke) uses random tensors instead of wilds data to
verify all training/eval code paths without the download.

WILDS API facts (verified from wilds/datasets/camelyon17_dataset.py):
  - metadata_fields = ['hospital', 'slide', 'y', 'from_source_domain']
    (the tumor label lives in metadata under name 'y', NOT 'tumor').
  - hospitals are 0-INDEXED: training hospitals are 0, 3, 4; the held-out
    ID-val hospital is 1; the OOD test hospital is 2 (the WILDS paper label
    calls the test hospital 5 in a 1-indexed scheme).
"""
import argparse
import csv
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

__version__ = "v2-fieldfix"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 96


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
    """Cheap scanner/staining shortcut proxy: per-channel mean/std + channel gaps."""
    m = imgs.mean(dim=(2, 3))
    s = imgs.std(dim=(2, 3))
    return torch.cat([m, s, m[:, :1] - m[:, 1:2], m[:, 1:2] - m[:, 2:3]], dim=1)


def shortcut_strength(imgs, labels, seed=0):
    torch.manual_seed(seed)
    f = proxy_features(imgs)
    y = torch.as_tensor(labels).long()
    n = len(y)
    perm = torch.randperm(n)
    tr, te = perm[:n // 2], perm[n // 2:]
    mu, sd = f[tr].mean(0, keepdim=True), f[tr].std(0, keepdim=True).clamp_min(1e-6)
    lin = nn.Linear(f.shape[1], 2)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        F.cross_entropy(lin((f[tr] - mu) / sd), y[tr]).backward()
        opt.step()
    with torch.no_grad():
        return float(lin((f[te] - mu) / sd).argmax(1).eq(y[te]).float().mean())


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


def load_smoke(n_per=600, seed=0):
    print("SMOKE MODE: random tensors instead of wilds data", flush=True)
    rng = np.random.default_rng(seed)
    data = {}
    for k in (1, 2, 3, "id", "ood"):
        imgs = torch.tensor(rng.random((n_per, 3, IMG, IMG)), dtype=torch.float32)
        labels = torch.tensor((rng.random(n_per) < 0.3).astype(np.int64))
        data[k] = (imgs, labels)
    return data, "smoke"


def load_wilds(subsample=5000, seed=0):
    print("loading camelyon17-wilds (reuses data/camelyon17_v1.0 if already downloaded)...", flush=True)
    from wilds import get_dataset
    ds = get_dataset("camelyon17", download=True)
    print(f"metadata fields: {list(ds.metadata_fields)}", flush=True)
    rng = np.random.default_rng(seed)
    fields = list(ds.metadata_fields)

    def field(*names):
        for nm in names:
            if nm in fields:
                return fields.index(nm)
        raise ValueError(f"none of {names} found in metadata fields {fields}")

    H = field("hospital", "center", "site")
    Y = field("y", "tumor")
    md = ds.metadata_array.numpy()
    y = ds.y_array.numpy().flatten().astype(np.int64)
    for h in sorted(set(md[:, H].tolist())):
        n_ok = int(((md[:, H] == h) & (y >= 0)).sum())
        print(f"  hospital {h}: {n_ok} labeled patches", flush=True)
    data = {}

    def grab(idxs, key):
        arrs = []
        for i in idxs:
            a = np.asarray(ds[int(i)][0])
            arrs.append(torch.tensor(a).permute(2, 0, 1) if a.ndim == 3 else torch.tensor(a))
        imgs = torch.stack(arrs)
        data[key] = (imgs.float()[:, :, :IMG, :IMG], torch.tensor(y[idxs]).long())
        print(f"  -> {key}: {len(idxs)} patches", flush=True)

    def pick(hospitals, n, key, exclude=None):
        mask = np.isin(md[:, H], hospitals) & (y >= 0)
        if exclude is not None and len(exclude):
            mask[exclude] = False
        idxs = rng.choice(np.where(mask)[0], size=min(n, int(mask.sum())), replace=False)
        grab(idxs, key)
        return idxs

    pool = pick((0, 3), 2 * subsample, "h1_pool")
    grab(pool[:subsample], 1)
    grab(pool[subsample:2 * subsample], 2)
    pick((4,), subsample, 3)
    pick((2,), subsample, "ood")
    pick((0, 3), subsample // 2, "id", exclude=pool)
    return data, "wilds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--subsample", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="wilds_pilot_results.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"device={DEVICE} harness={__version__} seeds={args.seeds} smoke={args.smoke}", flush=True)
    data, fmt = load_smoke(args.subsample) if args.smoke else load_wilds(args.subsample)

    print("\nStep 1 — per-hospital shortcut strength (proxy linear probe):", flush=True)
    strengths = {}
    for k in (1, 2, 3):
        strengths[k] = shortcut_strength(*data[k], seed=0)
        print(f"  hospital {k}: shortcut strength {strengths[k]:.3f}", flush=True)
    print(f"  {{1,2}} mean = {np.mean([strengths[1], strengths[2]]):.3f}", flush=True)

    print("\n(hospital keys 1,2 = wilds hospitals 0,3; 3rd = wilds 4; OOD = wilds 2; ID = disjoint {0,3} patches)", flush=True)
    print("\nStep 2 — paired training ({1,2} vs {1,2,3}):", flush=True)
    rows = []
    res = {c: {"id": [], "ood": []} for c in ("ERM-2h", "ERM-3h")}
    for seed in range(args.seeds):
        for c, hosp in (("ERM-2h", (1, 2)), ("ERM-3h", (1, 2, 3))):
            xs = torch.cat([data[h][0] for h in hosp]).to(DEVICE)
            ys = torch.cat([data[h][1] for h in hosp]).to(DEVICE)
            m = train_cnn(TorsoCNN().to(DEVICE), xs, ys, args.epochs, args.bs, args.lr, seed)
            aid = acc(m, data["id"][0].to(DEVICE), data["id"][1].to(DEVICE))
            aood = acc(m, data["ood"][0].to(DEVICE), data["ood"][1].to(DEVICE))
            res[c]["id"].append(aid)
            res[c]["ood"].append(aood)
            rows.append([seed, c, f"{aid:.4f}", f"{aood:.4f}"])
        print(f"seed {seed} done", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "config", "acc_id", "acc_ood"])
        w.writerows(rows)

    print(f"\n=== WILDS pilot ({args.seeds} seeds, data={fmt}) ===")
    for c in ("ERM-2h", "ERM-3h"):
        i, o = np.array(res[c]["id"]), np.array(res[c]["ood"])
        print(f"{c:7s} ID {i.mean():.4f}+-{i.std():.4f}   OOD {o.mean():.4f}+-{o.std():.4f}")
    delta = np.array(res["ERM-3h"]["ood"]) - np.array(res["ERM-2h"]["ood"])
    h3_vs = strengths[3] - np.mean([strengths[1], strengths[2]])
    pred = "weak shortcut -> adding 3rd should help" if h3_vs < 0 else "strong shortcut -> expect small/no gain"
    print(f"\ndelta OOD (3h - 2h): {delta.mean():+.4f} +- {delta.std():.4f}")
    print(f"hospital-3 shortcut vs {{1,2}} mean: {h3_vs:+.3f}  [{pred}]")
    print("\nPilot verdict = does sign(delta) match the prediction above.")
    print(f"wrote {args.out} — share it back.")


if __name__ == "__main__":
    main()

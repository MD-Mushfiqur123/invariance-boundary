"""Colab WILDS cross-hospital shortcut transfer matrix (EXP18b) — decides WHICH
regime of the decorrelation law Camelyon17's hospital shift lives in. NO new
download (reuses data/camelyon17_v1.0); probes only, ~2-3 minutes total.

Background: EXP17 showed uniform per-hospital shortcut STRENGTH (0.819-0.825)
yet catastrophic OOD collapse (0.886 ID -> 0.511 on hospital 2), and EXP18
showed hospital 4 has no weak-shortcut slides either (valid slides 0.96+).
Missing measurement: is the shortcut MAPPING P(tumor | color) the same function
in every hospital, or hospital-specific?

Method: per-hospital color/staining proxy features (8-dim channel stats);
train a linear tumor probe on half of hospital i's patches, evaluate on
hospital j (using hospital i's TRAIN-TIME normalization — transfer must not
re-fit the scaler). Diagonal = EXP17-style strength; off-diagonal = mapping
transfer. 5x5 matrix over all hospitals (base train {0,3}, third 4, ID-val 1,
OOD 2).

Reading:
  - Off-diagonals ≈ diagonal  -> shortcut mapping is STABLE across hospitals;
    hospital 2's collapse cannot come from the color channel, so no
    recombination of whole hospitals could decorrelate it (law boundary:
    the count of training environments is irrelevant if they all share rho).
  - Row 2 collapses toward chance -> hospital 2's P(tumor | color) differs
    (the rho-flip family the law describes). Then EXP17's null delta is
    explained: hospitals {0,3,4} share one mapping, so adding h4 leaves the
    training shortcut internally consistent (no decorrelation, no gain), and
    the fix would need a flipped training environment that Camelyon17 simply
    does not offer.

Usage in Colab:

    !pip install -q wilds
    !wget -qO colab_wilds_transfer.py https://raw.githubusercontent.com/MD-Mushfiqur123/invariance-boundary/<SHA>/colab_wilds_transfer.py
    !python colab_wilds_transfer.py

--smoke runs on random features without wilds.
"""
import argparse
import csv
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

__version__ = "v1-transfer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 96
HOSPITALS = (0, 1, 2, 3, 4)
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


def load_smoke(n_per=1000, seed=0):
    print("SMOKE MODE: random features instead of wilds data", flush=True)
    rng = np.random.default_rng(seed)
    feats, labels, tumor_rate = {}, {}, {}
    for h in HOSPITALS:
        labels[h] = torch.tensor((rng.random(n_per) < 0.3).astype(np.int64))
        shift = 0.02 * (h - 2)
        base = 0.5 + shift
        f = rng.random((n_per, 8)) + base
        feats[h] = torch.tensor(f, dtype=torch.float32)
        tumor_rate[h] = labels[h].float().mean().item()
    return feats, labels, tumor_rate, "smoke"


def to_tensor_img(a):
    t = torch.tensor(a)
    return t.permute(2, 0, 1) if t.ndim == 3 else t


def load_wilds(subsample=2500, seed=0):
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
    feats, labels, tumor_rate = {}, {}, {}
    for h in HOSPITALS:
        mask = (md[:, H] == h) & (y >= 0)
        n = min(subsample, int(mask.sum()))
        idxs = rng.choice(np.where(mask)[0], size=n, replace=False)
        imgs = torch.stack([to_tensor_img(np.asarray(ds[int(i)][0])) for i in idxs]).float()[:, :, :IMG, :IMG]
        labels[h] = torch.tensor(y[idxs]).long()
        feats[h] = proxy_features(imgs)
        tumor_rate[h] = labels[h].float().mean().item()
        print(f"  hospital {h} ({ROLES[h]}): {n} patches, tumor rate {tumor_rate[h]:.3f}", flush=True)
        del imgs
    return feats, labels, tumor_rate, "wilds"


def probe_row(feats, labels, src, steps=300, seed=0):
    torch.manual_seed(seed)
    f, y = feats[src], labels[src]
    n = len(y)
    if len(torch.unique(y)) < 2:
        return None
    perm = torch.randperm(n)
    tr, te = perm[:n // 2], perm[n // 2:]
    mu, sd = f[tr].mean(0, keepdim=True), f[tr].std(0, keepdim=True).clamp_min(1e-6)
    lin = nn.Linear(f.shape[1], 2)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin((f[tr] - mu) / sd), y[tr]).backward()
        opt.step()
    return lin, mu, sd, te, y[te]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=2500)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="wilds_transfer_results.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"device={DEVICE} harness={__version__} smoke={args.smoke}", flush=True)
    loader = load_smoke if args.smoke else load_wilds
    feats, labels, tumor_rate, fmt = loader(args.subsample)

    print("\nCross-hospital transfer matrix (row = probe trained on hospital i, col = tested on hospital j;", flush=True)
    print("train-time normalization from i; diagonal = within-hospital strength):", flush=True)
    print("        " + "".join(f"  h{j:<6d}" for j in HOSPITALS), flush=True)
    rows_out, mat = [], {}
    for i in HOSPITALS:
        probe = probe_row(feats, labels, i, steps=args.steps)
        cell_txt, row_acc = [], []
        for j in HOSPITALS:
            if probe is None:
                row_acc.append(float("nan"))
                cell_txt.append("   nan  ")
                continue
            lin, mu, sd, _, _ = probe
            with torch.no_grad():
                a = float(lin((feats[j] - mu) / sd).argmax(1).eq(labels[j]).float().mean())
            maj = max(tumor_rate[j], 1 - tumor_rate[j])
            row_acc.append(a)
            cell_txt.append(f" {a:.3f}/{a - maj:+.2f}")
        mat[i] = row_acc
        print(f"  h{i}  " + "".join(cell_txt), flush=True)
        for j, a in zip(HOSPITALS, row_acc):
            rows_out.append([i, j, f"{a:.4f}" if a == a else "nan", f"{tumor_rate[j]:.4f}", ROLES[i], ROLES[j]])
    print("  (cell = raw accuracy / majority-adjusted excess)", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_hospital", "test_hospital", "acc", "test_tumor_rate", "src_role", "test_role"])
        w.writerows(rows_out)

    diag2 = mat[2][2]
    transfers = [mat[2][j] for j in (0, 3, 4) if mat[2][j] == mat[2][j]]
    tmin, tmax = (min(transfers), max(transfers)) if transfers else (float("nan"), float("nan"))
    print(f"\nhospital 2 (OOD): own strength {diag2:.3f}; transfer from train hospitals {tmin:.3f}..{tmax:.3f}", flush=True)
    print(f"EXP17 reference: ERM OOD on hospital 2 = 0.511 (CNN), color probe transfer = above", flush=True)
    if transfers and tmin >= 0.9 * diag2:
        print("\nREGIME: MAPPING STABLE — P(tumor|color) is the same function in every hospital;")
        print("hospital 2's CNN collapse cannot be a color-shortcut failure, and no recombination")
        print("of whole hospitals could decorrelate it: all training envs share one rho.")
    else:
        print("\nREGIME: MAPPING SHIFT — hospital 2's P(tumor|color) differs from {0,3,4}.")
        print("This EXPLAINS EXP17's null delta: {0,3,4} share one internally consistent shortcut,")
        print("so adding h4 decorrelates nothing (no gain predicted, none observed); fixing OOD")
        print("would need a flipped training environment, which Camelyon17 does not offer.")
    print(f"\nwrote {args.out} — share it back (CSV has the full 5x5 matrix).")


if __name__ == "__main__":
    main()

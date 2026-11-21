"""Colab WILDS slide-level pilot (EXP18) — tests the decorrelation law across a
shortcut-strength RANGE on real data, with NO new download (reuses the cached
Camelyon17-WILDS at data/camelyon17_v1.0).

Motivation (EXP17): every whole hospital has near-identical shortcut strength
(0.819-0.825), so the hospital-paired design could only probe the law's null
regime (add 3rd env with ~zero decorrelation -> observed ~zero gain). Within
hospital 4, however, slides vary in tumor content, so per-slide strengths should
span a range. This harness:

  1. Measures per-SLIDE shortcut strength inside wilds hospital 4 (color/staining
     proxy probe, same 8-dim channel-statistics features).
  2. Splits hospital-4 slides into WEAK-shortcut and STRONG-shortcut arms
     (lowest vs highest valid strengths).
  3. Paired design (N seeds): ERM-2h on wilds hospitals {0,3} is the shared
     base; ERM-3h-weak adds hospital-4 weak-slide patches; ERM-3h-strong adds
     hospital-4 strong-slide patches. OOD = hospital 2, ID = disjoint {0,3} patches.
  4. Verdict: the law predicts delta(weak) > delta(strong) — the weak arm breaks
     the shortcut's label association, the strong arm preserves it.

Confound reported alongside: slide strength correlates with tumor content, so
arm label balance is printed and the verdict is interim until matched.

Usage in Colab (T4 GPU):

    !pip install -q wilds
    !wget -qO colab_wilds_slidepilot.py https://raw.githubusercontent.com/MD-Mushfiqur123/invariance-boundary/<SHA>/colab_wilds_slidepilot.py
    !python colab_wilds_slidepilot.py

--smoke runs random tensors end-to-end without wilds.
"""
import argparse
import csv
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

__version__ = "v1-slidepilot"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 96
BASE_MEAN = 0.822


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


def shortcut_strength(imgs, labels, seed=0):
    torch.manual_seed(seed)
    f = proxy_features(imgs)
    y = torch.as_tensor(labels).long()
    n = len(y)
    if n < 50 or len(torch.unique(y)) < 2:
        return float("nan")
    perm = torch.randperm(n)
    tr, te = perm[:n // 2], perm[n // 2:]
    if len(torch.unique(y[tr])) < 2 or len(torch.unique(y[te])) < 2:
        return float("nan")
    mu, sd = f[tr].mean(0, keepdim=True), f[tr].std(0, keepdim=True).clamp_min(1e-6)
    lin = nn.Linear(f.shape[1], 2)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        F.cross_entropy(lin((f[tr] - mu) / sd), y[tr]).backward()
        opt.step()
    with torch.no_grad():
        return float(lin((f[te] - mu) / sd).argmax(1).eq(y[te]).float().mean())


def minority_rate(labels):
    y = torch.as_tensor(labels).long()
    return min((y == 0).float().mean().item(), (y == 1).float().mean().item())


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
    for k in (1, 2, "weak", "strong", "id", "ood"):
        imgs = torch.tensor(rng.random((n_per, 3, IMG, IMG)), dtype=torch.float32)
        labels = torch.tensor((rng.random(n_per) < 0.3).astype(np.int64))
        data[k] = (imgs, labels)
    strengths = {"weak": 0.55, "strong": 0.75}
    return data, strengths, "smoke"


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
    S = field("slide")
    md = ds.metadata_array.numpy()
    y = ds.y_array.numpy().flatten().astype(np.int64)
    rng = np.random.default_rng(seed)
    data = {}

    def idxs_of(mask, n):
        return rng.choice(np.where(mask)[0], size=min(n, int(mask.sum())), replace=False)

    def grab(idxs, key):
        imgs = torch.stack([to_tensor_img(np.asarray(ds[int(i)][0])) for i in idxs])
        data[key] = (imgs.float()[:, :, :IMG, :IMG], torch.tensor(y[idxs]).long())
        print(f"  -> {key}: {len(idxs)} patches (tumor rate {data[key][1].float().mean():.3f})", flush=True)

    print("\nStep 1 — per-slide shortcut strength in hospital 4:", flush=True)
    h4 = md[:, H] == 4
    rows = []
    for s in sorted(set(md[h4, S].tolist())):
        sm = h4 & (md[:, S] == s) & (y >= 0)
        n = min(args.probe_n, int(sm.sum()))
        if n < 50:
            rows.append((s, float("nan"), n, float("nan")))
            continue
        sel = idxs_of(sm, n)
        imgs = torch.stack([to_tensor_img(np.asarray(ds[int(i)][0])) for i in sel]).float()[:, :, :IMG, :IMG]
        lab = torch.tensor(y[sel]).long()
        mr = minority_rate(lab)
        st = shortcut_strength(imgs, lab)
        rows.append((s, st, n, mr))
        st_txt = f"{st:.3f}" if st == st else "  nan"
        mr_txt = f"{mr:.3f}" if mr == mr else "  nan"
        print(f"  slide {s:>2}: strength {st_txt}  n={n}  minority={mr_txt}", flush=True)

    valid = [r for r in rows if r[1] == r[1] and r[3] == r[3] and r[3] >= args.min_minority]
    if len(valid) < 2 * args.min_slides:
        raise RuntimeError(f"only {len(valid)} usable slides (need {2 * args.min_slides}); lower --probe_n or --min_minority")
    valid.sort(key=lambda r: r[1])
    weak_rows, strong_rows = valid[:args.min_slides], valid[-args.min_slides:]
    weak_slides = [r[0] for r in weak_rows]
    strong_slides = [r[0] for r in strong_rows]
    strengths = {
        "weak": float(np.mean([r[1] for r in weak_rows])),
        "strong": float(np.mean([r[1] for r in strong_rows])),
    }
    print(f"\n  weak arm slides {weak_slides}  (strength {strengths['weak']:.3f})", flush=True)
    print(f"  strong arm slides {strong_slides}  (strength {strengths['strong']:.3f})", flush=True)
    print(f"  whole-hospital {0,3} mean from EXP17 ~{BASE_MEAN:.3f}", flush=True)

    pool_mask = ((md[:, H] == 0) | (md[:, H] == 3)) & (y >= 0)
    pool = idxs_of(pool_mask, 2 * args.n_base)
    used_mask = np.zeros(len(md), dtype=bool)
    used_mask[pool] = True
    grab(pool[:args.n_base], 1)
    grab(pool[args.n_base:2 * args.n_base], 2)

    arm_mask = h4 & np.isin(md[:, S], weak_slides + strong_slides) & (y >= 0)
    arm_idxs = idxs_of(arm_mask, 2 * args.n_arm)
    weak_sel = arm_idxs[:args.n_arm]
    strong_sel = arm_idxs[args.n_arm:2 * args.n_arm]
    grab(weak_sel, "weak")
    grab(strong_sel, "strong")

    grab(idxs_of((md[:, H] == 2) & (y >= 0), args.n_base), "ood")
    grab(idxs_of(pool_mask & ~used_mask, args.n_base // 2), "id")
    return data, strengths, "wilds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-base", type=int, default=3500)
    ap.add_argument("--n-arm", type=int, default=2500)
    ap.add_argument("--probe-n", type=int, default=600)
    ap.add_argument("--min-slides", type=int, default=2)
    ap.add_argument("--min-minority", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="wilds_slidepilot_results.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"device={DEVICE} harness={__version__} seeds={args.seeds} smoke={args.smoke}", flush=True)
    data, strengths, fmt = load_smoke(args.n_base) if args.smoke else load_wilds(args, seed=0)

    print("\nPaired design: base = wilds hospitals {0,3}; arms = hospital-4 weak- vs strong-shortcut slide patches; OOD = hospital 2", flush=True)
    print(f"arm strengths: weak {strengths['weak']:.3f}  strong {strengths['strong']:.3f}", flush=True)

    configs = (("ERM-2h", (1, 2)), ("ERM-3h-weak", (1, 2, "weak")), ("ERM-3h-strong", (1, 2, "strong")))
    rows = []
    res = {c: {"id": [], "ood": []} for c, _ in configs}
    for seed in range(args.seeds):
        for c, parts in configs:
            xs = torch.cat([data[p][0] for p in parts]).to(DEVICE)
            ys = torch.cat([data[p][1] for p in parts]).to(DEVICE)
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

    print(f"\n=== WILDS slide pilot ({args.seeds} seeds, data={fmt}) ===")
    for c, _ in configs:
        i, o = np.array(res[c]["id"]), np.array(res[c]["ood"])
        print(f"{c:13s} ID {i.mean():.4f}+-{i.std():.4f}   OOD {o.mean():.4f}+-{o.std():.4f}")
    d_weak = np.array(res["ERM-3h-weak"]["ood"]) - np.array(res["ERM-2h"]["ood"])
    d_strong = np.array(res["ERM-3h-strong"]["ood"]) - np.array(res["ERM-2h"]["ood"])
    print(f"\ndelta OOD weak arm:   {d_weak.mean():+.4f} +- {d_weak.std():.4f}")
    print(f"delta OOD strong arm: {d_strong.mean():+.4f} +- {d_strong.std():.4f}")
    print(f"\nLaw prediction: weak arm (strength gap {strengths['weak'] - BASE_MEAN:+.3f}) should beat strong arm (gap {strengths['strong'] - BASE_MEAN:+.3f}) on OOD.")
    diff = d_weak.mean() - d_strong.mean()
    verdict = "no difference (tie)" if abs(diff) < 1e-9 else ("MATCHES the law" if diff > 0 else "OPPOSITE of the law")
    print(f"Observed ordering: {verdict} (delta_weak - delta_strong = {d_weak.mean() - d_strong.mean():+.4f})")
    print(f"wrote {args.out} — share it back.")


if __name__ == "__main__":
    main()

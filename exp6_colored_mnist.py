"""EXP6: does the depth × invariance interaction replicate on image data?

2x2 factorial, ColoredMNIST, 5 seeds:
  head: shallow (2 linear layers) vs deep (3 linear layers)
  REx:  off vs lambda=30
Decision: if deep+REx loses most of the OOD gain that shallow+REx gets,
the synthetic depth finding replicates outside engineered data.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_colored_mnist import make_envs
from models import rex_penalty, evaluate as _noop  # reuse penalty


class CNN(nn.Module):
    def __init__(self, deep=False):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
        )  # -> (32, 7, 7)
        if deep:
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(1152, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
        else:
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(1152, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.backbone(x))


def train(model, data, epochs=12, bs=256, lr=1e-3, rex=None, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    envs = {k: (torch.tensor(v[0]), torch.tensor(v[1])) for k, v in data.items() if k in ("e0", "e1")}
    n = min(len(v[0]) for v in envs.values())
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            losses = []
            for k in envs:
                xb, yb = envs[k][0][idx], envs[k][1][idx]
                losses.append(F.cross_entropy(model(xb), yb))
            loss = sum(losses) / len(losses)
            if rex:
                loss = loss + rex * rex_penalty(losses)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def acc(model, imgs, labels, bs=2048):
    model.eval()
    x = torch.tensor(imgs)
    y = torch.tensor(labels)
    correct = 0
    for i in range(0, len(x), bs):
        correct += model(x[i:i + bs]).argmax(1).eq(y[i:i + bs]).sum().item()
    return correct / len(x)


def main(seeds=(0, 1, 2, 3, 4)):
    results = {}
    for seed in seeds:
        data = make_envs(n_per_env=6000, seed=seed)
        for depth in (False, True):
            for rex in (None, 30.0):
                key = f"{'deep' if depth else 'shallow'}/rex={rex or 'off'}"
                torch.manual_seed(seed)
                np.random.seed(seed)
                m = train(CNN(deep=depth), data, seed=seed, rex=rex)
                a_id, a_ood = acc(m, *data["test_id"]), acc(m, *data["test_ood"])
                results.setdefault(key, {"id": [], "ood": []})
                results[key]["id"].append(a_id)
                results[key]["ood"].append(a_ood)
        print(f"seed {seed} done")
    print("\n=== EXP6 ColoredMNIST 2x2 factorial (5 seeds) ===")
    for key, r in results.items():
        i, o = np.array(r["id"]), np.array(r["ood"])
        print(f"{key:18s} ID {i.mean():.4f}±{i.std():.4f}  OOD {o.mean():.4f}±{o.std():.4f}")


if __name__ == "__main__":
    main()

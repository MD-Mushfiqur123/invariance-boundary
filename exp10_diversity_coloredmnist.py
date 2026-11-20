"""EXP10: does environment diversity rescue OOD on image data? (+ DFR baseline)

ColoredMNIST. Training envs: e0 (rho=+0.9), e1 (rho=+0.3). OOD test: rho=-0.9.
Configs (shallow CNN head, 5 seeds):
  A. ERM, 2 envs                       (baseline)
  B. ERM + REx 30, 2 envs              (invariance, best from EXP6)
  C. ERM, 3 envs (+e2: rho=0)          (diversity, no invariance)
  D. ERM + REx 30, 3 envs              (diversity + invariance)
  E. DFR-style: ERM backbone + retrain head on decorrelated env e1
Headline question: does C alone beat B?  If yes: diversity > invariance on images.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_colored_mnist import make_envs
from exp6_colored_mnist import CNN
from models import rex_penalty

torch.set_num_threads(5)


def train(model, data, env_keys, epochs=12, bs=256, lr=1e-3, rex=None, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    envs = {k: (torch.tensor(data[k][0]), torch.tensor(data[k][1])) for k in env_keys}
    n = min(len(v[0]) for v in envs.values())
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            losses = []
            for k in env_keys:
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
    x = torch.tensor(imgs); y = torch.tensor(labels)
    c = 0
    for i in range(0, len(x), bs):
        c += model(x[i:i + bs]).argmax(1).eq(y[i:i + bs]).sum().item()
    return c / len(x)


def retrain_head_dfr(model, e1, epochs=6, bs=256, lr=1e-3, seed=0):
    m = CNN(deep=False)
    m.backbone.load_state_dict(model.backbone.state_dict())
    for p in m.backbone.parameters():
        p.requires_grad = False
    torch.manual_seed(seed)
    m.head = nn.Sequential(nn.Flatten(), nn.Linear(1152, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.Adam(m.head.parameters(), lr=lr)
    x, y = torch.tensor(e1[0]), torch.tensor(e1[1])
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(m(x[idx]), y[idx])
            loss.backward()
            opt.step()
    return m


def main():
    configs = ["ERM-2env", "REx-2env", "ERM-3env", "REx-3env", "DFR"]
    res = {c: {"id": [], "ood": []} for c in configs}
    for seed in range(5):
        data = make_envs(n_per_env=6000, seed=seed)
        torch.manual_seed(seed); np.random.seed(seed)
        m2 = train(CNN(), data, ("e0", "e1"), seed=seed)
        res["ERM-2env"]["id"].append(acc(m2, *data["test_id"]))
        res["ERM-2env"]["ood"].append(acc(m2, *data["test_ood"]))
        m3 = train(CNN(), data, ("e0", "e1"), rex=30.0, seed=seed)
        res["REx-2env"]["id"].append(acc(m3, *data["test_id"]))
        res["REx-2env"]["ood"].append(acc(m3, *data["test_ood"]))
        m4 = train(CNN(), data, ("e0", "e1", "e2"), seed=seed)
        res["ERM-3env"]["id"].append(acc(m4, *data["test_id"]))
        res["ERM-3env"]["ood"].append(acc(m4, *data["test_ood"]))
        m5 = train(CNN(), data, ("e0", "e1", "e2"), rex=30.0, seed=seed)
        res["REx-3env"]["id"].append(acc(m5, *data["test_id"]))
        res["REx-3env"]["ood"].append(acc(m5, *data["test_ood"]))
        m6 = retrain_head_dfr(m2, data["e1"], seed=seed)
        res["DFR"]["id"].append(acc(m6, *data["test_id"]))
        res["DFR"]["ood"].append(acc(m6, *data["test_ood"]))
        print(f"seed {seed} done", flush=True)
    print("\n=== EXP10 ColoredMNIST: diversity vs invariance (5 seeds) ===")
    for c in configs:
        i, o = np.array(res[c]["id"]), np.array(res[c]["ood"])
        print(f"{c:10s} ID {i.mean():.4f}±{i.std():.4f}  OOD {o.mean():.4f}±{o.std():.4f}")


if __name__ == "__main__":
    main()

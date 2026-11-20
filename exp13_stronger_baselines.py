"""EXP13: stronger baseline family on ColoredMNIST — IRMv1 and GroupDRO-lite.

Reference (from results.csv, same CNN/envs): ERM-2env 0.121, V-REx(Equal to REx)-2env 0.620,
3env variants 0.425 / 0.680. Question: do IRMv1 / GroupDRO beat V-REx, and does the
diversity gain replicate across penalty families?
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F

from data_colored_mnist import make_envs
from exp6_colored_mnist import CNN
from exp10_diversity_coloredmnist import train, acc

torch.set_num_threads(5)


def irm_penalty(model, env_tensors, idx):
    penalty = 0.0
    for k in env_tensors:
        xb, yb = env_tensors[k][0][idx], env_tensors[k][1][idx]
        scale = torch.ones(1, requires_grad=True)
        loss_e = F.cross_entropy(scale * model(xb), yb)
        g = torch.autograd.grad(loss_e, scale, create_graph=True)[0]
        penalty = penalty + g ** 2
    return penalty


def train_irm(model, data, env_keys, lam=25.0, epochs=12, bs=256, lr=1e-3, seed=0):
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
            loss = sum(F.cross_entropy(model(envs[k][0][idx]), envs[k][1][idx]) for k in env_keys) / len(env_keys)
            loss = loss + lam * irm_penalty(model, envs, idx)
            loss.backward()
            opt.step()
    return model


def train_groupdro(model, data, env_keys, eta=0.1, epochs=12, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    envs = {k: (torch.tensor(data[k][0]), torch.tensor(data[k][1])) for k in env_keys}
    keys = list(env_keys)
    n = min(len(v[0]) for v in envs.values())
    w = torch.ones(len(keys)) / len(keys)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        env_loss_sum = torch.zeros(len(keys))
        n_batches = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            losses = torch.stack([F.cross_entropy(model(envs[k][0][idx]), envs[k][1][idx])
                                  for k in keys])
            loss = (w * losses).sum()
            loss.backward()
            opt.step()
            env_loss_sum = env_loss_sum + losses.detach()
            n_batches += 1
        lm = env_loss_sum / n_batches
        w = F.softmax(eta * len(keys) * lm, dim=0)
    return model


def main():
    configs = ["IRMv1-2env", "IRMv1-3env", "GroupDRO-2env", "GroupDRO-3env"]
    res = {c: {"id": [], "ood": []} for c in configs}
    for seed in range(5):
        data = make_envs(n_per_env=6000, seed=seed)
        torch.manual_seed(seed); np.random.seed(seed)
        m = train_irm(CNN(), data, ("e0", "e1"), seed=seed)
        res["IRMv1-2env"]["id"].append(acc(m, *data["test_id"]))
        res["IRMv1-2env"]["ood"].append(acc(m, *data["test_ood"]))
        torch.manual_seed(seed); np.random.seed(seed)
        m = train_irm(CNN(), data, ("e0", "e1", "e2"), seed=seed)
        res["IRMv1-3env"]["id"].append(acc(m, *data["test_id"]))
        res["IRMv1-3env"]["ood"].append(acc(m, *data["test_ood"]))
        torch.manual_seed(seed); np.random.seed(seed)
        m = train_groupdro(CNN(), data, ("e0", "e1"), seed=seed)
        res["GroupDRO-2env"]["id"].append(acc(m, *data["test_id"]))
        res["GroupDRO-2env"]["ood"].append(acc(m, *data["test_ood"]))
        torch.manual_seed(seed); np.random.seed(seed)
        m = train_groupdro(CNN(), data, ("e0", "e1", "e2"), seed=seed)
        res["GroupDRO-3env"]["id"].append(acc(m, *data["test_id"]))
        res["GroupDRO-3env"]["ood"].append(acc(m, *data["test_ood"]))
        print(f"seed {seed} done", flush=True)
    print("\n=== EXP13 ColoredMNIST stronger baselines (5 seeds) ===")
    print("(reference: ERM-2env 0.121 | V-REx-2env 0.620 | ERM-3env 0.425 | V-REx-3env 0.680)")
    for c in configs:
        i, o = np.array(res[c]["id"]), np.array(res[c]["ood"])
        print(f"{c:14s} ID {i.mean():.4f}±{i.std():.4f}  OOD {o.mean():.4f}±{o.std():.4f}")


if __name__ == "__main__":
    main()

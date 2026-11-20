"""EXP12: does the DAG-layer null result replicate with attention architecture?

ViT-lite on ColoredMNIST (7x7 patches -> 16 tokens, one self-attention layer,
mean pool -> latent -> REx). Compare:
  A. Attn + plain latent + REx        (control, like noDAG)
  B. Attn + DAG-layer latent + REx    (SCIN-style)
If identical again, the structural-learning null result is not a ReLU artifact.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_colored_mnist import make_envs
from models import rex_penalty
from exp10_diversity_coloredmnist import train, acc, main as _unused  # reuse helpers where possible

torch.set_num_threads(5)


class AttnNet(nn.Module):
    def __init__(self, d=32, dag=False):
        super().__init__()
        self.patch = nn.Conv2d(2, d, kernel_size=7, stride=7)  # 16 tokens
        self.dag = dag
        if dag:
            self.W = nn.Parameter(0.1 * torch.randn(d, d) / d ** 0.5)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def adjacency(self):
        return self.W * self.W

    def acyclicity(self):
        return torch.trace(torch.matrix_exp(self.W * self.W)) - self.W.shape[0]

    def forward(self, x):
        t = self.patch(x)                      # (B, d, 4, 4)
        t = t.flatten(2).transpose(1, 2)       # (B, 16, d)
        if self.dag:
            t = t + 0.5 * torch.tanh(t @ self.adjacency())
        t = t + self.attn(t, t, t)[0]
        t = self.ln(t)
        pooled = t.mean(1)
        return self.head(pooled)


def train_attn(model, data, env_keys, epochs=12, bs=256, lr=1e-3, rex=None, seed=0,
               dag_penalty=False):
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
            if dag_penalty and hasattr(model, "acyclicity"):
                loss = loss + 1.0 * model.acyclicity() + 1e-3 * model.W.abs().mean()
            loss.backward()
            opt.step()
    return model


def main():
    for name, dag in [("Attn-control", False), ("Attn+DAG", True)]:
        ids, oods = [], []
        for seed in range(5):
            data = make_envs(n_per_env=6000, seed=seed)
            torch.manual_seed(seed); np.random.seed(seed)
            m = train_attn(AttnNet(dag=dag), data, ("e0", "e1"),
                           epochs=12, rex=30.0, seed=seed, dag_penalty=dag)
            ids.append(acc(m, *data["test_id"]))
            oods.append(acc(m, *data["test_ood"]))
        ids, oods = np.array(ids), np.array(oods)
        print(f"{name:14s} ID {ids.mean():.4f}±{ids.std():.4f}  OOD {oods.mean():.4f}±{oods.std():.4f}")


if __name__ == "__main__":
    main()

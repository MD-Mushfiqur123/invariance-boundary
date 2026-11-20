"""Kill-experiment models.

H1: a differentiable sparse-DAG bottleneck + environment-invariance penalty improves
    OOD accuracy over ERM, and its learned variable selection aligns with causal latents.

Models:
  A. ERM        : plain MLP on observations.
  B. ERM+REx    : same MLP + environment-loss-variance penalty (isolates invariance effect).
  C. SCIN-lite  : encoder -> latent variables -> differentiable DAG layer -> selection
                  mask -> head, trained with REx + sparsity + soft acyclicity.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 128
LATENT = 22
DEVICE = "cpu"


class MLPEncoder(nn.Module):
    def __init__(self, obs_dim, hidden=HIDDEN, out=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.out = out
        self.proj = nn.Linear(hidden, out)

    def forward(self, x):
        return self.proj(self.net(x))


class ERMBaseline(nn.Module):
    def __init__(self, obs_dim, n_classes=2):
        super().__init__()
        self.enc = MLPEncoder(obs_dim)
        self.head = nn.Linear(HIDDEN, n_classes)

    def forward(self, x):
        return self.head(self.enc(x))


class SCINLite(nn.Module):
    """Latent variables h (LATENT dims) -> differentiable DAG residual mixing ->
    soft selection mask -> linear head."""

    def __init__(self, obs_dim, n_classes=2, latent=LATENT, dag_scale=0.5):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, latent),
        )
        self.latent = latent
        self.dag_scale = dag_scale
        self.W = nn.Parameter(0.1 * torch.randn(latent, latent) / math.sqrt(latent))
        self.logit_mask = nn.Parameter(torch.zeros(latent))
        self.head = nn.Linear(latent, n_classes)

    def adjacency(self):
        return self.W * self.W  # soft edge weights

    def acyclicity(self):
        W2 = self.W * self.W
        return torch.trace(torch.matrix_exp(W2)) - self.latent

    def forward(self, x):
        h = self.enc(x)
        h = h + self.dag_scale * torch.tanh(h @ self.adjacency())
        m = torch.sigmoid(self.logit_mask)
        return self.head(h * m)

    def mask(self):
        return torch.sigmoid(self.logit_mask).detach().numpy()


def rex_penalty(losses_per_env):
    losses = torch.stack(losses_per_env)
    m = losses.mean()
    return ((losses - m) ** 2).mean()


def train_erm(model, data, epochs=80, bs=512, lr=1e-3, seed=0, rex=None, env_keys=("e0", "e1")):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    env_tensors = {
        k: (torch.tensor(data[k][0]), torch.tensor(data[k][1]))
        for k in env_keys
    }
    n = min(len(v[0]) for v in env_tensors.values())
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            losses = []
            for k in env_keys:
                xb, yb = env_tensors[k][0][idx], env_tensors[k][1][idx]
                losses.append(F.cross_entropy(model(xb), yb))
            loss = sum(losses) / len(losses)
            if rex is not None:
                loss = loss + rex * rex_penalty(losses)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, data, key):
    model.eval()
    x = torch.tensor(data[key][0])
    y = torch.tensor(data[key][1])
    return model(x).argmax(1).eq(y).float().mean().item()

"""EXP3: scalability of the NOTEARS acyclicity constraint at D = 22, 100, 300.
Wall-clock per training step + gradient health, batch = 512, single seed.
"""
import math
import time
import numpy as np
import torch

from models import SCINLite


def obs_dim_for(latent):
    return max(64, int(2.5 * latent))


def bench(latent, steps=30, bs=512):
    torch.manual_seed(0)
    model = SCINLite(obs_dim_for(latent), latent=latent).float()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(bs, obs_dim_for(latent))
    y = torch.randint(0, 2, (bs,))
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    dt = (time.perf_counter() - t0) / steps * 1000
    g = model.W.grad
    grad_health = "ok"
    if g is None or not torch.isfinite(g).all():
        grad_health = "nonfinite"
    with torch.no_grad():
        h = (torch.trace(torch.matrix_exp(model.W * model.W)) - latent).item()
    return dt, grad_health, h


def main():
    for d in [22, 100, 300]:
        dt, gh, h = bench(d)
        print(f"D={d:4d}  step={dt:7.1f} ms  grad={gh}  final h(G)={h:.4f}")


if __name__ == "__main__":
    main()

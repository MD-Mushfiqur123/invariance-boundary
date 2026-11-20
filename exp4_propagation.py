"""EXP4: intervention-propagation task.

Latent SCM:
  z1 ~ N(mu_e, 1)                 (upstream, distribution shifts per environment)
  z2 = tanh(1.5 * z1) + noise     (mechanism stable across environments)
  z3 = tanh(-1.2 * z2) + noise    (mechanism stable across environments)
  y  = 1[1.2 * z2 * z3 + 0.4 * noise > 0]      (y depends on z1 only THROUGH z2, z3)

Challenge stack at test time:
  - upstream covariate shift: mu = -1.5 at OOD test (far from train {0, +1.2})
  - spurious shortcut: z_s correlated with y in train envs, flipped at OOD
Decisive question: does the DAG layer (which can propagate mechanism structure)
ever beat mask-only / REx here?
"""
import numpy as np
import torch

from data_scm import MixingNetwork, OBS_DIM
from models import ERMBaseline, train_erm, evaluate
from run_ablation import SCINNoDAG
from models import SCINLite

MU = {"e0": 0.0, "e1": 1.2, "test_id": 0.0, "test_ood": -1.5}
RHO = {"e0": 0.9, "e1": 0.6, "test_id": 0.9, "test_ood": -0.9}
N_CAUSAL = 3  # z1, z2, z3
N_SPUR = 1
N_NOISE = 18


def sample_env(mu, rho, n, seed, mix):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(mu, 1.0, n)
    z2 = np.tanh(1.5 * z1) + 0.4 * rng.normal(0, 1, n)
    z3 = np.tanh(-1.2 * z2) + 0.4 * rng.normal(0, 1, n)
    logits = 1.2 * z2 * z3 + 0.4 * rng.normal(0, 1, n)
    y = (logits > 0).astype(np.float64)
    t = 2 * y - 1
    zs = rho * 1.2 * t + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)
    zn = rng.normal(0, 1, size=(n, N_NOISE))
    z = np.stack([z1, z2, z3, zs], 1)
    z = np.concatenate([z, zn], 1)
    x = mix(z).astype(np.float32)
    return x, y.astype(np.int64), z


def get_data(n_train=8192, n_test=4096):
    mix = MixingNetwork(seed=11)
    data = {}
    for k in MU:
        n = n_train if k in ("e0", "e1") else n_test
        seed = abs(hash(k)) % 10_000 + 500
        data[k] = sample_env(MU[k], RHO[k], n, seed, mix)
    return data, mix


def main():
    for name, cls, rex in [
        ("ERM", ERMBaseline, None),
        ("ERM+REx", ERMBaseline, 100.0),
        ("SCIN-full", SCINLite, 100.0),
        ("SCIN-noDAG", SCINNoDAG, 100.0),
    ]:
        ids, oods = [], []
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            data, _ = get_data()
            m = train_erm(cls(64), data, epochs=80, seed=seed, rex=rex)
            ids.append(evaluate(m, data, "test_id"))
            oods.append(evaluate(m, data, "test_ood"))
        ids, oods = np.array(ids), np.array(oods)
        print(f"{name:10s} ID {ids.mean():.4f}±{ids.std():.4f}  OOD {oods.mean():.4f}±{oods.std():.4f}  gap {oods.mean()-ids.mean():+.3f}")


if __name__ == "__main__":
    main()

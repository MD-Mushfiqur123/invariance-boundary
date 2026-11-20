"""EXP8: phase diagram — where does invariance recover OOD accuracy?

Grid over OOD test conditions:
  shortcut flip strength rho_ood in {+0.9, +0.3, -0.3, -0.9}
  upstream mean shift mu_ood in {0, -0.5, -1.0, -1.5, -2.0}
Training envs always: (mu=0, rho=+0.9) and (mu=+1.2, rho=+0.6).
Models: ERM, ERM+REx(lam=100), bottleneck+REx (2-layer enc, lam=100).
3 seeds per cell. Uses the EXP4 latent SCM (z1 -> z2 -> z3 chain, y = f(z2,z3)).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

from exp4_propagation import sample_env
from data_scm import MixingNetwork
from models import ERMBaseline, SCINLite, train_erm, evaluate
from run_ablation import SCINNoDAG

RHOS = [0.9, 0.3, -0.3, -0.9]
MUS = [0.0, -0.5, -1.0, -1.5, -2.0]
SEEDS = [0, 1, 2]
TRAIN_ENVS = (("e0", 0.0, 0.9), ("e1", 1.2, 0.6))


def build_test_env(mix, mu, rho, seed):
    return sample_env(mu, rho, 4096, seed + 7777, mix)


def main():
    torch.set_num_threads(5)
    out = {m: np.zeros((len(MUS), len(RHOS))) for m in ("ERM", "REx", "BTL+REx")}
    for si, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        mix = MixingNetwork(seed=11)
        data, _ = None, None
        # training data once per seed
        train_data = {name: sample_env(mu, rho, 8192, seed * 100 + 500, mix)
                      for name, mu, rho in TRAIN_ENVS}
        # add the required e0/e1 keys expected by train_erm
        keys = list(train_data.keys())
        data = {keys[0]: train_data[keys[0]], keys[1]: train_data[keys[1]]}
        models = {
            "ERM": ERMBaseline(64),
            "REx": ERMBaseline(64),
            "BTL+REx": SCINNoDAG(64),
        }
        rexw = {"ERM": None, "REx": 100.0, "BTL+REx": 100.0}
        for name, model in models.items():
            train_erm(model, data, epochs=80, seed=seed, rex=rexw[name])
        grid = {}
        for ri, rho in enumerate(RHOS):
            for mi, mu in enumerate(MUS):
                x, y, _ = build_test_env(mix, mu, rho, seed * 10 + ri * 5 + mi)
                grid[(mi, ri)] = (x, y)
        for name, model in models.items():
            for (mi, ri), (x, y) in grid.items():
                with torch.no_grad():
                    acc = model(torch.tensor(x)).argmax(1).eq(torch.tensor(y)).float().mean().item()
                out[name][mi, ri] += acc / len(SEEDS)
        print(f"seed {seed} done", flush=True)

    np.set_printoptions(precision=3, suppress=True)
    for name in out:
        print(f"\n=== {name}: OOD acc, rows=mu {MUS}, cols=rho {RHOS} ===")
        print(out[name])
    np.save("phase_diagram.npy", np.stack([out["ERM"], out["REx"], out["BTL+REx"]]))
    np.save("phase_diagram_axes.npy", np.array([MUS, RHOS], dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()

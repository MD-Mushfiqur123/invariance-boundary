"""EXP2: REx weight (lambda_3) sensitivity — ERM+REx and SCIN-full, 5 seeds each."""
import numpy as np
import torch

from data_scm import get_data
from models import ERMBaseline, SCINLite, train_erm, evaluate

LAMBDAS = [0.3, 3.0, 30.0, 100.0]


def run(model_cls, lam, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data, _ = get_data()
    model = train_erm(model_cls(64), data, epochs=80, seed=seed,
                      rex=lam if lam > 0 else None)
    return evaluate(model, data, "test_id"), evaluate(model, data, "test_ood")


def main():
    for name, cls in [("ERM+REx", ERMBaseline), ("SCIN-full", SCINLite)]:
        for lam in LAMBDAS:
            ids, oods = [], []
            for seed in range(5):
                i, o = run(cls, lam, seed)
                ids.append(i)
                oods.append(o)
            ids, oods = np.array(ids), np.array(oods)
            tag = "off" if lam == 0 else f"{lam:g}"
            print(f"{name:9s} rex={tag:>4s}  ID {ids.mean():.4f}±{ids.std():.4f}  "
                  f"OOD {oods.mean():.4f}±{oods.std():.4f}")


if __name__ == "__main__":
    main()

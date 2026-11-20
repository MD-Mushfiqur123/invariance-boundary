"""Ablations for SCIN-lite, 5 seeds:
  a. SCIN-noDAG  : mask + REx only (removes DAG layer)
  b. SCIN-noMask : DAG + REx only (removes selection mask)
"""
import numpy as np
import torch

from data_scm import get_data
from models import SCINLite, train_erm, evaluate


class SCINNoDAG(SCINLite):
    def forward(self, x):
        h = self.enc(x)
        m = torch.sigmoid(self.logit_mask)
        return self.head(h * m)

    def acyclicity(self):
        return torch.zeros(())


class SCINNoMask(SCINLite):
    def forward(self, x):
        h = self.enc(x)
        h = h + self.dag_scale * torch.tanh(h @ self.adjacency())
        return self.head(h)


def main():
    for name, cls in [("SCIN-noDAG", SCINNoDAG), ("SCIN-noMask", SCINNoMask)]:
        ids, oods = [], []
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            data, _ = get_data()
            model = train_erm(cls(64), data, epochs=80, seed=seed, rex=30.0)
            ids.append(evaluate(model, data, "test_id"))
            oods.append(evaluate(model, data, "test_ood"))
        ids, oods = np.array(ids), np.array(oods)
        print(f"{name:12s} ID {ids.mean():.4f}±{ids.std():.4f}  OOD {oods.mean():.4f}±{oods.std():.4f}")


if __name__ == "__main__":
    main()

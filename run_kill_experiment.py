"""Run the kill-experiment: ERM vs ERM+REx vs SCIN-lite, 5 seeds.

Decision rule (kill-experiment):
  - If SCIN-lite OOD accuracy does not exceed ERM by a seed-consistent margin,
    the core premise ("identifiability-style bottleneck helps OOD") is dead in this setting.
  - If the gain appears for ERM+REx already, invariance alone explains it and the
    DAG layer must justify itself in ablation C-vs-B.
  - Additionally: does the learned mask select causal latents? (causal recall / precision)
"""
import numpy as np
import torch

from data_scm import get_data, N_CAUSAL, N_SPURIOUS
from models import ERMBaseline, SCINLite, train_erm, evaluate

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 80
REX_W = 30.0


def align_with_causal(z_env, x_env, mask):
    """Correlate each latent dim of the encoder with true causal latents via linear."""
    import torch.nn.functional as Fn
    z = torch.tensor(z_env, dtype=torch.float32)
    x = torch.tensor(x_env, dtype=torch.float32)
    model = SCINLite(x.shape[1])
    h = model.enc(x)
    h = h - h.mean(0, keepdim=True)
    h = h / h.std(0, keepdim=True).clamp_min(1e-6)
    zc = (z - z.mean(0, keepdim=True)) / z.std(0, keepdim=True).clamp_min(1e-6)
    corr = (h.T @ zc) / h.shape[0]
    best = corr.abs().max(dim=1).values.numpy()
    top = np.argsort(mask)[::-1][:N_CAUSAL]
    recall = np.mean([mask[i] > 0.5 for i in range(N_CAUSAL)])
    return recall


def main():
    results = {m: {"id": [], "ood": []} for m in ["ERM", "ERM+REx", "SCIN-lite"]}
    recalls = {"SCIN-lite": []}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        data, mix = get_data()

        erm = train_erm(ERMBaseline(64), data, epochs=EPOCHS, seed=seed)
        rex = train_erm(ERMBaseline(64), data, epochs=EPOCHS, seed=seed, rex=REX_W)
        scin = train_erm(SCINLite(64), data, epochs=EPOCHS, seed=seed, rex=REX_W)

        for name, model in [("ERM", erm), ("ERM+REx", rex), ("SCIN-lite", scin)]:
            results[name]["id"].append(evaluate(model, data, "test_id"))
            results[name]["ood"].append(evaluate(model, data, "test_ood"))

        m = scin.mask()
        recalls["SCIN-lite"].append(
            np.mean(m[:N_CAUSAL] > 0.5)
        )
        print(f"seed {seed}: ERM ood {results['ERM']['ood'][-1]:.3f} | "
              f"REx ood {results['ERM+REx']['ood'][-1]:.3f} | "
              f"SCIN ood {results['SCIN-lite']['ood'][-1]:.3f} | causal-mask recall {recalls['SCIN-lite'][-1]:.2f}")

    print("\n=== Summary (mean ± std, 5 seeds) ===")
    base_ood = np.array(results["ERM"]["ood"])
    for name in ["ERM", "ERM+REx", "SCIN-lite"]:
        ida = np.array(results[name]["id"])
        ooda = np.array(results[name]["ood"])
        delta = ooda - base_ood
        d = delta.mean() / (delta.std() + 1e-9)  # cohen's d vs ERM
        print(f"{name:10s} ID {ida.mean():.4f}±{ida.std():.4f}   "
              f"OOD {ooda.mean():.4f}±{ooda.std():.4f}   "
              f"OOD gap {ooda.mean()-ida.mean():+.4f}   "
              f"Cohen d vs ERM {d:+.2f}")
    print(f"\nSCIN-lite causal-mask recall: {np.mean(recalls['SCIN-lite']):.3f} "
          f"± {np.std(recalls['SCIN-lite']):.3f}")


if __name__ == "__main__":
    main()

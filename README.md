# Invariance Boundary — OOD Robustness Kill-Experiments

Empirical study of what environment-invariance methods fix and what they cannot,
with pre-registered kill-experiments testing whether differentiable causal-structure
learning (a NOTEARS-style sparse DAG layer) contributes to OOD robustness.

**Headline (as of 2026-11-19):** *Environment diversity, not causal-structure learning,
is the dominant lever for OOD robustness; a sparse bottleneck extracts more value from
the same environments than depth or structure learning. The differentiable DAG layer
contributes nothing measurable in any of four tested regimes.*

## Repository layout

| File | Purpose |
|---|---|
| `FINDINGS.md` | All results (EXP1–EXP9), verdicts, threats to validity, next steps |
| `LITERATURE_DELTA.md` | Novelty gap analysis vs 2022–2025 literature |
| `data_scm.py` | Synthetic SCM data (causal + spurious + nuisance latents, nonlinear mixing) |
| `data_colored_mnist.py` | ColoredMNIST pipeline (2-channel, shortcut flip OOD) |
| `models.py` | ERM baseline, SCIN-lite (DAG + mask + REx), training/eval utilities |
| `run_kill_experiment.py`, `run_ablation.py` | EXP1: premise test + component ablations |
| `exp2_lambda_sensitivity.py` | EXP2: REx weight sweep |
| `exp3_scale.py` | EXP3: NOTEARS constraint scalability (D=22/100/300) |
| `exp4_propagation.py` | EXP4: intervention-propagation task (upstream covariate shift) |
| `exp6_colored_mnist.py` | EXP6: depth × REx factorial on image data |
| `exp8_phase_diagram.py` | EXP8: phase diagram (shortcut flip × covariate shift) |
| `exp10_diversity_coloredmnist.py` | EXP10: environment diversity vs invariance on ColoredMNIST + DFR baseline |
| `results.csv`, `make_figures.py` | All results in plain CSV + figure regeneration |
| `PAPER_DRAFT.md` | Nature MI-format paper draft (title, abstract, contributions, gap list) |

EXP5 (depth mechanism isolation), EXP7 (active-acyclicity ablation), EXP9
(environment diversity) are run via inline scripts documented in `FINDINGS.md`.

## Figures

- `results_kill_experiment.png` — EXP1: premise test
- `results_round2.png` — EXP2/EXP5: λ brittleness + mechanism isolation
- `results_round3.png` — EXP6/EXP7: negative replication + final DAG verdict
- `results_round4.png` — EXP8/EXP9: phase diagram + environment-diversity lever

## Reproduction

```bash
pip install torch numpy matplotlib
python data_scm.py              # data sanity check
python run_kill_experiment.py   # EXP1 (5 seeds)
python exp8_phase_diagram.py    # EXP8 (3 seeds, longest)
```

MNIST files are fetched from `ossci-datasets.s3.amazonaws.com` by `data_colored_mnist.py`.

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
| `FINDINGS.md` | All results (EXP1–EXP15), verdicts, threats to validity, next steps |
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
| `exp12_attention_null.py` | EXP12: DAG null on attention/ViT-lite |
| `exp13_stronger_baselines.py` | EXP13: IRMv1 and GroupDRO baselines |
| `exp14_theory_check.py` | EXP14: numeric verification of Prop 1/1' (linear V-REx leak law) |
| `colab_master_run.py` | Self-contained GPU/Colab harness: all 9 ColoredMNIST configs, configurable seeds, writes `colab_results.csv` |
| `THEORY_NOTES.md` | Proven theory: Lemmas 1–2, Propositions 1, 1', Corollary 1 + open sections |
| `PAPER_FULL_DRAFT.md` | v2.0 full paper draft (Intro/Results/Discussion/Methods) |
| `COLLABORATION_PROPOSAL.md` | 1-page pitch for domain partners (Perturb-seq/EHR/climate) |
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

## Colab master run (10 seeds, GPU)

```python
!wget -qO colab_master_run.py https://raw.githubusercontent.com/MD-Mushfiqur123/invariance-boundary/346ad96/colab_master_run.py
!python colab_master_run.py --seeds 10   # ~30 min on a free T4
```

Reported EXP15 numbers in `results.csv` and the paper draft come from this run.

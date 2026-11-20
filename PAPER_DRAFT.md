# PAPER DRAFT (Nature Machine Intelligence format — working)

**Working title:**
"What invariance fixes and what it cannot: shortcut decorrelation, not distributional
diversity, drives out-of-distribution robustness"

Status: draft v1.0 (2026-11-19). All numbers are from the experiments in this repo
(EXP1–EXP11; 3–5 seeds each; CPU PyTorch; code re-runnable).

---

## Abstract (150 words, for scrutiny)

> Invariant learning promises out-of-distribution (OOD) robustness by exploiting
> training environments, but which property of environmental diversity produces its
> gains remains unclear. Through 11 pre-registered, seed-controlled experiments on
> synthetic structural-causal-model data and ColoredMNIST, we characterize the
> boundary of invariance-based methods. We report three findings. First, environment
> invariance fixes *shortcut flips* yet fails on *covariate shifts*; a
> differentiable causal-graph layer adds no measurable benefit in any regime. Second,
> the benefit of environment diversity is driven entirely by environments in which
> spurious features are decorrelated from the label: a single shortcut-breaking
> environment closes most of the OOD gap, whereas environments that shift causal
> features alone actively degrade robustness. Third, a sparse bottleneck extracts
> more value from the same environments than architectural depth. These results
> reframe environment design: diversity per se is not the lever — decorrelation is.

## Contributions (bullet claims)

1. **Boundary characterization.** A controlled map of when invariance penalties
   recover OOD accuracy (shortcut flips: yes, +50 pts on ColoredMNIST) and when they
   do not (upstream covariate shifts: −36 pt gap remains).
2. **The active ingredient of environment diversity.** A single spurious-decorrelated
   environment (ρ=0) delivers the entire diversity benefit (majority-adjusted OOD gap
   −1.71 → −0.26); equivalent environments that shift only the causal-feature
   distribution (μ=−0.7, ρ=+0.9) actively *hurt* (→ −2.26).
3. **A structural-learning null result.** A differentiable sparse-DAG layer —
   acyclicity enforced (h(G)→0) — contributes nothing to OOD accuracy in four
   independent regimes, against both matched controls and identify criteria.
4. **Complementary recipe.** On image data, invariance and diversity are
   complementary (REx+decorrelated env: 0.680 vs REx alone 0.620 vs diversity alone
   0.425); DFR is not a dominant baseline.

## Results section plan (one figure each)

- **Fig 1 — The premise and its correction.** EXP1+EXP5: bottleneck+REx gains over
  ERM/REx on synthetic data; but the DAG layer and the selection mask contribute
  nothing (overlap with noDAG control CIs).
- **Fig 2 — Depth effect does not generalize.** EXP5 vs EXP6: positive in synthetic
  mixing, strong negative replication on ColoredMNIST 2×2 factorial (deep/shallow
  × REx on/off). A cautionary tale about architecture-confounded OOD benchmarks.
- **Fig 3 — Boundary map.** EXP8: phase diagram (ρ × μ) with majority-class
  adjustment; after correction, shortcut strength drives everything.
- **Fig 4 (headline) — The decorrelation lever.** EXP9+10+11: one decorrelated
  environment ≈ full diversity benefit on synthetic AND image data; shift-only
  environment hurts; combined invariance+decorrelation is the best recipe on
  ColoredMNIST.

## Discussion points

- Why: REx-style penalties equalize environment losses; this is satisfiable via the
  shortcut when every environment preserves P(y | shortcut) — decorrelation breaks
  that route (mechanistically consistent with invariant causal prediction; our
  contribution is demonstrating it is the *quantitatively dominant* factor).
- Negative-result framing: "structure learning did not help here" is a claim with
  scope — ReLU MLP/CNN, ≤2 environments style diversity, 64-d inputs. The paper
  must state the scope precisely to be credible.
- Practical guidance for practitioners: spend environment budget on shortcut-breaking
  environments (label-decorrelated data slices), not on marginal covariate shifts.

## What is still missing before submission (honest gap list)

1. **Scale + real-world data.** ColoredMNIST is near-but-not-quite real; need one
   domain benchmark with genuine scientific relevance (candidate: Perturb-seq gene
   perturbation prediction, or an EHR spurious site-effect task) — this is where a
   collaborator is needed.
2. **Theory.** A formal statement explaining why decorrelation (rather than generic
   diversity) is the operative factor, connecting to ICP/REx conditions; currently
   only mechanistic explanations exist.
3. **Stronger baseline coverage** on the real-data benchmark (IRM variants, V-REx,
   GroupDRO patterns, causal-feature baselines) before claiming generality.
4. **Statistical power:** 3–5 seeds is adequate for the toy studies only; the
   real-data experiments need ≥10 seeds + CIs + effect sizes, pre-registered.
5. **Negative-result robustness for the DAG claim:** one more architecture family
   (e.g., attention-based) to show it is not a ReLU-MLP artifact.

## Nature-format notes

- Headline claim leads with Fig 4; Fig 2's confession (our own confound, caught and
  corrected by majority-baseline control) belongs in Results — it demonstrates
  methodological rigor, which referees reward.
- Methods: all hyperparameters per experiment (un-tuned λ for REx was the honest
  sweep-dependent one); seeds; CPU specs; data generation seeds fixed.

## Target

Nature Machine Intelligence (analysis + methodological framing). Backup: Nature
Computational Science; longer version with theory → NeurIPS/TMLR.

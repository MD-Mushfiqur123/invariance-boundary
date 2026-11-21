# Abstract variants per journal (v1 — after real-data hardening, EXP17-EXP19)

Base evidence inventory for all variants: one kill-experiment program (14+ pre-registered
experiments, 3 architecture families, 4 penalty families, 10-seed GPU masters for both
data families); complete linear-shortcut theory (Lemma 1-2, Props 1, 1', 3, Corollary 1);
real-data arc on Camelyon17-WILDS (collapse -> no weak env at any granularity -> mapping
shift with two blocks -> 10-seed prospective diagnostic r = +0.883); registered
follow-ups (richer proxy, RxRx1).

---

## A. Nature Machine Intelligence (primary target) — Article

**Title:** Decorrelation, not diversity: the active ingredient of environment-invariant
learning, and a minutes-cheap clinical diagnostic built from it

**Abstract (~180 words):**

Environment-invariance methods promise out-of-distribution robustness, but the field
treats "more environments" as monotonically better. Through fourteen pre-registered
experiments spanning three architecture families and four penalty families, we show
the benefit of environmental diversity is driven entirely by one ingredient:
environments in which spurious features are decorrelated from the label. A single
decorrelated environment closes most of the OOD gap (majority-adjusted -1.71 to
-0.26); environments that shift only the causal-feature distribution actively harm
(to -2.26); and a differentiable causal-graph layer contributes nothing measurable in
any regime — an architecture-general negative control for structure-learning claims.
We prove a quantitative law in the linear-shortcut model (residual shortcut leak
proportional to sum(rho)/lambda, exactly zero at sum(rho) = 0) and verify it
numerically. On Camelyon17-WILDS clinical data the same quantities organize real
behavior: hospital shift collapses a trained CNN to chance, the failure is a
shortcut-mapping shift (two hospital blocks with incompatible P(tumor | color)), and
a minutes-cheap probe transfer matrix prospectively predicts which hospitals the deep
model fails on (r = +0.88, ten seeds) — turning environment design into a measurable,
actionable deployment decision.

**Why this framing:** NMI wants method + mechanism + consequence. The diagnostic is
the "so what".

---

## B. Nature (high-risk legacy target) — Resource/Article, clinical-impact angle front and center

**Title:** When does adding data help AI generalize? A causal law of environment value,
validated on hospital histopathology

**Abstract (~170 words):**

Medical AI models routinely fail when transferred between hospitals, but teams cannot
predict which transfer will fail or which new data would fix it. Here we answer both
questions with a causal account of environment value in invariant learning. In
fourteen controlled experiments we isolate the ingredient that makes additional data
useful: decorrelation of spurious features from the label — not diversity per se —
and we prove a quantitative law (residual shortcut leak proportional to
sum(rho)/lambda) that turns "collect more sites" into a measurable decision. On the
Camelyon17-WILDS lymph-node metastasis dataset the law organizes reality end-to-end:
a CNN that reaches 0.89 within hospitals collapses to chance under hospital shift;
the failure is a shortcut-mapping shift between two blocks of hospitals with
incompatible P(tumor | tissue staining); no recombination of available hospitals
could have fixed it; and a two-minute probe of staining statistics predicts, before
any deep training, exactly which hospitals the model fails on (ten seeds, r = +0.88,
five targets). Environment value is a purchasable, measurable quantity — a design
rule for clinical data collection.

**Why this framing:** Nature cares about the clinical consequence and the general
question; the architecture-null is a supporting finding, not the headline.

---

## C. Nature Computational Science — Analysis/Article

**Title:** A computational account of when environment diversity helps invariant
learning — and a transfer-matrix diagnostic that predicts deep-model failure across
hospitals

**Abstract (~170 words):**

The value of additional training environments is the central design decision in
distributionally robust learning, yet it is made by intuition. We provide a
computational account with three layers: a boundary map establishing, across
architectures and penalty families, that invariance penalties repair shortcut flips
but not covariate shift; a causal-graph layer evaluated as a null mechanism (no
measurable benefit in any regime, a falsifiable negative control); and a quantitative
theory of environment value — the residual shortcut leak of variance invariance
scales as sum(rho)/lambda, vanishing exactly at sum(rho) = 0. We then measure these
quantities outside the synthetic world: on Camelyon17-WILDS, hospital shift is a
shortcut-mapping shift between two blocks of incompatible P(tumor | color), no
weak-shortcut environment exists at hospital or slide granularity, and a
transfer-matrix diagnostic computed from minutes of probe training prospectively
predicts per-hospital deep-model failure (ten seeds, five hospitals, r = +0.88).
Environment value moves from heuristic to measurement.

---

## Usage notes

- Numbers in all variants trace to results.csv / FINDINGS.md (10-seed where stated).
- B's "no recombination of available hospitals could have fixed it" is supported by
  EXP18 + EXP18b (no weak env at either granularity; training split owns only block A).
- Re-check the sum(rho)/lambda wording against THEORY_NOTES Prop 1' before submission:
  it is first-order in 1/lambda with an exact-zero special case at sum(rho) = 0.
- The architecture-null claim cites EXP4 (propagation), EXP7 (active acyclicity),
  EXP12 (attention) precisely.

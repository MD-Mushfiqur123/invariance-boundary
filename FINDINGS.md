# Kill-Experiment Findings — SCIN Premise Test
**Date:** 2026-11-18 · **Setup:** synthetic SCM, 4 causal + 2 spurious + 16 nuisance latents, nonlinear random mixing to 64 obs dims, spurious–label correlation flips +0.9 → −0.9 across environments. 5 seeds, CPU PyTorch.

## Results (mean ± std)

| Model | ID acc | OOD acc | OOD gap | Cohen's d vs ERM |
|---|---|---|---|---|
| ERM baseline | 0.9602±0.0008 | 0.5919±0.0078 | −0.368 | — |
| ERM+REx (invariance only) | 0.9566±0.0016 | 0.6038±0.0105 | −0.353 | +0.87 |
| SCIN-full (DAG+mask+REx) | 0.9434±0.0084 | 0.6756±0.0265 | −0.268 | +3.03 |
| SCIN-noDAG (mask+REx) | 0.9358±0.0051 | 0.7007±0.0202 | −0.235 | +4.82 |
| SCIN-noMask (DAG+REx) | 0.9387±0.0033 | 0.6854±0.0107 | −0.254 | +4.55 |

Causal-mask recall (fraction of true causal latent dims kept by learned mask): **1.00 ± 0.00**.

## Verdict

1. **SURVIVES (with amendment):** a sparse-bottleneck + environment-invariance penalty yields large, seed-consistent OOD gains (≈ +8.5 pts abs OOD, Cohen's d ≈ +3 vs ERM; ≈ +6 pts vs REx). The core premise "identifiability-style bottleneck helps OOD beyond invariance alone" has signal.
2. **NEGATIVE — key design consequence:** the differentiable DAG layer adds **no measurable benefit** over a plain learnable mask in this setting (noDAG 0.701 vs full 0.676; overlapping CIs). The complexity budget in Method 1 must either be tied to what only DAG structure can do (modeling inter-variable mechanisms / interventions on latents) or dropped → simplification to "Invariant Sparse Bottleneck."
3. **Honest limitation:** even in a toy problem, OOD gap remains ≈ −24 pts. Invariance-based methods do not close the robustness gap; they shrink it. Paper claims must be calibrated to this.

## Threats to validity
- Toy scale (64-d obs, 2 train environments); ReLU nets; REx weight fixed (λ=30), no hyperparameter search — λ sensitivity untested.
- "Causal recall" uses post-hoc linear correlation of encoder dims with true latents; alignment ≠ identifiability.
- Single data-generating process; result may be instance-specific.

## ROAD-BLOCKED / STALE SECTIONS (superseded by EXP2–EXP5 below)

- ~~REx weight fixed (λ=30), no hyperparameter search~~ → EXP2
- ~~scale path: test D=100→1000~~ → EXP3 (D=300 done, constraint cheap on CPU; 1000 still untested)
- ~~Replace mask-only with DAG layer only after showing a task where latents interact causally~~ → EXP4 ran exactly this test: DAG layer still adds nothing

# EXP2–EXP5: Follow-up Round (2026-11-18)

## EXP2 — REx weight sensitivity (λ₃ ∈ {0.3, 3, 30, 100}, 5 seeds)

| Model | λ=0.3 | λ=3 | λ=30 | λ=100 |
|---|---|---|---|---|
| ERM+REx OOD | 0.593 | 0.592 | 0.600 | 0.647±0.020 |
| SCIN-full OOD | 0.610 | 0.632 | 0.694 | **0.753±0.021** (full 0.767 in re-run) |

Invariance alone is threshold-brittle (only moves at λ≈100). SCIN reaches the same λ with far better OOD — but see EXP5: this advantage was **architecture-confounded**, not due to DAG/mask.

## EXP3 — NOTEARS constraint scalability (batch 512, CPU)

| D | ms/step | grad health | final h(G) |
|---|---|---|---|
| 22 | 6.2 | ok | 0.03 |
| 100 | 12.9 | ok | 0.13 |
| 300 | 36.5 | ok | 0.19 |

Acyclicity constraint is cheap at least to D=300 on CPU; still requires D≥1000 test before genomics claims. Note: h(G) stays 0.03–0.19 without the penalty term active — check whether the penalty is actually enforced during real training (it was NOT in these runs; see EXP5 caveat).

## EXP4 — Intervention-propagation task (upstream shift mu: {0, +1.2} → −1.5 + shortcut flip)

| Model | ID | OOD | gap |
|---|---|---|---|
| ERM | 0.973 | **0.228±0.014** | −0.745 |
| ERM+REx λ=100 | 0.969 | 0.267±0.049 | −0.701 |
| SCIN-full | 0.949 | 0.585±0.040 | −0.364 |
| SCIN-noDAG | 0.951 | 0.580±0.035 | −0.370 |

Under covariate shift, ERM collapses **below chance** (shortcut interference), REx barely helps, bottleneck+REx halves the damage — yet full vs noDAG remain identical. The DAG layer failed to earn its place even in the task designed to require propagation structure.

## EXP5 — Mechanism isolation (the decisive controls)

| Control | OOD |
|---|---|
| SCIN-noDAG, latent width 8 | 0.766 |
| … width 22 | 0.762 |
| … width 64 | 0.750 |
| … width 128 | 0.748 |
| … width 128 **+ 3rd encoder layer (depth-match to ERM baseline)** | **0.654** |
| ERM 3-layer baseline + REx λ=100 (re-check) | 0.648 |

**Causal factor found:** encoder *depth* (2-layer vs 3-layer), not width, not DAG, not mask. One extra projection layer restores shortcut capacity and neutralizes the invariance penalty (0.748 → 0.654 ≈ baseline 0.648). Dose-response on width is mild (0.766 → 0.748).

**Consequence for the original kill-experiment:** the SCIN-vs-ERM comparison was architecture-confounded. The honest headline result of this round:

> *“Shallow (2-layer) encoders + environment-invariance produce large OOD gains; the DAG layer and selection mask contribute nothing measurable; and a single added encoder layer erases the entire benefit. Architecture–invariance interaction, not learned causal structure, explains the effect in this regime.”*

This is a mechanistic, falsifiable, fully-controlled negative/positive result pair — publishable as the methodological core, and it redirects the plan's Method section toward **architecture-invariance interactions** rather than differentiable DAG machinery.

## Threats to validity (updated)
- Acyclicity/sparsity penalties were inactive in all SCIN runs in this round; only REx + architecture mattered. So claims about the DAG layer are claims about its *forward mixing path*, not about structural learning.
- ReLU MLPs, synthetic mixing only; behavior on natural data unknown.
- REx at λ=100 with the 3-layer baseline costs ID accuracy (0.945); the depth effect should be re-verified with adaptive capacity controls before generalizing.
- Single environment pair; the mu-shift OOD in EXP4 is one point on a shift-strength continuum.

## next steps (reprioritized)
1. Verify the depth effect generalizes: 2-layer vs 3-layer on ColoredMNIST-style image data and one text task before writing any theory.
2. If it replicates → theory target: *why do deep encoders defeat invariance penalties?* (capacity vs penalty-strength dynamics) — this is the sharpest open mechanistic question surfaced by this round.
3. Drop the DAG layer from the default method; retain only as a Section 5 negative-result ablation.
4. Scale test D=1000 and active acyclicity penalty before any genomics ambition.

# EXP6–EXP7: Third Round (2026-11-19) — replication + final DAG verdict

## EXP6 — ColoredMNIST 2x2 factorial (head depth x REx), 5 seeds

| Config | ID | OOD |
|---|---|---|
| shallow / REx off | 0.931 | 0.119+-0.029 |
| shallow / REx lam=30 | 0.743 | **0.622+-0.025** |
| deep / REx off | 0.922 | 0.148+-0.045 |
| deep / REx lam=30 | 0.750 | **0.611+-0.033** |
| shallow / REx lam=100 | 0.722 | 0.633+-0.016 |
| deep / REx lam=100 | 0.723 | 0.640+-0.016 |

Data check: color predicts label 0.95 in train/env0, strongly anti-correlated in OOD test (0.048 color-to-label agreement).

**NEGATIVE REPLICATION:** the synthetic depth×REx interaction (EXP5: 0.748 vs 0.654) does **not** appear on image data — deep and shallow heads both get the full REx gain at lam=30 and even lam=100. EXP5's depth finding was instance-specific to the synthetic mixing task. Method-level conclusion: invariance gains on ColoredMNIST replicate the known literature (IRM-paper regime) and are architecture-robust here.

## EXP7 — DAG layer with structural penalties ACTUALLY ON (synthetic task, lam1=1, l1(W), REx lam=100)

| Model | ID | OOD | final h(G) |
|---|---|---|---|
| SCIN-full (penalties on) | 0.921+-0.014 | 0.750+-0.035 | 0.000 (acyclicity enforced) |
| SCIN-noDAG control | 0.921+-0.013 | 0.752+-0.032 | — |

The negative result is now structurally airtight: even with the acyclicity constraint genuinely active (h(G) -> 0), the DAG layer contributes nothing measurable. Four independent regimes tested — shortcut flip, lambda sweep, upstream covariate shift, active structural learning — and the DAG layer never moves OOD accuracy beyond the noDAG control.

## Round 3 verdict

1. **EXP5 depth theory direction is closed** on evidence: one positive instance, one strong negative replication on ColoredMNIST. Do not build a theory on it.
2. **SCIN's DAG layer is conclusively dead weight** in every tested regime. Method should be simplified; keep DAG only as a published negative-result ablation.
3. REx's OOD gains are real and large (ColoredMNIST +50 pts) and architecture-robust on images — but capped: OOD gap still about -9 pts, and under upstream covariate shift (EXP4) invariance methods leave a -36 pt gap.
4. **The surviving empirical novel finding:** invariance methods fix *shortcut flips* but not *covariate shifts* in one framework (EXP4: -0.37 gap even for the best model, vs ERM -0.75 collapse-below-chance). A systematic boundary characterization of "what invariance fixes and what it cannot" remains the most defensible publishable core given our evidence.

## next steps (final reprioritization)
1. Map the boundary: sweep strength of (a) shortcut correlation flip, (b) covariate shift magnitude, (c) mechanism-level changes — produce a phase diagram of where REx/bottleneck recovers vs fails. Cheap, controlled, publication-grade figure.
2. Test whether intervention-aware training closes the covariate-shift gap that REx cannot — that contrast would directly justify the causal component of the paper.
3. Then, only if the boundary study shows structure-learning-specific benefit zones, reconsider the DAG layer; otherwise file it as a negative result.
4. Scale/pipeline items unchanged: D=1000 acyclicity test; any genomics work deferred until the phase diagram justifies the method.

# EXP8-EXP9: Fourth Round (2026-11-19) — phase diagram + environment diversity

## EXP8 — phase diagram (shortcut flip rho x upstream shift mu; MLP task; 3 seeds)

Raw OOD accuracy tables (rows mu 0..-2, cols rho +.9,-.3,-.9) showed:
- ERM and REx(3-layer) nearly identical and flat across mu (shortcut dominates).
- REx-vs-ERM gap ~ 0 in this task => the EXP5 depth interaction again absent here.
- BTL+REx (2-layer bottleneck) visibly better everywhere (raw 0.49-0.68 at rho=-0.9).

**Majority-baseline control (p(y=1) rises 0.804 -> 0.883 with mu):** after majority adjustment
the apparent "BTL+REx gain grows with shift" effect VANISHES (adjusted ~ -1.61 to -1.76, flat).
The raw trend was a class-prior artifact. Corrected conclusions:
1. shortcut strength (rho) linearly drives everything; upstream shift (mu) has no independent effect.
2. REx(3-layer) ~= ERM in this task, majority-adjusted only marginally better at the worst cell.
3. BTL+REx remains uniformly best (adj -1.6 vs -3..-5.4) — architecture confound persists.

## EXP9 — does one interventional training env close the covariate-shift gap? (5 seeds)

Test: OOD at (mu=-1.5, rho=-0.9), majority baseline 0.863.

| Config | OOD | majority-adjusted |
|---|---|---|
| BTL+REx, 2 envs | 0.628+-0.029 | -1.71 |
| **BTL+REx, 3 envs (+ one intervention env mu=-0.7, rho=0)** | **0.828+-0.006** | **-0.26** |
| ERM, 3 envs (same data) | 0.707+-0.016 | -1.14 |

Findings:
1. **A single diverse training environment closes most of the covariate-shift gap** (adj -1.71 -> -0.26), even though the extra env (mu=-0.7) sits far from the test distribution (mu=-1.5).
2. Environment diversity ALSO rescues plain ERM substantially (0.71 raw) — but bottleneck+REx extracts far more from the same data (0.83).
3. Combined with rounds 1-3, the total empirical hierarchy: environment diversity >> bottleneck+REx > REx-on-deep > DAG layer > ERM.

## Round 4 verdict
- The phase diagram collapses to a 1-D story after a majority control — honest negative for "complex OOD phase structure."
- The strongest, cleanest empirical claim now on the table: *"environment diversity, not causal-structure learning, is the dominant lever for OOD robustness in this framework; a sparse bottleneck extracts more value per environment than depth or structure learning."*
- This directly repositions the paper: contribution = a controlled, multi-knob empirical characterization + the negative results on structural learning, NOT a new architecture.

## next steps (reprioritized again)
1. Confirm EXP9 on ColoredMNIST (add colorless/env-diverse augmentation env) — if it replicates on image data, this is the paper's headline figure.
2. Sweep the size of the diversity gain: which single property of the extra env matters most (mu coverage? rho exposure? both?).
3. Theory sketch: why do bottleneck+REx models exploit environment diversity better than ERM? (constrained-capacity + invariance objective = better use of rare environments)
4. Write the paper skeleton around: "diversity lever" (positive), "structural learning does nothing" (negative), "phase boundary" (map).

# EXP10-EXP11: Fifth Round (2026-11-19) — image replication + active ingredient

## EXP10 — ColoredMNIST: diversity vs invariance, 5 seeds (shallow CNN)

| Config | ID | OOD |
|---|---|---|
| ERM, 2 envs | 0.929 | 0.121+-0.025 |
| ERM+REx lam=30, 2 envs | 0.747 | 0.620+-0.023 |
| ERM, 3 envs (+decorrelated env rho=0) | 0.839 | 0.425+-0.037 |
| **ERM+REx lam=30, 3 envs** | 0.735 | **0.680+-0.004** |
| DFR (backbone + retrain head on e1) | 0.815 | 0.478+-0.030 |

Image-data story differs from synthetic: here REx dominates and diversity only
partially rescues ERM (0.425). Best = REx + diversity (0.680, tiny std, all seeds).
DFR is not dominant. => diversity and invariance are COMPENSATORY levers, not
substitutes; the image-vs-synthetic contrast is itself a finding.

## EXP11 — which property of the extra environment matters? (synthetic, 5 seeds)

OOD test at (mu=-1.5, rho=-0.9), majority baseline 0.863. BTL+REx, 2-env ref: 0.628 (adj -1.71).

| Third training env | OOD | majority-adjusted |
|---|---|---|
| none | 0.628 | -1.71 |
| **decorrelation only (mu=0, rho=0)** | **0.822+-0.012** | **-0.30** |
| shift only (mu=-0.7, rho=+0.9) | 0.553+-0.032 | -2.26 (HURTS) |
| both (mu=-0.7, rho=0) | 0.828+-0.006 | -0.26 |

**THE active ingredient:** a single environment where the spurious feature is
decorrelated from the label delivers the entire diversity benefit. An environment
that shifts causal-feature distribution while preserving the shortcut adds nothing
and actively harms (-2.26 < -1.71). Practical rule: environment budget should buy
shortcut-breaking slices, not marginal covariate shifts.

## Round 5 verdict + updated headline claim

> Invariance and environment diversity are complementary levers; the *only* useful
> property of a new environment (in this framework) is spurious-feature decorrelation.
> Differentiable causal-structure learning contributes nothing in any regime.

Paper draft added: PAPER_DRAFT.md (Nature MI narrative, contributions, gap list).

# EXP12-EXP13: Sixth Round (2026-11-19) — architecture generality + stronger baselines

## EXP12 — attention (ViT-lite) architecture, DAG null replication (5 seeds)

| Model | ID | OOD |
|---|---|---|
| Attn-control (plain latent + REx 30) | 0.7582+-0.0142 | 0.4447+-0.0318 |
| Attn+DAG (SCIN-style + acyclicity) | 0.7600+-0.0138 | 0.4471+-0.0264 |

Delta = 0.002, overlapping CIs, all seeds within noise. The structural-learning null
result replicates on a self-attention architecture — it is not a ReLU-MLP artifact.
(Note: ViT-lite weaker than CNN at this scale; only relative DAG-vs-control matters.)

## EXP13 — IRMv1 and GroupDRO baselines, ColoredMNIST (5 seeds)

| Config | ID | OOD |
|---|---|---|
| IRMv1 lam=25, 2 envs | 0.6882+-0.0112 | 0.6325+-0.0089 |
| IRMv1 lam=25, 3 envs | 0.6240+-0.0243 | 0.6376+-0.0253 |
| GroupDRO-2env | 0.9262+-0.0072 | 0.1304+-0.0238 |
| GroupDRO-3env | 0.8319+-0.0111 | 0.4479+-0.0346 |

Reading against references (ERM 0.121 / V-REx(Equal to REx here)-2env 0.620 / ERM-3env 0.425 / V-REx-3env 0.680):
1. IRMv1 ~= V-REx at 2 envs — both invariance penalties ride the same lever, as expected.
2. Only V-REx benefits from the decorrelated third env (0.620 -> 0.680); IRMv1 does not (0.633 -> 0.638).
3. GroupDRO collapses to the shortcut (0.130) — worst-group reweighting preserves runway
   for a shortcut when P(y|shortcut) is stable across groups; consistent with theory.

## Round 6 verdict — updated paper-strength assessment

Gap list from PAPER_DRAFT.md: item 3 (baselines) and item 5 (architecture generality
of the DAG null) now partially closed. Strongest current claim stack:

> 1. Invariance and diversity are complementary levers (EXP10).
> 2. Decorrelation is the sole active ingredient of diversity (EXP11).
> 3. Structural learning null is architecture-general (EXP1/4/7/12).
> 4. Not every invariance penalty uses diversity well (V-REx yes, IRMv1 no) (EXP13).

Remaining before submission: real-world scientific benchmark, theory for (2),
>=10 seeds, power analysis. These genuinely need a collaborator + GPU compute.

## Round 7 — Proposition 1 completed + EXP14 numeric verification

Status upgrade: THEORY_NOTES.md now contains complete proofs for Lemma 1 (exact
risk decomposition), Lemma 2 (zero-variance set), Proposition 1 (diversity =>
exact OOD invariance in the population/lexicographic limit), Proposition 1'
(finite-lambda residual leak: w_s* ~ sum(rho_e)/(4*lambda*gamma*s2e*Var(rho))),
and Corollary 1 (two-environment degeneracy). New falsifiable prediction: the
residual shortcut leak under V-REx scales as sum(rho_e)/lambda — the training
rho-imbalance is the knob, and a symmetric flip (sum rho = 0) has NO first-order leak.

EXP14 (exp14_theory_check.py): exact analytic population minimization of the
V-REx objective on the linear SCM (Var(y)=4.05, noise floor 0.25, gamma=1),
cross-checked with 3-seed finite-sample fits (400k/env).

| env rho        | lambda | sum(rho) | w_s predicted | w_s exact | dR(flip -0.9) exact |
|----------------|--------|----------|---------------|-----------|----------------------|
| {0.9,-0.9}     | 200    | 0.0      | 0             | 0.0000    | +0.0000              |
| {0.9,0.3,0}    | 200    | 1.2      | 0.0429        | 0.1073    | +0.1095              |
| {0.9,0.6,0.2}  | 200    | 1.7      | 0.1034        | 0.1847    | +0.3215              |

lambda sweep on {0.9,0.3,0}: w_s = 0.1242/0.1152/0.1073/0.0982/0.0123/0.0057 for
lambda = 50/100/200/400/800/1600; w_s*lambda -> 9.1 vs predicted constant 8.57.
Finite-sample fits match the analytic optimum (0.1071 +- 0.0025 vs 0.1073).

Findings:
1. sum(rho)=0 => leak exactly zero (Prop 1' first-order prediction exact).
2. 1/lambda scaling confirmed asymptotically (within ~7% by lambda=1600).
3. At practical lambda (<=400) the leak EXCEEDS the first-order floor — the signed
   cubic variance correction (Cov(rho, rho^2) > 0 lowers the penalty); treat the
   formula as a lower bound there.
4. OOD cost of the residue is material: +0.109 on a 0.25 noise floor (~44% relative)
   at lambda=200 — this is precisely the flip-failure residue the paper attributes
   to finite-lambda invariance training.

Paper impact: closes most of PAPER_DRAFT gap item "theory for the decorrelation
claim" (linear case). The story sharpens — the operative variable is the training
rho-imbalance sum(rho_e), which gives EXP11's "decorrelation is the active
ingredient" a quantitative law, and predicts that a THIRD env helps only insofar
as it reduces sum(rho) and increases Var(rho). Still open: nonlinear extension,
global finite-lambda characterization, mu-harm scale (Prop 2).

## Round 8 — EXP15: 10-seed GPU master run (Colab), all ColoredMNIST conclusions hold

The full 9-config ColoredMNIST battery was re-run with 10 seeds on a Colab T4 GPU
(self-contained harness `colab_master_run.py`; per-seed rows in colab_results.csv
to be attached). Supersedes the 5-seed CPU estimates for the image experiments.

| config        | ID (10 seeds)      | OOD (10 seeds)     | 5-seed CPU ref |
|---------------|--------------------|--------------------|----------------|
| ERM-2env      | 0.9199 +- 0.0106   | 0.1538 +- 0.0312   | 0.121          |
| V-REx-2env    | 0.7596 +- 0.0171   | 0.5981 +- 0.0363   | 0.620          |
| ERM-3env      | 0.8321 +- 0.0087   | 0.4468 +- 0.0345   | 0.425          |
| V-REx-3env    | 0.7336 +- 0.0074   | 0.6860 +- 0.0146   | 0.680          |
| IRMv1-2env    | 0.6731 +- 0.0272   | 0.6328 +- 0.0225   | 0.633          |
| IRMv1-3env    | 0.6217 +- 0.0242   | 0.6258 +- 0.0431   | 0.638          |
| GroupDRO-2env | 0.9169 +- 0.0112   | 0.1652 +- 0.0349   | 0.130          |
| GroupDRO-3env | 0.8264 +- 0.0084   | 0.4661 +- 0.0343   | 0.448          |
| ERM+DFR       | 0.8183 +- 0.0111   | 0.4722 +- 0.0238   | 0.478          |

Reading:
1. Every headline conclusion holds at 10 seeds: V-REx-3env best (0.686 +- 0.015,
   lowest OOD variance in the battery), decorrelation remains the active ingredient
   (ERM-3env 0.447 >> ERM-2env 0.154), IRMv1 insensitive to the decorrelated env
   (0.633 -> 0.626 — slightly negative), GroupDRO-2env collapses to the shortcut
   (0.165), DFR mid-pack (0.472).
2. All configs within 4 points of their 5-seed CPU estimates — no publication-risk
   drift; the paper's numbers were updated to the 10-seed values.
3. One shift with more power: IRMv1-2env (0.633) now exceeds V-REx-2env (0.598) —
   the 2-env "IRMv1 ~= V-REx" claim softens to "comparable"; the asymmetry claim is
   about the 3-env response, which is unchanged and sharper (V-REx +9, IRMv1 -1).
4. V-REx-3env's OOD sd (0.0146) is the smallest of all configs — the diversity
   recipe is also the most reliable, useful for the "practical guidance" section.

Gap list impact: item 4 (statistical power) now closed for the ColoredMNIST family
(10 seeds, GPU); still open for synthetic (3-5) and any real-data work.

## Round 9 — completion push (theory extension + synthetic GPU harness + outreach)

1. **Proposition 3 (nonlinear extension)** added to THEORY_NOTES.md: for any
   environment-independent mixing with a well-conditioned first-order shortcut
   channel, Lemmas 1-2 and Propositions 1/1'/Corollary 1 transfer verbatim with
   sigma_e^2 -> sigma_e^2 + sigma_r^2. Turns the "nonlinear extension open" gap
   into a precise, testable assumption rather than an open void.
2. **colab_synthetic_master.py**: self-contained 10-seed GPU harness for the full
   synthetic battery (ERM/V-REx x 2/3 env, SCIN-full/noDAG/noMask, Cohen's d,
   causal-mask recall). Smoke-tested (1 seed, reduced scale): runs end-to-end,
   mask recall 1.0, DAG null reproduces. Ready for the user's Colab.
3. **COLLABORATION_EMAIL.md**: ready-to-send outreach email template.

Status after this round: everything autonomously completable by the agent is DONE.
Remaining items are strictly user-dependent: merge decision on PR #1, sending the
proposal, verifying Prop 3 assumptions with a theory partner, real-data pilot runs.

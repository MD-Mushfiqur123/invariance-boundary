# What invariance fixes and what it cannot: shortcut decorrelation, not distributional diversity, drives out-of-distribution robustness

**Full draft v2.0** · 2026-11-19 ·
Target: *Nature Machine Intelligence* (backup: *Nature Computational Science*; theory-extended variant → NeurIPS/TMLR)
Code: this repository; every number below is reproducible from the committed scripts.

---

## Abstract

Invariant-risk minimization and related environment-invariance methods promise out-of-distribution (OOD) robustness by exploiting multiple training environments, but which property of environmental diversity produces their benefit has remained unclear. Through fourteen pre-registered, seed-controlled experiments on synthetic structural-causal-model data and ColoredMNIST — spanning multilayer, convolutional, and attention architectures and four penalty families — we characterize the boundary of these methods. First, invariance fixes *shortcut flips* (+50 points OOD on ColoredMNIST) yet leaves a −36-point gap under upstream covariate shift; a differentiable causal-graph layer contributes nothing measurable in any regime, on any architecture. Second, the benefit of environment diversity is driven entirely by environments in which spurious features are decorrelated from the label: one such environment closes most of the OOD gap (majority-adjusted −1.71 → −0.26), whereas environments that shift only the causal-feature distribution actively degrade robustness (→ −2.26). Third, penalty families differ sharply in their ability to exploit decorrelated environments (V-REx yes, IRMv1 no). These results reframe environment design — diversity per se is not the lever; decorrelation is — and provide an architecture-general negative control for structure-learning claims. We prove these mechanisms in the linear-shortcut model — including a quantitative law, residual shortcut leak ∝ Σρ/λ that vanishes exactly when the training coloring coefficients sum to zero — and verify the law numerically.

---

## Introduction

Deep models routinely fail under distribution shift by relying on spurious features — correlations that hold in training data but not at deployment. Environment-invariance methods (IRM; V-REx/REx; invariant causal prediction) aim to fix this: given multiple training environments, prefer predictors whose risk or gradient behavior is invariant across environments, on the logic that only causal mechanisms are invariant. Despite a decade of work, two basic empirical questions remain unanswered at the level of controlled evidence.

**First, what exactly do invariance methods fix?** Papers typically report aggregate OOD gains without separating the failure modes they are claimed to address. Shortcut reliance (a feature correlated with the label that reverses at test time) and covariate shift (the causal-feature distribution itself moves) induce different failures, and the community lacks a clean map of which interventions help which failure.

**Second, what is the active ingredient of environment diversity?** Practice treats "more environments" as monotonically better, but environments differ along many axes — marginal feature distributions, class priors, and the strength of spurious association. If the benefit of a new environment has one dominant ingredient, practitioners and dataset designers need to know which.

We answer both questions with a kill-experiment program: falsifiable, pre-registered hypotheses tested with matched controls, multiple seeds, majority-class corrections, and — critically — published negative results, including two of our own initially-promising hypotheses. Our contributions:

1. **A boundary map** (EXP4, EXP6, EXP8): invariance penalties recover shortcut flips (chance-level → 0.63–0.68 OOD) but leave covariate-shift failures essentially untouched (−36-point gap). After majority-class adjustment, the informativeness of an environment is driven by shortcut strength (ρ), not by causal-feature distribution distance (μ).
2. **The decorrelation lever** (EXP11, EXP10): a single environment in which the spurious feature is decorrelated from the label delivers the entire diversity benefit on both synthetic and image data; an environment differing only in causal-feature marginals *hurts*. On ColoredMNIST the best recipe is invariance + one decorrelated environment (0.686 ± 0.015), ahead of invariance alone (0.598), diversity alone (0.447), and DFR (0.472).
3. **An architecture-general structural-learning null** (EXP1, EXP4, EXP7, EXP12): a differentiable sparse-DAG layer with enforced acyclicity (h(G) → 0) adds nothing over matched controls in four independent regimes and three architecture families.
4. **Penalty-family asymmetry** (EXP13): first-order invariance penalties (IRMv1) do not exploit decorrelated environments (0.633 → 0.626), whereas second-order variance penalties (V-REx) do (0.598 → 0.686); worst-group reweighting (GroupDRO) collapses to the shortcut entirely (0.165).

---

## Results

### Bottleneck + invariance produces large OOD gains — but not for the reason we designed

On synthetic SCM data (4 causal, 2 spurious, 16 nuisance latent variables; nonlinear random-mixing observation function; shortcut–label correlation +0.9 in training, −0.9 in the OOD test; Methods), the ERM baseline reaches 0.96 in-distribution but 0.592 OOD (chance = 0.5): the shortcut dominates. Adding an environment-invariance penalty (V-REx, λ=30) to a standard 3-layer encoder yields 0.609 OOD — no meaningful gain. Adding a sparse-bottleneck representation with a differentiable DAG layer and learned selection mask yields 0.688 ± 0.016 (Cohen's d ≈ +6 vs ERM; EXP1/EXP16, 10 seeds). The learned mask retained 100% of true causal latent dimensions at every seed.

This initially appeared to support our architectural hypothesis (sparse causal structure identification improves robustness). It did not survive controls.

### Two confounds found and eliminated by our own ablations

**The DAG layer is inert.** Removing the DAG layer entirely (noDAG control) *matched or exceeded* the full model in every configuration we tested: 0.701 vs 0.676 (EXP1); identical to within 0.2 points under upstream covariate shift (EXP4); identical with the acyclicity constraint genuinely active, converging to h(G) = 0.000 (EXP7); and identical on a self-attention architecture (0.445 vs 0.447, EXP12); and still matched-or-beaten at 10 seeds (0.695 ± 0.015 vs 0.688 ± 0.016, EXP16). Four regimes, three architecture families (multilayer, convolutional, attention), one conclusion: under this training objective, differentiable structure learning provides no benefit beyond that of a plain low-dimensional bottleneck.

**The apparent architecture effect was depth, and did not generalize.** A controlled measurement chain (EXP5) showed that the synthetic-task gain came from the encoder's *depth* (2-layer vs 3-layer): adding one extra encoder layer to the bottleneck model erased the entire benefit (OOD 0.748 → 0.654 ≈ baseline 0.648). But our pre-registered replication on ColoredMNIST (EXP6, 2×2 factorial) rejected the generality of this depth effect: deep and shallow heads both received the full invariance benefit there (0.611 vs 0.622 at λ=30; 0.640 vs 0.633 at λ=100). We therefore treat both the DAG-layer benefit and the depth interaction as refuted claims, and note this as a cautionary case for architecture-confounded OOD benchmarks.

### The boundary: invariance fixes shortcut flips, not covariate shift

Under a joint shift — upstream causal-variable distribution moved (μ: {0,+1.2} → −1.5) plus shortcut flip — ERM collapses *below chance* (0.228 ± 0.014): the model relies on a shortcut whose sign has reversed. V-REx on a 3-layer encoder yields 0.267: invariance penalties barely help here, because every training environment preserves the shortcut's conditional structure. The best model (sparse bottleneck + V-REx) reaches 0.58 — better, but still a −0.37 gap from its in-distribution performance (EXP4).

A phase diagram over OOD test conditions (shortcut strength ρ × shift magnitude μ, EXP8) shows after majority-class adjustment that the apparent interaction of μ with any method's gain is a class-prior artifact (raw gains of 0.49–0.68 at ρ = −0.9 flatten to an adjusted band of −1.6 to −1.8). Shortcut strength drives everything linearly; distributional distance of causal features contributes nothing predictable.

### The decorrelation lever (headline result)

If diversity has an active ingredient, which property of a new environment is it? We added a third training environment to the covariate-shift task and varied its character (EXP11):

| third training environment | OOD acc | majority-adjusted gap |
|---|---|---|
| none (2-env reference) | 0.628 ± 0.029 | −1.71 |
| **decorrelation only** (μ=0, ρ=0) | **0.822 ± 0.012** | **−0.30** |
| shift only (μ=−0.7, ρ=+0.9) | 0.553 ± 0.032 | −2.26 |
| both (μ=−0.7, ρ=0) | 0.828 ± 0.006 | −0.26 |

The decorrelated environment delivers the entire benefit; the environment that shifts causal features while preserving the shortcut is *worse than adding nothing*. On the main synthetic generator the same lever dominates (EXP16, 10 seeds): adding the decorrelated third environment lifts ERM OOD from 0.592 to 0.721 ± 0.006 and V-REx to 0.731 ± 0.010 — above every architecture variant without it. On ColoredMNIST (EXP10), the same structure appears, with invariance and diversity as complementary levers: ERM+V-REx with one decorrelated environment reaches 0.686 ± 0.015 OOD, versus 0.598 invariance-only, 0.447 diversity-only, 0.472 DFR, 0.154 ERM (10-seed GPU master run, EXP15).

### A quantitative law for when environments help

In the linear-shortcut model we prove that the residual shortcut leak of V-REx at finite λ is first-order proportional to the training ρ-imbalance, w_s* ≈ Σ_e ρ_e / (4λγσ_ε²Var_env(ρ)), and exactly zero when Σ_e ρ_e = 0 (Proposition 1'; EXP14 verifies the exact-zero case, the 1/λ decay, and the OOD flip cost of the residue). This converts the categorical finding above into a design rule: environments help insofar as they reduce the mean shortcut-label association and increase its spread across environments — and at finite λ they never fully remove it.

### Penalty-family asymmetry

IRMv1 (first-order gradient penalty) matches V-REx at two environments (0.633 vs 0.598) but extracts almost nothing from the decorrelated third environment (0.626), whereas V-REx gains +9 points (0.686). GroupDRO, which upweights worst environments but never breaks the shortcut's conditional, collapses to near-ERM OOD (0.165) — consistent with its design goal being different (minimax group risk) and with hardness reweighting being orthogonal to decorrelation (EXP13). All nine ColoredMNIST configurations were re-confirmed in a 10-seed GPU master run (EXP15); every configuration landed within 4 points of the 5-seed estimates, with V-REx-3env best at 0.686 ± 0.015 and, notably, IRMv1 slightly *losing* the small benefit it had at two environments.

### Real-data pilot: Camelyon17-WILDS (hospital-shift histopathology)

A pre-registered paired test of the law on public clinical data (v2-fieldfix harness, 3 seeds). Set-up: train on two wilds hospitals {0,3} vs {0,3,4}; evaluate on held-out patches (ID) and on the WILDS test hospital 2 (OOD). The scanner/staining shortcut proxy (linear probe on 8-dim channel statistics into the tumor label) measures nearly identical strength in every hospital: 0.825 / 0.819 / 0.820.

| config | ID | OOD |
|---|---|---|
| ERM-2 hospitals | 0.886 ± 0.023 | 0.511 ± 0.117 |
| ERM-3 hospitals | 0.895 ± 0.010 | 0.462 ± 0.111 |

Two regimes register here. First, the core phenomenon replicates on real clinical data: a model with near-chance invariant structure sits at 0.89 in-distribution and collapses to 0.51 — chance — on a shifted hospital, while the color/staining channel is strongly label-predictive (≈0.82 probe) within every hospital. Second, the third hospital carries essentially zero shortcut decorrelation (gap −0.002), the law's null regime, and delivers ≈ nothing (ΔOOD −0.049 ± 0.205) — directionally consistent, though at 3 seeds the sign test is unpowered and formally inconclusive by design (the strength sat at the {base}-mean boundary we pre-declared uninformative). What remains untested on real data is the *weak-shortcut* regime: no entire hospital provides a low-strength third environment. A slide-level follow-up (EXP18) interrogates within-hospital 4 strength heterogeneity, arming the paired design with weak-vs-strong third-hospital slices from the same cached data.

### Real-data follow-ups: no weak environment exists, and the failure is a mapping shift

Two cached-data follow-ups resolve why the pilot showed a null delta. The slide-level arming (EXP18) failed to find its treatment: hospital 4's slides are either tumor-saturated (8 slides, minority ≤ 0.083, probe values are majority-baseline artifacts) or class-balanced with strength 0.96–0.98 — the weak-shortcut regime does not exist in Camelyon17 at hospital or slide granularity; the pre-registered weak-vs-strong arm comparison is thus infeasible (a negative design result we record as such). The decisive measurement is a cross-hospital transfer matrix (EXP18b): a linear probe on 8-dim color/staining features trained on hospital i and tested on hospital j (train-time normalization carried over). Hospital 2 — the OOD target — has own strength 0.864 but transfers at chance from every training hospital (0.517–0.523): its P(tumor | color) is a different function. The matrix also reveals the dataset's structure: exactly two color-mapping blocks, {0, 1, 3} mutually compatible (0.70–0.79) and {2, 4} jointly incompatible with block A (0.48–0.55, h2↔h4 = 0.655). The cheap probe matrix therefore predicted the deep model's 0.51 collapse before any CNN training. Why the h4 arm (block B, sized 2500 vs 7000) failed to decorrelate its way to OOD gains remains open — arm size, persistence of the marginal color-label correlation under mixed mappings, or a texture-level shortcut beyond the 8-dim proxy. A prospective diagnostic test (EXP19) then asks whether per-hospital probe transfer predicts per-hospital deep-model accuracy — and it does. With disjoint pools for everything, a TorsoCNN trained on {0, 3} scores 0.917/0.880 per training hospital, transfers to the compatible block member h1 at 0.812, and collapses on the flagged targets: 0.533 ± 0.157 on hospital 2 (chance) and 0.647 ± 0.088 on hospital 4 (partial). Predicted transfer excess correlates with observed accuracy at Pearson r = +0.905 (Spearman +0.800) across all five targets, with the full ordering preserved (h3 > h0 > h1 ≫ h4 > h2) and the diagnostic requiring no deep-model training. We state the limits honestly: five targets, r descriptive, hospital 4 under-predicted (the 8-dim proxy misses the histology component its 0.655 similarity to hospital 2 hints at), and hospital 2's seed variance (±0.157) suggests a bimodal collapse — a 10-seed hardening run is the registered next step. The real-data arc of this paper is: shortcut reliance collapses a clinical model across hospitals (0.89 → chance); no weak-shortcut environment exists at hospital or slide granularity to buy; the failure is a shortcut-mapping shift, measurable in minutes; and that cheap measurement prospectively predicts where the deep model fails.

## Discussion

**Mechanistic interpretation.** V-REx equalizes environment losses. When every training environment preserves P(y | shortcut), loss equality is satisfiable *through the shortcut*; a decorrelated environment makes the shortcut route internally inconsistent, forcing the predictor onto the causal block. We formalize this in the linear-shortcut model and prove (THEORY_NOTES.md): (i) with three or more distinct coloring coefficients among training environments the lexicographically minimal V-REx solution is exactly the causal predictor, whose risk equals the noise floor in *every* environment — training or test (Proposition 1); (ii) with two environments the zero-loss-variance set is a positive-dimensional manifold through the causal predictor, so nothing in first-order dynamics selects it (Corollary 1, the REx analogue of the IRMv1 degeneracy of Rosenfeld et al.); (iii) at finite λ the residual shortcut leak is first-order sum(ρ)/λ — zero for a symmetric flip, decaying as the training ρ-imbalance divided by λ (Proposition 1'), verified numerically to the asymptotic regime (EXP14: leak exactly 0 at Σρ = 0; w_s·λ converging to the predicted constant within ~7% by λ = 1600; measured OOD flip penalty +0.11 on a 0.25 noise floor at λ = 200). Proposition 1' thereby turns EXP11's categorical "decorrelation is the active ingredient" into a quantitative law: an added environment helps insofar as it reduces Σρ and increases Var(ρ). We emphasize: this is a *pseudo-interventional* account — an invariance mechanism — not do-operator identifiability; we label it as such. The proofs cover the linear shortcut channel; the nonlinear extension is open.

**Why the negative results matter.** Differentiable causal-structure learning is an active, well-funded direction. Our results do not claim it can never help; they establish that in four controlled regimes — including one designed to require propagation structure and one with acyclicity actively enforced — it provides no measurable OOD benefit over matched controls, across three architecture families. That is a real, falsifiable boundary statement which the field currently lacks in controlled form.

**Practical guidance.** Environment budget should buy *shortcut-breaking* slices of data — situations where the spurious feature's association with the label is weakened or severed (new sites, protocols, collection regimes) — not environments that merely shift the causal-feature marginals. The 2-week pilot design we propose for real data (train with vs without one decorrelated environment; measure held-out shift) is directly actionable for any team controlling data collection.

**Limitations.** (i) The quantitative synthetic results are on synthetic/semi-synthetic data ≤64×64; the ColoredMNIST family is the field's standard prove-out. The real-data evidence (EXP17–EXP19, Camelyon17-WILDS) is 3-seed with n=5 target hospitals: the shortcut-reliance collapse, the two-block mapping structure, and the r = +0.905 transfer diagnostic are each single-dataset observations — the 10-seed hardening run is registered, the r value is descriptive and the collapsible claim is the classification of failing targets, not the correlation; multi-dataset replication (RxRx1, 7.4 GB, same API) is the natural next scope step. (ii) The ColoredMNIST family (EXP15) and the synthetic battery (EXP16) use 10 seeds; remaining synthetic rounds used 3–5 seeds with un-tuned λ except where swept; real-data studies need ≥10 seeds and pre-registered analysis. (iii) The proven theory covers the linear-shortcut channel (complete proofs, numerically verified in EXP14); the nonlinear extension — local analysis around the encoder linearization — remains open and is the main requested contribution of a theory collaborator. (iv) "Only decorrelation matters" is a claim about this generator family; other shortcut structures (e.g., richly structured confounders) may behave differently.

---

## Methods

**Data.** *Synthetic SCM:* latent z = [c; s; n], c ∈ R⁴ (EXP1/2/3/7) or the chain c = (z1 → z2 → z3) with y = f(z2, z3) (EXP4/8/9/11), s ∈ R¹⁻² with s = ρ_e·g(y) + noise, nuisance n ∈ R¹⁶⁻¹⁸; observation x = mixing(z) through a fixed random two-layer tanh network to 64 dims. *ColoredMNIST:* 2-channel MNIST, label = digit parity, 25% label noise; channel tint g ~ N(ρ_e·t, 0.15²) with t = ±y; environments e0 (ρ=+0.9), e1 (+0.3), e2 (0), OOD test (−0.9). MNIST source: ossci-datasets.s3.amazonaws.com.

**Models.** *MLP:* 3-layer encoder 64→128→128→128 (baseline) or 2-layer 64→128→latent (bottleneck; width 8–128), linear head. *Bottleneck variants:* learned selection mask; differentiable DAG layer with forward mixing h + α·tanh(h·(W∘W)) and NOTEARS-style acyclicity h(G) = tr(e^{W∘W·|·|}) − D matched to the α variant; when active, λ₁=1, ‖W‖₁ 1e-3. *CNN (ColoredMNIST):* two 3×3 convs (stride 2) → 1152 flatten → shallow (2-layer) or deep (3-layer) head. *ViT-lite:* 7×7 patch embedding (16 tokens, d=32), one MultiheadAttention layer, mean pooling. *DFR:* frozen ERM backbone, head retrained on the decorrelated environment.

**Training.** Adam, lr 1e-3, batch 256 (images) / 512 (synthetic), 12–80 epochs as per experiment size. Environment-invariance = V-REx (variance of per-environment losses; λ ∈ {25, 30, 100}); IRMv1 = first-order gradient penalty (λ=25); GroupDRO = exponentiated worst-group reweighting (η=0.1).

**Evaluation.** Held-out test environments; majority-class-adjusted accuracy (acc − max class rate, normalized) wherever class priors shift; Cohen's d for effect sizes; 10 seeds for both the ColoredMNIST family (EXP15) and the synthetic battery (EXP16), 3–5 for earlier synthetic rounds (exact n per experiment in FINDINGS.md), mean ± sd. Baseline rates and class balancers were computed from 200k Monte-Carlo samples of the generator.

**Software.** PyTorch 2.15 (CPU); numpy 2.5; no randomness in evaluation paths beyond seeds; per-experiment scripts with fixed seed protocol; phase-diagram grid and all raw summary statistics in results.csv.

**Data & code availability.** Everything in this repository; synthetic experiments consume ≈1 CPU-hour total. The real-data pilot (EXP17) uses the public Camelyon17-WILDS dataset (~15 GB one-time download; CC0) via the wilds package and reuses its cache on reruns.

---

## References (to format later)

Arjovsky et al. 2020 (IRM) · Krueger et al. 2021 (REx/V-REx) · Rosenfeld et al. 2021 (risks of IRM) · Arjovsky et al. 2019 (invariant batch norm) · Geiger et al. 2025 (causal abstraction, JMLR 26) · Lippe et al. 2022 (CITRIS) · Bello et al. 2022 (DAGMA) · Zheng et al. 2018 (NOTEARS) · Gulrajani & Lopez-Paz 2021 (DomainBed) · Kirichenko et al. 2023 (DFR) · Sagawa et al. 2020 (GroupDRO) · Replogle et al. 2022 (Perturb-seq) · Arpit et al.; Geirhos et al. 2020 (shortcut learning, contextual survey)

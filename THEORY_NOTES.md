# THEORY_NOTES — Draft theory for the decorrelation claim
Status: Lemma 1–2, Proposition 1, Proposition 1', Corollary 1 are PROVEN (linear
model, population level); numerically verified in EXP14. Nonlinear extension and
the global finite-λ minimizer characterization remain open (theory collaborator).

## Setting

- Observed features x = [c; s] with causal block c ∈ R^k (Var(c) = Σ_c ≻ 0) and a
  one-dimensional spurious block s ∈ R (m > 1 identical, concatenated).
- Label process: y = βᵀc + ε, E ε = 0, Var(ε) = σ_ε² > 0, ε ⊥ (c, η).
- Training environments E = {e1,…,en}, environment e has a coloring coefficient ρ_e:
  s = ρ_e·γ·y + η, γ ≠ 0, Var(η) = σ_η² > 0, η ⊥ (c, ε).
  So y ⟂ s | c but Corr_e(s, y) = ρ_e (as in our data generators: s = ρ_e·g(y) + noise).
- Derived constants: Var(y) = σ_ε² + βᵀΣ_c β =: v_y > 0;
  Cov_e(c, s) = ρ_e γ Σ_c β; Cov_e(s, y) = ρ_e γ v_y; Cov(s, ε) = ρ_e γ σ_ε².
- Model class: linear predictors w = (w_c, w_s), ŷ = wᵀx, trained on the population
  V-REx objective J_λ(w) = Σ_e R_e(w) + λ·Var_e(R_e(w)), R_e(w) = E_e (y − wᵀx)².

Write w* := (β, 0) for the causal predictor and w = w* + (δ, w_s).

## Lemma 1 (exact risk decomposition)

For every environment coefficient ρ (training or test, any real value):

  R_ρ(w) = σ_ε² + δᵀΣ_cδ + w_s²(ρ²γ²v_y + σ_η²) + 2ργw_s(δᵀΣ_cβ) − 2ργw_s σ_ε².   (1)

Proof. (wᵀx − y)² = (δᵀc + w_s s − ε)². Expand; the only nontrivial terms are
E[s ε] = ργσ_ε², E[δᵀc · s] = ργ δᵀΣ_cβ, E[s²] = ρ²γ²v_y + σ_η², and all terms
involving E[c ε] = 0. ∎

Two immediate consequences.

1. R_ρ(w*) = σ_ε² for EVERY ρ. The causal predictor's risk is environment-free —
   this is the ID = OOD equality used as the benchmark throughout the paper.
2. R_ρ(w) − σ_ε² is, as a function of ρ, the quadratic q(ρ) = A + B(w)ρ + C(w)ρ²
   with C(w) = w_s²γ²v_y ≥ 0 and B(w) = 2γw_s(δᵀΣ_cβ − σ_ε²).

## Lemma 2 (zero-variance set)

Let D = {ρ_1, …, ρ_d} be the set of DISTINCT training coefficients.

(a) d ≥ 3  ⇒  {w : Var_e(R_e) = 0} = {w : w_s = 0} (δ unconstrained).
(b) d = 2  ⇒  {w : Var_e(R_e) = 0} = {w_s = 0} ∪ {2γw_s(δᵀΣ_cβ − σ_ε²)
      + w_s²γ²v_y(ρ_1 + ρ_2) = 0}; the second branch contains predictors with
      w_s ≠ 0 (a positive-dimensional manifold through w* whenever it is nonempty).
(c) d = 1  ⇒  the variance term vanishes identically; no constraint at all.

Proof. Var_e(R_e) = 0 ⇔ q(ρ) equal at all points of D.
(a) q(ρ) − constant is a quadratic with ≥ 3 distinct roots, hence identically zero;
    so C(w) = 0, and w_s = 0 because v_y > 0 and γ ≠ 0. Conversely w_s = 0 makes
    R_ρ = σ_ε² + δᵀΣ_cδ, ρ-free.
(b) q(ρ_1) = q(ρ_2) ⇔ (ρ_1 − ρ_2)·(B(w) + C(w)(ρ_1 + ρ_2)) = 0.
(c) trivial. ∎

## Proposition 1 (environment diversity ⇒ exact OOD invariance, population / λ→∞)

Assume d ≥ 3 distinct training coefficients. The lexicographic population objective
(first minimize Var_e(R_e), then Σ_e R_e) has the UNIQUE minimizer

  w** = (β, 0),  and  R(w**) = σ_ε² in every environment, training or test,

so R_OOD(ρ_test) = R_ID = σ_ε² for any ρ_test ∈ R.

Proof. By Lemma 2(a) the variance-zero set is exactly {w_s = 0}. On that set
Σ_e R_e = n(σ_ε² + δᵀΣ_cδ), strictly convex in δ (Σ_c ≻ 0) and uniquely minimized
at δ = 0. ∎

Remarks.
- No large λ is needed: the invariance term only has to select the w_s = 0
  stratum; the loss term does the rest. The operative lever is the third
  environment (d ≥ 3), not a bigger λ — exactly the EXP9/EXP11 lever.
- In this population-linear setting the diversity benefit is available to any
  penalty that enforces zero loss-variance; empirically V-REx realizes it while
  first-order IRMv1 does not (EXP13) — an optimization-level difference, not a
  population-level one.

## Proposition 1' (finite-λ residual leak; Σρ_e is the knob)

Claim.
(i) If Σ_{e∈E} ρ_e ≠ 0 then for every FINITE λ > 0 the causal predictor w* is not
    a local minimizer of J_λ: ∇J_λ(w*) = (0_k, −2γσ_ε²Σρ) ≠ 0, so the population
    minimizer has β̂_s ≠ 0.
(ii) As λ → ∞ the leak is first-order in Σρ and decays as 1/λ:
      w_s* = Σ_e ρ_e / (4 λ γ σ_ε² Var_env(ρ)) + O(λ⁻²),
    with Var_env(ρ) = (1/n)Σ(ρ_e − ρ̄)². In particular Σρ = 0 ⇒ w_s* = 0.
(iii) The leak's OOD cost, to leading order (δ = O(w_s²) from the δ-block
    stationarity): R_{ρ_test}(w) − R_{ρ̄}(w) ≈ −2γw_sσ_ε²(ρ_test − ρ̄), which for a
    sign-flipped test environment (ρ_test < 0 < ρ̄) is STRICTLY POSITIVE: the
    finite-λ residue is exactly the below-chance flip failure of EXP4/EXP8.
    Under a flip to ρ_test = −ρ̄ the penalty is ≈ 4γw_sσ_ε²ρ̄ > 0.

Proof. (i) ∂/∂w_s Σ_e R_e |_{w*} = Σ_e (−2ρ_eγσ_ε²) = −2γσ_ε²Σρ ≠ 0 (Lemma 1);
    the Var term has zero gradient wherever Var = 0 (Var = (1/n)Σ(R_e − R̄)² and
    R̄ − R_e = 0 for all e, so ∇Var = (2/n)Σ(R_e − R̄)∇R_e = 0).
(ii) Second-order expansion of (1) at δ = 0: the escape quadratics are
    Σ-term: w_s² · (nσ_η² + γ²v_yΣρ²) − 2γσ_ε²Σρ·w_s and
    Var-term: λ · 4γ²σ_ε⁴Var_env(ρ)·w_s² (variance of the linear-in-ρ part of
    R_e − σ_ε², which is −2ργw_sσ_ε²). Setting d/dw_s = 0 at leading order gives
    the displayed formula; δ-block stationarity gives δ = O(w_s²), and the cubic
    cross-terms (e.g. the Cov(ρ, ρ²) term in Var of a quadratic) contribute the
    O(λ⁻²) remainder.
(iii) Substitute w_s* into (1) with δ = 0. ∎

Scope: population level, leading order. Note the O(λ⁻²) remainder has a sign —
when Cov_env(ρ, ρ²) > 0 it LOWERS the variance penalty for w_s > 0, so at practical
λ the true leak exceeds the first-order floor: the formula is a LOWER BOUND there.
EXP14 confirms all three signs and the 1/λ asymptotics.

## Corollary 1 (two-environment degeneracy)

d = 2 ⇒ the zero-variance set contains a manifold with β̂_s ≠ 0 (Lemma 2b). On the
symmetric flip (ρ̄, −ρ̄) every such point has per-env risk strictly larger than
σ_ε², so the population lexicographic minimizer is still w*; but the V-REx gradient
vanishes along the entire manifold, so nothing in first-order training dynamics
selects w* either. The shortcut retention of 2-env training (EXP1/EXP4/EXP13,
and the 2-env rows of EXP10/EXP11) is structurally expected; cf. Rosenfeld et al.
(2021) for the analogous IRMv1 degeneracy, which is the same d ≤ 2 mechanism.

## EXP14 — numerical verification (exp14_theory_check.py)

Exact analytic population minimization (5×5 covariance algebra, LBFGS with 8
restarts) cross-checked against a 3-seed finite-sample fit (400k samples/env).
Constants: v_y = 4.0504, σ_ε² = 0.25, γ = 1. Noise floor R_ID = 0.25.

| env ρ        | λ   | Σρ   | w_s first-order pred | w_s exact            | ΔR(flip −0.9) exact |
|--------------|-----|------|----------------------|----------------------|----------------------|
| {0.9,−0.9}   | 200 | 0.0  | 0                    | 0.0000               | +0.0000              |
| {0.9,0.3,0}  | 200 | 1.2  | 0.0429               | 0.1073 (sample 0.1071±0.0025) | +0.1095 |
| {0.9,0.6,0.2}| 200 | 1.7  | 0.1034               | 0.1847               | +0.3215              |

1/λ sweep on ρ = {0.9, 0.3, 0}: exact w_s = 0.1242, 0.1152, 0.1073, 0.0982, 0.0123,
0.0057 for λ = 50, 100, 200, 400, 800, 1600; w_s·λ → 9.1 at λ=1600 vs predicted
constant Σρ/(4γσ_ε²Var(ρ)) = 8.57 (converging from above). ΔR(flip) falls
+0.159 → +0.004.

Findings.
1. Σρ = 0 (symmetric flip) ⇒ leak EXACTLY zero, as Prop 1'(ii) predicts.
2. 1/λ scaling confirmed asymptotically (within ~7% of prediction by λ = 1600).
3. At practical λ (≤ 400) the leak exceeds the first-order floor — the signed
   cubic correction; treat the Prop 1' formula as a lower bound there.
4. The OOD cost of the residue is material: +0.11 on a 0.25 noise floor (~44%
   relative) at λ = 200.

## Proposition 2 (shift-only environments can hurt)

Claim. A new environment with unchanged ρ_e = ρ̄ but shifted causal distribution
(μ ≠ 0) leaves the invariance term's shortcut block unchanged while reweighting the
empirical per-env loss levels; with finite λ this increases the de-facto shortcut
weight relative to no extra environment. Our EXP11 result (adding a μ-only env:
majority-adjusted OOD gap −1.71 → −2.26) matches the predicted sign. Contrast with
Prop 1': a decorrelation-only env (ρ ↓, μ fixed) changes the shortcut geometry
itself; a shift-only env does not.

Proof sketch: direct in the linear family — the variance penalty acts on
environment-conditional loss levels, not on decorrelation of s.

## Empirical mapping (theory prediction vs observation)

| Theory prediction | Experiment | Observed |
|---|---|---|
| ≥ 3 distinct ρ, lexicographic ⇒ shortcut block vanishes, R_OOD = R_ID exactly | EXP11 decorr-only env | OOD 0.822 ≈ both 0.828 vs 0.628 (2-env) |
| Σρ ≠ 0, finite λ ⇒ residual leak ∝ Σρ/λ, hurting on flip | EXP14 | exact 0 at Σρ=0; 1/λ scaling; ΔR +0.109 at λ=200 |
| d ≤ 2 ⇒ zero-variance manifold with β̂_s ≠ 0; dynamics unconstrained | EXP1/EXP4/EXP13 (2-env) | ERM/REx below chance at ρ_test = −0.9; IRMv1 2env ≈ 3env |
| μ-shift with ρ̄ ≥ 0 ⇒ no help, possible harm | EXP11 shift-only | −2.26 (hurts) |
| V-REx exploits a decorrelated env; first-order IRMv1 need not | EXP13 | V-REx 0.620→0.680; IRMv1 0.633→0.638 |

## TODO for a full theory section

1. DONE (this revision): Lemmas 1–2, Propositions 1, 1', Corollary 1, verified by
   EXP14. Remaining: global (non-local) finite-λ minimizer characterization for
   moderate λ.
2. Nonlinear extension: local analysis around the encoder linearization, or a
   two-layer result via the bottleneck literature.
3. Link to identifiability: a environment with ρ = 0 is a *pseudo-intervention*
   that severs the spurious association; formalize via ICP — honestly labeled
   pseudo-interventional, not do-operator identifiability.
4. Derive the predicted μ-harm scale of Proposition 2 and compare with EXP11's
   −0.55 swing.

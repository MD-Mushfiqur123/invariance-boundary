# THEORY_NOTES — Draft theory for the decorrelation claim
Status: sketch (needs a theory collaborator; not yet proofs).

## Setting

- Observed features split into causal block c ∈ R^k and spurious block s ∈ R^m
  (after a mixing/observation process that preserves their spans).
- Label process: y = f(c) + ε, ε independent of (s, c).
- Training environments E = {e1,…,en}; each environment has a coloring coefficient
  ρ_e that controls the association between s and y: y ⟂ s | c, but marginally
  Corr_e(s, y) = ρ_e (as in our data generators: s = ρ_e·g(y) + independent noise).
- Model class: predictors g_θ trained by minimizing
  Σ_e L_e(g_θ) + λ · Var_e(L_e(g_θ))   (V-REx / REx).

## Proposition 1 (decorrelation silo, linear case)

Assume a linear-generative model: x = [c; s], y = β_c·c + ε, s = ρ_e·γ·y + η_e.

Claim. If the training environments include at least one with ρ_e = 0, then (for
full-rank γ and λ large) the population minimizers of the objective are exactly the
predictors with zero spurious block (β̂_s = 0); their OOD risk under any test flip
ρ_test equals their ID risk. If instead all training environments have ρ_e = ρ̄ > 0,
there exists a minimizer with β̂_s ≠ 0 whose OOD risk under ρ_test = −ρ̄ is strictly
larger, and gradient descent from standard initialization attains it for suitable λ.

Proof sketch.
1. Per-environment population optimum: w_e* = Σ_e^{-1} Cov_e(x, y). Direct Gram-matrix
   computation gives Cov_e(x_s, y) = ρ_e γ Var(y).
2. The shortcut block of w_e* is proportional to ρ_e; the causal block is
   environment-invariant (y|c stable across environments).
3. Var_e(L_e) → 0 forces equal losses across environments; with one ρ=0 environment
   the only de-correlated optimum has β̂_s = 0 (the ρ=0 environment pins down the
   causal-optimal predictor, which is then the unique variance-free solution).
4. With all ρ_e = ρ̄ > 0, the shortcut block cancels only if per-env risk is equalized
   at a shifted level; the resulting minimizer keeps β̂_s ≠ 0 with sign error under
   ρ_test = −ρ̄, producing the below-chance failure we observe. R_OOD − R_ID is
   elementary in the closed-form family.

Scope to make explicit: linear shortcut channel; full-rank γ; λ large enough for the
invariance term to dominate empirical perturbation (cite and contrast Rosenfeld et al.
2021, linear-max-margin caveat); population-to-empirical gap is the usual n-term.

## Proposition 2 (shift-only environments can hurt)

Claim. A new environment with unchanged ρ_e = ρ̄ but shifted causal distribution
(μ ≠ 0) leaves the invariance term's shortcut block unchanged while reweighting the
empirical per-env loss levels; with finite λ this increases the de-facto shortcut
weight relative to no extra environment. Our EXP11 result (adding a μ-only env:
majority-adjusted OOD gap −1.71 → −2.26) matches the predicted sign.

Proof sketch: direct in the linear family — the variance penalty acts on environment-
conditional loss levels, not on decorrelation of s.

## Empirical mapping (what the theory predicts vs what we observed)

| Theory prediction | Experiment | Observed |
|---|---|---|
| ≥1 ρ=0 env ⇒ shortcut block vanishes (full diversity benefit) | EXP11 decorr-only env | OOD 0.822 ≈ both 0.828 vs 0.628 (2-env) |
| No decorrelated env ⇒ flipping ρ_test keeps shortcut sign error | EXP4/EXP8 (all train ρ>0) | ERM/REx below chance at ρ_test = −0.9 |
| μ-shift with ρ̄ ≥ 0 ⇒ no help, possible harm | EXP11 shift-only | −2.26 (hurts) |
| V-REx exploits a new decorrelated env; first-order IRMv1 need not | EXP13 | V-REx 0.620→0.680; IRMv1 0.633→0.638 |

## TODO for a full theory section
1. Rigorous Proposition 1 with minimal conditions and complete linear proof.
2. Nonlinear extension: local analysis around the encoder linearization, or a
   two-layer result via the bottleneck literature.
3. Link to identifiability: a ρ=0 environment is a *pseudo-intervention* that severs
   the spurious association; formalize via ICP — honestly labeled pseudo-
   interventional, not do-operator identifiability.
4. Derive the predicted μ-harm scale and compare with EXP11's −0.55 swing.

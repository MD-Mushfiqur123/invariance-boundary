# COLLABORATION_PROPOSAL — Invariance Boundary study
**For:** potential collaborators (computational biology / clinical data science)
**From:** MD Mushfiqur (+ automated experimentation) · 2026-11-19

## One-paragraph pitch

We have a completed, reproducible experimental program (13 controlled experiments,
5-seed replicates, code public) characterizing exactly when environment-invariance
methods deliver OOD robustness — and when they cannot. The headline: the *only*
property of training-environment diversity that matters is whether it breaks the
spurious (shortcut) correlation; environments that shift the causal-feature
distribution alone provide nothing and can actively hurt. We verified the
structural-learning null (differentiable causal-DAG layers give no OOD benefit) on
three architecture families. We are now seeking a domain collaborator to run this on
one scientifically meaningful benchmark and co-author the resulting paper
(target: Nature Machine Intelligence / Nature Computational Science).

## What is already done (evidence in repo)

- Boundary map of invariance methods (shortcut flips fixed; covariate shift not).
- Architecture- and penalty-family-validated results (MLP, CNN, ViT-lite; V-REx / IRMv1 / GroupDRO / DFR).
- A reproducible harness: data generators, models, per-experiment scripts, results.csv.
- Paper draft (PAPER_DRAFT.md) with an honest remaining-gap list.
- Theory sketch (THEORY_NOTES.md) with linear-case propositions matching all observations.

## What we need from a domain partner

1. **Benchmark + ground truth.** A real dataset with (a) a known spurious
   site/channel/batch effect, and (b) a defensible holdout where the spurious
   feature's association with the outcome flips or is absent. Candidate domains:
   - *Genomics:* Perturb-seq (Replogle et al. 2022) — cell line/plate as environment;
     test whether the decorrelated-environment hypothesis predicts which held-out
     plates transfer.
   - *Clinical EHR / imaging:* site/scanner-effect tasks (site = environment).
   - *Climate / remote sensing:* station/sensor as environment.
2. **Domain interpretation** of the shortcut-decorrelation hypothesis on your data.
3. **Co-authorship involvement:** theory + real data together are the paper.

## What we bring

- Full empirical harness; hypothesis is falsifiable on your data with a 2-week pilot:
  *train with vs without one decorrelated environment; measure OOD on the held-out shift.*
- Negative-result discipline: three of our own hypotheses were killed by our controls
  (documented in FINDINGS.md) — we will hold your data to the same standard.

## Roles

| Who | Contribution |
|---|---|
| Us | Theory sketch, multi-architecture experiment design, all ML implementation |
| Domain partner | Dataset curation, ground-truth validation, interpretation, co-writing |
| Optional: theory collaborator | Rigorize THEORY_NOTES.md propositions |

## Contact

Open an issue in this repository tagged `collab`.

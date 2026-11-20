# Literature Delta Analysis — SCIN Plan v1.0
**Date:** 2026-11-18 · Purpose: justify every novelty claim against 2022–2025 literature before any more compute is spent.

## Claim-by-claim

| Plan claim | Overlapping prior work | What remains genuinely open |
|---|---|---|
| Causal graphs inside neural nets | Geiger et al., *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability* (JMLR 2025, arXiv:2301.04709) — unifies patching, causal scrubbing, circuit discovery, DAS, SAEs under one causal framework | A **trainable** (not post-hoc) path from causal abstraction to OOD robustness |
| Differentiable causal structure | NOTEARS→DAGMA line; DCDI/DECI; invariant-causal-subgraph GNNs with learnable structural masks (e.g., InsGNN, WIREs DMKD 2025 review) | Joint structure learning + invariance **with identifiability guarantees in non-temporal high-dim settings** |
| Causal disentanglement & robustness | CRL field: CITRIS/iCITRIS (Lippe et al. 2022), LEAP, TDRL, CaRiNG (2024), identifiability from purely observational data (NeurIPS 2024), general multi-distribution setting (ICML 2024); survey arXiv:2310.11011 | Most identifiability results need temporal structure or interventions. **A general observational identifiability route to OOD gains** is still open |
| Environment invariance | IRM (Arjovsky 2020) + documented failures: Rosenfeld et al. 2021 (*Risks of IRM*). **Plan must cite these** — reviewers will check | Principled fixes to IRM's failure modes — our kill-experiment directly contributes evidence here |

## Consequence for the plan
The "we do all four jointly for the first time" framing is **not defensible**. Defensible framing after kill-experiment:

> *We show empirically that combining a learnable sparse bottleneck with environment-invariance produces large OOD gains where invariance alone fails — but that the popular differentiable-DAG machinery contributes nothing in this regime. We characterize when structure learning does and does not matter for OOD robustness.*

That is a *mechanistic* question (why does bottleneck+REx work where REx alone fails?) — arguably more interesting to a Nature MI audience than another "we beat SOTA" pipeline.

## Required citations to add to the plan (non-negotiable for submission)
- Geiger et al. 2025 (JMLR 26) — causal abstraction
- Rosenfeld et al. 2021 — risks of IRM
- Lippe et al. 2022 (CITRIS), Brehmer et al. 2022, von Kügelgen et al. 2021
- arXiv:2310.11011 survey (causal generative modeling)
- Replogle et al. 2022 (Perturb-seq) + BEELINE critique for the genomics section
- DAGMA (Bello et al. 2022) — replaces NOTEARS as the scalable DAG baseline
- Zhang et al. 2024 identifiability works (NeurIPS 2024 observational identifiability)

# Draft email — reach out to a domain collaborator

Subject: Collaboration: a controlled test of when environment diversity helps AI
generalization — looking for a bio/clinical data partner

Dear Dr. [Name],

I am [Mushfiqur Rahman / your full name], an independent researcher working on
out-of-distribution generalization for machine learning. I am writing to invite
you to co-author a study in which your domain data would provide the real-world
test of a finding we have established rigorously in controlled settings.

**The finding.** Across 15 seed-controlled experiments (10 seeds on the image
benchmark, pre-registered kill questions, negative results reported in full), we
show that the benefit of "environment diversity" in invariance-based OOD training
is driven by exactly one property: whether the added environment *decorrelates the
spurious feature from the label*. Environments that shift the causal-feature
distribution instead actively hurt; a differentiable causal-graph layer adds
nothing in any regime or architecture; and we can now predict the residual
shortcut error of invariance training quantitatively (it scales with the training
spurious-association imbalance divided by the penalty weight).

**What we need from a partner.** One dataset in which (a) a shortlist of known
confounders/shortcuts exists, and (b) data can be grouped by collection context
(site, protocol, batch, time period). Candidates we consider ideal: Perturb-seq
gene-perturbation screens batched by experiment; EHR prediction across hospital
sites; single-cell atlases across donors/protocols; remote-sensing or climate
reanalysis across stations.

**What we offer.** A 2-week pilot requiring no data movement on your side: we run
a paired analysis (models trained with vs. without one decorrelated group), and
judge whether held-out performance under a flipped/stressful environment improves
the way our theory predicts. If it does, the follow-up is a full study
(Nature Machine Intelligence–shaped) with you as senior author. One-page summary
of the program is attached (COLLABORATION_PROPOSAL.md).

Would you be open to a 30-minute call?

Best regards,
[Your name, affiliation, contact]

---

# ইমেইল পাঠানোর নিয়ম (নিজের জন্য)

- এই ড্রাফটে [Name] আর [Your name...] বদলে নিও
- সাথে COLLABORATION_PROPOSAL.md (PDF/export করে) attach করো
- প্রার্থী: যেকোনো computational biology / clinical AI গ্রুপ, বা বিশ্ববিদ্যালয়ের
  অধ্যাপক যার কাছে Perturb-seq বা মাল্টি-সাইট EHR ডেটা আছে

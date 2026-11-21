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

---

# কেউ চেনা না থাকলে: তিনটা self-serve রাস্তা

## A. Open Problems in Single-Cell Analysis (সবচেয়ে বাস্তবসম্মত, আজই শুরু করা যায়)

[openproblems.bio](https://openproblems.bio) — ওপেন-সোর্স single-cell benchmarking
কমিউনিটি (CZI-supported, Nature Biotechnology 2025 paper)। **নতুন contribution-এর
জন্য স্পষ্টভাবে ওপেন**, প্রথমবারের contributors-দের স্বাগতম বলে নিজেরাই বলে।

ধাপ:
1. তাদের [Discord](https://openproblems.bio)-এ ঢুকে নিচের intro-টা পোস্ট করো:

   > 👋 Hi! I'm an independent ML researcher. I've built and proven (linear-case)
   > a controlled theory of WHEN adding a training environment helps invariance-based
   > OOD generalization: the active ingredient is shortcut decorrelation, not generic
   > diversity — verified at 10 seeds on two benchmark families. I'd love to
   > contribute a new benchmark task here: "predict-from-a-decorrelated-batch" —
   > a task where spurious batch-label correlation flips across collection contexts.
   > Happy to draft the task spec + methods. Which channel/issue is best to start?

2. তাদের [GitHub রিপোতে](https://github.com/openproblems-bio/openproblems) একটা issue
   খোলো, টাইটেল: `Proposal: new task — batch/protocol-shift with decorrelated label
   structure`. বডিতে COLLABORATION_PROPOSAL.md-এর পাইলট ডিজাইন সংক্ষেপ দাও।
3. community@openproblems.bio-তে ওই একই প্রস্তাব ইমেইল করো (ওরা code of conduct-সহ
   স্বাগতম-সংস্কৃতির কমিউনিটি)।

## B. Public ডেটাতে পাইলট — কোনো সহযোগীই লাগবে না

WILDS বেঞ্চমার্ক ([wilds.stanford.edu](https://wilds.stanford.edu), Camelyon17 =
৫টা হাসপাতাল; RxRx1 = lab-batch confound) ওপেন ডাউনলোড, Colab-এ চলে। আমাদের
EXP15/16-এর মতো paired design ওখানেই চালানো যায় — তুমি চাইলে আমি
`colab_wilds_pilot.py` হারনেস বানাব। **আপডেট: পাইলট এখন চলে গেছে (EXP17)** —
WILDS Camelyon17-এ hospital-shift collapse রেপ্লিকেট হয়েছে (ID 0.886 → OOD 0.511,
chance), আর তৃতীয় হাসপাতাল null-regime (gap −0.002 → ΔOOD ≈ 0) আইনের সাথে
সামঞ্জস্যপূর্ণ। slide-level EXP18, ট্রান্সফার-ম্যাট্রিক্স EXP18b, প্রোসপেক্টিভ
diagnostic EXP19 (r = +0.88, 10 seeds) এবং দ্বিতীয় ডেটাসেট RxRx1-এর
প্রি-রেজিস্টার্ড রেপ্লিকেশন EXP20 (floor-bound weak replication, unseen-target
Spearman +0.800) সম্পন্ন; তাই "real-data pilot" স্ট্যাটাস =
**COMPLETE, ৪টি ডেটা রেজিম, ২০টি প্রি-রেজিস্টার্ড এক্সপেরিমেন্ট**।

## C. Cold email — পরিচিত হওয়া লাগে না

উপরের টেমপ্লেট পাঠাও: (i) single-cell/OOD-এ সদ্য পাবলিশ করা assistant professors
(ওরা সবচেয়ে দ্রুত রিপ্লাই দেয়), (ii) যেসব ল্যাবের পেপার আমরা LITERATURE_DELTA.md-
তে cite করেছি — তাদের কাছে আমাদের ফল নিজেই গুরুত্বপূর্ণ। প্রতি বার্তায় এক লাইনে
ওদের নিজের পেপারের রেফারেন্স যোগ করো।

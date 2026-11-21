"""Regenerate all result figures from results.csv (tables reproduced in FINDINGS.md)."""
import warnings
warnings.filterwarnings("ignore")
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = list(csv.DictReader(open("results.csv", encoding="utf-8")))


def vals(exp, col="ood"):
    r = [float(row[col]) for row in ROWS if row["exp"] == exp]
    s = [float(row["sd_" + col]) for row in ROWS if row["exp"] == exp]
    n = [row["config"] for row in ROWS if row["exp"] == exp]
    return n, np.array(r), np.array(s)


def bars(ax, exp, title, ylim=(0, 0.9)):
    names, m, s = vals(exp)
    ax.bar(names, m, yerr=s, capsize=4, color="#d1704f")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", labelsize=7)


fig, (a1, a2, a3, a4) = plt.subplots(1, 4, figsize=(16, 4))
bars(a1, "EXP1", "EXP1: premise test (synthetic)")
bars(a2, "EXP6", "EXP6: depth 2x2 on ColoredMNIST")
bars(a3, "EXP10", "EXP10: diversity vs invariance (images)")
bars(a4, "EXP11", "EXP11: active ingredient (synthetic)")
for ax in (a1, a2, a3, a4):
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
fig.tight_layout()
fig.savefig("results_all.png", dpi=160)
print("saved results_all.png")

def realdata_fig():
    import re
    pred, cnn, sd = {}, [], []
    for h in range(5):
        for row in ROWS:
            if row["exp"] == "EXP19" and row["config"] == f"cnn10-target-h{h}":
                cnn.append(float(row["id"]))
                sd.append(float(row["sd_id"]))
                pred[h] = float(re.search(r"pred ([+-][\d.]+)", row["note"]).group(1))
    excess = [c - 0.5 for c in cnn]
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for h in range(5):
        b1.scatter(pred[h], excess[h], s=90, color="#d1704f")
        b1.annotate(f"h{h}", (pred[h], excess[h]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    xs = np.linspace(0, 0.34, 10)
    b1.plot(xs, xs, ls="--", c="gray", lw=0.8)
    b1.set_xlabel("predicted transfer excess (probe matrix)")
    b1.set_ylabel("observed CNN excess (10 seeds)")
    b1.set_title("EXP19: 2-minute probe predicts deep-model OOD\nPearson r = +0.883, Spearman +0.800", fontsize=9)
    b1.set_xlim(-0.03, 0.34)
    b1.set_ylim(-0.10, 0.44)
    b1.axhline(0, ls=":", c="gray", lw=0.8)
    b2.bar([f"h{h}" for h in range(5)], cnn, yerr=sd, capsize=4, color=["#7a9e7e", "#7a9e7e", "#d1704f", "#7a9e7e", "#e0b050"])
    b2.axhline(0.5, ls="--", c="gray", lw=0.8)
    b2.set_ylim(0, 1)
    b2.set_title("Camelyon17-WILDS CNN accuracy by hospital\n(train {h0,h3}; 10 seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig("exp19_realdata.png", dpi=160)
    print("saved exp19_realdata.png")


realdata_fig()

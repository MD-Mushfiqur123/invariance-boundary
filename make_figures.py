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

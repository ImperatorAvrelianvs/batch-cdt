"""Plot the two oracle savings across the original cost-quantile sweep."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.use14corefonts"] = True
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D

base = Path(__file__).resolve().parent
with (base / "thresholds.csv").open() as f:
    rows = list(csv.DictReader(f))
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.55), sharey=True)
cases = [("knapsack", 1, "Knapsack", "#225b81", "o"),
         ("set_cover", 1, "Set cover (T = 1)", "#ac561d", "s"),
         ("set_cover", 10, "Set cover (T = 10)", "#438067", "^")]
for ax, metric, title in zip(axes, ["lp_saved_pct", "losses_saved_pct"],
                             ["Planner LP solves saved", "Loss evaluations saved"]):
    for family, realizations, label, color, marker in cases:
        subset = [r for r in rows if r["family"] == family and int(r["realizations"]) == realizations]
        x = [100*float(r["quantile"]) for r in subset]
        y = [float(r[metric]) for r in subset]
        ax.plot(x, y, color=color, marker=marker, markersize=3.2, lw=1.45, label=label)
        chosen = next(i for i, r in enumerate(subset) if r["chosen"] == "True")
        ax.plot(x[chosen], y[chosen], marker="o", ms=10, mfc="none", mec=color, mew=1.0)
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.set_xlabel("Cost percentile defining the loss threshold")
    ax.set_xticks([10, 30, 50, 70, 90])
    ax.set_ylim(-2, 102)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", lw=.5, alpha=.3)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Reduction from independent stopping")
handles, labels = axes[0].get_legend_handles_labels()
handles.append(Line2D([], [], marker="o", color="#444444", mfc="none", ls="none", ms=8))
labels.append("Published operating point")
fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5,.035), frameon=False, fontsize=9)
fig.text(.5, .01, "T is the number of demand realizations per calibration example.", ha="center", fontsize=8)
fig.subplots_adjust(left=.092, right=.985, bottom=.28, top=.92, wspace=.14)
fig.savefig(base / "threshold_savings.pdf", bbox_inches="tight")
fig.savefig(base / "threshold_savings.png", dpi=160, bbox_inches="tight")

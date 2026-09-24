"""Redraw Chapter 1's validation / rollout / monitoring diagram.

Usage (from the repository root):  python tools/make_rollout_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY, GRAY, INK = "#17324d", "#667085", "#1f2937"
EDGE = {"blue": "#2f6b9a", "teal": "#2f7d78", "amber": "#d68a00", "green": "#4a7c59", "red": "#c24a4a", "slate": "#667085"}
FILL = {"blue": "#eaf1f8", "teal": "#eaf5f3", "amber": "#fff4df", "green": "#eef6ec", "red": "#fbecec", "slate": "#f4f6f8"}


def box(ax, x, y, w, h, text, c, size=13, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                fc=FILL[c], ec=EDGE[c], lw=2.2))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=size, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=size - 3, color=GRAY)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight="bold", color=INK)


def arrow(ax, p, q, color=GRAY, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=color, lw=2,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(15.5, 7.2))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 15.5); ax.set_ylim(0, 7.2); ax.axis("off")
ax.text(0.5, 6.65, "Validation, safe rollout, and monitoring", fontsize=24, weight="bold", color=NAVY)
ax.text(0.5, 6.2, "Each stage answers a different question; rollback stays available at every step.",
        fontsize=14, color=GRAY)

stages = [("Offline\nevaluation", "blue", "Is it better on held-out data?"),
          ("Shadow\ndeployment", "teal", "Does it run correctly on live traffic?"),
          ("Canary\nrollout", "amber", "Is it healthy for a few real users?"),
          ("A/B test", "green", "Does it improve the product?"),
          ("Ramp to\n100%", "green", "Gradual, with guardrails")]
w, h, y, gap = 2.55, 1.35, 4.1, 0.43
for i, (t, c, q) in enumerate(stages):
    x = 0.5 + i * (w + gap)
    box(ax, x, y, w, h, t, c, size=14)
    ax.text(x + w / 2, y - 0.3, q, ha="center", va="center", fontsize=10.5, color=GRAY)
    if i:
        arrow(ax, (x - gap, y + h / 2), (x, y + h / 2))
ax.text(0.5 + 3 * (w + gap) + w / 2, y + h + 0.22, "e.g. 95% control / 5% treatment",
        ha="center", fontsize=10.5, color=EDGE["green"], weight="bold")

box(ax, 8.7, 1.45, 6.3, 1.45, "Monitor production signals", "blue", size=14,
    sub="ranking quality · feature & label drift · engagement · latency · availability")
arrow(ax, (0.5 + 4 * (w + gap) + w / 2, y - 0.5), (0.5 + 4 * (w + gap) + w / 2, 2.9))
box(ax, 5.6, 1.6, 2.4, 1.15, "Threshold\ncrossed?", "amber", size=14)
arrow(ax, (8.7, 2.17), (8.0, 2.17))
box(ax, 0.5, 2.3, 4.3, 1.0, "Roll back to the last stable model", "red", size=12.5,
    sub="when quality or stability degrades")
box(ax, 0.5, 0.95, 4.3, 1.0, "Trigger retraining", "amber", size=12.5,
    sub="when data or product conditions drift")
arrow(ax, (5.6, 2.4), (4.8, 2.8), EDGE["red"])
arrow(ax, (5.6, 1.95), (4.8, 1.45), EDGE["amber"])
ax.text(0.5, 0.35, "Retraining options: fine-tune with regularization · add adapters / new layers · "
        "replay a sample of older data to limit forgetting", fontsize=11.5, color=INK)
fig.savefig("chapters/01_ranking_model/diagrams/08_validation_monitoring.png", dpi=125, facecolor="white")

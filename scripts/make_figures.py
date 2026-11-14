"""Regenerate the paper figures from the files in results/.

Every number plotted is read from a result JSON, so a figure cannot drift away from
the run that produced it. Writes PDFs into paper/figures/.

Style follows the publication guidance in the scientific-visualization and
ieee-figure-table skills: the Okabe-Ito colorblind-safe palette, redundant encoding
by marker and line style so the panels survive grayscale, no text below 6.5 pt at
final size, bold panel labels, and vector output. Body text is serif to match the
IEEEtran column it sits in.

Reads:
  multigold_benchmark.json        -> Fig. 2
  multigold_pool_sweep.json       -> Fig. 3b, 3c
  multigold_instrumentation.json  -> Fig. 3a
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
FIGDIR = PROJECT_ROOT / "paper" / "figures"

# Okabe-Ito, distinguishable under all common forms of colour blindness.
OI = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}
INK = "#1a1a1a"
GREY = "#6b6b6b"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
})
SMALL = 6.5  # floor for any annotation, per the publication guidance


def load(name: str):
    return json.loads((RESULTS / name).read_text())


def tidy(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, lw=0.4, color="#d9d9d9", zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, letter):
    ax.text(-0.17, 1.06, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


# ------------------------------------------------------------------ Fig. 2
def fig_main() -> None:
    """Claim: both diversity-aware rules score below Top-k on held-out repos."""
    d = load("multigold_benchmark.json")
    ceil = d["pool_ceiling"]["test"]

    order = [
        ("Top-$k$", "top_k", OI["blue"], "o"),
        ("MMR\n$\\lambda{=}0.5$", "mmr_lambda0.5", OI["orange"], "s"),
        ("FL\n$\\alpha{=}0.5$", "fl_alpha0.5", OI["vermillion"], "^"),
        ("FL\n$\\alpha{=}0$", "fl_alpha0", OI["purple"], "v"),
        ("Random", "random", GREY, "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    for ax, k, lab in ((axes[0], "k5", "(a)"), (axes[1], "k10", "(b)")):
        fx = d["fixed"][k]
        x = np.arange(len(order))
        rec = [fx[key]["test"]["recall"] for _, key, _, _ in order]
        com = [fx[key]["test"]["recall_compl"] for _, key, _, _ in order]
        cols = [c for _, _, c, _ in order]

        ax.bar(x - 0.19, rec, 0.36, color=cols, zorder=3, label="recall")
        ax.bar(x + 0.19, com, 0.36, color=cols, zorder=3, alpha=0.45,
               hatch="////", edgecolor=INK, linewidth=0.4,
               label="complementary recall")

        ax.axhline(ceil["recall"], ls=(0, (5, 2)), lw=0.8, color=OI["green"], zorder=2)
        # Left-aligned in both panels: the right end is occupied by the legend in (a).
        ax.text(-0.42, ceil["recall"] + 0.008,
                f"pool ceiling {ceil['recall']:.2f}", ha="left", va="bottom",
                fontsize=SMALL, color=OI["green"])

        for xi, (r, c) in enumerate(zip(rec, com)):
            ax.text(xi - 0.19, r + 0.006, f"{r:.2f}", ha="center", va="bottom",
                    fontsize=SMALL)
            ax.text(xi + 0.19, c + 0.006, f"{c:.2f}", ha="center", va="bottom",
                    fontsize=SMALL)

        ax.set_xticks(x)
        ax.set_xticklabels([n for n, _, _, _ in order])
        ax.set_ylabel("Held-out recall")
        ax.set_ylim(0, max(ceil["recall"], max(rec)) * 1.22)
        ax.set_title(f"{lab} budget $k={k[1:]}$, pool $N=30$, $n=84$", loc="left", pad=6)
        tidy(ax)
        if k == "k5":
            ax.legend(frameon=False, loc="upper right", handlelength=1.3,
                      borderpad=0.2, labelspacing=0.3)

    fig.tight_layout(pad=0.4)
    fig.savefig(FIGDIR / "fig2_multigold.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_multigold.pdf")


# ------------------------------------------------------------------ Fig. 3
def fig_mechanism() -> None:
    """Claims: (a) ratio scales with N; (b) deficit grows with N; (c) tuning -> w=1."""
    from matplotlib.ticker import NullFormatter, NullLocator

    instr = load("multigold_instrumentation.json")
    sweep = load("multigold_pool_sweep.json")
    bench = load("multigold_benchmark.json")

    first = sorted([r for r in instr if r["step"] == 1], key=lambda r: r["pool_n"])
    Ns = np.array([r["pool_n"] for r in first], float)
    ratio = np.array([r["ratio_median"] for r in first])

    def log_x_pool(ax):
        """Log pool axis labelled only at the sizes actually run.

        Matplotlib draws decade subdivisions on a log axis by default, and those
        minor labels land on top of the major ones when the range is narrow.
        """
        ax.set_xscale("log")
        ax.set_xlim(12, 260)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([15, 30, 50, 100, 200])
        ax.set_xticklabels(["15", "30", "50", "100", "200"])

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75))

    # ---- (A) ratio vs N, log-log, against the proportional prediction
    ax = axes[0]
    slope = float(np.polyfit(np.log(Ns), np.log(ratio), 1)[0])
    ax.plot(Ns, ratio[0] * (Ns / Ns[0]), ls=(0, (4, 2)), lw=1.0, color=GREY,
            zorder=2, label=r"prediction $\propto N$")
    ax.plot(Ns, ratio, "-o", color=OI["blue"], lw=1.3, ms=3.8, zorder=4,
            label="measured")
    log_x_pool(ax)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([20, 50, 100, 250])
    ax.set_yticklabels(["20", "50", "100", "250"])
    ax.set_ylim(14, 420)
    ax.set_xlabel("Candidate pool size $N$")
    ax.set_ylabel(r"median $\Delta_c/\mathrm{rel}(c)$ at step 1")
    ax.text(0.04, 0.95, f"fitted slope {slope:.2f}", transform=ax.transAxes,
            fontsize=SMALL, va="top", ha="left", color=INK)
    ax.text(0.04, 0.875, "(1.00 is exact proportionality)", transform=ax.transAxes,
            fontsize=SMALL, va="top", ha="left", color=GREY)
    ax.legend(frameon=False, loc="lower right", handlelength=1.7,
              borderpad=0.2, labelspacing=0.25)
    panel_label(ax, "A")
    ax.set_title("coverage outgrows relevance", loc="left", pad=5, color=GREY,
                 fontsize=7.5)
    tidy(ax, grid_axis="both")

    # ---- (B) recall deficit vs N at the conventional weight
    ax = axes[1]
    series = (
        ("fl_alpha0.5_vs_topk", 5, r"FL $\alpha{=}0.5$, $k{=}5$", OI["vermillion"], "^", "-"),
        ("fl_alpha0.5_vs_topk", 10, r"FL $\alpha{=}0.5$, $k{=}10$", OI["vermillion"], "^", (0, (1, 1.6))),
        ("mmr_lambda0.5_vs_topk", 5, r"MMR $\lambda{=}0.5$, $k{=}5$", OI["orange"], "s", (0, (4, 2))),
        ("mmr_lambda0.5_vs_topk", 10, r"MMR $\lambda{=}0.5$, $k{=}10$", OI["orange"], "s", (0, (1, 1.6))),
    )
    for key, k, lab, col, mk, ls in series:
        rows = sorted([e for e in sweep if e["k"] == k], key=lambda e: e["pool_n"])
        ax.plot([e["pool_n"] for e in rows],
                [e[key]["recall"]["mean_diff"] for e in rows],
                ls=ls, marker=mk, color=col, lw=1.2, ms=3.4,
                mfc="white" if k == 10 else col, mew=1.0, zorder=4, label=lab)
    ax.axhline(0.0, lw=0.9, color=INK, zorder=3)
    log_x_pool(ax)
    ax.set_ylim(-0.265, 0.055)
    ax.text(0.98, 0.985, "Top-$k$ baseline", transform=ax.transAxes,
            fontsize=SMALL, ha="right", va="top", color=INK)
    ax.set_xlabel("Candidate pool size $N$")
    ax.set_ylabel(r"$\Delta$ recall vs Top-$k$ (held out)")
    ax.legend(frameon=False, loc="lower left", handlelength=2.0, borderpad=0.2,
              labelspacing=0.22, bbox_to_anchor=(-0.01, -0.02))
    panel_label(ax, "B")
    ax.set_title("so the deficit widens", loc="left", pad=5, color=GREY, fontsize=7.5)
    tidy(ax)

    # ---- (C) weight sweep on the training repositories
    ax = axes[2]
    for key, lab, col, mk, ls in (
        ("fl_alpha", r"facility location, $\alpha$", OI["vermillion"], "^", "-"),
        ("mmr_lambda", r"MMR, $\lambda$", OI["orange"], "s", (0, (4, 2))),
    ):
        rows = bench["sweeps"]["k5"][key]
        ax.plot([r["w"] for r in rows], [r["recall"] for r in rows],
                ls=ls, marker=mk, color=col, lw=1.2, ms=3.6, zorder=4, label=lab)
    tk = bench["fixed"]["k5"]["top_k"]["train"]["recall"]
    ax.axhline(tk, ls=(0, (1, 2)), lw=1.0, color=OI["blue"], zorder=3)
    ax.scatter([1.0], [tk], s=55, facecolor="none", edgecolor=OI["green"],
               linewidth=1.3, zorder=6)
    ax.set_xlim(-0.05, 1.18)
    ax.set_ylim(0.118, 0.288)
    # Top-k line labelled above its own line at the left, clear of the curves.
    ax.text(0.0, tk + 0.004, f"Top-$k$ {tk:.3f}", fontsize=SMALL,
            ha="left", va="bottom", color=OI["blue"])
    # The selected weight is called out in the empty lower-right quadrant, with no
    # leader line, because any arrow to the marker crosses both curves.
    ax.text(1.14, 0.133, "selected on train:\n$\\alpha^{*}=\\lambda^{*}=1.0$",
            fontsize=SMALL, ha="right", va="bottom", color=OI["green"],
            linespacing=1.4)
    ax.set_xlabel("Weight on the relevance term")
    ax.set_ylabel("Training recall, $k=5$")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, loc="upper left", handlelength=2.0, borderpad=0.2,
              labelspacing=0.25, bbox_to_anchor=(-0.02, 0.69))
    panel_label(ax, "C")
    ax.set_title("and tuning removes the term", loc="left", pad=5, color=GREY,
                 fontsize=7.5)
    tidy(ax)

    fig.tight_layout(pad=0.5, w_pad=1.6)
    fig.savefig(FIGDIR / "fig3_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig3_mechanism.pdf (fitted log-log slope {slope:.3f})")


# ------------------------------------------------------------------ Fig. 1
def fig_pipeline() -> None:
    """Schematic of the shared pipeline. Hand-laid out; no data dependency.

    Geometry notes, since the coordinates are absolute: boxes are sized so each
    inter-box gap is wide enough for its edge label at 6.5 pt, and the Top-k bypass
    arc is routed below the boxes with its own clear band underneath so the arc and
    its caption cannot cross.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.16, 2.35))
    # the rounded boxstyle adds 0.35 of padding outside the nominal rectangle, so the
    # limits carry a margin or the first and last boxes lose their outer border
    ax.set_xlim(-1.0, 101.0); ax.set_ylim(0, 33); ax.axis("off")

    BOX_Y, BOX_H = 11.0, 18.0          # boxes occupy y 11..29
    MID = BOX_Y + BOX_H / 2            # arrow height between boxes
    BYPASS_Y = 9.0                     # arc leaves from just under the boxes
    CAPTION_Y = 0.6                    # arc dips to about y=4, caption sits below

    # x extents chosen so every gap is 5 units wide
    boxes = [
        (0.0, 16.0, "Corpus and question",
         "3502 definitions from\n12 Python libraries\n\n315 questions,\nmean 4.52 gold each"),
        (21.0, 15.0, "Dense retrieval",
         "all-mpnet-base-v2\nL2-normalized\n\nper-repository\ncosine search"),
        (41.0, 19.0, "Selection",
         "graph over the pool,\nedges carry sim$(i,j)$\n\nscore$(c\\mid S) =$\n"
         "$(1-\\alpha)\\Delta_c + \\alpha\\,$rel$(c)$"),
        (65.0, 14.0, "Generation",
         "Gemini 3.6 Flash\n$T = 0.1$\n\n(pilot only,\nSec. V-F)"),
        (84.0, 16.0, "Scoring",
         "recall\nanchor recall\ncomplementary\nrecall"),
    ]

    for i, (x, w, title, body) in enumerate(boxes):
        highlight = (i == 2)
        ax.add_patch(FancyBboxPatch(
            (x, BOX_Y), w, BOX_H, boxstyle="round,pad=0.35,rounding_size=1.1",
            facecolor="#eaf3fb" if highlight else "#f7fafc",
            edgecolor=OI["blue"] if highlight else INK,
            linewidth=0.9 if highlight else 0.7, zorder=3))
        ax.text(x + w / 2, BOX_Y + BOX_H - 2.2, title, ha="center", va="top",
                fontsize=7.4, weight="bold", zorder=4)
        ax.text(x + w / 2, BOX_Y + BOX_H - 6.0, body, ha="center", va="top",
                fontsize=SMALL, linespacing=1.45, color="#333333", zorder=4)

    # single-symbol edge labels; the sizes are given in the caption, which keeps the
    # labels inside their 5-unit gaps instead of running under the box borders
    for i, lab in enumerate(["$q$", "$C$", "$S$", ""]):
        x_end = boxes[i][0] + boxes[i][1]
        x_start = boxes[i + 1][0]
        ax.add_patch(FancyArrowPatch(
            (x_end + 0.4, MID), (x_start - 0.4, MID), arrowstyle="-|>",
            mutation_scale=7, linewidth=0.7, color=INK, zorder=5,
            shrinkA=0, shrinkB=0))
        if lab:
            ax.text((x_end + x_start) / 2, MID + 1.0, lab, ha="center", va="bottom",
                    fontsize=7, color=INK, zorder=5)

    # Top-k bypass: leaves under "Dense retrieval", lands under "Scoring"
    ax.add_patch(FancyArrowPatch(
        (28.0, BYPASS_Y), (92.0, BYPASS_Y), arrowstyle="-|>", mutation_scale=7,
        linewidth=0.7, color=GREY, linestyle=(0, (3, 2)), zorder=2,
        connectionstyle="arc3,rad=0.16"))
    ax.text(60.0, CAPTION_Y,
            "Top-$k$ bypasses selection: the $k$ best-ranked passages go straight "
            "to the generator", ha="center", va="bottom", fontsize=SMALL,
            color=GREY, style="italic", zorder=5)

    ax.text(50.0, 30.4,
            "MMR, facility location and the coverage-only ablation all select from "
            "the same pool $C$ of $N$ candidates, and every rule passes $k$ passages on",
            ha="center", va="bottom", fontsize=SMALL, color="#444444", zorder=5)

    fig.tight_layout(pad=0.2)
    fig.savefig(FIGDIR / "fig1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_pipeline.pdf")


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_main()
    fig_mechanism()

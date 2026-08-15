"""The three figures.

Pure matplotlib over numpy inputs. No GPU, no model, no network: everything
here must render from `artifacts/` on a laptop, because that is what a judge
does with the Colab.

Colours are chosen to survive greyscale printing and the common forms of
colour blindness: the cascade segments differ in lightness as well as hue, and
nothing is distinguished by red-versus-green alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURVIVE = "#1b3a6b"      # darkest: the mass that makes it all the way through
LOSS_F1 = "#c96a1f"      # representation failure
LOSS_F2 = "#d9a441"      # verbalizability failure
LOSS_F3 = "#8fb3d9"      # channel closure
GRID = "#d6d6d6"
INK = "#222222"


def _tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(True, alpha=0.25, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def cascade_bar(f1: float, f2: float, f3: float, n_entering: int,
                title: str = "") -> plt.Figure:
    """Figure 1 -- the poster. 100% decomposed into three losses and a survivor.

    Widths are the actual mass lost at each stage, so the segments sum to 100
    by construction: (1-f1) + f1(1-f2) + f1f2(1-f3) + f1f2f3 = 1.
    """
    lost_1 = (1 - f1) * 100
    lost_2 = f1 * (1 - f2) * 100
    lost_3 = f1 * f2 * (1 - f3) * 100
    survives = f1 * f2 * f3 * 100

    fig, ax = plt.subplots(figsize=(10, 3.1))
    left = 0.0
    segments = [
        (lost_1, LOSS_F1, "representation failure", f"1 − f₁ = {1 - f1:.3f}"),
        (lost_2, LOSS_F2, "verbalizability failure", f"f₁(1 − f₂) = {f1 * (1 - f2):.3f}"),
        (lost_3, LOSS_F3, "channel closure", f"f₁f₂(1 − f₃) = {f1 * f2 * (1 - f3):.3f}"),
        (survives, SURVIVE, "reported", f"f₁f₂f₃ = {f1 * f2 * f3:.3f}"),
    ]
    for width, colour, _label, _detail in segments:
        if width <= 0:
            continue
        ax.barh([0], [width], left=left, color=colour, edgecolor="white",
                linewidth=1.4, height=0.55)
        if width > 6:
            ax.text(left + width / 2, 0, f"{width:.1f}%", ha="center",
                    va="center", color="white" if colour in (SURVIVE, LOSS_F1) else INK,
                    fontsize=11, fontweight="bold")
        left += width

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _, _ in segments]
    labels = [f"{lab}  ({det})" for _, _, lab, det in segments]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, fontsize=9.5)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("% of injection trials", fontsize=11)
    ax.set_title(title or "Where the injected concept is lost", fontsize=13,
                 color=INK, pad=12)
    _tidy(ax)
    ax.text(0.995, 1.16, f"n = {n_entering} injection trials", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#666666")
    fig.tight_layout()
    return fig


def order_contrast(per_order: dict, differences: dict,
                   title: str = "") -> plt.Figure:
    """Figure 2 -- the same cascade under both orders, plus a CI on the DIFFERENCE.

    `per_order`: {order: {"f1":..,"f2":..,"f3":..}}
    `differences`: {"f1": (point, lo, hi), ...} for order_a - order_b

    Two overlapping marginal intervals do not settle whether a difference
    straddles zero, which is why the right panel plots the difference itself.
    """
    orders = list(per_order)
    names = ["f1", "f2", "f3"]
    pretty = ["f₁", "f₂", "f₃"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})

    x = np.arange(len(names))
    width = 0.36
    for i, order in enumerate(orders):
        values = [per_order[order][n] for n in names]
        ax.bar(x + (i - 0.5) * width, values, width,
               label=order.replace("_", " "),
               color=[SURVIVE, LOSS_F2][i % 2], edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=12)
    ax.set_ylabel("conditional rate", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title("cascade by order", fontsize=11, color=INK)
    _tidy(ax)

    points = [differences[n][0] for n in names]
    lows = [differences[n][0] - differences[n][1] for n in names]
    highs = [differences[n][2] - differences[n][0] for n in names]
    ax2.errorbar(points, x, xerr=[lows, highs], fmt="o", color=SURVIVE,
                 capsize=4, markersize=7, linewidth=2)
    ax2.axvline(0, color="#999999", linewidth=1.2, linestyle="--")
    ax2.set_yticks(x)
    ax2.set_yticklabels(pretty, fontsize=12)
    ax2.invert_yaxis()
    label_a = orders[0].replace("_", " ")
    label_b = orders[1].replace("_", " ") if len(orders) > 1 else "?"
    ax2.set_xlabel(f"difference  ({label_a} − {label_b})", fontsize=10)
    ax2.set_title("CI on the difference, not two marginals", fontsize=11, color=INK)
    _tidy(ax2)

    fig.suptitle(title or "Order contrast", fontsize=13, color=INK)
    fig.tight_layout()
    return fig


def f2_sensitivity(ks, f2_observed, null_lo, null_hi, null_median,
                   title: str = "") -> plt.Figure:
    """Figure 3 -- f₂ over k with the matched-norm random-vector null band.

    f₂ rises monotonically with k by construction, so a scalar f₂ is not a
    reportable quantity. Cosine has no absolute scale at d=5120 either, so the
    only meaningful reading is the distance from this null band.
    """
    ks = np.asarray(ks, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(ks, null_lo, null_hi, color=LOSS_F3, alpha=0.45,
                    label="random-vector null band (95%)")
    ax.plot(ks, null_median, color="#5b7fa6", linewidth=1.6, linestyle="--",
            label="null median")
    ax.plot(ks, f2_observed, color=SURVIVE, marker="o", linewidth=2.4,
            markersize=7, label="f₂ observed")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(int(k)) for k in ks])
    ax.set_xlabel("k  (concept token inside top-k of the lens readout)", fontsize=11)
    ax.set_ylabel("f₂", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title(title or "f₂ is a curve, not a scalar", fontsize=13, color=INK)
    _tidy(ax)
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, path) -> list:
    """Write PNG and PDF beside each other; return what was written."""
    from pathlib import Path

    path = Path(path)
    written = []
    for suffix in (".png", ".pdf"):
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=200, bbox_inches="tight",
                    facecolor="white")
        written.append(target)
    plt.close(fig)
    return written

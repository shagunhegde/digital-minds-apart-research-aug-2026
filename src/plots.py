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


def _rate(value: float, entered: int, expr: str) -> str:
    """A conditional rate for a legend, or why there isn't one.

    A conditional whose denominator is empty is undefined, not zero, and
    printing `nan` in a legend invites a reader to treat it as a measured
    null. Say what happened instead.
    """
    if entered == 0:
        return "n/a (0 entered)"
    if not np.isfinite(value):
        return "n/a (undefined)"
    return f"{expr} = {value:.3f}"


def cascade_bar(f1: float, f2: float, f3: float, n_entering: int,
                title: str = "", subtitle: str = "",
                n_surviving_f1: int | None = None,
                n_surviving_f2: int | None = None) -> plt.Figure:
    """Figure 1 -- the poster. 100% decomposed into three losses and a survivor.

    Widths are the actual mass lost at each stage, so the segments sum to 100
    by construction: (1-f1) + f1(1-f2) + f1f2(1-f3) + f1f2f3 = 1.

    `subtitle` names the f2 slot. It is not decoration: "f2 at the naming slot
    read from the output distribution" and "f2 at the boolean slot read
    through the J-lens at L59" are different measurements of different things,
    and a figure that does not say which invites the wrong one to be quoted.
    """
    f2_defined = np.isfinite(f2)
    f3_defined = np.isfinite(f3)
    lost_1 = (1 - f1) * 100
    lost_2 = f1 * (1 - f2) * 100 if f2_defined else 0.0
    lost_3 = f1 * f2 * (1 - f3) * 100 if f2_defined and f3_defined else 0.0
    survives = f1 * f2 * f3 * 100 if f2_defined and f3_defined else 0.0

    fig, ax = plt.subplots(figsize=(10, 3.4))
    left = 0.0
    entered_2 = (n_surviving_f1 if n_surviving_f1 is not None
                 else int(round(f1 * n_entering)))
    entered_3 = (n_surviving_f2 if n_surviving_f2 is not None
                 else int(round(f1 * f2 * n_entering)) if f2_defined else 0)
    segments = [
        (lost_1, LOSS_F1, "representation failure", f"1 − f₁ = {1 - f1:.3f}"),
        (lost_2, LOSS_F2, "verbalizability failure",
         _rate(f1 * (1 - f2) if f2_defined else float("nan"), entered_2,
               "f₁(1 − f₂)")),
        (lost_3, LOSS_F3, "channel closure",
         _rate(f1 * f2 * (1 - f3) if f2_defined and f3_defined else float("nan"),
               entered_3, "f₁f₂(1 − f₃)")),
        (survives, SURVIVE, "reported",
         _rate(f1 * f2 * f3 if f2_defined and f3_defined else float("nan"),
               entered_3, "f₁f₂f₃")),
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
    # Title, subtitle and the n annotation used to be stacked into the same
    # ~0.16 axes-fraction of headroom and overlapped. The title now owns the
    # figure's suptitle slot, the subtitle sits under it, and n moves inside
    # the axes.
    ax.set_title("", pad=2)
    fig.suptitle(title or "Where the injected concept is lost", fontsize=13,
                 color=INK, y=0.99)
    if subtitle:
        fig.text(0.5, 0.905, subtitle, ha="center", va="top", fontsize=9.5,
                 color="#555555")
    _tidy(ax)
    ax.text(0.995, 0.97, f"n = {n_entering} injection trials",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 0.88 if subtitle else 0.93))
    return fig


def order_contrast(per_order: dict, differences: dict,
                   title: str = "", subtitle: str = "") -> plt.Figure:
    """Figure 2 -- the same cascade under both orders, plus a CI on the DIFFERENCE.

    `per_order`: {order: {"f1":..,"f2":..,"f3":.., "n_surviving_f1":.., ...}}
    `differences`: {"f1": (point, lo, hi), ...} for order_a - order_b

    Two overlapping marginal intervals do not settle whether a difference
    straddles zero, which is why the right panel plots the difference itself.

    A factor whose denominator is empty in either order is DROPPED from both
    panels and named in a footnote. Plotting it as 0.00 with no interval reads
    as a measured null -- "the model never verbalises it" -- when what
    happened is that nothing reached the stage and the conditional is
    undefined.
    """
    orders = list(per_order)
    names = ["f1", "f2", "f3"]
    pretty = {"f1": "f₁", "f2": "f₂", "f3": "f₃"}
    denominators = {"f2": "n_surviving_f1", "f3": "n_surviving_f2"}

    def defined(name: str) -> bool:
        for order in orders:
            row = per_order[order]
            if not np.isfinite(row.get(name, float("nan"))):
                return False
            key = denominators.get(name)
            if key is not None and row.get(key, 1) == 0:
                return False
        return True

    shown = [n for n in names if defined(n)]
    dropped = [n for n in names if n not in shown]
    if not shown:                       # nothing survived anywhere
        shown, dropped = ["f1"], [n for n in names if n != "f1"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})

    x = np.arange(len(shown))
    width = 0.36
    for i, order in enumerate(orders):
        values = [per_order[order][n] for n in shown]
        ax.bar(x + (i - 0.5) * width, values, width,
               label=order.replace("_", " "),
               color=[SURVIVE, LOSS_F2][i % 2], edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[n] for n in shown], fontsize=12)
    ax.set_ylabel("conditional rate", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title("cascade by order", fontsize=11, color=INK)
    _tidy(ax)

    points = [differences[n][0] for n in shown]
    lows = [differences[n][0] - differences[n][1] for n in shown]
    highs = [differences[n][2] - differences[n][0] for n in shown]
    ax2.errorbar(points, x, xerr=[lows, highs], fmt="o", color=SURVIVE,
                 capsize=4, markersize=7, linewidth=2)
    ax2.axvline(0, color="#999999", linewidth=1.2, linestyle="--")
    ax2.set_yticks(x)
    ax2.set_yticklabels([pretty[n] for n in shown], fontsize=12)
    ax2.invert_yaxis()
    label_a = orders[0].replace("_", " ")
    label_b = orders[1].replace("_", " ") if len(orders) > 1 else "?"
    ax2.set_xlabel(f"difference  ({label_a} − {label_b})", fontsize=10)
    ax2.set_title("CI on the difference, not two marginals", fontsize=11, color=INK)
    _tidy(ax2)

    fig.suptitle(title or "Order contrast", fontsize=13, color=INK, y=0.99)
    note = subtitle
    if dropped:
        missing = ", ".join(pretty[n] for n in dropped)
        reason = (f"{missing} omitted: no trials reached the stage in at "
                  f"least one order, so the conditional is undefined")
        note = f"{note}  ·  {reason}" if note else reason
    if note:
        fig.text(0.5, 0.905, note, ha="center", va="top", fontsize=9,
                 color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.88 if note else 0.94))
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

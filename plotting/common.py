"""Shared loading and styling for the MUSHRA results plotting scripts."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

FILENAME_RE = re.compile(r"^(R\d+T\d+)_([^_]+)_([^_]+)\.wav$")

# Panel title used when ratings are pooled across excitation signals
ALL_SIGNALS = None
# Display names for the excitation signal names used in the stimulus filenames
SIGNAL_NAMES = {"Beet": "Classical", "Hand": "Handclaps", "Slow": "Pop"}

CONDITIONS = ["target", "fulloptim", "geq", "passive"]
# Display names for the condition names used in the stimulus filenames
CONDITION_LABELS = {"target": "Target", "fulloptim": "Full Optim.", "geq": "Graphic EQ", "passive": "Passive"}
# Sampled from matplotlib's magma colormap at np.linspace(0.15, 0.8, 4)
CONDITION_COLOURS = {
    "target": "#251255",
    "fulloptim": "#7e2482",
    "geq": "#d8456c",
    "passive": "#fe9f6d",
}
CONDITION_MARKERS = {"target": "o", "fulloptim": "s", "geq": "D", "passive": "^"}
GRID_COLOUR = "#e5e4df"

# Figure sizes for an IEEE Transactions paper (IEEEtran journal layout). IEEE asks for 8-10 pt text in
# figures. Every figure is laid out on the twocolumn canvas, so both sizes have the same aspect ratio,
# markers and line widths; the onecolumn figure only has larger fonts and fewer pixels, so that when it
# is scaled down to the column width its fonts appear at font_size and its resolution is dpi. Include
# the figures in LaTeX with width=\columnwidth (onecolumn) or width=\textwidth (twocolumn).
#   onecolumn: fits one column (\columnwidth = 21pc), for a figure environment
#   twocolumn: spans both columns (\textwidth = 43pc), for a figure* environment
FIGURE_SIZES = {
    "onecolumn": {"width": 252 / 72.27, "font_size": 8, "small_font_size": 8, "dpi": 800},
    "twocolumn": {"width": 516 / 72.27, "font_size": 9, "small_font_size": 8, "dpi": 800},
}
CANVAS_WIDTH = FIGURE_SIZES["twocolumn"]["width"]


def apply_figure_style(figure_size):
    """Set the matplotlib style (LaTeX text in Times, font sizes, DPI) for a figure size."""
    spec = FIGURE_SIZES[figure_size]
    # How much the figure is scaled when included in the paper
    scale = spec["width"] / CANVAS_WIDTH
    font_size = spec["font_size"] / scale
    small_font_size = spec["small_font_size"] / scale
    plt.rcParams.update({
        "text.usetex": True,
        # Times text and maths via newtx, matching the IEEEtran body font
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
        "font.family": "serif",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": small_font_size,
        "ytick.labelsize": small_font_size,
        "legend.fontsize": small_font_size,
        "legend.title_fontsize": font_size,
        "figure.labelsize": font_size,
        "figure.titlesize": font_size,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "savefig.dpi": spec["dpi"] * scale,
        "figure.dpi": 150,
    })
    return spec


def figure_dimensions(height):
    """Return the (width, height) in inches of the canvas every figure is laid out on."""
    return CANVAS_WIDTH, height


def add_figure_size_arg(parser, default):
    parser.add_argument("--figure-size", default=default, choices=list(FIGURE_SIZES),
                        help="Size the figure (and its font sizes and DPI) to fit one column (onecolumn) "
                             "or span both columns (twocolumn) of an IEEE Transactions paper "
                             "(default: %(default)s)")


def set_combination_ticks(ax, x, combinations, figure_size):
    """Label the x-axis with the passive/target combinations, rotated if they would overlap."""
    ax.set_xticks(x)
    ax.set_xticklabels(combinations, rotation=90 if figure_size == "onecolumn" and len(combinations) > 8 else 0)


def add_legend(fig, handles, labels, title="Condition", top=False):
    """Add a figure legend to the right of the axes, or above them on a single row if top is True.

    The figure must use the constrained layout so the legend's space is reserved.
    """
    if top:
        return fig.legend(handles, labels, loc="outside upper center", ncols=len(handles), frameon=False)
    return fig.legend(handles, labels, title=title, loc="outside right upper", frameon=False)


def load_ratings(paths, include_repeats, split_signals=True):
    """Return {signal: {combination: {condition: [ratings]}}} pooled over the given results CSVs.

    If split_signals is False, all excitation signals are pooled under ALL_SIGNALS.
    """
    ratings = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if not include_repeats and row["repeat"] == "1":
                    continue
                match = FILENAME_RE.match(row["filename"])
                if match is None:
                    print(f"Skipping unrecognised filename {row['filename']} in {path.name}")
                    continue
                combination, signal, condition = match.groups()
                key = signal if split_signals else ALL_SIGNALS
                ratings[key][combination][condition].append(float(row["rating"]))
    return ratings


def mean_ci95(values):
    values = np.asarray(values)
    mean = values.mean()
    if len(values) < 2:
        return mean, 0.0
    half_width = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return mean, half_width


def combination_sort_key(combination):
    r, t = map(int, re.findall(r"\d+", combination))
    return r, t


def plot_layout(ratings):
    """Return (signals, combinations, conditions) in display order."""
    signals = sorted(ratings)
    combinations = sorted({k for s in signals for k in ratings[s]}, key=combination_sort_key)
    present = {c for s in signals for k in ratings[s] for c in ratings[s][k]}
    conditions = [c for c in CONDITIONS if c in present] + sorted(present - set(CONDITIONS))
    return signals, combinations, conditions


def style_axis(ax, signal, ylabel):
    ax.set_title(SIGNAL_NAMES.get(signal, signal), loc="left")
    # Leave headroom so points/CIs at the ends of the 0-100 scale aren't clipped
    ax.set_ylim(-10, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def add_common_args(parser, results_dir, include_repeats, split_signals, figure_size):
    """Add the arguments shared by all results plotting scripts, using the given defaults."""
    add_figure_size_arg(parser, figure_size)
    parser.add_argument("--results-dir", default=str(results_dir),
                        help="Folder containing the results CSVs (default: %(default)s)")
    parser.add_argument("--include-repeats", default=include_repeats, action=argparse.BooleanOptionalAction,
                        help="Include repeated trials (repeat=1) (default: %(default)s)")
    parser.add_argument("--split-signals", default=split_signals, action=argparse.BooleanOptionalAction,
                        help="One panel per excitation signal; otherwise all signals are pooled "
                             "into a single panel (default: %(default)s)")

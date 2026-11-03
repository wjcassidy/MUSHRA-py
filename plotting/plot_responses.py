"""Plot every individual rating for each passive/target combination (e.g. R1T5)
in the MUSHRA results CSVs, as jittered points per condition with a median line.

Defaults are set in the settings block below; command-line arguments override them.

Usage:
    python plotting/plot_responses.py [--results-dir DIR] [--[no-]include-repeats]
                                      [--[no-]split-signals] [--figure-size {onecolumn,twocolumn}]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (
    CONDITION_COLOURS, CONDITION_LABELS, CONDITION_MARKERS, DEFAULT_RESULTS_DIR, GRID_COLOUR,
    add_common_args, add_legend, set_combination_ticks, apply_figure_style, figure_dimensions, load_ratings, plot_layout, style_axis,
)
from exclusion_criteria import included_results

# Settings (defaults for the command-line arguments)
RESULTS_DIR = DEFAULT_RESULTS_DIR             # Folder containing the results CSVs
INCLUDE_REPEATS = False                       # Include repeated trials (repeat=1)
SPLIT_SIGNALS = False                          # One panel per excitation signal; False pools all signals
FIGURE_SIZE = "twocolumn"                     # "onecolumn" or "twocolumn" (IEEE Transactions)
PANEL_HEIGHT = 2.3                            # Height of each panel at twocolumn width (inches)


def plot(ratings, figure_size=FIGURE_SIZE):
    apply_figure_style(figure_size)
    signals, combinations, conditions = plot_layout(ratings)

    x = np.arange(len(combinations))
    offset_step = 0.2
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2) * offset_step
    jitter_width = 0.06
    median_half_width = 0.08
    # Fixed seed so the jitter is the same every time the plot is regenerated
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(
        len(signals), 1, sharex=True, squeeze=False, layout="constrained",
        figsize=figure_dimensions(PANEL_HEIGHT * len(signals) + 0.5),
    )
    axes = axes[:, 0]
    for ax, signal in zip(axes, signals):
        for condition, offset in zip(conditions, offsets):
            colour = CONDITION_COLOURS.get(condition)
            xs, ys = [], []
            for i, combination in enumerate(combinations):
                values = ratings[signal][combination].get(condition)
                if not values:
                    continue
                centre = x[i] + offset
                xs.extend(centre + rng.uniform(-jitter_width, jitter_width, len(values)))
                ys.extend(values)
                median = np.median(values)
                ax.plot([centre - median_half_width, centre + median_half_width], [median, median],
                        color=colour, linewidth=1.2, zorder=3)
            ax.scatter(
                xs, ys,
                marker=CONDITION_MARKERS.get(condition, "o"),
                color=colour, alpha=0.55, s=6, linewidths=0,
                label=CONDITION_LABELS.get(condition, condition), zorder=2,
            )

        for boundary in x[:-1] + 0.5:
            ax.axvline(boundary, color=GRID_COLOUR, linewidth=0.8, zorder=0)

        style_axis(ax, signal, "Rating (line = median)")

    set_combination_ticks(axes[-1], x, combinations, figure_size)
    axes[-1].set_xlim(-0.5, len(combinations) - 0.5)
    axes[-1].set_xlabel("Installation (R) and target (T) room combination")
    legend = add_legend(fig, *axes[0].get_legend_handles_labels())
    for handle in legend.legend_handles:
        handle.set_alpha(1)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, RESULTS_DIR, INCLUDE_REPEATS, SPLIT_SIGNALS, FIGURE_SIZE)
    args = parser.parse_args()

    results = included_results(args.results_dir)
    ratings = load_ratings(results, args.include_repeats, args.split_signals)
    if not ratings:
        raise SystemExit(f"No results found in {args.results_dir}")
    plot(ratings, args.figure_size)


if __name__ == "__main__":
    main()

"""Plot mean ratings with 95% confidence intervals for each passive/target
combination (e.g. R1T5) in the MUSHRA results CSVs.

Defaults are set in the settings block below; command-line arguments override them.

Usage:
    python plotting/plot_means_and_confidence.py [--results-dir DIR] [--[no-]include-repeats]
                                                 [--[no-]split-signals] [--[no-]error-bars]
                                                 [--figure-size {onecolumn,twocolumn}]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (
    CONDITION_COLOURS, CONDITION_LABELS, CONDITION_MARKERS, DEFAULT_RESULTS_DIR, GRID_COLOUR,
    add_common_args, add_legend, set_combination_ticks, apply_figure_style, figure_dimensions, load_ratings, mean_ci95, plot_layout,
    style_axis,
)
from exclusion_criteria import included_results

# Settings (defaults for the command-line arguments)
RESULTS_DIR = DEFAULT_RESULTS_DIR             # Folder containing the results CSVs
INCLUDE_REPEATS = False                       # Include repeated trials (repeat=1)
SPLIT_SIGNALS = False                         # One panel per excitation signal; False pools all signals
SHOW_ERROR_BARS = True                        # Draw 95% confidence interval error bars around the means
FIGURE_SIZE = "twocolumn"                     # "onecolumn" or "twocolumn" (IEEE Transactions)
PANEL_HEIGHT = 1.7                            # Height of each panel at twocolumn width (inches)
SORT_CONDITION = "fulloptim"                  # Order combinations by decreasing mean rating of this condition
RATING_LABELS = [                             # Y-axis labels, top (100) to bottom (0), equally spaced
    "Sounds the same", "Very similar", "Similar", "Somewhat similar", "Not at all similar",
]


def sort_by_condition_mean(ratings, signals, combinations, condition):
    """Return the combinations ordered by decreasing mean rating of the given condition, pooled over
    all signals so that every panel shares the same order. Combinations without that condition go last.
    """
    def key(combination):
        values = [v for s in signals for v in ratings[s][combination].get(condition, [])]
        return -np.mean(values) if values else np.inf

    return sorted(combinations, key=key)


def plot(ratings, show_error_bars=True, figure_size=FIGURE_SIZE):
    apply_figure_style(figure_size)
    signals, combinations, conditions = plot_layout(ratings)
    combinations = sort_by_condition_mean(ratings, signals, combinations, SORT_CONDITION)

    x =np.arange(len(combinations))
    offset_step = 0.18
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2) * offset_step

    fig, axes = plt.subplots(
        len(signals), 1, sharex=True, squeeze=False, layout="constrained",
        figsize=figure_dimensions(PANEL_HEIGHT * len(signals) + 0.5),
    )
    axes = axes[:, 0]
    for ax, signal in zip(axes, signals):
        for condition, offset in zip(conditions, offsets):
            means, errors, xs = [], [], []
            for i, combination in enumerate(combinations):
                values = ratings[signal][combination].get(condition)
                if not values:
                    continue
                mean, half_width = mean_ci95(values)
                xs.append(x[i] + offset)
                means.append(mean)
                errors.append(half_width)
            ax.errorbar(
                xs, means, yerr=errors if show_error_bars else None,
                fmt=CONDITION_MARKERS.get(condition, "o"),
                color=CONDITION_COLOURS.get(condition),
                markersize=4, capsize=2, elinewidth=1, capthick=1,
                label=CONDITION_LABELS.get(condition, condition),
            )

        for boundary in x[:-1] + 0.5:
            ax.axvline(boundary, color=GRID_COLOUR, linewidth=0.8, zorder=0)

        style_axis(ax, signal, r"Rating (mean $\pm$ 95\% CI)" if show_error_bars else "Rating (mean)")
        ax.set_yticks(np.linspace(0, 100, len(RATING_LABELS)), RATING_LABELS[::-1])

    set_combination_ticks(axes[-1], x, combinations, figure_size)
    axes[-1].set_xlim(-0.5, len(combinations) - 0.5)
    axes[-1].set_ylim(0, 105)
    axes[-1].set_xlabel("Installation (R) and target (T) room combination")
    add_legend(fig, *axes[0].get_legend_handles_labels(), top=True)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, RESULTS_DIR, INCLUDE_REPEATS, SPLIT_SIGNALS, FIGURE_SIZE)
    parser.add_argument("--error-bars", dest="show_error_bars", default=SHOW_ERROR_BARS,
                        action=argparse.BooleanOptionalAction,
                        help="Draw 95%% confidence interval error bars around the means (default: %(default)s)")
    args = parser.parse_args()

    results = included_results(args.results_dir)
    ratings = load_ratings(results, args.include_repeats, args.split_signals)
    if not ratings:
        raise SystemExit(f"No results found in {args.results_dir}")
    plot(ratings, args.show_error_bars, args.figure_size)


if __name__ == "__main__":
    main()

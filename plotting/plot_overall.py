"""Plot the overall comparison of conditions, pooling the ratings over all
passive/target combinations (e.g. R1T5) in the MUSHRA results CSVs.

Shows the rating distribution for each condition (as the individual responses, a
violin or a box plot), with the mean (and optionally the 95% confidence interval)
marked, grouped by excitation signal.

Defaults are set in the settings block below; command-line arguments override them.

Usage:
    python plotting/plot_overall.py [--results-dir DIR] [--[no-]include-repeats]
                                    [--[no-]split-signals] [--[no-]error-bars]
                                    [--distribution {points,violin,box}]
                                    [--figure-size {onecolumn,twocolumn}] [--output FILE]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (
    ALL_SIGNALS, CONDITION_COLOURS, CONDITION_LABELS, CONDITION_MARKERS, DEFAULT_RESULTS_DIR, GRID_COLOUR, SIGNAL_NAMES,
    add_common_args, add_legend, apply_figure_style, figure_dimensions, load_ratings, mean_ci95, plot_layout,
    style_axis,
)
from exclusion_criteria import included_results

# Settings (defaults for the command-line arguments)
RESULTS_DIR = DEFAULT_RESULTS_DIR             # Folder containing the results CSVs
INCLUDE_REPEATS = False                       # Include repeated trials (repeat=1)
SPLIT_SIGNALS = True                          # One group per excitation signal; False pools all signals
SHOW_ERROR_BARS = False                        # Draw 95% confidence interval error bars around the means
DISTRIBUTION = "box"                       # How to show the ratings: "points", "violin" or "box"
DISTRIBUTIONS = ["points", "violin", "box"]
DISTRIBUTION_LABELS = {"points": "responses", "violin": "violin", "box": "box"}
BOX_WIDTH = 0.15                              # Box width (x units; boxes are 0.2 apart, so this leaves a small gap)
FIGURE_SIZE = "onecolumn"                  # "onecolumn" or "twocolumn" (IEEE Transactions)
FIGURE_HEIGHT = 2.2                           # Printed height (inches), as plot_means_and_confidence
ALL_SIGNALS_LABEL = "All excitations"         # Label for the group pooling all excitation signals
ALL_SIGNALS_GAP = 0.03                        # Space between the pooled-group axes and the per-signal axes (fraction of figure width)
SIGNAL_ORDER =["Beet", "Slow", "Hand"]       # Left-to-right order of the signals (Classical, Pop, Handclaps)
OUTPUT = None                                 # Save the figure to this file (e.g. "overall.pdf") at the print DPI; None only shows it


def group_ratings(ratings, signals, combinations, conditions):
    """Return [(label, {condition: [ratings]})] for each x-axis group, pooled over the combinations.

    When the ratings are split by signal, a group pooling all signals is added first.
    """
    groups = {}
    for signal in signals:
        groups[signal] = {condition: [v for combination in combinations
                                      for v in ratings[signal][combination].get(condition, [])]
                          for condition in conditions}
    if signals != [ALL_SIGNALS]:
        pooled = {condition: [v for signal in signals for v in groups[signal][condition]]
                  for condition in conditions}
        ordered = sorted(signals, key=lambda s: SIGNAL_ORDER.index(s) if s in SIGNAL_ORDER else len(SIGNAL_ORDER))
        return [(ALL_SIGNALS_LABEL, pooled)] + [(SIGNAL_NAMES.get(s, s), groups[s]) for s in ordered]
    return [(ALL_SIGNALS_LABEL, groups[ALL_SIGNALS])]


def plot(ratings, show_error_bars=True, distribution=DISTRIBUTION, figure_size=FIGURE_SIZE, output=OUTPUT):
    # Laid out at the printed width so the markers and lines match plot_means_and_confidence in the paper
    apply_figure_style(figure_size, actual_width=True)
    signals, combinations, conditions = plot_layout(ratings)
    groups = group_ratings(ratings, signals, combinations, conditions)

    # The pooled group gets its own axes, separate from the per-signal groups
    panels = [groups[:1], groups[1:]] if len(groups) > 1 else [groups]
    offset_step = 0.2
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2) * offset_step
    violin_width = 0.18
    jitter_width = 0.06
    # Fixed seed so the jitter is the same every time the plot is regenerated
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(panels), sharey=True, squeeze=False,
                             width_ratios=[len(panel) for panel in panels],
                             figsize=figure_dimensions(FIGURE_HEIGHT, figure_size), layout="constrained")
    axes = axes[0]
    fig.get_layout_engine().set(wspace=ALL_SIGNALS_GAP)
    for panel_index, (ax, panel) in enumerate(zip(axes, panels)):
        plot_panel(ax, panel, conditions, offsets, distribution, show_error_bars, violin_width, jitter_width, rng)
        summary = r"mean $\pm$ 95\% CI" if show_error_bars else "mean"
        style_axis(ax, None, f"Rating ({DISTRIBUTION_LABELS[distribution]}, {summary})" if panel_index == 0 else "")
        ax.set_ylim(-5, 105)

    fig.supxlabel("Excitation signal", fontsize=plt.rcParams["axes.labelsize"])
    # Tighter spacing so the single-row legend fits the onecolumn width
    add_legend(fig, *axes[0].get_legend_handles_labels(), top=True, columnspacing=1.0, handletextpad=0.3,
               markerscale=1.25)

    if output:
        # Saved here rather than from a viewer, which may capture the figure at the screen DPI
        fig.savefig(output, dpi=plt.rcParams["savefig.dpi"])
        print(f"Saved {output}")
    plt.show()


def plot_panel(ax, groups, conditions, offsets, distribution, show_error_bars, violin_width, jitter_width, rng):
    """Draw the rating distributions and means of each condition for the given groups on one axes."""
    x = np.arange(len(groups), dtype=float)
    for condition, offset in zip(conditions, offsets):
        colour = CONDITION_COLOURS.get(condition)
        marker = CONDITION_MARKERS.get(condition, "o")
        means, errors, xs = [], [], []
        for i, (_, group) in enumerate(groups):
            values = group[condition]
            if not values:
                continue
            centre = x[i] + offset
            if distribution == "points":
                ax.scatter(centre + rng.uniform(-jitter_width, jitter_width, len(values)), values,
                           marker=marker, color=colour, alpha=0.25, s=6, linewidths=0, zorder=1)
            elif distribution == "box":
                box = ax.boxplot(values, positions=[centre], widths=BOX_WIDTH, patch_artist=True,
                                 showfliers=False, manage_ticks=False, zorder=1)
                for patch in box["boxes"]:
                    patch.set_facecolor(colour)
                    patch.set_edgecolor(colour)
                for line in box["whiskers"] + box["caps"] + box["medians"]:
                    line.set_color("black")
                    line.set_linestyle("-")
                    line.set_linewidth(0.8)
            elif np.ptp(values) > 0:
                violin = ax.violinplot(values, positions=[centre], widths=violin_width, showextrema=False)
                for body in violin["bodies"]:
                    body.set_facecolor(colour)
                    body.set_edgecolor(colour)
                    body.set_alpha(1)
                    body.set_zorder(1)
            else:
                # The density estimate is undefined when every rating is identical, so draw a flat line
                ax.plot([centre - violin_width / 2, centre + violin_width / 2], [values[0], values[0]],
                        color=colour, alpha=0.6, linewidth=1.2, zorder=1)
            mean, half_width = mean_ci95(values)
            xs.append(centre)
            means.append(mean)
            errors.append(half_width)
        ax.errorbar(
            xs, means, yerr=errors if show_error_bars else None,
            fmt=marker, color=colour, markeredgecolor="white", markeredgewidth=0.6,
            markersize=4, ecolor="black", capsize=2, elinewidth=1, capthick=1,
            label=CONDITION_LABELS.get(condition, condition), zorder=3,
        )

    for boundary in x[:-1] + 0.5:
        ax.axvline(boundary, color=GRID_COLOUR, linewidth=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in groups])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, RESULTS_DIR, INCLUDE_REPEATS, SPLIT_SIGNALS, FIGURE_SIZE)
    parser.add_argument("--error-bars", dest="show_error_bars", default=SHOW_ERROR_BARS,
                        action=argparse.BooleanOptionalAction,
                        help="Draw 95%% confidence interval error bars around the means (default: %(default)s)")
    parser.add_argument("--distribution", default=DISTRIBUTION, choices=DISTRIBUTIONS,
                        help="How to show the rating distribution (default: %(default)s)")
    parser.add_argument("--output", default=OUTPUT,
                        help="Save the figure to this file at the print DPI (default: %(default)s)")
    args = parser.parse_args()

    results = included_results(args.results_dir)
    ratings = load_ratings(results, args.include_repeats, args.split_signals)
    if not ratings:
        raise SystemExit(f"No results found in {args.results_dir}")
    plot(ratings, args.show_error_bars, args.distribution, args.figure_size, args.output)


if __name__ == "__main__":
    main()

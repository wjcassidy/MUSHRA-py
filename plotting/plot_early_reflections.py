"""Plot the early reflections of the echograms of the passive room, the target room
and the full optimisation (EnvFOA) for the listening test source/receiver position.

Each column is one passive/target combination, titled with the mean MUSHRA rating of the
full optimisation for that combination. Echograms use the W channel, with
time and level relative to the direct sound, as the rooms have different
source/receiver distances and the RIRs have different gains.

Usage:
    python plotting/plot_early_reflections.py [--figure-size {onecolumn,twocolumn}] [--results-dir DIR]
                                              [--[no-]include-repeats]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from common import (
    ALL_SIGNALS, CONDITION_COLOURS, CONDITION_LABELS, DEFAULT_RESULTS_DIR, GRID_COLOUR, add_figure_size_arg,
    apply_figure_style, figure_dimensions, load_ratings,
)
from exclusion_criteria import included_results

# Settings
COMBINATIONS = ["R2T2", "R2T4"]               # Passive/target combinations to plot
# 0-indexed, as in the optimisation outputs (the dataset H_SR_TEST files are 1-indexed). Matches
# PyRES-ClosedLoop/examples/convolve_*_irs_with_excitation.py. S3 is the Handclaps source, and is
# also used by the Classical and Pop excitations alongside S0-S2
LISTENING_TEST_RECEIVER = 1
LISTENING_TEST_SOURCE = 3
TIME_LIMITS_MS = (-2, 30)                    # Time window, relative to the direct sound
LEVEL_LIMITS_DB = (-25, 6)                    # Level range, relative to the direct sound
# The direct sound is the first arrival within this range of the RIR's peak. The peak alone isn't
# enough, as an early reflection can be stronger than the direct sound (e.g. R2T4, S3)
DIRECT_SOUND_THRESHOLD_DB = -12
DIRECT_SOUND_WINDOW_MS = 1                    # Window after the onset used to measure the direct sound level
DATASET_ROOT = Path("/Volumes/WillFootage/PhD/AAESOptimisationDatasetRIRs")
FIGURE_SIZE = "onecolumn"                     # "onecolumn" or "twocolumn" (IEEE Transactions)
ROW_HEIGHT = 1.2                              # Height of each row at twocolumn width (inches)

CONDITIONS = ["passive", "target", "fulloptim"]  # One row per condition, in this order
RESULTS_DIR = DEFAULT_RESULTS_DIR             # Folder containing the results CSVs
INCLUDE_REPEATS = False                       # Include repeated trials (repeat=1) in the mean ratings
RATING_CONDITION = "fulloptim"                # Condition whose mean rating titles each column (pooled over signals)


def condition_files(combination):
    """Return {condition: listening test RIR path} for a combination such as "R2T4"."""
    room, target = combination[1:].split("T")
    r, s = LISTENING_TEST_RECEIVER, LISTENING_TEST_SOURCE
    dataset_dir = DATASET_ROOT / "AAES Dataset"
    return {
        "passive": dataset_dir / f"aaes_optimisation_dataset_room_{room}" / f"H_SR_TEST_R{r + 1}_S{s + 1}.wav",
        "target": dataset_dir / f"aaes_optimisation_dataset_target_room_{target}" / f"H_SR_TEST_R{r + 1}_S{s + 1}.wav",
        "fulloptim": DATASET_ROOT / "Optimisation Results" / combination / "EnvFOA" / f"opt_ambi_test_R{r}_S{s}.wav",
    }


def echogram(path):
    """Return (times in ms, levels in dB), both relative to the direct sound, of an RIR's W channel."""
    sample_rate, spatial_rir = wavfile.read(path)
    magnitude = np.abs(spatial_rir[:, 0].astype(float))
    threshold = magnitude.max() * 10 ** (DIRECT_SOUND_THRESHOLD_DB / 20)
    onset = np.argmax(magnitude >= threshold)
    window = int(DIRECT_SOUND_WINDOW_MS * 1e-3 * sample_rate)
    direct_level = magnitude[onset:onset + window].max()

    times = (np.arange(len(magnitude)) - onset) / sample_rate * 1e3
    with np.errstate(divide="ignore"):
        levels = 20 * np.log10(magnitude / direct_level)
    return times, levels


def plot(echograms, mean_ratings, figure_size=FIGURE_SIZE):
    apply_figure_style(figure_size)
    combinations = list(echograms)
    conditions = CONDITIONS
    fig, axes = plt.subplots(
        len(conditions), len(combinations), sharex=True, sharey=True, squeeze=False, layout="constrained",
        figsize=figure_dimensions(ROW_HEIGHT * len(conditions) + 0.6),
    )
    for column, combination in enumerate(combinations):
        # Column title, above the top row's own (left-aligned) title
        axes[0, column].annotate(
            rf"Mean Rating: {mean_ratings[combination]:.0f}\%", xy=(0.5, 1), xycoords="axes fraction",
            xytext=(0, 1.6 * plt.rcParams["axes.titlesize"]), textcoords="offset points", ha="center",
            va="bottom", fontsize=plt.rcParams["axes.titlesize"], annotation_clip=False,
        ).set_in_layout(True)
        for row, condition in enumerate(conditions):
            ax = axes[row, column]
            times, levels = echograms[combination][condition]
            visible = (times >= TIME_LIMITS_MS[0]) & (times <= TIME_LIMITS_MS[1])
            ax.fill_between(times[visible], LEVEL_LIMITS_DB[0], levels[visible], color=CONDITION_COLOURS[condition],
                            linewidth=1.5)
            ax.set_title(f"{combination}: {CONDITION_LABELS[condition]}", loc="left")
            ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    axes[0, 0].set_xlim(TIME_LIMITS_MS)
    axes[0, 0].set_ylim(LEVEL_LIMITS_DB)
    fig.supylabel("Magnitude (dB)")
    fig.supxlabel("Time after direct sound (ms)")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_figure_size_arg(parser, FIGURE_SIZE)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help="Folder containing the results CSVs (default: %(default)s)")
    parser.add_argument("--include-repeats", default=INCLUDE_REPEATS, action=argparse.BooleanOptionalAction,
                        help="Include repeated trials (repeat=1) in the mean ratings (default: %(default)s)")
    args = parser.parse_args()

    ratings = load_ratings(included_results(args.results_dir), args.include_repeats, split_signals=False)
    mean_ratings = {combination: np.mean(ratings[ALL_SIGNALS][combination][RATING_CONDITION])
                    for combination in COMBINATIONS}

    echograms = {}
    for combination in COMBINATIONS:
        echograms[combination] = {condition: echogram(path) for condition, path in condition_files(combination).items()}
    plot(echograms, mean_ratings, args.figure_size)


if __name__ == "__main__":
    main()

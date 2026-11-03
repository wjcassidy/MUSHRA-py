"""Plot the unpleasantness features and overall predicted unpleasantness of the
test RIRs for one passive/target combination (e.g. R1T5).

Each box pools all test source/receiver combinations (or, with LISTENING_TEST_ONLY,
only those used for the listening test stimuli), comparing the passive room, the
target room and the full optimisation (EnvFOA).

Usage:
    python plotting/plot_unpleasantness.py [--figure-size {onecolumn,twocolumn}]
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.io import wavfile

from common import (
    CONDITION_COLOURS, CONDITION_LABELS, GRID_COLOUR, add_figure_size_arg, add_legend, apply_figure_style,
    figure_dimensions,
)

# Settings
COMBINATION = "R2T4"                          # Passive/target combination to plot
LISTENING_TEST_ONLY = False                   # Only the source/receiver RIRs convolved with the excitations
# 0-indexed, as in the optimisation outputs (the dataset H_SR_TEST files are 1-indexed). Matches
# PyRES-ClosedLoop/examples/convolve_*_irs_with_excitation.py, pooling the sources of all excitations
LISTENING_TEST_RECEIVER = 1
LISTENING_TEST_SOURCES = [0, 1, 2, 3]
# When LISTENING_TEST_ONLY is False: include the stress-test positions (the last receivers/sources of each test set)
INCLUDE_NON_CENTRAL_POSITIONS = False
NUM_STRESS_TEST_RECEIVERS = 4
NUM_STRESS_TEST_SOURCES = 2
DATASET_ROOT = Path("/Volumes/WillFootage/PhD/AAESOptimisationDatasetRIRs")
MODEL_SRC_DIR = Path(__file__).resolve().parents[2] / "AAESUnpleasantnessModel" / "Src"
FIGURE_SIZE = "twocolumn"                     # "onecolumn" or "twocolumn" (IEEE Transactions)
FIGURE_HEIGHT = 2.6                           # Height at twocolumn width (inches)

sys.path.append(str(MODEL_SRC_DIR))
import Colouration
import DSE
import FlutterEcho
import HFDamping
import PredictUnpleasantness
import SDM

METRIC_LABELS = ["Spatial\nAsymmetry", "Flutter\nEcho", "Colouration", "HF\nDamping", "DSE\nCurvature", "Unpleasantness"]

YLIM = (0.0, 1.0)


def exclude_stress_test_positions(files):
    """Drop the RIRs of the last NUM_STRESS_TEST_RECEIVERS receivers and NUM_STRESS_TEST_SOURCES sources."""
    indices = {f: tuple(map(int, re.search(r"_R(\d+)_S(\d+)\.wav$", f.name).groups())) for f in files}
    receivers = sorted({r for r, _ in indices.values()})
    sources = sorted({s for _, s in indices.values()})
    central_receivers = set(receivers[:len(receivers) - NUM_STRESS_TEST_RECEIVERS])
    central_sources = set(sources[:len(sources) - NUM_STRESS_TEST_SOURCES])
    return [f for f in files if indices[f][0] in central_receivers and indices[f][1] in central_sources]


def condition_files(combination, listening_test_only=False, include_non_central_positions=True):
    """Return {condition: [test RIR paths]} for a combination such as "R1T5".

    If listening_test_only is True, only the RIRs used for the listening test stimuli are returned.
    Otherwise, if include_non_central_positions is False, the stress-test positions are excluded.
    """
    match = re.fullmatch(r"R(\d+)T(\d+)", combination)
    if match is None:
        raise ValueError(f"Unrecognised combination {combination!r}, expected e.g. 'R1T5'")
    room, target = match.groups()
    dataset_dir = DATASET_ROOT / "AAES Dataset"
    dirs_and_patterns = {
        "passive": (dataset_dir / f"aaes_optimisation_dataset_room_{room}", "H_SR_TEST_R{r}_S{s}.wav", 1),
        "target": (dataset_dir / f"aaes_optimisation_dataset_target_room_{target}", "H_SR_TEST_R{r}_S{s}.wav", 1),
        "fulloptim": (DATASET_ROOT / "Optimisation Results" / combination / "EnvFOA", "opt_ambi_test_R{r}_S{s}.wav", 0),
    }
    files = {}
    for condition, (directory, pattern, index_offset) in dirs_and_patterns.items():
        if listening_test_only:
            r = LISTENING_TEST_RECEIVER + index_offset
            files[condition] = [directory / pattern.format(r=r, s=s + index_offset) for s in LISTENING_TEST_SOURCES]
        else:
            files[condition] = sorted(directory.glob(pattern.format(r="*", s="*")))
            if not include_non_central_positions:
                files[condition] = exclude_stress_test_positions(files[condition])
    return files


def compute_scores(path):
    sample_rate, spatial_rir = wavfile.read(path)
    spatial_rir = spatial_rir.astype(float) / np.max(np.abs(spatial_rir))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        asymmetry = SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, False)
        flutter_echo = FlutterEcho.getFlutterEchoScore(spatial_rir, sample_rate, False)
        colouration = Colouration.getColouration(spatial_rir[:, 0], sample_rate, False)
        hf_damping = HFDamping.getHFDampingScore(spatial_rir[:, 0], sample_rate, False)
        curvature = DSE.getCurvature(spatial_rir[:, 0], sample_rate)
        # Same as PredictUnpleasantness.predictUnpleasantnessFromRIR, reusing the features above
        unpleasantness = PredictUnpleasantness.predictUnpleasantnessFromFeatures(
            colouration, asymmetry, flutter_echo, curvature, hf_damping, 0, use_max_from_both=True) / 100.0
    scores = [asymmetry, flutter_echo, colouration, hf_damping, curvature, unpleasantness]
    return [float(s) if np.isfinite(s) else np.nan for s in scores]


def add_overflow_markers(ax, bp, positions):
    """Mark boxes, whiskers or outliers that extend beyond the y-axis limits."""
    low, high = YLIM
    kw = dict(ms=4, mfc="white", mew=0.8, zorder=6, clip_on=False, color="k")
    for i, x in enumerate(positions):
        y = np.concatenate([
            bp["boxes"][i].get_path().vertices[:, 1],
            bp["caps"][2 * i].get_ydata(),
            bp["caps"][2 * i + 1].get_ydata(),
            bp["fliers"][i].get_ydata(),
        ])
        if np.any(y > high):
            ax.plot(x, high, marker="^", **kw)
        if np.any(y < low):
            ax.plot(x, low, marker="v", **kw)


def plot(combination, all_scores, figure_size=FIGURE_SIZE):
    apply_figure_style(figure_size)
    conditions = list(all_scores)
    box_spacing = 0.5
    group_width = box_spacing * len(conditions) + 0.25
    group_centers = np.arange(len(METRIC_LABELS)) * group_width
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2) * box_spacing

    data, positions, colours = [], [], []
    for metric_index, centre in enumerate(group_centers):
        for condition, offset in zip(conditions, offsets):
            column = all_scores[condition][:, metric_index]
            data.append(column[np.isfinite(column)])
            positions.append(centre + offset)
            colours.append(CONDITION_COLOURS[condition])

    fig, ax = plt.subplots(figsize=figure_dimensions(FIGURE_HEIGHT), layout="constrained")
    bp = ax.boxplot(data, positions=positions, widths=0.42, patch_artist=True, capwidths=0.3)
    for i, colour in enumerate(colours):
        bp["boxes"][i].set(facecolor=colour, edgecolor=colour)
        bp["medians"][i].set(color="black", linewidth=1)
        bp["fliers"][i].set(markerfacecolor=colour, markeredgecolor=colour, markersize=2)
    for line in bp["whiskers"] + bp["caps"]:
        line.set_linewidth(0.8)
    add_overflow_markers(ax, bp, positions)

    for boundary in group_centers[:-1] + group_width / 2:
        ax.axvline(boundary, color=GRID_COLOUR, linewidth=0.8, zorder=0)

    ax.set_title(combination, loc="left")
    ax.set_xticks(group_centers)
    if figure_size == "onecolumn":
        # The metric names overlap when the figure fits one column, so angle them on a single line
        ax.set_xticklabels([label.replace("\n", " ") for label in METRIC_LABELS], rotation=35, ha="right",
                           rotation_mode="anchor")
    else:
        ax.set_xticklabels(METRIC_LABELS)
    ax.set_xlim(group_centers[0] - group_width / 2, group_centers[-1] + group_width / 2)
    ax.set_ylim(YLIM)
    ax.set_ylabel("Score")
    ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend_elements = [Patch(facecolor=CONDITION_COLOURS[c], edgecolor=CONDITION_COLOURS[c], label=CONDITION_LABELS[c])
                       for c in conditions]
    add_legend(fig, legend_elements, [h.get_label() for h in legend_elements])

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_figure_size_arg(parser, FIGURE_SIZE)
    args = parser.parse_args()

    all_scores = {}
    for condition, files in condition_files(COMBINATION, LISTENING_TEST_ONLY, INCLUDE_NON_CENTRAL_POSITIONS).items():
        if not files:
            raise SystemExit(f"No test RIRs found for {CONDITION_LABELS[condition]} ({COMBINATION})")
        print(f"Computing scores for {CONDITION_LABELS[condition]} ({len(files)} files)...")
        all_scores[condition] = np.array([compute_scores(f) for f in files])
    plot(COMBINATION, all_scores, args.figure_size)


if __name__ == "__main__":
    main()

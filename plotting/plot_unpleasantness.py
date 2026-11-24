"""Plot the unpleasantness features and overall predicted unpleasantness of the
test RIRs for one passive/target combination (e.g. R1T5).

Each box pools all test source/receiver combinations (or, with LISTENING_TEST_ONLY,
only those used for the listening test stimuli), comparing the passive room, the
target room, the full optimisation (EnvFOA) and the graphic EQ baseline (GEQBaseline).

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
# When LISTENING_TEST_ONLY is False, which test positions to pool (the stress-test positions are the last
# receivers/sources of each test set):
#   "central"          - standard receivers x standard sources
#   "all"              - every receiver x every source
#   "stress_receivers" - stress-test receivers x standard sources
#   "stress_sources"   - standard receivers x stress-test sources
POSITIONS = "central"
NUM_STRESS_TEST_RECEIVERS = 4
NUM_STRESS_TEST_SOURCES = 2
DATASET_ROOT = Path("/Volumes/WillFootage/PhD/AAESOptimisationDatasetRIRs")
MODEL_SRC_DIR = Path(__file__).resolve().parents[2] / "AAESUnpleasantnessModel" / "Src"
FIGURE_SIZE = "twocolumn"                     # "onecolumn" or "twocolumn" (IEEE Transactions)
FIGURE_HEIGHT = 2.0                           # Height at twocolumn width (inches)

sys.path.append(str(MODEL_SRC_DIR))
import Colouration
import DSE
import FlutterEcho
import HFDamping
import PredictUnpleasantness
import SDM

METRIC_LABELS = ["Spatial\nAsymmetry", "Flutter\nEcho", "Colouration", "HF\nDamping", "DSE\nCurvature", "Unpleasantness"]

YLIM = (0.0, 1.0)


POSITION_MODES = ("central", "all", "stress_receivers", "stress_sources")


def select_positions(files, positions):
    """Keep the RIRs whose receiver/source fall in the given POSITION_MODES selection.

    The stress-test positions are the last NUM_STRESS_TEST_RECEIVERS receivers and NUM_STRESS_TEST_SOURCES sources.
    """
    if positions not in POSITION_MODES:
        raise ValueError(f"Unrecognised positions {positions!r}, expected one of {POSITION_MODES}")
    if positions == "all":
        return files
    indices = {f: tuple(map(int, re.search(r"_R(\d+)_S(\d+)\.wav$", f.name).groups())) for f in files}
    receivers = sorted({r for r, _ in indices.values()})
    sources = sorted({s for _, s in indices.values()})
    central_receivers = set(receivers[:len(receivers) - NUM_STRESS_TEST_RECEIVERS])
    central_sources = set(sources[:len(sources) - NUM_STRESS_TEST_SOURCES])
    stress_receivers = set(receivers) - central_receivers
    stress_sources = set(sources) - central_sources
    allowed_receivers, allowed_sources = {
        "central": (central_receivers, central_sources),
        "stress_receivers": (stress_receivers, central_sources),
        "stress_sources": (central_receivers, stress_sources),
    }[positions]
    return [f for f in files if indices[f][0] in allowed_receivers and indices[f][1] in allowed_sources]


def condition_files(combination, listening_test_only=False, positions="all"):
    """Return {condition: [test RIR paths]} for a combination such as "R1T5".

    If listening_test_only is True, only the RIRs used for the listening test stimuli are returned.
    Otherwise, the RIRs are filtered by positions (one of POSITION_MODES).
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
        "geq": (DATASET_ROOT / "Optimisation Results" / combination / "GEQBaseline", "opt_ambi_test_R{r}_S{s}.wav", 0),
    }
    files = {}
    for condition, (directory, pattern, index_offset) in dirs_and_patterns.items():
        if listening_test_only:
            r = LISTENING_TEST_RECEIVER + index_offset
            files[condition] = [directory / pattern.format(r=r, s=s + index_offset) for s in LISTENING_TEST_SOURCES]
        else:
            files[condition] = sorted(directory.glob(pattern.format(r="*", s="*")))
            files[condition] = select_positions(files[condition], positions)
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


def print_box_statistics(data, box_conditions, num_conditions):
    """Print the median and maximum of each box, grouped by metric."""
    for metric_index, metric in enumerate(METRIC_LABELS):
        print(f"\n{metric.replace(chr(10), ' ')}:")
        for i in range(metric_index * num_conditions, (metric_index + 1) * num_conditions):
            values = data[i]
            median, maximum = (np.median(values), np.max(values)) if values.size else (np.nan, np.nan)
            print(f"  {CONDITION_LABELS[box_conditions[i]]:<20} median = {median:.3f}, max = {maximum:.3f}")


def plot(all_scores, figure_size=FIGURE_SIZE):
    apply_figure_style(figure_size)
    conditions = list(all_scores)
    box_spacing = 0.5
    group_width = box_spacing * len(conditions) + 0.25
    group_centers = np.arange(len(METRIC_LABELS)) * group_width
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2) * box_spacing

    data, positions, colours, box_conditions = [], [], [], []
    for metric_index, centre in enumerate(group_centers):
        for condition, offset in zip(conditions, offsets):
            column = all_scores[condition][:, metric_index]
            data.append(column[np.isfinite(column)])
            positions.append(centre + offset)
            colours.append(CONDITION_COLOURS[condition])
            box_conditions.append(condition)

    fig, ax = plt.subplots(figsize=figure_dimensions(FIGURE_HEIGHT), layout="constrained")
    bp = ax.boxplot(data, positions=positions, widths=0.42, patch_artist=True, capwidths=0.3)
    for i, (colour, condition) in enumerate(zip(colours, box_conditions)):
        bp["boxes"][i].set(facecolor=colour, edgecolor=colour)
        # White on the dark target colour so the median stays visible
        bp["medians"][i].set(color="white" if condition == "target" else "black", linewidth=1)
        bp["fliers"][i].set(markerfacecolor=colour, markeredgecolor=colour, markersize=2)
    for line in bp["whiskers"] + bp["caps"]:
        line.set_linewidth(0.8)
    add_overflow_markers(ax, bp, positions)
    print_box_statistics(data, box_conditions, len(conditions))

    for boundary in group_centers[:-1] + group_width / 2:
        ax.axvline(boundary, color=GRID_COLOUR, linewidth=0.8, zorder=0)

    ax.set_xticks(group_centers)
    if figure_size == "onecolumn":
        # The metric names overlap when the figure fits one column, so angle them on a single line
        ax.set_xticklabels([label.replace("\n", " ") for label in METRIC_LABELS], rotation=35, ha="right",
                           rotation_mode="anchor")
    else:
        ax.set_xticklabels(METRIC_LABELS)
    ax.set_xlim(group_centers[0] - group_width / 2, group_centers[-1] + group_width / 2)
    ax.set_ylim(YLIM)
    ax.set_ylabel("Severity")
    ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend_elements = [Patch(facecolor=CONDITION_COLOURS[c], edgecolor=CONDITION_COLOURS[c], label=CONDITION_LABELS[c])
                       for c in conditions]
    # Tighter spacing so the single-row legend fits the onecolumn width, and sits close to the axes
    add_legend(fig, legend_elements, [h.get_label() for h in legend_elements], top=True, columnspacing=1.0,
               handletextpad=0.3, borderpad=0, borderaxespad=0.2)
    fig.get_layout_engine().set(h_pad=0)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_figure_size_arg(parser, FIGURE_SIZE)
    args = parser.parse_args()

    all_scores = {}
    for condition, files in condition_files(COMBINATION, LISTENING_TEST_ONLY, POSITIONS).items():
        if not files:
            raise SystemExit(f"No test RIRs found for {CONDITION_LABELS[condition]} ({COMBINATION})")
        print(f"Computing scores for {CONDITION_LABELS[condition]} ({len(files)} files)...")
        all_scores[condition] = np.array([compute_scores(f) for f in files])
    plot(all_scores, args.figure_size)


if __name__ == "__main__":
    main()

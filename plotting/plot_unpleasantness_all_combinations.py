"""Plot the unpleasantness features and overall predicted unpleasantness of the full
optimisation (EnvFOA) test RIRs, pooled over every passive/target combination (R1T1, R1T2, ...).

Each feature has two boxes:
    (a) the standard test positions (standard receivers x standard sources)
    (b) the off-centre (stress-test) receivers x standard sources

If any feature can't be computed (or isn't finite) for a source/receiver position in any combination, that
position is excluded for that passive room in every combination, so every box has the same number of values.

Usage:
    python plotting/plot_unpleasantness_all_combinations.py [--figure-size {onecolumn,twocolumn}]
        [--skip COMBINATION ...]
"""

import argparse
import multiprocessing
import re
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.io import wavfile

from common import (
    CONDITION_COLOURS, GRID_COLOUR, add_figure_size_arg, add_legend, apply_figure_style, combination_sort_key,
    figure_dimensions,
)
from plot_unpleasantness import (
    DATASET_ROOT, METRIC_LABELS, YLIM, Colouration, DSE, FlutterEcho, HFDamping, PredictUnpleasantness, SDM,
    add_overflow_markers, select_positions,
)

# Settings
CONDITION_DIR = "EnvFOA"                      # Optimisation result folder to evaluate in each combination
FIGURE_SIZE = "onecolumn"                     # "onecolumn" or "twocolumn" (IEEE Transactions)
FIGURE_HEIGHT = 4.0                           # Height at twocolumn width (inches)
MAX_WORKERS = None                            # Processes used to compute the scores (None: one per CPU)
SKIP_COMBINATIONS = ["R1T6","R4T7"]                        # Passive/target combinations to leave out, e.g. ["R4T1", "R4T7"]

POSITION_GROUPS = {
    "central": "(a) Standard positions",
    "stress_receivers": "(b) Off-centre receivers",
}
COLOUR = CONDITION_COLOURS["fulloptim"]
# The off-centre boxes use a lighter tint of the same colour
GROUP_COLOURS = {
    "central": COLOUR,
    "stress_receivers": tuple([0.5 * to_rgb(COLOUR)[0] + 0.2,
                               0.5 * to_rgb(COLOUR)[1] + 0.2,
                               to_rgb(COLOUR)[2] + 0.4]),
}


def compute_scores(path):
    """As plot_unpleasantness.compute_scores, but a feature that can't be computed for an RIR (e.g. HF damping
    when its EDC never reaches the late-decay range) is NaN rather than an error, as is its unpleasantness.

    Returns (scores, names of the failed features).
    """
    sample_rate, spatial_rir = wavfile.read(path)
    spatial_rir = spatial_rir.astype(float) / np.max(np.abs(spatial_rir))
    omni = spatial_rir[:, 0]
    extractors = {
        "asymmetry": lambda: SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, False),
        "flutter_echo": lambda: FlutterEcho.getFlutterEchoScore(spatial_rir, sample_rate, False),
        "colouration": lambda: Colouration.getColouration(omni, sample_rate, False),
        "hf_damping": lambda: HFDamping.getHFDampingScore(omni, sample_rate, False),
        "curvature": lambda: DSE.getCurvature(omni, sample_rate),
    }
    features, failed = {}, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, extract in extractors.items():
            try:
                features[name] = float(extract())
            except Exception:
                features[name] = np.nan
                failed.append(name)
        unpleasantness = np.nan
        if not failed:
            unpleasantness = PredictUnpleasantness.predictUnpleasantnessFromFeatures(
                features["colouration"], features["asymmetry"], features["flutter_echo"], features["curvature"],
                features["hf_damping"], 0, use_max_from_both=True) / 100.0
    scores = [*features.values(), unpleasantness]
    return [float(s) if np.isfinite(s) else np.nan for s in scores], failed


def combinations(skip=()):
    """Return the combinations with a CONDITION_DIR folder, except those in skip."""
    results_dir = DATASET_ROOT / "Optimisation Results"
    names = [d.name for d in results_dir.iterdir()
             if d.is_dir() and re.fullmatch(r"R\d+T\d+", d.name) and (d / CONDITION_DIR).is_dir()]
    unknown = set(skip) - set(names)
    if unknown:
        print(f"Warning: skipped combinations not found: {', '.join(sorted(unknown, key=combination_sort_key))}")
    return sorted(set(names) - set(skip), key=combination_sort_key)


def group_files(combination_names):
    """Return {position group: [test RIR paths]} pooled over the given combinations."""
    files = {group: [] for group in POSITION_GROUPS}
    for combination in combination_names:
        directory = DATASET_ROOT / "Optimisation Results" / combination / CONDITION_DIR
        rirs = sorted(directory.glob("opt_ambi_test_R*_S*.wav"))
        for group in POSITION_GROUPS:
            selected = select_positions(rirs, group)
            if not selected:
                raise SystemExit(f"No {group} test RIRs found in {directory}")
            files[group].extend(selected)
    return files


def position_key(path):
    """Return (passive room, receiver, source) for an optimisation result RIR path."""
    room = re.fullmatch(r"R(\d+)T\d+", path.parent.parent.name).group(1)
    receiver, source = re.search(r"_R(\d+)_S(\d+)\.wav$", path.name).groups()
    return f"R{room}", int(receiver), int(source)


def exclude_invalid_positions(files, scores):
    """Drop every RIR whose (passive room, receiver, source) has a non-finite score in any combination.

    files and scores are {group: paths} and {group: array (num files x num metrics)}. Returns the filtered
    scores and the sorted excluded positions.
    """
    invalid = {position_key(path)
               for group in files
               for path, row in zip(files[group], scores[group]) if not np.all(np.isfinite(row))}
    filtered = {}
    for group in files:
        keep = np.array([position_key(path) not in invalid for path in files[group]], dtype=bool)
        filtered[group] = scores[group][keep]
    return filtered, sorted(invalid)


def plot(all_scores, figure_size=FIGURE_SIZE):
    apply_figure_style(figure_size)
    groups = list(all_scores)
    box_spacing = 0.5
    group_width = box_spacing * len(groups) + 0.5
    group_centers = np.arange(len(METRIC_LABELS)) * group_width
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * box_spacing

    data, positions, box_groups = [], [], []
    for metric_index, centre in enumerate(group_centers):
        for group, offset in zip(groups, offsets):
            column = all_scores[group][:, metric_index]
            data.append(column[np.isfinite(column)])
            positions.append(centre + offset)
            box_groups.append(group)

    fig, ax = plt.subplots(figsize=figure_dimensions(FIGURE_HEIGHT), layout="constrained")
    bp = ax.boxplot(data, positions=positions, widths=0.42, patch_artist=True, capwidths=0.3)
    for i, group in enumerate(box_groups):
        colour = GROUP_COLOURS[group]
        bp["boxes"][i].set(facecolor=colour, edgecolor=colour)
        # White on the dark colour so the median stays visible
        bp["medians"][i].set(color="white")# if group == "central" else "black", linewidth=1)
        bp["fliers"][i].set(markerfacecolor=colour, markeredgecolor=colour, markersize=2)
    for line in bp["whiskers"] + bp["caps"]:
        line.set_linewidth(1.2)
    add_overflow_markers(ax, bp, positions)

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

    legend_elements = [Patch(facecolor=GROUP_COLOURS[g], edgecolor=GROUP_COLOURS[g], label=POSITION_GROUPS[g])
                       for g in groups]
    add_legend(fig, legend_elements, [h.get_label() for h in legend_elements], top=True, columnspacing=1.0,
               handletextpad=0.3, borderpad=0, borderaxespad=0.2)
    fig.get_layout_engine().set(h_pad=0)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_figure_size_arg(parser, FIGURE_SIZE)
    parser.add_argument("--skip", nargs="*", default=SKIP_COMBINATIONS, metavar="COMBINATION",
                        help="Passive/target combinations to leave out, e.g. --skip R4T1 R4T7 "
                             "(default: %(default)s)")
    args = parser.parse_args()

    combination_names = combinations(args.skip)
    if not combination_names:
        raise SystemExit(f"No combinations with a {CONDITION_DIR} folder found under {DATASET_ROOT}")
    print(f"Combinations: {', '.join(combination_names)}")

    files = group_files(combination_names)
    all_scores = {}
    # Fork so the workers reuse the parent's imported model modules: the model's PredictUnpleasantness adds the
    # AAESUnpleasantnessModel-Diff path to sys.path, so freshly spawned workers would import its modules instead
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=multiprocessing.get_context("fork")) as executor:
        for group, group_paths in files.items():
            print(f"Computing scores for {POSITION_GROUPS[group]} ({len(group_paths)} files)...")
            results = list(executor.map(compute_scores, group_paths))
            all_scores[group] = np.array([scores for scores, _ in results])
            for path, (scores, failed) in zip(group_paths, results):
                if failed or not np.all(np.isfinite(scores)):
                    invalid = failed or [name for name, score in zip(METRIC_LABELS, scores) if not np.isfinite(score)]
                    print(f"  Invalid {', '.join(n.replace(chr(10), ' ') for n in invalid)} for "
                          f"{path.parent.parent.name}/{path.name}")

    all_scores, excluded = exclude_invalid_positions(files, all_scores)
    if excluded:
        print("Excluding these positions from every combination of their passive room (0-indexed, as in the "
              "result filenames): " + ", ".join(f"{room} R{r}_S{s}" for room, r, s in excluded))
    for group, scores in all_scores.items():
        print(f"{POSITION_GROUPS[group]}: {len(scores)} RIRs per box")
    plot(all_scores, args.figure_size)


if __name__ == "__main__":
    main()

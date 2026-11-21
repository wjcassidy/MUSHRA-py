"""Plot the strength G (ISO 3382-1) of the test RIRs, pooled over every passive/target combination (R1T1,
R1T2, ...) that has both a full optimisation (EnvFOA) and a graphic EQ baseline (GEQBaseline) result.

Each box pools the selected test source/receiver positions (see plot_unpleasantness.py's
LISTENING_TEST_ONLY and POSITIONS) of every combination, comparing the passive room, the target room, the
full optimisation and the graphic EQ baseline, in each octave band of BANDS and their mean. The passive and
target room RIRs are included once per combination, so each room is weighted the same as in the
optimisation boxes (each unique RIR is only computed once). With --style means, each box is replaced by
the mean and its 95% confidence interval (t-distribution) instead.

G = 10 log10( integral p^2 dt / integral p10^2 dt ), where p10 is the free-field impulse response of the
same source at 10 m. The free-field reference is taken from the direct sound of the dataset RIRs: the
simulations have no pre-delay (the direct sound arrives at d / c) and the direct sound falls off as 1/d,
so each RIR's direct sound energy E_dir, in each octave band, gives E_10 = E_dir * (d / 10)^2. As each
room's simulations share a source level, E_dir * d^2 is a constant of the room folder, taken as its median
over all of its test RIRs so that early reflections arriving within the direct sound window don't bias it.
The optimisation outputs are peak normalised, so they're rescaled by matching their direct sound (which is
the passive room's) to that of the passive RIR at the same source/receiver position.

Only the W channel is used, and the RIRs are integrated to their ends (the optimisation outputs are 4 s).

Usage:
    python plotting/plot_strength.py [--figure-size {onecolumn,twocolumn}] [--skip COMBINATION ...]
        [--style {boxes,means}]
"""

import argparse
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import signal, stats
from scipy.io import wavfile

from common import (
    CANVAS_WIDTH, CONDITION_COLOURS, CONDITION_LABELS, CONDITION_MARKERS, FIGURE_SIZES, GRID_COLOUR,
    add_figure_size_arg, add_legend,
    apply_figure_style, combination_sort_key, figure_dimensions, mean_ci,
)
from plot_unpleasantness import DATASET_ROOT, condition_files

# Settings
SKIP_COMBINATIONS = ["R1T6", "R4T7"]                        # Passive/target combinations to leave out, e.g. ["R1T6", "R4T7"]
STYLE = "means"                               # "boxes" (box plots) or "means" (means with CI error bars)
CONFIDENCE = 95                               # Confidence level (%) of the error bars in the "means" style
MAX_WORKERS = None                            # Processes used to compute G (None: one per CPU)
LISTENING_TEST_ONLY = False                   # Only the source/receiver RIRs convolved with the excitations
POSITIONS = "central"                         # "central", "all", "stress_receivers" or "stress_sources"
BANDS = [500, 1000]                           # Octave band centre frequencies (Hz); ISO 3382-1's G_M is
                                              # the mean of 500 Hz and 1 kHz
FILTER_ORDER = 4                              # Butterworth order of the octave bandpass (applied zero-phase)
SPEED_OF_SOUND = 343.0                        # m/s
# The direct sound is the first arrival within this range of the broadband RIR's peak (see plot_edc.py)
DIRECT_SOUND_THRESHOLD_DB = -12
DIRECT_SOUND_HALF_WINDOW_S = 0.5e-3           # The direct sound window spans this either side of its onset
FIGURE_SIZE = "onecolumn"                     # "onecolumn" or "twocolumn" (IEEE Transactions)
FIGURE_HEIGHT = 2.4                           # Height at twocolumn width (inches)

CONDITIONS = ["passive", "target", "fulloptim", "geq"]  # Plotted in this order
OPTIMISATION_NAME_RE = re.compile(r"opt_ambi_test_R(\d+)_S(\d+)\.wav$")


def read_w_channel(path):
    sample_rate, rir = wavfile.read(path)
    rir = rir[:, 0] if rir.ndim > 1 else rir
    return sample_rate, rir.astype(float)


def octave_bandpass(x, sample_rate, centre_freq):
    """Filter with a zero-phase octave bandpass centred on centre_freq."""
    low, high = centre_freq / np.sqrt(2), centre_freq * np.sqrt(2)
    if high >= sample_rate / 2:
        raise ValueError(f"The octave band at {centre_freq} Hz exceeds the Nyquist frequency ({sample_rate / 2} Hz)")
    sos = signal.butter(FILTER_ORDER, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, x)


def direct_sound(rir, sample_rate):
    """Return (onset sample, the RIR with everything outside the direct sound window zeroed)."""
    magnitude = np.abs(rir)
    onset = int(np.argmax(magnitude >= magnitude.max() * 10 ** (DIRECT_SOUND_THRESHOLD_DB / 20)))
    half_window = int(round(DIRECT_SOUND_HALF_WINDOW_S * sample_rate))
    windowed = np.zeros_like(rir)
    start, stop = max(onset - half_window, 0), onset + half_window + 1
    windowed[start:stop] = rir[start:stop]
    return onset, windowed


def band_energies(x, sample_rate):
    return np.array([np.sum(octave_bandpass(x, sample_rate, f) ** 2) for f in BANDS])


def free_field_constant(directory):
    """Return E_dir * d^2 in each band, the median over a dataset room folder's test RIRs."""
    constants = []
    for path in sorted(directory.glob("H_SR_TEST_R*_S*.wav")):
        sample_rate, rir = read_w_channel(path)
        onset, direct = direct_sound(rir, sample_rate)
        distance = onset / sample_rate * SPEED_OF_SOUND
        constants.append(band_energies(direct, sample_rate) * distance ** 2)
    if not constants:
        raise SystemExit(f"No test RIRs found in {directory}")
    return np.median(constants, axis=0)


def passive_counterpart(path, passive_dir):
    """Return the passive dataset RIR at the same position as an (0-indexed) optimisation output."""
    receiver, source = map(int, OPTIMISATION_NAME_RE.search(path.name).groups())
    return passive_dir / f"H_SR_TEST_R{receiver + 1}_S{source + 1}.wav"


def strength(path, constant, passive_path=None):
    """Return G (dB) in each band, rescaling by the direct sound of passive_path if given."""
    sample_rate, rir = read_w_channel(path)
    if passive_path is not None:
        passive_sample_rate, passive = read_w_channel(passive_path)
        assert passive_sample_rate == sample_rate
        _, direct = direct_sound(passive, sample_rate)
        # Least-squares gain from the passive RIR's direct sound to the optimisation output's
        window = np.flatnonzero(direct)
        gain = np.dot(rir[window], passive[window]) / np.dot(passive[window], passive[window])
        rir = rir / gain
    reference = constant / 10 ** 2
    return 10 * np.log10(band_energies(rir, sample_rate) / reference)


def combinations(skip=()):
    """Return the combinations with both EnvFOA and GEQBaseline results, except those in skip."""
    results_dir = DATASET_ROOT / "Optimisation Results"
    names = [d.name for d in results_dir.iterdir()
             if d.is_dir() and re.fullmatch(r"R\d+T\d+", d.name)
             and (d / "EnvFOA").is_dir() and (d / "GEQBaseline").is_dir()]
    unknown = set(skip) - set(names)
    if unknown:
        print(f"Warning: skipped combinations not found: {', '.join(sorted(unknown, key=combination_sort_key))}")
    return sorted(set(names) - set(skip), key=combination_sort_key)


def strength_tasks(combination_names):
    """Return ({condition: [(combination, path)]}, {path: (dataset folder of its reference, passive path)}).

    The dataset folder is that of the passive room for the optimisation outputs, whose passive path is the
    passive RIR at the same position (None for the dataset RIRs).
    """
    files = {condition: [] for condition in CONDITIONS}
    tasks = {}
    for combination in combination_names:
        combination_files = condition_files(combination, LISTENING_TEST_ONLY, POSITIONS)
        room = re.fullmatch(r"R(\d+)T\d+", combination).group(1)
        passive_dir = DATASET_ROOT / "AAES Dataset" / f"aaes_optimisation_dataset_room_{room}"
        for condition in CONDITIONS:
            paths = combination_files[condition]
            if not paths:
                raise SystemExit(f"No test RIRs found for {CONDITION_LABELS[condition]} ({combination})")
            for path in paths:
                files[condition].append((combination, path))
                if condition in ("passive", "target"):
                    tasks[path] = (path.parent, None)
                else:
                    tasks[path] = (passive_dir, passive_counterpart(path, passive_dir))
    return files, tasks


def _strength_task(args):
    path, constant, passive_path = args
    return strength(path, constant, passive_path)


def compute_strengths(files, tasks):
    """Return {condition: (num_rirs, len(BANDS)) array of G}, computing each unique RIR once."""
    directories = sorted({directory for directory, _ in tasks.values()})
    paths = sorted(tasks)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=multiprocessing.get_context("fork")) as executor:
        print(f"Computing the free-field references of {len(directories)} dataset folders...")
        constants = dict(zip(directories, executor.map(free_field_constant, directories)))
        print(f"Computing G for {len(paths)} unique RIRs...")
        args = [(path, constants[tasks[path][0]], tasks[path][1]) for path in paths]
        results = dict(zip(paths, executor.map(_strength_task, args)))
    return {condition: np.array([results[path] for _, path in entries]) for condition, entries in files.items()}


def print_combination_medians(files, strengths):
    """Print each combination's median G (mean over BANDS) for each condition."""
    print(f"\nMedian G (mean over {', '.join(format_frequency(f) for f in BANDS)}) per combination:")
    print(f"  {'':<6}" + "".join(f"{CONDITION_LABELS[c]:>13}" for c in CONDITIONS))
    combination_names = sorted({c for entries in files.values() for c, _ in entries}, key=combination_sort_key)
    for combination in combination_names:
        medians = []
        for condition in CONDITIONS:
            mask = np.array([c == combination for c, _ in files[condition]])
            medians.append(np.median(strengths[condition][mask].mean(axis=1)))
        print(f"  {combination:<6}" + "".join(f"{m:>10.2f} dB" for m in medians))


def format_frequency(freq):
    return f"{freq / 1000:g} kHz" if freq >= 1000 else f"{freq:g} Hz"


STYLES = ("boxes", "means")


def paired_key(combination, path):
    """Return (combination, receiver, source) for an optimisation output, to pair the conditions."""
    return (combination, *map(int, OPTIMISATION_NAME_RE.search(path.name).groups()))


def compare_fulloptim_to_geq(files, strengths):
    """Print one-tailed Wilcoxon signed-rank tests of whether the G of the full optimisation, averaged over
    BANDS (ISO 3382-1's G_M), exceeds the graphic EQ's.

    The two are paired by combination and source/receiver position. The test is run over every position, and
    over the per-combination means (positions within a combination aren't independent, so this is the more
    conservative test).
    """
    geq_rows = {paired_key(c, p): i for i, (c, p) in enumerate(files["geq"])}
    keys = [paired_key(c, p) for c, p in files["fulloptim"]]
    if set(keys) != set(geq_rows):
        raise SystemExit("The full optimisation and graphic EQ RIRs are not at the same positions")
    fulloptim = strengths["fulloptim"].mean(axis=1)
    geq = strengths["geq"][[geq_rows[k] for k in keys]].mean(axis=1)

    combination_names = sorted({k[0] for k in keys}, key=combination_sort_key)
    masks = [np.array([k[0] == c for k in keys]) for c in combination_names]
    tests = {
        f"per position (n = {len(keys)})": (fulloptim, geq),
        f"per combination means (n = {len(combination_names)})": (
            np.array([fulloptim[m].mean() for m in masks]), np.array([geq[m].mean() for m in masks])),
    }
    bands = ", ".join(format_frequency(f) for f in BANDS)
    print(f"\nOne-tailed Wilcoxon signed-rank test on G averaged over {bands}, "
          f"H1: {CONDITION_LABELS['fulloptim']} > {CONDITION_LABELS['geq']}")
    for test_name, (a, b) in tests.items():
        difference = a - b
        result = stats.wilcoxon(a, b, alternative="greater")
        print(f"  {test_name:<30} median difference = {np.median(difference):5.2f} dB, "
              f"{np.sum(difference > 0)}/{len(difference)} positive, W = {result.statistic:g}, "
              f"p = {result.pvalue:.3g}")


def plot(strengths, figure_size=FIGURE_SIZE, style=STYLE):
    apply_figure_style(figure_size)
    # The figure is laid out on the twocolumn canvas and scaled down to the printed width, so scale the
    # markers and line widths up to print at the same size as plot_means_and_confidence.py's (twocolumn)
    scale = CANVAS_WIDTH / FIGURE_SIZES[figure_size]["width"]
    group_labels = [format_frequency(f) for f in BANDS]
    if len(BANDS) > 1:
        group_labels.append("Mean")
        # Mean of the band levels in dB, as ISO 3382-1 G_M
        strengths = {c: np.column_stack([g, g.mean(axis=1)]) for c, g in strengths.items()}

    box_spacing = 0.5
    group_width = box_spacing * len(CONDITIONS) + 0.25
    group_centers = np.arange(len(group_labels)) * group_width
    offsets = (np.arange(len(CONDITIONS)) - (len(CONDITIONS) - 1) / 2) * box_spacing

    data, positions, box_conditions = [], [], []
    for group_index, centre in enumerate(group_centers):
        print(f"\n{group_labels[group_index]}:")
        for condition, offset in zip(CONDITIONS, offsets):
            column = strengths[condition][:, group_index]
            data.append(column)
            positions.append(centre + offset)
            box_conditions.append(condition)
            mean, half_width = mean_ci(column, CONFIDENCE)
            print(f"  {CONDITION_LABELS[condition]:<12} median = {np.median(column):6.2f} dB, "
                  f"mean = {mean:6.2f} dB +/- {half_width:.2f} dB ({CONFIDENCE}% CI), "
                  f"range = [{column.min():6.2f}, {column.max():6.2f}] dB")

    fig, ax = plt.subplots(figsize=figure_dimensions(FIGURE_HEIGHT), layout="constrained")
    if style == "boxes":
        bp = ax.boxplot(data, positions=positions, widths=0.42, patch_artist=True, capwidths=0.3)
        for i, condition in enumerate(box_conditions):
            colour = CONDITION_COLOURS[condition]
            bp["boxes"][i].set(facecolor=colour, edgecolor=colour)
            # White on the dark target colour so the median stays visible
            bp["medians"][i].set(color="white" if condition == "target" else "black", linewidth=1 * scale)
            bp["fliers"][i].set(markerfacecolor=colour, markeredgecolor=colour, markersize=2 * scale)
        for line in bp["whiskers"] + bp["caps"]:
            line.set_linewidth(0.8 * scale)
        legend_elements = [Patch(facecolor=CONDITION_COLOURS[c], edgecolor=CONDITION_COLOURS[c],
                                 label=CONDITION_LABELS[c]) for c in CONDITIONS]
    elif style == "means":
        # Styled as plot_means_and_confidence.py's means (matplotlib's default 1 pt marker edge, scaled)
        for values, x, condition in zip(data, positions, box_conditions):
            mean, half_width = mean_ci(values, CONFIDENCE)
            ax.errorbar(x, mean, yerr=half_width, fmt=CONDITION_MARKERS[condition], color=CONDITION_COLOURS[condition],
                        markeredgewidth=1 * scale, markersize=4 * scale,
                        capsize=2 * scale, elinewidth=1 * scale, capthick=1 * scale, zorder=3)
        legend_elements = [Line2D([], [], linestyle="none", marker=CONDITION_MARKERS[c], color=CONDITION_COLOURS[c],
                                  markeredgewidth=1 * scale, markersize=4 * scale,
                                  label=CONDITION_LABELS[c])
                           for c in CONDITIONS]
    else:
        raise ValueError(f"Unrecognised style {style!r}, expected one of {STYLES}")

    for boundary in group_centers[:-1] + group_width / 2:
        ax.axvline(boundary, color=GRID_COLOUR, linewidth=0.8 * scale, zorder=0)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{chr(ord('a') + i)}. {label}" for i, label in enumerate(group_labels)])
    ax.set_xlim(group_centers[0] - group_width / 2, group_centers[-1] + group_width / 2)
    ax.set_ylabel(r"Strength $G$ (dB)")
    ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=0.8 * scale)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(plt.rcParams["axes.linewidth"] * scale)
    ax.tick_params(width=plt.rcParams["xtick.major.width"] * scale, length=plt.rcParams["xtick.major.size"] * scale)

    add_legend(fig, legend_elements, [h.get_label() for h in legend_elements], top=True, columnspacing=1.0,
               handletextpad=0.3, borderpad=0, borderaxespad=0.2)
    fig.get_layout_engine().set(h_pad=0)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_figure_size_arg(parser, FIGURE_SIZE)
    parser.add_argument("--skip", nargs="*", default=SKIP_COMBINATIONS, metavar="COMBINATION",
                        help="Passive/target combinations to leave out, e.g. --skip R1T6 R4T7 "
                             "(default: %(default)s)")
    parser.add_argument("--style", default=STYLE, choices=STYLES,
                        help=f"Box plots, or means with {CONFIDENCE}%% confidence interval error bars "
                             "(default: %(default)s)")
    args = parser.parse_args()

    combination_names = combinations(args.skip)
    if not combination_names:
        raise SystemExit(f"No combinations with EnvFOA and GEQBaseline folders found under {DATASET_ROOT}")
    print(f"Combinations: {', '.join(combination_names)}")

    files, tasks = strength_tasks(combination_names)
    strengths = compute_strengths(files, tasks)
    print_combination_medians(files, strengths)
    compare_fulloptim_to_geq(files, strengths)
    for condition, values in strengths.items():
        print(f"{CONDITION_LABELS[condition]}: {len(values)} RIRs per box")
    plot(strengths, args.figure_size, args.style)


if __name__ == "__main__":
    main()

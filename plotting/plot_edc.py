"""Plot the energy decay curves (Schroeder backward integration) of the passive room, the target
room, the full optimisation (EnvFOA) and the graphic EQ baseline (GEQBaseline) for one
passive/target combination, source/receiver position and octave band.

Each RIR's W channel is filtered with an octave bandpass centred on the given frequency, then
backward integrated. Time is relative to the direct sound and each EDC is normalised to 0 dB at
the direct sound, as the rooms have different source/receiver distances and the RIRs have
different gains.

Source and receiver indices are 0-indexed, as in the optimisation outputs (the dataset H_SR_TEST
files are 1-indexed).

Usage:
    python plotting/plot_edc.py R2T4 [--source 3] [--receiver 1] [--centre-freq 1000]
                                     [--figure-size {onecolumn,twocolumn}]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

from common import (
    CANVAS_WIDTH, CONDITION_COLOURS, CONDITION_LABELS, GRID_COLOUR, add_figure_size_arg, apply_figure_style, figure_dimensions,
)

# Settings
SOURCE = 3                                    # Default source index (0-indexed); S3 is the listening test source
RECEIVER = 1                                  # Default receiver index (0-indexed); R1 is the listening test receiver
ROOM_TARGET_COMBO = "R2T2"
CENTRE_FREQ = 1000                            # Default octave band centre frequency (Hz)
FILTER_ORDER = 4                              # Butterworth order of the octave bandpass (applied zero-phase)
LEVEL_LIMITS_DB = (-60, 1)                    # Level range, relative to the energy at the direct sound
TIME_LIMIT_S = None                           # Upper time limit (s); None to stop just after every EDC reaches
                                              # the bottom of LEVEL_LIMITS_DB
# The direct sound is the first arrival within this range of the broadband RIR's peak (see
# plot_early_reflections.py: an early reflection can be stronger than the direct sound)
DIRECT_SOUND_THRESHOLD_DB = -12
DATASET_ROOT = Path("/Volumes/WillFootage/PhD/AAESOptimisationDatasetRIRs")
FIGURE_SIZE = "onecolumn"                     # "onecolumn" or "twocolumn" (IEEE Transactions)
ASPECT_RATIO = 3 / 5                          # Figure height:width

CONDITIONS = ["passive", "target", "fulloptim", "geq"]  # Plotted in this order
LINE_STYLES = {"passive": ":", "target": "--", "fulloptim": "-", "geq": "-."}
LINE_COLOURS = {"passive": "black", "target": "black",
                "fulloptim": CONDITION_COLOURS["fulloptim"], "geq": CONDITION_COLOURS["geq"]}

COMBINATION_RE = re.compile(r"^R(\d+)T(\d+)$")


def condition_files(combination, source, receiver):
    """Return {condition: RIR path} for a combination such as "R2T4" and 0-indexed source/receiver."""
    match = COMBINATION_RE.match(combination)
    if match is None:
        raise ValueError(f"Combination must look like R2T4, got {combination!r}")
    room, target = match.groups()
    r, s = receiver, source
    dataset_dir = DATASET_ROOT / "AAES Dataset"
    return {
        "passive": dataset_dir / f"aaes_optimisation_dataset_room_{room}" / f"H_SR_TEST_R{r + 1}_S{s + 1}.wav",
        "target": dataset_dir / f"aaes_optimisation_dataset_target_room_{target}" / f"H_SR_TEST_R{r + 1}_S{s + 1}.wav",
        "fulloptim": DATASET_ROOT / "Optimisation Results" / combination / "EnvFOA" / f"opt_ambi_test_R{r}_S{s}.wav",
        "geq": DATASET_ROOT / "Optimisation Results" / combination / "GEQBaseline" / f"opt_ambi_test_R{r}_S{s}.wav",
    }


def octave_bandpass(rir, sample_rate, centre_freq):
    """Filter an RIR with a zero-phase octave bandpass centred on centre_freq."""
    low, high = centre_freq / np.sqrt(2), centre_freq * np.sqrt(2)
    if high >= sample_rate / 2:
        raise ValueError(f"The octave band at {centre_freq} Hz exceeds the Nyquist frequency ({sample_rate / 2} Hz)")
    sos = signal.butter(FILTER_ORDER, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, rir)


def energy_decay_curve(path, centre_freq):
    """Return (times in s, EDC in dB), relative to the direct sound, of an RIR's W channel in an octave band."""
    sample_rate, spatial_rir = wavfile.read(path)
    rir = spatial_rir[:, 0] if spatial_rir.ndim > 1 else spatial_rir
    rir = rir.astype(float)

    # Find the direct sound on the broadband RIR, so it's at the same time in every band
    magnitude = np.abs(rir)
    onset = np.argmax(magnitude >= magnitude.max() * 10 ** (DIRECT_SOUND_THRESHOLD_DB / 20))

    filtered = octave_bandpass(rir, sample_rate, centre_freq)[onset:]
    edc = np.cumsum(filtered[::-1] ** 2)[::-1]
    with np.errstate(divide="ignore"):
        edc_db = 10 * np.log10(edc / edc[0])
    times = np.arange(len(edc)) / sample_rate
    return times, edc_db


def format_frequency(freq):
    return f"{freq / 1000:g} kHz" if freq >= 1000 else f"{freq:g} Hz"


def plot(edcs, figure_size=FIGURE_SIZE):
    apply_figure_style(figure_size)
    fig, ax = plt.subplots(layout="constrained", figsize=figure_dimensions(CANVAS_WIDTH * ASPECT_RATIO))
    for condition in CONDITIONS:
        times, edc_db = edcs[condition]
        ax.plot(times, edc_db, color=LINE_COLOURS[condition], label=CONDITION_LABELS[condition], linewidth=2,
                linestyle=LINE_STYLES[condition])

    time_limit = TIME_LIMIT_S
    if time_limit is None:
        time_limit = 1.05 * max(times[edc_db >= LEVEL_LIMITS_DB[0]][-1] for times, edc_db in edcs.values())
    ax.set_xlim(0, time_limit)
    ax.set_ylim(LEVEL_LIMITS_DB)
    ax.set_xlabel("Time after direct sound (s)")
    ax.set_ylabel("Energy Decay Curve (dB)")
    ax.grid(True, color=GRID_COLOUR, linewidth=1.2)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combination", type=str, default=ROOM_TARGET_COMBO, help="Passive/target combination, e.g. R2T4")
    parser.add_argument("--source", type=int, default=SOURCE, help="Source index, 0-indexed (default: %(default)s)")
    parser.add_argument("--receiver", type=int, default=RECEIVER,
                        help="Receiver index, 0-indexed (default: %(default)s)")
    parser.add_argument("--centre-freq", type=float, default=CENTRE_FREQ,
                        help="Octave band centre frequency in Hz (default: %(default)s)")
    add_figure_size_arg(parser, FIGURE_SIZE)
    args = parser.parse_args()

    files = condition_files(args.combination, args.source, args.receiver)
    for path in files.values():
        if not path.exists():
            parser.error(f"RIR not found: {path}")
    edcs = {condition: energy_decay_curve(path, args.centre_freq) for condition, path in files.items()}
    plot(edcs, args.figure_size)


if __name__ == "__main__":
    main()

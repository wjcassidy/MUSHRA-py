"""Plot the early reflections of the echograms of the passive room, the target room
and the full optimisation (EnvFOA) for the listening test source/receiver position.

Each column is one passive/target combination. Echograms use the W channel, with
time and level relative to the direct sound, as the rooms have different
source/receiver distances and the RIRs have different gains.

Usage:
    python plotting/plot_early_reflections.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from common import CONDITION_COLOURS, GRID_COLOUR

# Settings
COMBINATIONS = ["R2T2", "R2T4"]               # Passive/target combinations to plot
# 0-indexed, as in the optimisation outputs (the dataset H_SR_TEST files are 1-indexed). Matches
# PyRES-ClosedLoop/examples/convolve_*_irs_with_excitation.py. S3 is the Handclaps source, and is
# also used by the Classical and Pop excitations alongside S0-S2
LISTENING_TEST_RECEIVER = 1
LISTENING_TEST_SOURCE = 3
TIME_LIMITS_MS = (-5, 30)                    # Time window, relative to the direct sound
LEVEL_LIMITS_DB = (-40, 6)                    # Level range, relative to the direct sound
# The direct sound is the first arrival within this range of the RIR's peak. The peak alone isn't
# enough, as an early reflection can be stronger than the direct sound (e.g. R2T4, S3)
DIRECT_SOUND_THRESHOLD_DB = -12
DIRECT_SOUND_WINDOW_MS = 1                    # Window after the onset used to measure the direct sound level
DATASET_ROOT = Path("/Volumes/WillFootage/PhD/AAESOptimisationDatasetRIRs")

CONDITION_LABELS = {"passive": "Passive", "target": "Target", "fulloptim": "Full optimisation (EnvFOA)"}


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


def plot(echograms):
    combinations = list(echograms)
    conditions = list(CONDITION_LABELS)
    fig, axes = plt.subplots(
        len(conditions), len(combinations), sharex=True, sharey=True, squeeze=False,
        figsize=(5.5 * len(combinations) + 1, 2.2 * len(conditions) + 0.8),
    )
    for column, combination in enumerate(combinations):
        for row, condition in enumerate(conditions):
            ax = axes[row, column]
            times, levels = echograms[combination][condition]
            visible = (times >= TIME_LIMITS_MS[0]) & (times <= TIME_LIMITS_MS[1])
            ax.fill_between(times[visible], LEVEL_LIMITS_DB[0], levels[visible], color=CONDITION_COLOURS[condition],
                            linewidth=0.4)
            ax.set_title(f"{combination}: {CONDITION_LABELS[condition]}", loc="left", fontsize=11)
            ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    axes[0, 0].set_xlim(TIME_LIMITS_MS)
    axes[0, 0].set_ylim(LEVEL_LIMITS_DB)
    for ax in axes[:, 0]:
        ax.set_ylabel("Level (dB re direct)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time after direct sound (ms)")

    fig.tight_layout()
    plt.show()


def main():
    echograms = {}
    for combination in COMBINATIONS:
        echograms[combination] = {condition: echogram(path) for condition, path in condition_files(combination).items()}
    plot(echograms)


if __name__ == "__main__":
    main()

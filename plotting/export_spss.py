"""Combine the results CSVs of the included subjects into a single CSV for SPSS.

Subjects are selected by exclusion_criteria.included_results(), so the settings in
exclusion_criteria.py (including BYPASS) apply here as they do to the plots.

Two layouts are available:
  - long: one row per rating, with the factors as columns. Use with Linear Mixed
    Models (MIXED).
  - wide: one row per subject, with one rating column per cell of the design (e.g.
    R1T5_Beet_target). Use with GLM Repeated Measures. With repeats included,
    repeated trials get their own columns, suffixed _repeat.

In both layouts, the subject column is the only subject identifier: a counter from
1 to N over the included subjects, so a subject's number changes if the exclusions
change. Results filenames are not exported.

The output must not go directly in the results folder, as every CSV there is read
as a subject's results.

Defaults are set in the settings block below; command-line arguments override them.

Usage:
    python plotting/export_spss.py [--results-dir DIR] [--output FILE]
                                   [--format {long,wide}] [--[no-]include-repeats]
"""

import argparse
import csv
from pathlib import Path

from common import CONDITIONS, DEFAULT_RESULTS_DIR, FILENAME_RE, combination_sort_key
from exclusion_criteria import included_results

# Settings (defaults for the command-line arguments)
RESULTS_DIR = DEFAULT_RESULTS_DIR             # Folder containing the results CSVs
OUTPUT = DEFAULT_RESULTS_DIR / "spss" / "mushra_results.csv"  # Combined CSV to write
FORMAT = "long"                               # Layout: "long" or "wide"
INCLUDE_REPEATS = False                       # Include repeated trials (repeat=1)
FORMATS = ["long", "wide"]

LONG_COLUMNS = ["subject", "order", "page", "room_target_combo", "excitation", "condition",
                "repeated", "rating"]


def load_rows(paths, include_repeats):
    """Return one dict per rating, in LONG_COLUMNS, for the given results CSVs."""
    rows = []
    subject = 1
    for path in paths:
        first_row = len(rows)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if not include_repeats and row["repeat"] == "1":
                    continue
                match = FILENAME_RE.match(row["filename"])
                if match is None:
                    print(f"Skipping unrecognised filename {row['filename']} in {path.name}")
                    continue
                combination, signal, condition = match.groups()
                rows.append({
                    "subject": subject, "order": row["order"], "page": row["page"],
                    "room_target_combo": combination, "excitation": signal, "condition": condition,
                    "repeated": row["repeat"], "rating": row["rating"],
                })
        if len(rows) > first_row:  # Keep the numbering contiguous if a file has no usable rows
            subject += 1
    return rows


def cell_sort_key(row):
    condition = row["condition"]
    condition_index = CONDITIONS.index(condition) if condition in CONDITIONS else len(CONDITIONS)
    return row["repeated"], combination_sort_key(row["room_target_combo"]), row["excitation"], condition_index, condition


def cell_name(row):
    name = f"{row['room_target_combo']}_{row['excitation']}_{row['condition']}"
    return f"{name}_repeat" if row["repeated"] == "1" else name


def to_wide(rows):
    """Return (columns, rows) with one row per subject and one column per design cell."""
    cells = list(dict.fromkeys(cell_name(row) for row in sorted(rows, key=cell_sort_key)))
    subjects = {}
    for row in rows:
        subject = subjects.setdefault(row["subject"], {"subject": row["subject"]})
        subject[cell_name(row)] = row["rating"]
    return ["subject"] + cells, list(subjects.values())


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help="Folder containing the results CSVs (default: %(default)s)")
    parser.add_argument("--output", default=str(OUTPUT), help="Combined CSV to write (default: %(default)s)")
    parser.add_argument("--format", default=FORMAT, choices=FORMATS, help="Layout (default: %(default)s)")
    parser.add_argument("--include-repeats", default=INCLUDE_REPEATS, action=argparse.BooleanOptionalAction,
                        help="Include repeated trials (repeat=1) (default: %(default)s)")
    args = parser.parse_args()

    output = Path(args.output)
    if output.resolve().parent == Path(args.results_dir).resolve():
        raise SystemExit("--output can't be directly in the results folder: it would be read as a subject")

    rows = load_rows(included_results(args.results_dir), args.include_repeats)
    if not rows:
        raise SystemExit(f"No results found in {args.results_dir}")

    columns, rows = to_wide(rows) if args.format == "wide" else (LONG_COLUMNS, rows)
    write_csv(output, columns, rows)
    print(f"Wrote {len(rows)} rows and {len(columns)} columns to {output}")


if __name__ == "__main__":
    main()

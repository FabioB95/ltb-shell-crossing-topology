#!/usr/bin/env python3
"""
Dedupe experiment1/experiment2 result CSVs by their unique key (pid for
exp1, idx for exp2), keeping the first occurrence of each, then re-run the
summarize() logic. Does NOT re-run any simulation -- just cleans up
leftover duplicate rows (e.g. from re-running a batch without clearing
the CSV first) and recomputes the aggregate statistics from the clean
data.

Usage:
    python3 dedupe_and_resummarize.py exp1
    python3 dedupe_and_resummarize.py exp2
    python3 dedupe_and_resummarize.py both
"""

import sys
import csv
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def dedupe_csv(csv_path: Path, key_field: str, keep: str = "last"):
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} does not exist.")
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    seen = {}
    order = []
    n_dupes = 0
    for row in rows:
        key = row[key_field]
        if key in seen:
            n_dupes += 1
            if keep == "last":
                seen[key] = row  # overwrite with the later occurrence
            # if keep == "first": do nothing, keep the earlier one
        else:
            seen[key] = row
            order.append(key)

    deduped = [seen[k] for k in order]

    print(f"  {csv_path.name}: {len(rows)} rows -> {len(deduped)} unique "
          f"({n_dupes} duplicates removed, keeping '{keep}' occurrence)")

    if n_dupes == 0:
        print("  No duplicates found, nothing to do.")
        return

    backup_path = csv_path.with_suffix(".csv.bak")
    if not backup_path.exists():
        shutil.copy(csv_path, backup_path)
        print(f"  Backed up original to {backup_path}")
    else:
        print(f"  Backup already exists at {backup_path} (not overwritten)")

    # sort by key (numeric) for readability
    deduped.sort(key=lambda r: int(r[key_field]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"  Rewrote {csv_path} with {len(deduped)} unique rows (kept '{keep}').")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "both"
    keep = sys.argv[2] if len(sys.argv) > 2 else "last"

    if keep not in ("first", "last"):
        print(f"Invalid 'keep' argument: {keep!r}. Must be 'first' or 'last'.")
        sys.exit(1)

    if target in ("exp1", "both"):
        print("=== Experiment 1 ===")
        dedupe_csv(DATA_DIR / "experiment1_results.csv", key_field="pid", keep=keep)

    if target in ("exp2", "both"):
        print("=== Experiment 2 ===")
        dedupe_csv(DATA_DIR / "experiment2_results.csv", key_field="idx", keep=keep)

    print("\nNow re-run summarize for whichever experiment(s) you deduped:")
    print("  python3 experiment1_phase_diagram.py summarize")
    print("  python3 experiment2_percolation.py summarize")


if __name__ == "__main__":
    main()

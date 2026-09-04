#!/usr/bin/env python3
"""Print the Experiment 2 aggregate table in a readable format."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
json_path = PROJECT_ROOT / "data" / "experiment2_summary.json"

with open(json_path, "r") as f:
    s = json.load(f)

print(f"sigma_crit estimate: {s.get('sigma_crit_estimate')}")
print(f"n_total_profiles: {s.get('n_total_profiles')}")
print()
header = f"{'sigma':>7} {'n_br':>7} {'n_scs':>7} {'frac_scs>=1':>12} {'f_lcc':>8} {'fully_conn':>11}"
print(header)
print("-" * len(header))
for a in s["aggregate"]:
    print(f"{a['sigma']:>7.3f} {a['mean_n_branches']:>7.2f} {a['mean_n_scs_strong']:>7.2f} "
          f"{a['frac_at_least_one_strong_scs']:>12.2f} {a['mean_f_lcc']:>8.3f} "
          f"{a['frac_fully_connected']:>11.3f}")

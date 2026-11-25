"""Validate live timings against event replay and print the ablation summaries."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("results", type=Path)
    cli.add_argument("--expected-runs", type=int, help="Check for this many paired repetitions on every task")
    args = cli.parse_args()
    rows = list(csv.DictReader((args.results / "thresholds.csv").open()))
    assert len(rows) == 51
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], int(r["realizations"]))].append(r)
        assert float(r["lp_saved_pct"]) >= 0 and float(r["losses_saved_pct"]) >= 0
        if r["published_cutoff"] and r["cutoff_matches_published"] != "True":
            print("Published cutoff discrepancy:", r["family"], r["realizations"], r["quantile"], r["cutoff"], r["published_cutoff"])
        if r["published_joint_lp"] and (r["published_joint_lp"] != r["joint_lp"] or
                                        r["published_joint_losses"] != r["joint_losses"]):
            print("Published count discrepancy:", r["family"], r["realizations"], r["quantile"],
                  r["joint_lp"], r["published_joint_lp"], r["joint_losses"], r["published_joint_losses"])
    times = defaultdict(list)
    repetitions = defaultdict(set)
    with (args.results / "timings.jsonl").open() as stream:
        for line in stream:
            t = json.loads(line)
            key = (t["family"], t["realizations"])
            method = t["method"]
            assert method in ("independent", "joint") and t["seconds"] > 0
            assert t["rep"] not in repetitions[key, method], (key, method, t["rep"])
            repetitions[key, method].add(t["rep"])
            chosen, = [r for r in groups[key] if r["chosen"] == "True"]
            assert math.isclose(t["cutoff"], float(chosen["cutoff"]), rel_tol=0, abs_tol=1e-12)
            assert t["lp_solves"] == int(chosen[t["method"]+"_lp"])
            assert t["loss_evals"] == int(chosen[t["method"]+"_losses"])
            times[key, t["method"]].append(t["seconds"])
    for key, group in groups.items():
        assert len(group) == 17
        print(key)
        for col in ["lp_saved_pct", "losses_saved_pct"]:
            vals = [float(r[col]) for r in group]
            print(col, "min / median / max:", *[round(f(vals), 3) for f in [min, median, max]])
        for method in ["independent", "joint"]:
            vals = times[key, method]
            if args.expected_runs is not None:
                assert repetitions[key, method] == set(range(1, args.expected_runs + 1)), (key, method)
            if vals:
                print(method, "wall time (s)", [round(x, 3) for x in vals], "median", round(median(vals), 3))
    set_cover = [r for r in rows if r["family"] == "set_cover"]
    assert len(set_cover) == 34
    for metric in ["lp_saved_pct", "losses_saved_pct"]:
        print("Set-covering median across 34 settings:", metric, round(median(float(r[metric]) for r in set_cover), 3))


if __name__ == "__main__":
    main()

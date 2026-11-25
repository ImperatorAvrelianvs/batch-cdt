"""Reproduce saved operation counts; --full also rebuilds the synthetic benchmarks."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
JOINT = BASE/"joint_stopping"
EXT = BASE/"extensions"
RESULTS = EXT/"results"


def run(script, *args):
    print(f"Running {script.name}", flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable,str(script),*map(str,args)],cwd=ROOT,env=env,check=True)


def compare_csv(actual, reference, varying=()):
    result = pd.read_csv(actual)
    expected = pd.read_csv(reference)
    pd.testing.assert_frame_equal(result.drop(columns=list(varying)),expected.drop(columns=list(varying)),
                                  check_dtype=False, check_exact=False, rtol=1e-8, atol=1e-9)


def verify(out, full):
    run(JOINT/"compare.py","replay","--repo",ROOT,"--traces",JOINT/"traces","--out",out)
    compare_csv(out/"thresholds.csv",JOINT/"thresholds.csv")
    run(JOINT/"check_results.py", JOINT, "--expected-runs", 3)
    run(EXT/"graded_comparison.py","--traces",JOINT/"traces","--thresholds",JOINT/"thresholds.csv",
        "--out",out/"graded_counts.csv")
    compare_csv(out/"graded_counts.csv",RESULTS/"graded_counts.csv")
    run(EXT/"check_edge_cases.py")
    if full:
        for script,table in [("selection.py","scaling.csv"),("split_study.py","pool_splits.csv")]:
            run(EXT/script,"--out",out/table)
            compare_csv(out/table,RESULTS/table)
        run(EXT/"planner_study.py","--out",out)
        compare_csv(out/"planner_study.csv",RESULTS/"planner_study.csv",("full_seconds","joint_seconds"))
        compare_csv(out/"milp_latency.csv",RESULTS/"milp_latency.csv",("seconds",))
        run(EXT/"exhaustive_milp.py","--out",out)
        for table in ("exhaustive_milp_scenes.csv","exhaustive_milp_cutoffs.csv"):
            compare_csv(out/table,RESULTS/table)
    print("All requested checks passed.",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    with TemporaryDirectory() as work:
        verify(Path(work),args.full)

"""Record true costs of every event once, using the public, unmodified benchmark."""
import argparse
import multiprocessing as mp
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=2)
def family_data(family):
    import run_timing
    return run_timing.make_family(family)


def initialize(repo, cores):
    sys.path.insert(0, repo)
    if cores:
        worker = mp.current_process()._identity[-1] - 1
        os.sched_setaffinity(0, {cores[worker % len(cores)]})


def record(args):
    import run_timing
    family, i, out = args
    target = out / f"{family}_{i:02d}.npz"
    if target.exists():
        return family, i
    scenes, realizations, realization_cost = family_data(family)
    scene = scenes[i]
    stats = {"lp": 0}
    endpoints, solves, costs = [], [], []
    for endpoint, x in run_timing.lazy_path(scene, stats):
        endpoints.append(endpoint)
        solves.append(stats["lp"])
        costs.append([realization_cost(scene, x, d) for d in range(realizations)])
    tmp = target.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, endpoints=endpoints, solves=solves, costs=costs, total_lp=stats["lp"])
    tmp.replace(target)
    return family, i


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=1)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(f, i, args.out) for f in ("knapsack", "set_cover") for i in range(48)]
    finished = {"knapsack": 0, "set_cover": 0}
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    with mp.Pool(args.workers, initializer=initialize, initargs=(str(args.repo.resolve()), cores)) as pool:
        for family, i in pool.imap_unordered(record, tasks):
            finished[family] += 1
            if finished[family] % 4 == 0:
                print(f"{family}: {finished[family]}/48 scenes", flush=True)

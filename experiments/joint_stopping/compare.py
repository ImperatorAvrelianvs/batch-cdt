"""Compare a global heap sweep with independent right-to-left scans on the same data."""
import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy


def independent(n, k, first, nxt, loss):
    """Stop each scene at its first unsafe event, then select its joint rank."""
    boundaries = []
    evaluations = 0
    for i in range(n):
        event = first(i)
        boundary = -math.inf
        while event is not None:
            endpoint, decision = event
            evaluations += 1
            if loss(i, decision):
                boundary = endpoint
                break
            event = nxt(i)
        boundaries.append(boundary)
    return sorted(boundaries, reverse=True)[k], evaluations


def live_run(benchmark, scenes, realizations, realization_cost, tau, method):
    start = time.perf_counter()
    cpu_start = time.process_time()
    stats = [{"lp": 0} for _ in scenes]
    generators = [benchmark.lazy_path(scene, count) for scene, count in zip(scenes, stats)]
    k = math.floor((len(scenes) + 1) * benchmark.ALPHA - 1)

    def loss(i, decision):
        return np.mean([realization_cost(scenes[i], decision, d) for d in range(realizations)]) > tau

    routine = benchmark.sweep if method == "joint" else independent
    boundary, losses = routine(len(scenes), k, lambda i: next(generators[i]),
                               lambda i: next(generators[i], None), loss)
    return dict(method=method, seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu_start,
                loss_evals=losses,
                lp_solves=sum(s["lp"] for s in stats), cutoff=boundary)


def fresh(args, benchmark):
    import random
    for family, realizations, reps in args.case:
        realizations, reps = int(realizations), int(reps)
        published = json.loads((args.repo / f"timing_{family}.json").read_text())
        chosen = next(r for r in published["results"] if r["T"] == realizations)["chosen"]
        scenes, _, realization_cost = benchmark.make_family(family)
        # Both methods get the same initialized solver before timing starts.
        _, _, x = benchmark.P.lp(scenes[0], 1.0)
        realization_cost(scenes[0], x, 0)
        method_order = ["independent", "joint"]
        random.Random(f"{family}/{realizations}/42").shuffle(method_order)
        for rep in range(reps):
            for method in method_order[::(-1 if rep % 2 else 1)]:
                res = live_run(benchmark, scenes, realizations, realization_cost, chosen["tau"], method)
                res.update(family=family, realizations=realizations, rep=rep+1, tau=chosen["tau"])
                with (args.out / "timings.jsonl").open("a") as f:
                    f.write(json.dumps(res) + "\n")
                print(res, flush=True)


def load_traces(trace_dir, family):
    paths = []
    for i in range(48):
        with np.load(trace_dir / f"{family}_{i:02d}.npz") as d:
            paths.append({key: d[key] for key in d})
    return paths


def replay(paths, mean_costs, tau, k, routine):
    # Preserve generator lookahead and charge final unsuccessful next() calls.
    positions = [-1] * len(paths)
    lp_by_scene = [0] * len(paths)

    def pull(i):
        pos = positions[i] + 1
        positions[i] = pos
        p = paths[i]
        if pos == len(p["endpoints"]):
            lp_by_scene[i] = int(p["total_lp"])
            return None
        lp_by_scene[i] = int(p["solves"][pos])
        return float(p["endpoints"][pos]), pos

    boundary, losses = routine(len(paths), k, pull, pull, lambda i, pos: mean_costs[i][pos] > tau)
    return boundary, int(losses), int(sum(lp_by_scene))


def full_analysis(args, benchmark):
    rows = []
    for family in ("knapsack", "set_cover"):
        paths = load_traces(args.traces, family)
        published = json.loads((args.repo / f"timing_{family}.json").read_text())
        n = len(paths)
        k = math.floor((n + 1) * benchmark.ALPHA - 1)
        for p in paths:
            assert np.all(np.diff(p["endpoints"]) <= 1e-10)
        for setting in published["results"]:
            realizations = setting["T"]
            mean_costs = [p["costs"][:, :realizations].mean(axis=1) for p in paths]
            quantiles = np.linspace(.1, .9, 17)
            thresholds = np.quantile(np.concatenate(mean_costs), quantiles)
            # Use the exact published machine thresholds on archived interior rows,
            # including the headline rows; regenerate any omitted boundary row.
            archived = iter(setting["threshold_sweep"])
            for quantile, generated_tau in zip(quantiles, thresholds):
                # Upstream archives all 17 quantiles for cover. Knapsack q=0.10
                # hits 1.0 (fallback), so the archive's interior-only filter omits it.
                boundary_only = family == "knapsack" and quantile == quantiles[0]
                prior = None if boundary_only else next(archived)
                tau = float(generated_tau) if prior is None else prior["tau"]
                bounds = []
                for p, costs in zip(paths, mean_costs):
                    unsafe = np.flatnonzero(costs > tau)
                    bounds.append(float(p["endpoints"][unsafe[0]]) if len(unsafe) else -math.inf)
                exhaustive = sorted(bounds, reverse=True)[k]
                joint, j_evals, j_lp = replay(paths, mean_costs, tau, k, benchmark.sweep)
                local, l_evals, l_lp = replay(paths, mean_costs, tau, k, independent)
                assert joint == local == exhaustive
                chosen = math.isclose(tau, setting["chosen"]["tau"], rel_tol=0, abs_tol=1e-10)
                rows.append(dict(family=family, realizations=realizations, quantile=round(float(quantile), 2), tau=tau,
                                 cutoff=exhaustive, chosen=chosen, joint_lp=j_lp, independent_lp=l_lp,
                                 joint_losses=j_evals, independent_losses=l_evals,
                                 lp_saved_pct=100*(1-j_lp/l_lp), losses_saved_pct=100*(1-j_evals/l_evals),
                                 scenes_always_safe=sum(v == -math.inf for v in bounds),
                                 full_lp=sum(int(p["total_lp"]) for p in paths),
                                 full_losses=sum(len(p["endpoints"]) for p in paths),
                                 tau_from_fresh_trace=float(generated_tau),
                                 published_cutoff=(prior["cutoff"] if prior else None),
                                 published_joint_lp=(prior["lp_lazy"] if prior else None),
                                 published_joint_losses=(prior["loss_evals_sweep"] if prior else None),
                                 cutoff_matches_published=(math.isclose(joint, prior["cutoff"], abs_tol=1e-8)
                                                           if prior else None)))
            assert next(archived, None) is None
    with (args.out / "thresholds.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["chosen"]:
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["timing", "replay"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--traces", type=Path, default=Path("traces"))
    parser.add_argument("--case", action="append", nargs=3, metavar=("FAMILY", "REALIZATIONS", "REPEATS"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import run_timing as benchmark
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}", flush=True)
    if args.mode == "timing":
        fresh(args, benchmark)
    else:
        full_analysis(args, benchmark)

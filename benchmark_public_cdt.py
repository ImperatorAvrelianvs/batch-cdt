"""Two small public-data tests of black-box batch CDT calibration.

Inputs (unmodified numeric data):
  OR-Library mknap2: 48 multidimensional knapsack instances.
  OR-Library scp41--scp410: ten set-cover instances, sampled requirements.

Both planners are linear programs.  The integer benchmark problems are
relaxed to allow fractional decisions; results must be described as LP
applications of the published coefficient data, not original MIP scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


HERE = Path(__file__).resolve().parent / "public_benchmarks"


def read_knapsack(path=HERE / "orlib_mknap2.txt"):
    """Return the 48 named instances in the OR-Library text file."""
    pieces = re.split(r"(?m)^\s*problem\s+([^\n]+)\n", path.read_text())
    instances = []
    for name, block in zip(pieces[1::2], pieces[2::2]):
        numbers = []
        for line in block.splitlines():
            line = line.strip()
            if line and any(char.isdigit() for char in line) and re.fullmatch(r"[\d+\-.\s]+", line):
                numbers.extend(float(token) for token in line.split())
        if not numbers:
            continue
        n_resources, n_items = map(int, numbers[:2])
        count = 2 + n_items + n_resources + n_resources * n_items
        assert len(numbers) == count + 1, (name, len(numbers), count + 1)
        values = np.asarray(numbers[2:], dtype=float)
        profit = values[:n_items]
        capacity = values[n_items:n_items + n_resources]
        resource = values[n_items + n_resources:-1].reshape(n_resources, n_items)
        instances.append((name.strip(), profit, resource, capacity))
    assert len(instances) == 48, len(instances)
    return instances


def read_set_cover(path=HERE / "orlib_scp41.txt"):
    """Return the 200-by-1000 coverage matrix and column costs."""
    values = np.fromstring(path.read_text(), sep=" ", dtype=np.int64)
    n_rows, n_cols = map(int, values[:2])
    costs = values[2:2+n_cols].astype(float)
    position = 2+n_cols
    rows, cols = [], []
    for i in range(n_rows):
        count = int(values[position]); position += 1
        chosen = values[position:position+count] - 1
        assert len(chosen) == count and np.all((0 <= chosen) & (chosen < n_cols))
        rows.extend([i]*count)
        cols.extend(chosen.tolist())
        position += count
    assert position == len(values), (position, len(values))
    matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_rows, n_cols))
    assert matrix.nnz == len(rows)
    return matrix, costs


def build_knapsack():
    scenes = []
    for name, profit, resource, capacity in read_knapsack():
        # Fractional project selection: maximize normalized profit, with a
        # penalty for using the first constrained resource.
        base = -profit / profit.sum()
        risk = resource[0] / capacity[0]
        scenes.append({"name": name, "base": base, "risk": risk,
                       "A": resource, "b": capacity,
                       "bound": (0, 1), "threshold": 0.75})
    return scenes


def build_cover(seed=20261116):
    sources = [(f"scp4{i}", *read_set_cover(
        HERE / ("orlib_scp41.txt" if i == 1 else f"scp4{i}.txt")))
        for i in range(1, 11)]
    rng = np.random.default_rng(seed)
    scenes = []
    for i in range(48):
        source, matrix, costs = sources[i % len(sources)]
        required = np.sort(rng.choice(matrix.shape[0], 80, replace=False))
        # Fractional set covering.  The first objective is the number of
        # selected columns; the second is their original monetary cost.
        A = -matrix[required]
        base = np.ones(len(costs)) / 25
        risk = costs / 150
        scenes.append({"name": f"{source}_requirements_{i:02d}",
                       "base": base, "risk": risk, "A": A,
                       "b": -np.ones(len(required)), "bound": (0, 1),
                       "threshold": 2.5})
    return scenes


def solve(scene, lam):
    result = linprog(scene["base"] + lam * scene["risk"],
                     A_ub=scene["A"], b_ub=scene["b"],
                     bounds=scene["bound"], method="highs")
    if not result.success:
        raise RuntimeError((scene["name"], lam, result.message))
    return float(scene["risk"] @ result.x)


class Oracle:
    """One cached binary observation per scene and parameter."""

    def __init__(self, scenes, grid, precomputed_risks=None):
        self.scenes, self.grid, self.memo = scenes, grid, {}
        self.precomputed_risks = precomputed_risks

    def __call__(self, i, j):
        if j == len(self.grid):
            return 0  # An external safe fallback, not a solver call.
        key = (i, j)
        if key not in self.memo:
            scene = self.scenes[i]
            risk = (solve(scene, self.grid[j]) if self.precomputed_risks is None
                    else self.precomputed_risks[i, j])
            self.memo[key] = int(risk > scene["threshold"] + 1e-8)
        return self.memo[key]

    @property
    def calls(self):
        return len(self.memo)


def bisection(oracle, k, certified=False, anchor=False):
    """Return the first feasible grid index, optionally using partial sums."""
    n, J = len(oracle.scenes), len(oracle.grid)
    if anchor:
        for i in range(n):
            oracle(i, 0)
    lo, hi = -1, J
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if not certified:
            collisions = sum(oracle(i, mid) for i in range(n))
            feasible = collisions <= k
        else:
            lower = np.zeros(n, dtype=int)
            upper = np.ones(n, dtype=int)
            for (i, j), value in oracle.memo.items():
                if j >= mid: lower[i] = max(lower[i], value)
                if j <= mid: upper[i] = min(upper[i], value)
            while lower.sum() <= k < upper.sum():
                i = int(np.flatnonzero(lower < upper)[0])
                lower[i] = upper[i] = oracle(i, mid)
            feasible = upper.sum() <= k
        if feasible: hi = mid
        else: lo = mid
    return hi


def selection(oracle, k, seed):
    """Randomized Quickselect on hidden last-unsafe grid indices."""
    n, J = len(oracle.scenes), len(oracle.grid)
    rng = np.random.default_rng(seed)
    active = [i for i in range(n) if oracle(i, 0)]
    if len(active) <= k:
        return 0
    rank = k + 1
    while True:
        pivot_scene = int(rng.choice(active))
        lo, hi = 0, J
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if oracle(pivot_scene, mid): lo = mid
            else: hi = mid
        pivot = lo
        higher, equal, lower = [], [], []
        for i in active:
            if oracle(i, pivot + 1): higher.append(i)
            elif oracle(i, pivot): equal.append(i)
            else: lower.append(i)
        if rank <= len(higher): active = higher
        elif rank <= len(higher) + len(equal): return pivot + 1
        else:
            rank -= len(higher) + len(equal)
            active = lower


def benchmark_family(label, scenes, repeats=7, J=129, seed=20261116):
    grid = np.linspace(0, 1, J)
    k = int(math.floor((len(scenes) + 1) * 0.1 - 1))
    methods = {
        "full_bisection": lambda q, s: bisection(q, k),
        "certified_bisection": lambda q, s: bisection(q, k, certified=True),
        "anchored_bisection": lambda q, s: bisection(q, k, certified=True, anchor=True),
        "anchored_selection": lambda q, s: selection(q, k, s),
    }
    order_rng = np.random.default_rng(seed)
    records = []
    # Untimed warmup removes the first HiGHS initialization from comparisons.
    solve(scenes[0], 0.5)
    for repeat in range(repeats):
        for method in order_rng.permutation(list(methods)):
            oracle = Oracle(scenes, grid)
            t0 = time.perf_counter()
            answer = methods[method](oracle, seed + repeat)
            duration = time.perf_counter() - t0
            records.append({"family": label, "method": method, "repeat": repeat,
                            "cutoff_index": int(answer), "lambda":
                            float(grid[answer]) if answer < J else None,
                            "solves": oracle.calls, "seconds": duration})
    answers = {r["cutoff_index"] for r in records}
    assert len(answers) == 1, (label, answers)
    table = []
    for method in methods:
        runs = [r for r in records if r["method"] == method]
        counts = [r["solves"] for r in runs]
        table.append({"method": method,
                      "mean_solves": float(np.mean(counts)),
                      "sd_solves": float(np.std(counts, ddof=1)),
                      "mean_seconds": float(np.mean([r["seconds"] for r in runs])),
                      "sd_seconds": float(np.std([r["seconds"] for r in runs], ddof=1))})
    return {"family": label, "scenes": len(scenes), "grid_points": J,
            "target_alpha": 0.1, "allowed_losses": k,
            "cutoff": records[0]["lambda"], "summary": table, "runs": records}


def audit_grid(scenes, expected_cutoff, J=129):
    """Independently solve every grid problem and verify monotonicity."""
    grid = np.linspace(0, 1, J)
    risks = np.array([[solve(scene, lam) for lam in grid] for scene in scenes])
    assert np.all(np.diff(risks, axis=1) <= 1e-7)
    unsafe = np.array([risk > scene["threshold"] + 1e-8
                       for risk, scene in zip(risks, scenes)])
    k = int(math.floor((len(scenes) + 1) * .1 - 1))
    answer = int(np.flatnonzero(unsafe.sum(axis=0) <= k)[0])
    assert np.isclose(grid[answer], expected_cutoff)
    return {"solves": len(scenes)*J, "all_risks_monotone": True,
            "cutoff_index": answer}


def pilot():
    for label, scenes in (("knapsack", build_knapsack()),
                          ("set_cover", build_cover())):
        t0 = time.perf_counter()
        raw0 = np.array([solve(s, 0) for s in scenes])
        raw1 = np.array([solve(s, 1) for s in scenes])
        print(label, "n", len(scenes), "time", round(time.perf_counter()-t0, 2),
              "risk median at 0 and 1", np.median(raw0), np.median(raw1),
              "unsafe at 0 and 1", sum(raw0 > scenes[0]["threshold"]),
              sum(raw1 > scenes[0]["threshold"]))
        print("range at 0", (raw0.min(), raw0.max()),
              "range at 1", (raw1.min(), raw1.max()))


def threshold_sweep():
    """Study query counts across risk thresholds, reusing LP solutions."""
    results = {}
    grid = np.linspace(0, 1, 129)
    thresholds = {"knapsack": [0.35, 0.50, 0.65, 0.75, 0.85, 0.95, 1.01],
                  "set_cover": [2.1, 2.3, 2.5, 2.7, 3.0, 4.0, 5.0, 6.0, 7.0]}
    for label, scenes in (("knapsack", build_knapsack()),
                          ("set_cover", build_cover())):
        risks = np.array([[solve(s, lam) for lam in grid] for s in scenes])
        assert np.all(np.diff(risks, axis=1) <= 1e-7)
        k = int(math.floor((len(scenes) + 1) * 0.1 - 1))
        rows = []
        for threshold in thresholds[label]:
            chosen_scenes = [{**s, "threshold": threshold} for s in scenes]
            methods = {"full": lambda q: bisection(q, k),
                       "certified": lambda q: bisection(q, k, certified=True),
                       "anchored": lambda q: bisection(q, k, certified=True, anchor=True)}
            counts, answers = {}, set()
            for name, method in methods.items():
                oracle = Oracle(chosen_scenes, grid, risks)
                answers.add(method(oracle))
                counts[name] = oracle.calls
            selection_counts = []
            for seed in range(25):
                oracle = Oracle(chosen_scenes, grid, risks)
                answers.add(selection(oracle, k, seed))
                selection_counts.append(oracle.calls)
            assert len(answers) == 1, (label, threshold, answers)
            index = answers.pop()
            rows.append({"threshold": threshold,
                         "cutoff": float(grid[index]) if index < len(grid) else None,
                         **counts,
                         "selection_mean": float(np.mean(selection_counts)),
                         "selection_sd": float(np.std(selection_counts, ddof=1))})
        results[label] = rows
        print(label, "threshold  cutoff   full  certified  anchored  selection(mean±sd)")
        for row in rows:
            print("  %.2f    %-7s  %3d    %3d        %3d      %5.1f ± %.1f" %
                  (row["threshold"], str(row["cutoff"]), row["full"],
                   row["certified"], row["anchored"],
                   row["selection_mean"], row["selection_sd"]))
    return results


def main(repeats, audit=False):
    output = []
    for label, scenes in (("knapsack", build_knapsack()),
                          ("set_cover", build_cover())):
        result = benchmark_family(label, scenes, repeats=repeats)
        if audit:
            result["exhaustive_grid_audit"] = audit_grid(scenes, result["cutoff"])
        output.append(result)
        print(f"{label}: {result['scenes']} scenes, cutoff {result['cutoff']}")
        for row in result["summary"]:
            print(f"  {row['method']:24s} {row['mean_solves']:6.1f} solves"
                  f"   {row['mean_seconds']:.4f} +/- {row['sd_seconds']:.4f} s")
    target = Path(__file__).with_name("public_cdt_benchmark_results.json")
    inputs = sorted(HERE.glob("*.txt"))
    metadata = {"python": platform.python_version(), "platform": platform.platform(),
                "processor": platform.processor(), "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
                "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in inputs},
                "seed": 20261116, "repetitions": repeats,
                "model": "LP relaxations of OR-Library instances; binary losses from resource/cost thresholds"}
    target.write_text(json.dumps({"metadata": metadata, "benchmarks": output}, indent=2) + "\n")
    print("Saved", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.pilot: pilot()
    elif args.threshold_sweep: threshold_sweep()
    else: main(args.repeats, audit=args.audit)

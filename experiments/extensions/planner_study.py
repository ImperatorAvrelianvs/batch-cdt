"""Benchmark LP and binary-integer planner paths and audit the 129-point grid baseline."""
import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from paths import (ParametricOracle, algorithm_2g_v2, make_interior_instances,
                   make_loss_fn, reversed_eisner_severance_lazy_v2, run_full_es_reference)
from point_sweep import sweep


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def interval_events(pieces):
    """Choose the right-hand optimizer at interior breaks and inward optimizers at 0 and 1."""
    events = {upper: (min(i+1, len(pieces)-1), i)
              for i, (_, upper, *_) in enumerate(pieces)}
    events[0.0] = (0, None)
    return events


def evaluate_reference(curves, grid):
    losses, profits = [], []
    for pieces, values in curves:
        right = [p[1] for p in pieces]
        idx = np.minimum(np.searchsorted(right, grid, side="right"), len(pieces)-1)
        losses.append(np.asarray(values)[idx])
        profits.append(np.asarray([p[3] for p in pieces])[idx])
    return np.array(losses), np.array(profits)


def run_study():
    rows = []
    for mode, d, constraints in [("lp", 30, 5), ("milp", 22, 4)]:
        instances = make_interior_instances(100, d, constraints, seed=2026)
        for loss_type in ("binary", "continuous"):
            alpha = .10
            budget = (len(instances)+1)*alpha-1
            ref = run_full_es_reference(instances, mode, loss_type, alpha)
            curves = ref["all_pieces"]
            event_rows = [interval_events(pieces) for pieces, _ in curves]
            scorers = [values.__getitem__ for _, values in curves]
            ref_boundary, _ = sweep(event_rows, scorers, 0., 1., budget)
            start = perf_counter()
            oracles = [ParametricOracle(a,b,A,c,mode) for a,b,A,c,_,_ in instances]
            streams = [reversed_eisner_severance_lazy_v2(oracle) for oracle in oracles]
            costs = [make_loss_fn(r,t,loss_type) for _,_,_,_,r,t in instances]
            boundary, right, evaluations, retired, status = algorithm_2g_v2(streams,costs,alpha)
            elapsed = perf_counter()-start
            assert status == "certified" and abs(boundary-ref_boundary) < 1e-8
            assert abs(boundary-ref["ref_lam"]) < 1e-8
            deployed = boundary + min(1e-6,(right-boundary)/2)
            grid = np.linspace(0.,1.,129)
            losses, profit = evaluate_reference(curves,grid)
            grid_majorant = np.maximum.accumulate(losses[:,::-1],axis=1)[:,::-1]
            j = np.flatnonzero(grid_majorant.sum(axis=0) <= budget)[0]
            _, evaluated_profit = evaluate_reference(curves, [deployed])
            # Check the entire continuous-domain suffix at the selected interior point.
            maxima = [max(v[i] for i,p in enumerate(pcs) if p[1] > deployed)
                      for pcs,v in curves]
            assert sum(maxima) <= budget + 1e-9
            gain = 100 * (evaluated_profit.mean()/profit[:,j].mean()-1)
            rows.append(dict(planner=mode, loss=loss_type, scenes=len(curves),
                             nonmonotone=ref["nonmono"], pieces=ref["pieces"], boundary=boundary,
                             deployed=deployed, grid_majorant_lambda=grid[j],
                             calibration_profit_gain_pct=gain, full_solves=ref["solves"],
                             joint_solves=sum(o.solves for o in oracles), full_losses=ref["evals"],
                             joint_losses=evaluations, retired=retired,
                             full_seconds=ref["time"], joint_seconds=elapsed))
    return rows


def latency_probe():
    rows = []
    for i, (a,b,A,c,_,_) in enumerate(make_interior_instances(60,22,4,seed=2026)):
        oracle = ParametricOracle(a,b,A,c,"milp")
        for lam in (.05,.15,.30,.50,.70,.85):
            start=perf_counter()
            oracle.solve(lam)
            elapsed=perf_counter()-start
            unconstrained = (a+lam*b)>0
            rows.append(dict(scene=i, parameter=lam, seconds=elapsed,
                             sign_optimum_feasible=bool(np.all(A@unconstrained<=c)),
                             empty_sign_optimum=not np.any(unconstrained)))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    write_rows(args.out/"planner_study.csv",run_study())
    write_rows(args.out/"milp_latency.csv",latency_probe())

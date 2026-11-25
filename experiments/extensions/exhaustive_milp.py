"""Check 100 22-variable planner paths against enumeration of every binary vector."""
import argparse
from pathlib import Path

import numpy as np

from paths import ParametricOracle, full_eisner_severance, make_interior_instances, make_loss_fn
from planner_study import interval_events, write_rows
from point_sweep import sweep


def winning_pieces(a, b, A, rhs):
    # Meet the two halves at the constraints. Every pair corresponds to one 22-bit vector.
    half = len(a)//2
    bits = ((np.arange(1 << half)[:,None] >> np.arange(half)) & 1).astype(float)
    a_left, a_right = bits @ A[:,:half].T, bits @ A[:,half:].T
    feasible = np.ones((len(bits),len(bits)), dtype=bool)
    for j, limit in enumerate(rhs):
        feasible &= a_left[:,None,j] + a_right[None,:,j] <= limit
    left, right = np.nonzero(feasible)
    intercept = (bits @ a[:half])[left] + (bits @ a[half:])[right]
    slope = (bits @ b[:half])[left] + (bits @ b[half:])[right]

    # For lambda >= 0 a smaller intercept and no larger slope cannot win.
    order = np.lexsort((-slope, -intercept))
    increasing = np.r_[True,slope[order][1:] > np.maximum.accumulate(slope[order])[:-1]]
    hull, starts = [], []
    for ix in order[increasing]:
        while hull:
            prev = hull[-1]
            crossing = (intercept[prev]-intercept[ix])/(slope[ix]-slope[prev])
            if crossing > starts[-1]:
                break
            hull.pop()
            starts.pop()
        starts.append(crossing if hull else -np.inf)
        hull.append(ix)
    pieces = []
    for pos, ix in enumerate(hull):
        lo = max(0.,starts[pos])
        hi = min(1.,starts[pos+1] if pos+1<len(starts) else np.inf)
        if hi > lo:
            x = np.r_[bits[left[ix]], bits[right[ix]]]
            pieces.append((lo,hi,x,intercept[ix],slope[ix]))
    return pieces,len(left)


def calibrate(paths, scorers):
    curves = [[fn(p[2]) for p in pcs] for pcs,fn in zip(paths,scorers)]
    return sweep([interval_events(pcs) for pcs in paths],
                 [curve.__getitem__ for curve in curves], 0.,1.,(len(paths)+1)*.10-1)[0]


def run(directory):
    scenes = make_interior_instances(100,22,4,seed=2026)
    references, candidates, rows = [], [], []
    for i,(a,b,A,rhs,_,_) in enumerate(scenes):
        exact, feasible = winning_pieces(a,b,A,rhs)
        solved = full_eisner_severance(ParametricOracle(a,b,A,rhs,"milp"))
        assert len(exact)==len(solved)
        edge_error = max(abs(p[1]-q[1]) for p,q in zip(exact,solved))
        gap = max(abs(p[3]-q[3]+.5*(p[0]+p[1])*(p[4]-q[4])) for p,q in zip(exact,solved))
        assert gap < 1e-8 and edge_error < 1e-8
        assert all(np.array_equal(p[2],q[2]) for p,q in zip(exact,solved))
        references.append(exact)
        candidates.append(solved)
        rows.append(dict(scene=i,vectors=2**22,feasible=feasible,winning_pieces=len(exact),
                         max_midpoint_objective_error=gap,max_breakpoint_error=edge_error))
    write_rows(directory/"exhaustive_milp_scenes.csv",rows)
    cutoffs = []
    for loss in ("binary","continuous"):
        scorers = [make_loss_fn(r,tau,loss) for *_,r,tau in scenes]
        gold = calibrate(references,scorers)
        actual = calibrate(candidates,scorers)
        assert abs(gold-actual)<1e-8
        cutoffs.append(dict(loss=loss,exhaustive=gold,solver=actual))
    write_rows(directory/"exhaustive_milp_cutoffs.csv",cutoffs)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)

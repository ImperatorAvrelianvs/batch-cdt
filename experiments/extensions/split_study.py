"""Check bounded-loss calibration on 500 held-out splits at each of three risk levels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ParametricOracle, full_eisner_severance, make_interior_instances, make_loss_fn
from planner_study import interval_events
from point_sweep import sweep


def build_pool():
    rows, curves = [], []
    for a, b, A, rhs, r, tau in make_interior_instances(300,25,4,seed=999):
        pcs = full_eisner_severance(ParametricOracle(a,b,A,rhs,"lp"))
        loss = make_loss_fn(r,tau,"continuous")
        values = np.array([loss(p[2]) for p in pcs])
        rights = np.array([p[1] for p in pcs])
        rows.append(interval_events(pcs))
        curves.append((rights,values,np.maximum.accumulate(values[::-1])[::-1]))
    return rows,curves


def run():
    events,curves = build_pool()
    scores = [v.__getitem__ for _,v,_ in curves]
    rng = np.random.default_rng(777)
    results = []
    for alpha in (.05,.10,.20):
        for rep in range(500):
            perm = rng.permutation(len(curves))
            cal,test = perm[:150],perm[150:]
            budget = (len(cal)+1)*alpha-1
            boundary,deployed = sweep([events[i] for i in cal],[scores[i] for i in cal],0.,1.,budget)
            if deployed is None:
                avg_loss = avg_majorant = 0.0
            else:
                def values(i):
                    rights,loss,majorant = curves[i]
                    j = min(np.searchsorted(rights,deployed,side="right"),len(rights)-1)
                    return loss[j],majorant[j]
                assert sum(values(i)[1] for i in cal) <= budget+1e-9
                avg_loss,avg_majorant = np.mean([values(i) for i in test],axis=0)
            results.append((alpha,rep,boundary,deployed,avg_loss,avg_majorant))
    return pd.DataFrame(results,columns=["alpha","rep","boundary","deployed","heldout_loss","heldout_majorant"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    run().to_csv(args.out,index=False)

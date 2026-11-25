"""Regression checks for tied ranks, singleton tie decisions, and prefix solve counts."""
from fractions import Fraction
from itertools import combinations, combinations_with_replacement
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from point_sweep import regression, sweep
from selection import quickselect_2way

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import run_timing


def check_ranks():
    rng = np.random.default_rng(0)
    cases = 0
    for n in (3,4,5):
        for rho in combinations_with_replacement(range(5), n):
            for k in range(n):
                alpha = (k+1.25)/(n+1)
                for anchored in (False,True):
                    answer,_ = quickselect_2way(np.array(rho),4,alpha,rng,anchored)
                    assert answer == sorted(rho,reverse=True)[k]
                    cases += 1
    return cases


def check_prefixes():
    seen = set()

    def solve(lines, x):
        winner = min(range(len(lines)),key=lambda j:lines[j][0]+x*lines[j][1])
        seen.add(winner)
        return *lines[winner],winner

    paths = yields = 0
    grid = [Fraction(i,11) for i in range(1,11)]
    with patch.object(run_timing.P,"lp",solve):
        for n in range(len(grid)+1):
            for points in combinations(grid,n):
                h = len(points)+1
                lines = [(Fraction(0),Fraction(h))]
                for bp in points:
                    a,b = lines[-1]
                    lines.append((a+bp,b-1))
                seen.clear()
                emitted = set()
                stats = {"lp":0}
                generator = run_timing.lazy_path(lines,stats,lo=Fraction(0),hi=Fraction(1),tol=0)
                for m,(_,winner) in enumerate(generator,1):
                    assert winner == h-m
                    emitted.add(winner)
                    pending = len(seen - {0,h-1} - emitted)
                    expected = 1 if m==1 else 2*m+pending-int(m==h)
                    assert stats["lp"]==expected
                    yields += 1
                assert stats["lp"]==(2 if h==1 else 2*h-1)
                paths += 1
    return paths,yields


def check_singletons():
    # Both rows are safe on every open cell. At one shared point the selected
    # optimizer has loss one. Its exact position changes the feasible suffix.
    zero,half,one = Fraction(0),Fraction(1,2),Fraction(1)
    for spike in (zero,half,one):
        events = {one:("safe","safe"),half:("safe","safe"),zero:("safe",None)}
        events[spike] = ("unsafe",events[spike][1])
        score = {"safe":0,"unsafe":1}.__getitem__
        boundary,deployed = sweep([events,events],[score,score],zero,one,Fraction(1,5))
        if spike == one:
            assert deployed is None
        else:
            assert boundary==spike and deployed>spike


if __name__ == "__main__":
    assert check_ranks()==2030
    assert check_prefixes()==(1024,6144)
    check_singletons()
    regression()
    print("Passed 2,030 tied-rank cases, 1,024 rational paths (6,144 prefixes), and three singleton positions.")

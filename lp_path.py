"""Solution paths of parametric LPs (Eisner-Severance) and an expensive realized loss (integer repair)."""
import sys, time, heapq, math
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import issparse

from pathlib import Path
EXTRACT = str(Path(__file__).resolve().parent)  # needs benchmark_public_cdt.py and public_benchmarks/ here
sys.path.insert(0, EXTRACT)
import benchmark_public_cdt as pub

JITTER = 1e-3


def build(family, seed=20261117, sigma=0.15):
    """Scenes from the public benchmark, with a fixed objective jitter (unique optima on open
    intervals) and noisy true risk coefficients (prediction error)."""
    scenes = pub.build_knapsack() if family == "knapsack" else pub.build_cover()
    rng = np.random.default_rng(seed)
    for s in scenes:
        s["base"] = s["base"] * (1.0 + JITTER * rng.uniform(-1, 1, len(s["base"])))
        s["risk_true"] = s["risk"] * np.exp(sigma * rng.standard_normal(len(s["risk"])) - 0.5 * sigma ** 2)
    return scenes


def lp(scene, lam):
    r = linprog(scene["base"] + lam * scene["risk"], A_ub=scene["A"], b_ub=scene["b"],
                bounds=scene["bound"], method="highs")
    if not r.success:
        raise RuntimeError(r.message)
    x = r.x
    return float(scene["base"] @ x), float(scene["risk"] @ x), x


def solution_path(scene, lo=0.0, hi=1.0, tol=1e-9):
    """Return (breakpoints, decisions, n_lp). decisions[j] is optimal on the j-th open interval."""
    count = [0]

    def S(l):
        count[0] += 1
        return lp(scene, l)

    out = []  # (breakpoint, left solution, right solution)

    def rec(l, L, u, U):
        if abs(L[1] - U[1]) <= tol * (1 + abs(L[1])):
            return
        lam = (U[0] - L[0]) / (L[1] - U[1])
        lam = min(max(lam, l), u)
        M = S(lam)
        val = M[0] + lam * M[1]
        if val >= L[0] + lam * L[1] - tol * (1 + abs(val)):
            out.append((lam, L, U))
            return
        rec(l, L, lam, M)
        rec(lam, M, u, U)

    L, U = S(lo), S(hi)
    rec(lo, L, hi, U)
    out.sort(key=lambda t: t[0])
    breaks = [t[0] for t in out]
    decisions = [L[2]] if not out else [out[0][1][2]] + [t[2][2] for t in out]
    return np.array(breaks), decisions, count[0]


def repair(scene, x):
    """Nearest feasible 0/1 plan in L1: min sum_j (1-2x_j) z_j  s.t.  A z <= b, z binary."""
    A = scene["A"]
    con = LinearConstraint(A, -np.inf, scene["b"])
    r = milp(1.0 - 2.0 * x, constraints=con, integrality=np.ones(len(x)), bounds=Bounds(0, 1))
    if r.x is None:
        raise RuntimeError(r.message)
    return np.round(r.x)


def realized_risk(scene, x):
    return float(scene["risk_true"] @ repair(scene, x))

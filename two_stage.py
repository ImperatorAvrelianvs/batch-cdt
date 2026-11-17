"""Two-stage set covering on OR-Library scp41-scp410.

First stage (the planner): LP relaxation covering 80 forecast requirements, objective
(#columns)/25 + lambda*(cost)/150, as in the paper's public-data task, with a fixed objective jitter.
Realized outcome: the requirements that actually materialize. Evaluating a plan means solving the
integer recourse problem: cover every realized requirement the plan leaves uncovered at minimum cost.
Loss = 1{ (first-stage cost + recourse cost)/150 > tau }.
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import lp_path as P

pub = P.pub


def build(p_keep=0.8, q_new=0.5, seed=20261117):
    sources = [(f"scp4{i}", *pub.read_set_cover(pub.HERE / ("orlib_scp41.txt" if i == 1 else f"scp4{i}.txt")))
               for i in range(1, 11)]
    rng = np.random.default_rng(20261116)          # same requirement draws as the paper's task
    rng2 = np.random.default_rng(seed)
    scenes = []
    for i in range(48):
        name, matrix, costs = sources[i % len(sources)]
        required = np.sort(rng.choice(matrix.shape[0], 80, replace=False))
        base = np.ones(len(costs)) / 25
        base = base * (1.0 + P.JITTER * rng2.uniform(-1, 1, len(costs)))
        is_req = np.zeros(matrix.shape[0], bool); is_req[required] = True
        u = rng2.uniform(size=matrix.shape[0])
        realized = np.flatnonzero(np.where(is_req, u < p_keep, u < q_new))
        scenes.append({"name": f"{name}_{i:02d}", "base": base, "risk": costs / 150, "A": -matrix[required],
                       "b": -np.ones(len(required)), "bound": (0, 1), "matrix": matrix, "costs": costs,
                       "realized": realized, "required": required})
    return scenes


def realized_cost(scene, x):
    """(first-stage cost + optimal integer recourse cost)/150."""
    M = scene["matrix"][scene["realized"]]
    cover = M @ x
    open_rows = np.flatnonzero(cover < 1 - 1e-7)
    first = float(scene["costs"] @ x)
    if len(open_rows) == 0:
        return first / 150
    con = LinearConstraint(M[open_rows], 1, np.inf)
    r = milp(scene["costs"], constraints=con, integrality=np.ones(len(x)), bounds=Bounds(0, 1))
    if r.x is None or r.status != 0:
        raise RuntimeError(r.message)
    return (first + float(scene["costs"] @ np.round(r.x))) / 150

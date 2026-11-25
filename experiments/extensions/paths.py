"""Affine LP and binary-integer planners used in the synthetic benchmarks.

The floating-point interval sweep assumes inward endpoint choices and a stable
adjacent optimizer at interior ties. point_sweep.py handles separately supplied
point decisions. All timing, loss and boundary comparisons use the same scenes.
"""
import heapq
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp



class ParametricOracle:
    """Black-box solver for max_{x in X} (a^T x + lambda * b^T x).
    mode='lp'   : continuous polytope 0 <= x <= 1, A_ub x <= b_ub (HiGHS LP)
    mode='milp' : binary integer set x in {0, 1}^d, A_ub x <= b_ub (HiGHS MILP)
    """
    def __init__(self, a, b, A_ub, b_ub, mode="lp"):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.A_ub = np.asarray(A_ub, dtype=float)
        self.b_ub = np.asarray(b_ub, dtype=float)
        self.mode = mode
        self.d = len(self.a)
        self.solves = 0
        self.solve_time = 0.0
        if mode == "milp":
            self.integrality = np.ones(self.d, dtype=int)
            self.bounds = Bounds(np.zeros(self.d), np.ones(self.d))
            self.constraints = LinearConstraint(self.A_ub, -np.inf, self.b_ub)

    def solve(self, lam):
        t0 = time.perf_counter()
        c_min = -(self.a + lam * self.b)
        if self.mode == "lp":
            res = linprog(c_min, A_ub=self.A_ub, b_ub=self.b_ub, bounds=(0.0, 1.0), method="highs")
            assert res.success, f"LP failed: {res.message}"
            x = res.x
        elif self.mode == "milp":
            res = milp(c=c_min, integrality=self.integrality, bounds=self.bounds, constraints=self.constraints)
            assert res.success, f"MILP failed: {res.message}"
            x = np.round(res.x)
        else:
            raise ValueError(self.mode)
        self.solve_time += time.perf_counter() - t0
        self.solves += 1
        return x, float(self.a @ x), float(self.b @ x)


def full_eisner_severance(oracle, lam_min=0.0, lam_max=1.0, tol=1e-9):
    """Standard full Eisner-Severance returning all maximal optimality intervals left-to-right."""
    x_l, a_l, b_l = oracle.solve(lam_min)
    x_r, a_r, b_r = oracle.solve(lam_max)
    raw = []

    def recurse(ll, xl, al, bl, lr, xr, ar, br):
        if abs(br - bl) <= tol or (lr - ll) <= tol:
            raw.append((ll, lr, xr, ar, br))
            return
        lam_m = (al - ar) / (br - bl)
        if lam_m <= ll + tol or lam_m >= lr - tol:
            raw.append((ll, lr, xr, ar, br))
            return
        xm, am, bm = oracle.solve(lam_m)
        if am + lam_m * bm <= ar + lam_m * br + tol:
            raw.append((ll, lam_m, xl, al, bl))
            raw.append((lam_m, lr, xr, ar, br))
        else:
            recurse(ll, xl, al, bl, lam_m, xm, am, bm)
            recurse(lam_m, xm, am, bm, lr, xr, ar, br)

    recurse(lam_min, x_l, a_l, b_l, lam_max, x_r, a_r, b_r)

    merged = []
    for ll, lr, x, av, bv in raw:
        if lr - ll <= tol:
            continue
        if merged and abs(merged[-1][3] - av) <= 1e-7 and abs(merged[-1][4] - bv) <= 1e-7:
            p = merged[-1]
            merged[-1] = (p[0], lr, p[2], p[3], p[4])
        else:
            merged.append((ll, lr, x, av, bv))
    return merged


def reversed_eisner_severance_lazy_v2(oracle, lam_min=0.0, lam_max=1.0, tol=1e-9):
    """Two-Phase Lazy Reversed Eisner-Severance generator:
    Phase 0: yields rightmost decision at lam_max (1 solve) before touching lam_min.
    Phase 1: yields maximal optimality intervals strictly right-to-left on demand.
    """
    x_r, a_r, b_r = oracle.solve(lam_max)
    yield ("INIT_RIGHT", x_r, a_r, b_r)

    x_l, a_l, b_l = oracle.solve(lam_min)
    stack = [(lam_min, x_l, a_l, b_l, lam_max, x_r, a_r, b_r)]
    cur_l, cur_r, cur_x, cur_a, cur_b = lam_max, lam_max, x_r, a_r, b_r

    while stack:
        ll, xl, al, bl, lr, xr, ar, br = stack.pop()
        if abs(br - bl) <= tol or (lr - ll) <= tol:
            cur_l = ll
            continue
        lam_m = (al - ar) / (br - bl)
        if lam_m <= ll + tol or lam_m >= lr - tol:
            cur_l = ll
            continue
        xm, am, bm = oracle.solve(lam_m)
        if am + lam_m * bm <= ar + lam_m * br + tol:
            emit_l, emit_r, emit_x, emit_a, emit_b = lam_m, cur_r, cur_x, cur_a, cur_b
            cur_l, cur_r, cur_x, cur_a, cur_b = ll, lam_m, xl, al, bl
            if emit_r - emit_l > tol:
                yield ("INTERVAL", emit_l, emit_r, emit_x, emit_a, emit_b)
        else:
            stack.append((ll, xl, al, bl, lam_m, xm, am, bm))
            stack.append((lam_m, xm, am, bm, lr, xr, ar, br))

    if cur_r - cur_l > tol:
        yield ("INTERVAL", cur_l, cur_r, cur_x, cur_a, cur_b)


def algorithm_2g_v2(generators, loss_fns, alpha, lam_min=0.0, lam_max=1.0, tol=1e-9):
    """Interval version of Algorithm 2G with floating-point comparisons and adjacent ties."""
    n = len(generators)
    B_n = (n + 1) * alpha - 1.0
    if B_n < -tol:
        return lam_max, lam_max, 0, 0, "fallback"

    m = np.zeros(n, dtype=float)
    S = 0.0
    loss_evals = 0
    retired = 0

    for i in range(n):
        tag, x_r, _, _ = next(generators[i])
        assert tag == "INIT_RIGHT"
        li = float(loss_fns[i](x_r))
        loss_evals += 1
        m[i] = li
        S += li
        if m[i] >= 1.0 - 1e-12:
            retired += 1

    if S > B_n + tol:
        return lam_max, lam_max, loss_evals, retired, "fallback"

    heap = []
    for i in range(n):
        if m[i] >= 1.0 - 1e-12:
            continue
        tag, lam_l, lam_r, _, _, _ = next(generators[i])
        assert tag == "INTERVAL"
        if lam_l > lam_min + tol:
            heapq.heappush(heap, (-lam_l, i))

    last_safe_lam = lam_max
    while heap:
        lam_star = -heap[0][0]
        batch = []
        while heap and abs((-heap[0][0]) - lam_star) <= tol:
            _, i = heapq.heappop(heap)
            batch.append(i)

        for i in batch:
            nxt = next(generators[i], None)
            if nxt is None:
                continue
            _, lam_l, lam_r, x, _, _ = nxt
            li = float(loss_fns[i](x))
            loss_evals += 1
            if li > m[i]:
                S += (li - m[i])
                m[i] = li
            if m[i] >= 1.0 - 1e-12:
                retired += 1
            elif lam_l > lam_min + tol:
                heapq.heappush(heap, (-lam_l, i))

        if S > B_n + tol:
            return lam_star, last_safe_lam, loss_evals, retired, "certified"

        last_safe_lam = lam_star

    return lam_min, last_safe_lam, loss_evals, retired, "safe_at_min"


def make_interior_instances(n_inst, d, m_cons, seed=2026, b_scale=2.5, tau=0.65, sigma=0.45):
    """Generates n_inst multi-dimensional resource allocation instances with noisy realized weights."""
    rng = np.random.default_rng(seed)
    instances = []
    for _ in range(n_inst):
        profits = rng.uniform(10.0, 100.0, size=d)
        A = rng.uniform(5.0, 35.0, size=(m_cons, d))
        b_ub = 0.45 * A.sum(axis=1)
        r_pred = A[0] / b_ub[0]
        a = profits / profits.sum() + rng.uniform(-1e-4, 1e-4, size=d)
        b_obj = -b_scale * r_pred
        noise = np.exp(sigma * rng.standard_normal(d) - 0.5 * sigma**2)
        r_true = r_pred * noise
        instances.append((a, b_obj, A, b_ub, r_true, tau))
    return instances


def make_loss_fn(r_true, tau, loss_type="continuous"):
    def fn(x):
        excess = float(r_true @ x) - tau
        if loss_type == "binary":
            return 1.0 if excess > 0.0 else 0.0
        return 0.0 if excess <= 0.0 else min(1.0, excess / 0.25)
    return fn


def run_full_es_reference(instances, mode, loss_type, alpha):
    n = len(instances)
    B_n = (n + 1) * alpha - 1.0
    t0 = time.perf_counter()
    total_solves, total_evals, total_pieces, nonmono = 0, 0, 0, 0
    all_pieces, all_events = [], []
    init_sum = 0.0

    for a, b_obj, A, b_ub, r_true, tau in instances:
        orc = ParametricOracle(a, b_obj, A, b_ub, mode=mode)
        pieces = full_eisner_severance(orc, 0.0, 1.0)
        total_solves += orc.solves
        total_pieces += len(pieces)
        lfn = make_loss_fn(r_true, tau, loss_type)
        losses = [lfn(p[2]) for p in pieces]
        total_evals += len(losses)
        if any(losses[j + 1] > losses[j] + 1e-12 for j in range(len(losses) - 1)):
            nonmono += 1

        cur_m = losses[-1]
        init_sum += cur_m
        for j in range(len(pieces) - 2, -1, -1):
            if losses[j] > cur_m + 1e-12:
                all_events.append((pieces[j][1], losses[j] - cur_m))
                cur_m = losses[j]
        all_pieces.append((pieces, losses))

    all_events.sort(key=lambda t: -t[0])
    S = init_sum
    ref_lam = 0.0
    if S > B_n + 1e-9:
        ref_lam = 1.0
    else:
        idx = 0
        while idx < len(all_events):
            lam_star = all_events[idx][0]
            while idx < len(all_events) and abs(all_events[idx][0] - lam_star) <= 1e-9:
                S += all_events[idx][1]
                idx += 1
            if S > B_n + 1e-9:
                ref_lam = lam_star
                break
    return {
        "ref_lam": ref_lam, "solves": total_solves, "evals": total_evals,
        "pieces": total_pieces, "nonmono": nonmono, "time": time.perf_counter() - t0,
        "all_pieces": all_pieces,
    }



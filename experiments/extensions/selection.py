"""Compare grid calibration algorithms on three distributions of last-unsafe indices."""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def full_binary_search(rho, J, alpha):
    """Standard binary search over grid {0, ..., J-1} (representing lambda_1..lambda_J).
    rho[i] in {0, ..., J}: last unsafe 1-indexed grid index (0 = safe at lambda_1).
    Loss L_i(j) = 1 iff rho[i] >= j + 1 (0-indexed j in 0..J-1).
    """
    n = len(rho)
    k = int(math.floor((n + 1) * alpha - 1.0 + 1e-12))
    lo, hi = 0, J
    queries = 0
    while lo < hi:
        mid = (lo + hi) // 2
        queries += n
        cnt = int(np.sum(rho >= (mid + 1)))
        if cnt <= k:
            hi = mid
        else:
            lo = mid + 1
    return lo, queries


def certified_bisection(rho, J, alpha):
    """Proposition 3.2: Certified bisection with O(1) incremental bound updates per query."""
    n = len(rho)
    B = (n + 1) * alpha - 1.0
    min_safe = np.full(n, J, dtype=np.int64)
    max_unsafe = np.full(n, -1, dtype=np.int64)

    lo, hi = 0, J
    queries = 0
    while lo < hi:
        mid = (lo + hi) // 2
        known_one = max_unsafe >= mid
        known_zero = min_safe <= mid
        unknown_idx = np.flatnonzero(~(known_one | known_zero))

        s_lo = int(known_one.sum())
        s_hi = s_lo + len(unknown_idx)

        ptr = 0
        while not (s_hi <= B + 1e-12 or s_lo > B + 1e-12):
            i = unknown_idx[ptr]
            ptr += 1
            queries += 1
            if rho[i] >= mid + 1:
                max_unsafe[i] = mid
                s_lo += 1
            else:
                min_safe[i] = mid
                s_hi -= 1

        if s_hi <= B + 1e-12:
            hi = mid
        else:
            lo = mid + 1
    return lo, queries


def quickselect_3way(rho, J, alpha, rng, anchored=True):
    """Algorithm 1 (3-way partition): compares each row against t using up to 2 queries."""
    n = len(rho)
    k = int(math.floor((n + 1) * alpha - 1.0 + 1e-12))
    if k < 0:
        return J, 0
    rank = k + 1

    min_safe = np.full(n, J, dtype=np.int64)
    max_unsafe = np.full(n, -1, dtype=np.int64)
    queries = 0

    def query(i, j):
        nonlocal queries
        if j < 0:
            return 1
        if j >= J:
            return 0
        if max_unsafe[i] >= j:
            return 1
        if min_safe[i] <= j:
            return 0
        queries += 1
        if rho[i] >= j + 1:
            max_unsafe[i] = j
            return 1
        else:
            min_safe[i] = j
            return 0

    def find_rho(i, lo_r, hi_r):
        l, r = lo_r, hi_r
        while l < r:
            m = (l + r + 1) // 2
            if query(i, m - 1) == 1:
                l = m
            else:
                r = m - 1
        return l

    if anchored:
        active = [i for i in range(n) if query(i, 0) == 1]
        if len(active) <= k:
            return 0, queries
        lo_val, hi_val = 1, J
    else:
        active = list(range(n))
        lo_val, hi_val = 0, J

    while active:
        p = active[int(rng.integers(0, len(active)))]
        t = find_rho(p, lo_val, hi_val)
        greater, equal, less = [], [p], []
        for i in active:
            if i == p:
                continue
            if query(i, t) == 1:
                greater.append(i)
            elif t == 0 or query(i, t - 1) == 1:
                equal.append(i)
            else:
                less.append(i)
        if rank <= len(greater):
            active = greater
            lo_val = t + 1
        elif rank <= len(greater) + len(equal):
            return t, queries
        else:
            rank -= len(greater) + len(equal)
            active = less
            hi_val = t - 1
    return 0, queries


def quickselect_2way(rho, J, alpha, rng, anchored=True):
    """Refined 2-way (1-query-per-row) Randomized Quickselect."""
    n = len(rho)
    k = int(math.floor((n + 1) * alpha - 1.0 + 1e-12))
    if k < 0:
        return J, 0
    rank = k + 1

    min_safe = np.full(n, J, dtype=np.int64)
    max_unsafe = np.full(n, -1, dtype=np.int64)
    queries = 0

    def query(i, j):
        nonlocal queries
        if j < 0:
            return 1
        if j >= J:
            return 0
        if max_unsafe[i] >= j:
            return 1
        if min_safe[i] <= j:
            return 0
        queries += 1
        if rho[i] >= j + 1:
            max_unsafe[i] = j
            return 1
        else:
            min_safe[i] = j
            return 0

    def find_rho(i, lo_r, hi_r):
        l, r = lo_r, hi_r
        while l < r:
            m = (l + r + 1) // 2
            if query(i, m - 1) == 1:
                l = m
            else:
                r = m - 1
        return l

    if anchored:
        active = [i for i in range(n) if query(i, 0) == 1]
        if len(active) <= k:
            return 0, queries
        lo_val, hi_val = 1, J
    else:
        active = list(range(n))
        lo_val, hi_val = 0, J

    while active:
        if lo_val == hi_val:
            return lo_val, queries
        p = active[int(rng.integers(0, len(active)))]
        t = find_rho(p, lo_val, hi_val)

        if t == hi_val:
            equal, less = [p], []
            for i in active:
                if i == p:
                    continue
                if query(i, t - 1) == 1:
                    equal.append(i)
                else:
                    less.append(i)
            if rank <= len(equal):
                return t, queries
            rank -= len(equal)
            active = less
            hi_val = t - 1
        else:
            greater, leq = [], []
            for i in active:
                if i == p:
                    continue
                if query(i, t) == 1:
                    greater.append(i)
                else:
                    leq.append(i)
            if rank <= len(greater):
                active = greater
                lo_val = t + 1
            elif rank == len(greater) + 1:
                return t, queries
            else:
                rank -= (len(greater) + 1)
                active = leq
                hi_val = t
    return lo_val, queries


# Three threshold distribution regimes
def gen_sparse_small(rng, n, J):
    unsafe = rng.random(n) < 0.35
    rho = np.zeros(n, dtype=np.int64)
    m = int(unsafe.sum())
    if m > 0:
        rho[unsafe] = np.clip(np.ceil(rng.beta(1.2, 25.0, size=m) * J).astype(np.int64), 1, J)
    return rho

def gen_dense_small(rng, n, J):
    unsafe = rng.random(n) < 0.90
    rho = np.zeros(n, dtype=np.int64)
    m = int(unsafe.sum())
    if m > 0:
        rho[unsafe] = np.clip(np.ceil(rng.beta(1.2, 25.0, size=m) * J).astype(np.int64), 1, J)
    return rho

def gen_dense_uniform(rng, n, J):
    unsafe = rng.random(n) < 0.95
    rho = np.zeros(n, dtype=np.int64)
    m = int(unsafe.sum())
    if m > 0:
        rho[unsafe] = rng.integers(1, J + 1, size=m)
    return rho

REGIMES = [
    ("Sparse Small-Cutoff (SDD-like, 35% unsafe)", gen_sparse_small),
    ("Dense Small-Cutoff (90% unsafe)",            gen_dense_small),
    ("Dense Uniform (High-Cutoff, 95% unsafe)",    gen_dense_uniform),
]

N_LIST = [48, 100, 250, 500, 1000, 2000]
J_LIST = [129, 1025, 8193]
ALPHA = 0.10
N_TRIALS = 40

def run():
    rng_scale = np.random.default_rng(2026)
    scaling_records = []

    for r_name, gen_fn in REGIMES:
        for J in J_LIST:
            for n in N_LIST:
                qf, qc, qa3, qa2, qu2 = [], [], [], [], []
                for _ in range(N_TRIALS):
                    rho = gen_fn(rng_scale, n, J)
                    c0, q0 = full_binary_search(rho, J, ALPHA)
                    c1, q1 = certified_bisection(rho, J, ALPHA)
                    c2, q2 = quickselect_3way(rho, J, ALPHA, rng_scale, anchored=True)
                    c3, q3 = quickselect_2way(rho, J, ALPHA, rng_scale, anchored=True)
                    c4, q4 = quickselect_2way(rho, J, ALPHA, rng_scale, anchored=False)
                    assert c0 == c1 == c2 == c3 == c4, f"Exactness mismatch: {c0},{c1},{c2},{c3},{c4}"
                    qf.append(q0); qc.append(q1); qa3.append(q2); qa2.append(q3); qu2.append(q4)
                lb = n - 1 + math.log2(J - 1)
                scaling_records.append({
                    "Regime": r_name, "J": J, "n": n, "LB": lb,
                    "Full": np.mean(qf), "CertBisect": np.mean(qc),
                    "AnchQS_3w": np.mean(qa3), "AnchQS_2w": np.mean(qa2), "UnanchQS_2w": np.mean(qu2),
                    "Anch2w_std": np.std(qa2), "Unanch2w_std": np.std(qu2),
                })
    return pd.DataFrame(scaling_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run().to_csv(args.out, index=False)

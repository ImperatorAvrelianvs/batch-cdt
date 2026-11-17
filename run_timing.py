"""Compare exact continuous calibration procedures for parametric LPs.

  (1) full path, all losses: sum of measured stage times
  (2) full path, backward sweep: sum of measured stage times
  (3) on-demand path, backward sweep: a separately timed complete run

Usage: python run_timing.py knapsack|set_cover
"""
import sys, time, json, math, heapq
import numpy as np
import lp_path as P
import two_stage as T2

ALPHA, LO, HI = 0.1, 0.0, 1.0
T_DAYS = 10


def lazy_path(scene, stats, lo=LO, hi=HI, tol=1e-9):
    """Yield (right_endpoint, decision) from the most to the least conservative interval."""
    def S(l):
        stats["lp"] += 1
        return P.lp(scene, l)
    U = S(hi)
    yield hi, U[2]
    L = S(lo)
    stack = [(lo, L, hi, U)]
    while stack:
        l, Lf, u, Uf = stack.pop()
        if abs(Lf[1] - Uf[1]) <= tol * (1 + abs(Lf[1])):
            continue
        lam = min(max((Uf[0] - Lf[0]) / (Lf[1] - Uf[1]), l), u)
        M = S(lam)
        val = M[0] + lam * M[1]
        if val >= Lf[0] + lam * Lf[1] - tol * (1 + abs(val)) or u - l < 1e-13:
            yield lam, Lf[2]
            continue
        stack.append((l, Lf, lam, M))      # explored later
        stack.append((lam, M, u, Uf))      # explored first: right to left


def make_family(name):
    if name == "knapsack":
        scenes = P.build("knapsack")
        days = 1

        def day_cost(s, x, d):
            return P.realized_risk(s, x)
    else:
        scenes = T2.build()
        rng = np.random.default_rng(777)
        for s in scenes:                     # T_DAYS realized requirement sets per data point
            m = s["matrix"].shape[0]
            is_req = np.zeros(m, bool)
            is_req[s["required"]] = True
            s["realized_days"] = []
            for d in range(T_DAYS):
                u = rng.uniform(size=m)
                s["realized_days"].append(np.flatnonzero(np.where(is_req, u < 0.8, u < 0.5)))
        days = T_DAYS

        def day_cost(s, x, d):
            s["realized"] = s["realized_days"][d]
            return T2.realized_cost(s, x)
    return scenes, days, day_cost


def sweep(n, k, first, nxt, loss):
    """Backward sweep. first(i)->(r,x); nxt(i)->(r,x) or None; loss(i,x)->bool.
    Returns (V_(k+1), number of loss evaluations)."""
    heap = []
    for i in range(n):
        r, x = first(i)
        heap.append((-r, i, x))
    heapq.heapify(heap)
    found, evals = 0, 0
    while heap:
        negr, i, x = heapq.heappop(heap)
        evals += 1
        if loss(i, x):
            found += 1
            if found == k + 1:
                return -negr, evals
        else:
            item = nxt(i)
            if item is not None:
                heapq.heappush(heap, (-item[0], i, item[1]))
    return -math.inf, evals


def main(name):
    scenes, days, day_cost = make_family(name)
    n = len(scenes)
    k = int(math.floor((n + 1) * ALPHA - 1))
    out = {"family": name, "n": n, "k": k, "days_available": days}

    def log(*a):
        print(*a, flush=True)

    # ---- Full paths and cumulative solve counts
    paths = []
    t_all = time.perf_counter()
    for s in scenes:
        st = {"lp": 0}
        t0 = time.perf_counter()
        rights, decs, cum_lp, cum_t = [], [], [], []
        for r, x in lazy_path(s, st):
            rights.append(r)
            decs.append(x)
            cum_lp.append(st["lp"])
            cum_t.append(time.perf_counter() - t0)
        paths.append({"rights": np.array(rights), "decs": decs, "cum_lp": np.array(cum_lp),
                      "cum_t": np.array(cum_t), "lp_total": st["lp"]})
    out["path_seconds_full"] = time.perf_counter() - t_all
    out["path_lp_full"] = int(sum(p["lp_total"] for p in paths))
    out["intervals"] = int(sum(len(p["rights"]) for p in paths))
    log(f"[{name}] full paths: {out['intervals']} intervals, {out['path_lp_full']} LP solves, {out['path_seconds_full']:.1f}s")

    # verification against direct solves and the non-lazy routine (untimed)
    rng = np.random.default_rng(3)
    mism = checks = 0
    for i in rng.choice(n, 12, replace=False):
        p = paths[i]
        edges = np.concatenate(([LO], p["rights"][::-1]))     # increasing lambda
        h = len(p["rights"])
        for _ in range(4):
            j = int(rng.integers(h))
            lam = rng.uniform(edges[j], edges[j + 1])
            x = P.lp(scenes[i], lam)[2]
            mism += int(np.max(np.abs(x - p["decs"][h - 1 - j])) > 1e-6)
            checks += 1
    br, dec, _ = P.solution_path(scenes[0])
    assert len(dec) == len(paths[0]["rights"]) and np.allclose(br, paths[0]["rights"][::-1][:-1])
    out["path_decision_mismatches"] = f"{mism}/{checks}"
    log(f"[{name}] decisions vs direct solves at random parameters: {mism}/{checks} mismatches; generator equals non-lazy path on scene 0")

    # ---- Loss on every interval, with individual evaluation times
    cost = [np.zeros((len(p["rights"]), days)) for p in paths]
    tcost = [np.zeros((len(p["rights"]), days)) for p in paths]
    t_all = time.perf_counter()
    for i, (s, p) in enumerate(zip(scenes, paths)):
        for j, x in enumerate(p["decs"]):
            for d in range(days):
                t0 = time.perf_counter()
                cost[i][j, d] = day_cost(s, x, d)
                tcost[i][j, d] = time.perf_counter() - t0
        if i % 8 == 7:
            log(f"[{name}] losses: data point {i + 1}/{n}, elapsed {time.perf_counter() - t_all:.0f}s")

    results = []
    for T in sorted({1, days}):
        mean_cost = [c[:, :T].mean(axis=1) for c in cost]
        eval_t = [t[:, :T].sum(axis=1) for t in tcost]
        allc = np.concatenate(mean_cost)

        def analyse(tau):
            V = []
            for p, c in zip(paths, mean_cost):
                unsafe = np.flatnonzero(c > tau)             # index 0 = most conservative interval
                V.append(p["rights"][unsafe[0]] if unsafe.size else -math.inf)
            V = np.array(V)
            r = np.sort(V)[::-1][k]
            lp_lazy = 0
            lp_t_lazy = ev_t = 0.0
            ev = 0
            for p, et, v in zip(paths, eval_t, V):
                h = len(p["rights"])
                m_safe = int(np.sum(p["rights"] > max(v, r)))
                hit = bool(np.isfinite(v) and v >= r)
                m = m_safe + int(hit)
                ev += m
                ev_t += et[:m].sum()
                pulled = min(m_safe + 1, h)
                lp_lazy += int(p["cum_lp"][pulled - 1])
                lp_t_lazy += float(p["cum_t"][pulled - 1])
            nonmono = int(sum(np.any(np.diff((c > tau).astype(int)[::-1]) > 0) for c in mean_cost))
            return dict(tau=float(tau), cutoff=float(r), loss_evals_sweep=int(ev), loss_seconds_sweep=float(ev_t),
                        lp_lazy=int(lp_lazy), lp_seconds_lazy=float(lp_t_lazy), nonmonotone_scenes=nonmono)

        taus = np.quantile(allc, np.linspace(0.1, 0.9, 17))
        table = [analyse(t) for t in taus]
        table = [r for r in table if LO < r["cutoff"] < HI]
        chosen = min(table, key=lambda r: abs(r["cutoff"] - 0.5 * (LO + HI)))
        exhaustive_loss_t = float(sum(t.sum() for t in eval_t))
        res = {"T": T, "chosen": chosen, "threshold_sweep": table,
               "exhaustive": {"lp": out["path_lp_full"], "loss_evals": out["intervals"],
                              "seconds": out["path_seconds_full"] + exhaustive_loss_t,
                              "lp_seconds": out["path_seconds_full"], "loss_seconds": exhaustive_loss_t},
               "full_path_sweep_from_recorded_times": {"lp": out["path_lp_full"], "loss_evals": chosen["loss_evals_sweep"],
                                                       "seconds": out["path_seconds_full"] + chosen["loss_seconds_sweep"]}}
        # ---- Fresh, timed on-demand sweep
        tau = chosen["tau"]
        reps = 3 if (name == "knapsack" or T == 1) else 1
        real = []
        for rep in range(reps):
            stats = [{"lp": 0} for _ in range(n)]
            gens = [lazy_path(s, st) for s, st in zip(scenes, stats)]
            lt = [0.0]

            def loss(i, x):
                t0 = time.perf_counter()
                c = np.mean([day_cost(scenes[i], x, d) for d in range(T)])
                lt[0] += time.perf_counter() - t0
                return c > tau

            t0 = time.perf_counter()
            r, ev = sweep(n, k, lambda i: next(gens[i]), lambda i: next(gens[i], None), loss)
            tot = time.perf_counter() - t0
            assert abs(r - chosen["cutoff"]) < 1e-9, (r, chosen["cutoff"])
            real.append({"seconds": tot, "loss_seconds": lt[0], "lp": int(sum(st["lp"] for st in stats)),
                         "loss_evals": ev, "cutoff": float(r)})
        res["lazy_sweep_measured"] = real
        results.append(res)
        e = res["exhaustive"]
        f = res["full_path_sweep_from_recorded_times"]
        z = real[0]
        log(f"[{name}] T={T} tau={tau:.4f} cutoff={chosen['cutoff']:.4f} nonmonotone data points={chosen['nonmonotone_scenes']}")
        log(f"    (1) full path + all losses : {e['lp']} LP, {e['loss_evals']} loss evals, {e['seconds']:.2f}s  (LP {e['lp_seconds']:.2f}s, loss {e['loss_seconds']:.2f}s)")
        log(f"    (2) full path + sweep      : {f['lp']} LP, {f['loss_evals']} loss evals, {f['seconds']:.2f}s  (from recorded per-evaluation times)")
        log(f"    (3) lazy path + sweep      : {z['lp']} LP, {z['loss_evals']} loss evals, {np.mean([q['seconds'] for q in real]):.2f}s measured over {reps} run(s)"
            f"  [predicted {chosen['lp_lazy']} LP, {chosen['loss_evals_sweep']} evals]")
        sw = [(round(r["cutoff"], 3), r["loss_evals_sweep"], r["lp_lazy"]) for r in table]
        log(f"    threshold sweep (cutoff, sweep loss evals, lazy LP solves): {sw}")
    out["results"] = results
    json.dump(out, open(f"timing_{name}.json", "w"), indent=1)
    log(f"[{name}] saved timing_{name}.json")


if __name__ == "__main__":
    main(sys.argv[1])

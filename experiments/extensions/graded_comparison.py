"""Compare joint and independent graded-loss stopping on the set-cover traces."""
import argparse
import csv
from itertools import groupby
from pathlib import Path

import numpy as np
from point_sweep import sweep as point_sweep
from paths import algorithm_2g_v2


def path_events(right):
    yield ('INIT_RIGHT', 0, 0, 0)
    for j, r in enumerate(right):
        left = right[j + 1] if j + 1 < len(right) else 0.0
        yield ('INTERVAL', left, r, j, 0, 0)


def exhaustive(rights, curves, budget):
    # The reference uses the open pieces from the traces, with inward endpoint
    # decisions. Finite positive crossing coordinates are the returned infima.
    value = sum(loss[0] for loss in curves)
    if value > budget:
        return 1.0, 'fallback'
    changes = sorted(((r, inc) for right, loss in zip(rights, curves)
                      for r, inc in zip(right[1:], np.diff(np.maximum.accumulate(loss))) if inc),
                     reverse=True)
    for r, ties in groupby(changes, key=lambda item: item[0]):
        value += sum(inc for _, inc in ties)
        if value > budget:
            return r, 'certified'
    return 0.0, 'safe_at_min'


def compare(args):
    rights, costs = [], []
    for p in sorted(args.traces.glob('set_cover_*.npz')):
        with np.load(p) as data:
            rights.append(data['endpoints'])
            costs.append(data['costs'])
    assert len(rights) == 48
    with args.thresholds.open() as f:
        thresholds = [r for r in csv.DictReader(f) if r['family'] == 'set_cover']
    event_rows = []
    for right in rights:
        steps = {r: (max(0, j-1), j) for j, r in enumerate(right)}
        steps[0.0] = (len(right)-1, None)
        event_rows.append(steps)
    rows = []
    for row in thresholds:
        realizations, tau = int(row['realizations']), float(row['tau'])
        losses = [np.clip(c[:, :realizations].mean(axis=1) / tau - 1, 0, 1) for c in costs]
        assert not any(np.any((v >= 1 - 1e-12) & (v < 1)) for v in losses)
        bound, upper, joint, _, status = algorithm_2g_v2([path_events(r) for r in rights],
                                                    [v.__getitem__ for v in losses], .10)
        oracle, check_status = exhaustive(rights, losses, (len(losses) + 1) * .10 - 1)
        assert abs(bound - oracle) < 1e-8 and status == check_status
        scored = [0]
        def scorer(curve):
            def score(index):
                scored[0] += 1
                return curve[index]
            return score
        ref_bound, deployed = point_sweep(event_rows, list(map(scorer, losses)), 0.0, 1.0,
                                          (len(losses)+1)*.10-1)
        assert ref_bound is not None and abs(ref_bound - oracle) < 1e-8
        assert scored[0] == joint, (scored[0], joint)
        suffix = sum(np.max(v[r > deployed]) if deployed < 1 else v[0]
                     for r, v in zip(rights, losses))
        assert suffix <= (len(losses)+1)*.10-1+1e-10
        # Independent calculation of the whole per-row suffix majorant can stop
        # when it first sees loss one. Subsequent values cannot change the majorant.
        saturation = [np.flatnonzero(v == 1) for v in losses]
        independent = sum(int(ix[0])+1 if len(ix) else len(v) for ix, v in zip(saturation, losses))
        full = sum(map(len, losses))
        if status != 'fallback':
            point = (bound + upper) / 2 if status == 'certified' else 0.
            value = sum(np.max(v[r > point]) if point < 1 else v[0]
                        for r, v in zip(rights, losses))
            assert value <= (len(losses)+1)*.10-1+1e-10
        rows.append(dict(realizations=realizations, quantile=row['quantile'], tau=tau, paper_threshold=row['chosen'],
                         status=status, boundary=bound, joint_count=joint, separate_count=independent,
                         all_count=full, percent_saved=100*(1-joint/independent),
                         nonmonotone=sum(bool(np.any(np.diff(v)<-1e-9)) for v in losses)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    for d in (1, 10):
        group = [r for r in rows if r['realizations']==d]
        print('realizations',d,'savings min, median, max',np.percentile([r['percent_saved'] for r in group],[0,50,100]),
              'status counts',{s:sum(r['status']==s for r in group) for s in ('certified','fallback','safe_at_min')})
        print(next(r for r in group if r['paper_threshold']=='True'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ('traces','thresholds','out'):
        parser.add_argument('--'+opt, type=Path, required=True)
    compare(parser.parse_args())

"""Exact reference for graded calibration on a finite ordered event partition.

A row maps each of its decision changes to (point decision, decision just to
its left). Every row also has the two domain endpoints. At the lower endpoint
use None for the decision on the left. A score callback accepts a decision ID.
Coordinates and losses can be Fraction for exact arithmetic. A lazy path oracle
can supply the same two event types on demand, subject to its tie convention.
"""
from fractions import Fraction
import heapq
from itertools import product


def sweep(rows, scores, low, high, budget):
    states = [iter(sorted(row, reverse=True)) for row in rows]
    cache = [{} for row in rows]
    maxima = [0] * len(rows)
    heap = []
    for i, stream in enumerate(states):
        assert low in rows[i] and high in rows[i] and next(stream) == high
        heapq.heappush(heap, (-high, i))
    total = 0
    last_safe = None
    while heap:
        key = -heap[0][0]
        batch = []
        while heap and -heap[0][0] == key:
            batch.append(heapq.heappop(heap)[1])
        for phase in (0, 1):
            if phase == 1 and key == low:
                break
            for i in batch:
                decision = rows[i][key][phase]
                if maxima[i] == 1:
                    continue
                if decision not in cache[i]:
                    cache[i][decision] = scores[i](decision)
                value = cache[i][decision]
                if value > maxima[i]:
                    total += value - maxima[i]
                    maxima[i] = value
            if total > budget:
                if last_safe is None:
                    return None, None
                return key, last_safe
            if phase == 0:
                last_safe = key
            else:
                for i in batch:
                    nxt = next(states[i], None)
                    if nxt is not None:
                        heapq.heappush(heap, (-nxt, i))
                if heap:
                    last_safe = (key - heap[0][0]) / 2
    return low, low


def regression():
    low, mid, high = map(Fraction, (0, 1, 2))
    rows = [{high: (0, 1), mid: (2, 3), low: (4, None)}] * 2
    # Rows with coincident boundaries, arbitrary point-only tie choices and
    # independently varying open segments. Target budget is one half.
    tested = 0
    for values in product((Fraction(0), Fraction(1,2), Fraction(1)), repeat=6):
        a,b,c,d,e,f = values
        first = (a,b,c,d,e); second = (f,c,e,a,b)
        atoms = (high, (high+mid)/2, mid, (mid+low)/2, low)
        feasible = [x for j,x in enumerate(atoms) if max(first[:j+1])+max(second[:j+1]) <= Fraction(1,2)]
        boundary, deploy = sweep(rows, [first.__getitem__,second.__getitem__],low,high,Fraction(1,2))
        if not feasible:
            assert deploy is None
        else:
            assert deploy in feasible
            last = feasible[-1]
            infimum = mid if last==(high+mid)/2 else low if last==(mid+low)/2 else last
            assert boundary == infimum
        tested += 1
    print('Exact event-partition checks:', tested)

if __name__ == '__main__':
    regression()

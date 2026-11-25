# Batch CDT ablation of joint stopping

The independent baseline starts each of the 48 on-demand paths at its most conservative endpoint. It stops at that example's first unsafe event, or at exhaustion for an always-safe example. It then returns the fourth-largest of the 48 boundaries. The joint method uses the original heap and stops the batch at the fourth unsafe example. Both use the same generator, tie order within each path, loss, realized outcomes, risk budget, and numerical tolerances.

## Files

- `thresholds.csv` contains both counts, the common cutoff, and the percentage reduction relative to independent stopping for all 51 task/threshold settings. The `chosen` flag identifies the paper's headline thresholds. The cutoff in the file is the last-unsafe boundary. A cutoff of 1 invokes the benchmark's fallback.
- `timings.jsonl` records uncached, end-to-end calibration runs at the headline thresholds. A loss evaluation at T=10 tests one decision with ten realization-specific cost calculations. The solver only calls the recourse MILP when requirements are left uncovered.
- `traces/` contains the inputs to the count comparison, one compressed NumPy file per family and scene. They include the right endpoints, cumulative LP solves at each yielded decision, total LP solves on exhaustion, and the unthresholded costs for all sampled demand realizations.
- `trace.py` regenerates those traces. `compare.py` implements the independent baseline, runs the original joint sweep, calculates all threshold counts, and measures both methods on fresh solver calls. `check_results.py` checks fresh counts and cutoffs against replay and reports any differences from the archived public results.

## Reproduce

The experiment uses commit `63039ac30e54c5a723043cedc54f7c3069d2c391` of [the public benchmark](https://github.com/ImperatorAvrelianvs/batch-cdt). The benchmark's data files, scene generator, LP solver, cost function for a single demand realization and joint sweep are used directly. The run used Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0.

```sh
# From experiments/joint_stopping
python -m pip install numpy==2.3.5 scipy==1.17.0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python compare.py replay --repo ../.. --out rerun --traces traces
python compare.py timing --repo ../.. --out rerun --case knapsack 1 3 --case set_cover 1 3 --case set_cover 10 3
python check_results.py rerun --expected-runs 3
```

To rebuild every event and realization-specific cost from the original optimization problems, run `python trace.py --repo ../.. --out new_traces --workers 4` and point the replay at `new_traces`. Timings append to the output file, so use a fresh directory for a separate run.

## Measurement

The three tasks are knapsack and two-stage set covering with one or ten demand realizations per calibration example. Each realization specifies the requirements that materialize. The first-stage plan stays fixed across realizations; the loss uses its cost plus mean integer recourse cost. The 17 cut levels per task range from the 10th through the 90th cost percentiles in five-percentage-point increments. We use the exact threshold numbers in the public result files for their 50 archived interior rows. The omitted tenth-percentile knapsack threshold is regenerated using the same quantile calculation. Replay runs each stopping algorithm event by event, including generation of an unevaluated lookahead and unsuccessful requests at the end of a path. Every cutoff is checked independently against exhaustive evaluation of every event. Generating the full traces is an audit cost and is excluded from headline runtimes.

For fresh timings, each method uses the serial path and recourse oracles from the public repository, with fresh generators and no per-decision cache. Data generation and solver warm-up precede timing. The timer includes generator construction, LP solves, loss evaluations, selection, and for the joint method, heap operations. Code that records the result runs after the timer. Per-example scanning could also be run across multiple cores. This comparison measures the benefit of joint stopping with serial calibration in both methods.

## Counts at the paper's operating points

| Task | Common boundary | Independent LP solves | Joint LP solves | Independent loss evaluations | Joint loss evaluations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Knapsack | 0.498484 | 1,665 | 999 | 803 | 420 |
| Set cover, one demand realization | 0.429481 | 3,702 | 961 | 1,780 | 368 |
| Set cover, ten demand realizations | 0.545421 | 2,526 | 750 | 1,197 | 257 |

Joint and independent calibration agreed with exhaustive evaluation in all 51 settings. All 50 archived interior cutoffs and joint operation counts were reproduced. The remaining setting is the knapsack fallback at the tenth cost percentile. Both counts fell in all 51 comparisons; both fell by at least 25% in 40 of 51. These thresholds reuse the same underlying problem instances and are a sensitivity analysis.

| Task | Median reduction in LP solves | Range | Median reduction in loss evaluations | Range |
| --- | ---: | ---: | ---: | ---: |
| Knapsack | 32.1% | 25.9–60.5% | 44.5% | 37.3–84.7% |
| Set cover, one demand realization | 30.1% | 17.0–78.6% | 31.5% | 17.8–86.9% |
| Set cover, ten demand realizations | 28.0% | 15.7–70.3% | 29.1% | 16.5–78.5% |

`threshold_savings.pdf` and `threshold_savings.png` plot the savings over the 17 percentiles. Open rings mark the paper's published operating points. Run `plot.py` with Matplotlib installed to regenerate both figures.

## Headline runtimes

Median end-to-end seconds on one pinned CPU (Intel Xeon Platinum 8573C), after the exhaustive trace generation and first timing pass finished. Each method has three timings per task. Method order was seeded per task and alternated over repeated pairs. All 18 measurements reproduced the cutoffs and oracle counts from replay.

| Task | Runs per method | Independent (s) | Joint (s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Knapsack | 3 | 8.00 | 4.09 | 1.96× |
| Set cover, one demand realization | 3 | 65.06 | 13.54 | 4.80× |
| Set cover, ten demand realizations | 3 | 176.92 | 40.12 | 4.41× |

The original full-enumeration records give **estimated speedups of 44.5× and 74.6×** for one and ten demand realizations per example. These divide the sum of the measured exhaustive stages (388.96 s and 1,358.31 s) by the median end-to-end sweep time from the same experiment (8.749 s and 18.206 s). Generating all paths first and then sweeping their losses gives estimated ratios of 32.5× and 16.6×. The 4.80× and 4.41× comparisons use separate, fully timed runs of independent and joint stopping.

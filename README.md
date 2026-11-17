# Batch Conformal Decision Theory with Fewer Counterfactual Evaluations

Code and reproducible experiments for
[the paper](paper/main.pdf) ([LaTeX source](paper/main.tex);
[alphaXiv preprint](https://www.alphaxiv.org/abs/2611.batch-conformal-counterfactual-evaluations)).

We study the number of counterfactual planner calls needed for exact batch
calibration. The paper gives an order-statistic reduction for monotone binary
losses, black-box query bounds, and a backward search along a planner's
solution path. For parametric linear programs the path is generated only
as needed, reducing optimization solves as well as loss evaluations.

## Reproduce the public-data experiments

```bash
python -m pip install -r requirements.txt
python benchmark_public_cdt.py --repeats 11 --audit
python benchmark_public_cdt.py --threshold-sweep
```

The script uses original coefficients from two families in Beasley's
[OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/):

- `mknap2`: 48 multidimensional knapsack instances;
- `scp41`--`scp410`: ten set-covering instances, used to generate 48
  fixed-seed subsets of required rows.

The planners are **LP relaxations**, not the original integer programs.
Their binary losses are constructed for this experiment and do not constitute
observed safety outcomes. Every method uses the same matrices and SciPy
HiGHS solver. The optional `--audit` independently solves all 12,384 grid
problems and verifies the common calibrated cutoffs.
The threshold study compares query counts using cached LP solutions;
it is not a separate timing experiment.

| Problem | Ordinary bisection | Certified bisection |
| --- | ---: | ---: |
| Knapsack | 336 LP solves | 158 LP solves |
| Set covering | 336 LP solves | 115 LP solves |

Wall-clock measurements and environment versions are in
[`public_cdt_benchmark_results.json`](public_cdt_benchmark_results.json).
Run-to-run times will vary by machine.

## Continuous calibration with costly losses

```bash
python run_timing.py knapsack       # typically tens of seconds
python run_timing.py set_cover      # may take about 30 minutes
```

The scripts use the same OR-Library data but evaluate each fractional
decision by an additional integer optimization problem:

- `lp_path.py` constructs the parametric LP solution path and rounds
  knapsack decisions to feasible binary ones by integer programming.
- `two_stage.py` adds random requirements to set-covering decisions;
  unmet requirements are covered by integer optimization.
- `run_timing.py` compares complete path enumeration, a backward sweep
  over that path, and a path generated on demand from its conservative end.

All methods recover the same last-unsafe boundary. For 48 calibration
observations with risk level `alpha=0.1`, the supplied records contain:

| Task | Complete path LP solves | On-demand LP solves | All loss evaluations | Backward sweep |
| --- | ---: | ---: | ---: | ---: |
| Knapsack | 2,498 | 999 | 1,273 | 420 |
| Set covering, one realized day | 23,128 | 961 | 11,588 | 368 |
| Set covering, ten realized days | 23,128 | 750 | 11,588 | 257 |

The full-path times in [`timing_knapsack.json`](timing_knapsack.json)
and [`timing_set_cover.json`](timing_set_cover.json) are sums of recorded
LP and integer-program timings. The on-demand method was timed as a complete
run, three times for knapsack and one-day covering, and once for ten-day
covering. These comparisons should not be treated as independent
end-to-end timings for every method. Both JSON files include a threshold
sweep; the reported threshold for each task was chosen from 17
quantiles so that its cutoff is nearest `0.5`.

[`timing_knapsack_recheck.json`](timing_knapsack_recheck.json) is an
independent run on a second machine. It reproduces the solve counts
and cutoff; timings differ with the hardware and solver environment.

The [executed notebook](experiments.ipynb) contains these experiments,
synthetic tests of the continuous loss-evaluation procedure, the timed
LP results above, and a separate robot-planner diagnostic, with cached
outputs. Its original experiments are self-contained but have
additional dependencies (`pandas`, `matplotlib`, `numba`,
`opencv-python-headless`, `scikit-image`, and `scikit-fmm`).

## Data attribution and scope

The public optimization coefficients originate from the OR-Library
([knapsack](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html),
[set covering](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html)).
The notebook contains archived Stanford Drone Dataset material with its own
attribution and noncommercial terms. Neither the use of public benchmark
coefficients nor the synthetic study establishes risk control on real-world
decisions; the statistical guarantee in the paper is conditional on its
stated assumptions.

No repository-wide software license has been selected.

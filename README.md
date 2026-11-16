# Batch Conformal Decision Theory with Fewer Counterfactual Evaluations

Research draft and executable experiments accompanying
[the paper](https://www.alphaxiv.org/pdf/2611.batch-conformal-counterfactual-evaluations).

We study the number of counterfactual planner calls needed for exact batch
calibration. The paper gives an order-statistic reduction for monotone binary
losses, black-box query bounds, and a backward procedure for evaluating losses
along a planner's parameter-dependent solution path.

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

The [executed notebook](experiments.ipynb) contains these experiments,
synthetic tests of the continuous loss-evaluation procedure, and a separate
robot-planner diagnostic, with cached outputs. It is self-contained but has
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

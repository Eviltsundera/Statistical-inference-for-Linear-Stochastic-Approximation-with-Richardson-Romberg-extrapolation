# Theory-aligned RR stepsize sweep

**Date:** 2026-05-26
**Script:** `code/run_comparison.py`
**Machine:** `beleriand`
**Raw outputs:**

- `code/results/theory_rr_sweep/main_T1000000_base0p02_pair0p04_0p02_w24.csv`
- `code/results/theory_rr_sweep/main_T1000000_base0p05_pair0p10_0p05_w24.csv`
- `code/results/theory_rr_sweep/main_T1000000_base0p10_pair0p20_0p10_w24.csv`
- `code/results/theory_rr_sweep/theory_rr_sweep_T1000000_w24.log`

## Motivation

The first main comparison report used a deliberately wide RR pair,
`alpha in {0.2, 0.02}`.  This is useful as a stress test, but it is not the
canonical first-order Richardson--Romberg pair from the theory, where the two
stepsizes are usually adjacent scales, `(2 alpha, alpha)`.

This sweep checks whether the RR advantage persists when the pair is chosen in
the theory-aligned form `(2 alpha, alpha)` and the base stepsize `alpha` is
varied over `{0.02, 0.05, 0.10}`.

## Setup

- 100 problems x 100 trajectories, `T = 1_000_000`, `d = 5`,
  `n_states = 10`.
- Problem generation: `eig_min = 0.25`, `eig_max = 0.60`,
  `noise_target = 0.35`, seed `42`.
- Same problem seeds were used for all three runs, so the rows are directly
  comparable across RR pairs.
- PR schedule: `c0 = 200`, `k0 = 20000`, `gamma = 0.65`.
- Batch settings: `K = 63`, `burn_in = 1000`, OBM block size
  `b_n = floor(T^0.6)`.
- Projection direction: random unit direction per problem.
- Final run used `24` workers.  An earlier `80` worker attempt saturated the
  machine with blocked worker processes and was stopped before producing final
  CSV output.

Problem diagnostics were stable across the three runs:

| RR pair | median max `||A(x)||_2` | max max `||A(x)||_2` | median max `rho(I + alpha_hi A)` | max max `rho(I + alpha_hi A)` |
|---|---:|---:|---:|---:|
| `(0.04, 0.02)` | 0.770 | 0.847 | 0.994 | 0.999 |
| `(0.10, 0.05)` | 0.770 | 0.847 | 0.986 | 0.997 |
| `(0.20, 0.10)` | 0.770 | 0.847 | 0.972 | 0.994 |

## Main results

`L2` and `Width` are reported in units of `1e-3`.  Coverage target is 95%.
All methods had zero divergences.

| RR pair | Method | L2 | Width | Cov median | Cov mean |
|---|---|---:|---:|---:|---:|
| `(0.04, 0.02)` | alpha `0.04` const | 3.25 | 5.37 | 92.0% | 91.1% |
| `(0.04, 0.02)` | alpha `0.02` const | 3.05 | 5.36 | 94.0% | 93.3% |
| `(0.04, 0.02)` | **RR** | **2.97** | 5.36 | 94.0% | 94.2% |
| `(0.04, 0.02)` | RR + OBM | 2.97 | 5.33 | 94.0% | 94.3% |
| `(0.04, 0.02)` | RR + OBM-RR | 2.97 | 5.38 | 95.0% | 94.5% |
| `(0.10, 0.05)` | alpha `0.10` const | 4.46 | 5.37 | 86.0% | 77.4% |
| `(0.10, 0.05)` | alpha `0.05` const | 3.41 | 5.37 | 91.5% | 89.0% |
| `(0.10, 0.05)` | **RR** | **2.97** | 5.37 | 94.0% | 94.2% |
| `(0.10, 0.05)` | RR + OBM | 2.97 | 5.40 | 95.0% | 94.6% |
| `(0.10, 0.05)` | RR + OBM-RR | 2.97 | 5.39 | 95.0% | 94.5% |
| `(0.20, 0.10)` | alpha `0.20` const | 7.53 | 5.38 | 67.5% | 53.8% |
| `(0.20, 0.10)` | alpha `0.10` const | 4.46 | 5.37 | 86.0% | 77.4% |
| `(0.20, 0.10)` | **RR** | **2.97** | 5.38 | 94.0% | 94.2% |
| `(0.20, 0.10)` | RR + OBM | 2.97 | 5.42 | 95.0% | 94.6% |
| `(0.20, 0.10)` | RR + OBM-RR | 2.97 | 5.39 | 95.0% | 94.4% |

The PR baseline is unchanged across the three runs because its schedule does
not use the configured RR pair:

| Method | L2 | Width | Cov median | Cov mean |
|---|---:|---:|---:|---:|
| PR + OBM | 3.48 | 5.39 | 92.0% | 88.9% |
| PR + MSB | 3.48 | 5.35 | 91.0% | 88.5% |
| PR + OBM-RR | 3.48 | 5.40 | 92.0% | 88.7% |

## Interpretation

The main qualitative signal is stable: RR removes most of the constant-step
bias without increasing the CI width.  The larger single-alpha branches
undercover more severely as the base stepsize grows: median coverage falls from
92.0% at `alpha = 0.04` to 86.0% at `alpha = 0.10` and 67.5% at
`alpha = 0.20`.  The smaller branches are better, but still keep a visible bias
floor for `alpha >= 0.05`.

The RR rows are almost invariant across the three theory-aligned pairs.  Median
L2 stays at about `2.97e-3`, CI width stays at about `5.36e-3`--`5.38e-3`, and
median coverage stays at 94%.  This is stronger than merely finding one good
pair: within this range, the RR estimator is not especially sensitive to the
exact base stepsize.

OBM and lugsail change the RR confidence intervals only mildly at this horizon.
For all three pairs, `RR + OBM` and `RR + OBM-RR` move median coverage from
94% to at most 95%, with width changes below roughly 1%.  This agrees with the
earlier main comparison: at `T = 10^6`, the long-run variance estimator is not
the dominant error source for the RR method.

## Takeaways for the thesis

1. The thesis can present `(2 alpha, alpha)` as the natural RR choice without
   losing the empirical advantage seen in the earlier wide-pair comparison.
2. RR is robust over the tested base stepsizes: `alpha = 0.02`, `0.05`, and
   `0.10` give essentially the same L2 error, width, and coverage after
   extrapolation.
3. The experiment cleanly separates two bias sources.  Single-alpha
   constant-step branches undercover because of point-estimator bias; OBM and
   lugsail can only adjust long-run variance estimation and therefore cannot
   fix that undercoverage.  RR in `alpha` fixes the point-estimator bias first.
4. Lugsail remains a secondary effect in the long-horizon comparison.  It is
   worth keeping in the experiments, but the stronger empirical motivation for
   lugsail still comes from the short-horizon bias-variance report.


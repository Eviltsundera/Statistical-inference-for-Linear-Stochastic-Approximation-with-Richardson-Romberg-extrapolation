# Oracle-variance comparison for RR confidence intervals

**Date:** 2026-05-26
**Script:** `code/run_oracle_variance_comparison.py`
**Machine:** `beleriand`
**Raw outputs:**

- `code/results/oracle_variance/oracle_rr_T1000000_pair0p20_0p10_w24.csv`
- `code/results/oracle_variance/oracle_rr_T1000000_pair0p20_0p10_w24_summary.csv`
- `code/results/oracle_variance/oracle_rr_T1000000_pair0p20_0p10_w24.log`

## Motivation

The main comparison and theory-aligned stepsize sweep show that RR intervals
have near-nominal coverage at `T = 10^6`.  Those intervals still mix several
finite-sample effects:

- point-estimator bias of the RR center;
- normal approximation error;
- error from estimating the long-run variance with batch means, OBM, OBM-RR,
  or MSB.

This experiment removes the third source by replacing the data-driven variance
estimator with the analytic finite-state long-run variance

$$
\sigma^2_\infty(u) =
u^\top \bar A^{-1} \Gamma_\epsilon \bar A^{-\top} u,
$$

computed by `compute_asymptotic_variance`.  The interval center is unchanged:
all rows use the same RR-averaged constant-stepsize estimator.

## Setup

- 100 problems x 100 trajectories, `T = 1_000_000`, `d = 5`,
  `n_states = 10`.
- Problem generation: `eig_min = 0.25`, `eig_max = 0.60`,
  `noise_target = 0.35`, seed `42`.
- RR pair: `(0.20, 0.10)`, i.e. the largest adjacent pair from
  `reports/2026-05-26_theory_rr_alpha_sweep.md`.
- Burn-in and batching: `burn_in = 1000`, `K = 63`, effective averaging length
  `T_eff = 999000`.
- OBM block size: `b_n = floor(T^0.6) = 3981`.
- MSB bootstrap replications: `500`.
- Projection direction: one random unit direction per problem.
- Runtime: 971 s, about 16.2 min with 24 workers.

The runner was checked against `run_comparison.py` on a smoke test so that the
reported L2/width aggregation convention matches the earlier reports: first
average over trajectories within a problem, then take medians across problems.

## Results

`L2` and `Width` are reported in units of `1e-3`.  Coverage target is 95%.
All methods had zero divergences.

| Method | L2 | Width | Cov median | Cov mean | Width / oracle |
|---|---:|---:|---:|---:|---:|
| RR + batch means | 2.97 | 5.38 | 94.0% | 94.2% | 0.990 |
| RR + oracle variance | 2.97 | 5.43 | 95.0% | 94.8% | 1.000 |
| RR + OBM | 2.97 | 5.42 | 95.0% | 94.6% | 0.998 |
| RR + OBM-RR | 2.97 | 5.39 | 95.0% | 94.4% | 0.993 |
| RR + MSB | 2.97 | 5.36 | 95.0% | 94.4% | 0.987 |

Coverage percentiles across problems:

| Method | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| RR + batch means | 91.9 | 93.0 | 94.0 | 96.0 | 97.0 |
| RR + oracle variance | 92.0 | 93.8 | 95.0 | 96.0 | 97.0 |
| RR + OBM | 92.0 | 93.8 | 95.0 | 96.0 | 97.0 |
| RR + OBM-RR | 92.0 | 93.0 | 95.0 | 96.0 | 97.0 |
| RR + MSB | 92.0 | 93.0 | 95.0 | 96.0 | 97.0 |

## Interpretation

The oracle row is almost indistinguishable from the practical variance
estimators.  OBM is only about 0.2% narrower than the oracle interval at the
median, OBM-RR about 0.7% narrower, and MSB about 1.3% narrower.  Coverage
changes by at most 0.4 percentage points in mean coverage and not at all at the
median for OBM, OBM-RR, and MSB.

This indicates that, for the RR pair `(0.20, 0.10)` at `T = 10^6`,
long-run-variance estimation is not the bottleneck.  Once the RR center has
removed the dominant constant-stepsize bias, the analytic oracle interval and
the data-driven intervals all deliver near-nominal coverage.

The batch-means row is slightly below the oracle row, with median coverage 94%
instead of 95%.  The difference is small and consistent with the earlier
theory-aligned sweep: at this horizon, the choice among batch means, OBM,
OBM-RR, and MSB only mildly affects RR intervals.

## Takeaways for the thesis

1. The remaining coverage error of RR at `T = 10^6` is not primarily caused by
   the practical long-run variance estimator.
2. OBM is already close to the oracle variance benchmark at this horizon.
3. Lugsail/OBM-RR is again neutral in the long-horizon setting: it does not
   hurt, but it also does not materially improve coverage over OBM.
4. The next useful diagnostic is therefore not another long-horizon oracle
   comparison, but a horizon sweep.  At shorter `T`, the lugsail bias-variance
   report suggests that variance-estimator bias should become visible.


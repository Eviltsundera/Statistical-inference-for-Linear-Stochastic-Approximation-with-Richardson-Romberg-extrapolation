# RR coverage sweep over trajectory length

**Date:** 2026-05-26
**Script:** `code/run_oracle_variance_comparison.py`
**Machine:** `beleriand`
**Raw outputs:**

- `code/results/oracle_variance/oracle_rr_Tsweep_pair0p20_0p10_w24.csv`
- `code/results/oracle_variance/oracle_rr_Tsweep_pair0p20_0p10_w24_summary.csv`
- `code/results/oracle_variance/oracle_rr_Tsweep_pair0p20_0p10_w24.log`

## Motivation

The fixed-horizon oracle comparison at `T = 10^6` showed that OBM, OBM-RR,
MSB, and the analytic oracle variance give nearly identical RR confidence
intervals.  This sweep asks where that agreement begins.  The goal is to
separate three effects over trajectory length:

- finite-horizon point-estimator and normal-approximation error, visible in
  the oracle row;
- long-run-variance estimation error, visible as a gap between oracle and
  OBM/MSB/OBM-RR rows;
- possible lugsail improvement, visible if OBM-RR moves coverage or width
  closer to the oracle row than OBM.

## Setup

- 100 problems x 100 trajectories for each `T`.
- Horizons:
  `T in {20_000, 50_000, 100_000, 300_000, 1_000_000}`.
- Problem generation: `d = 5`, `n_states = 10`, `eig_min = 0.25`,
  `eig_max = 0.60`, `noise_target = 0.35`, seed `42`.
- RR pair: `(0.20, 0.10)`.
- Same problem seeds and projection directions are reused across horizons.
- OBM block size: `b_n = floor(T^0.6)`.
- MSB bootstrap replications: `500`.
- Runtime: 1210 s, about 20.2 min with 24 workers.

## Results

`L2` and `Width` are reported in units of `1e-3`.  Coverage target is 95%.
All rows had zero divergences.

| T | Method | L2 | Width | Cov median | Cov mean | Width / oracle |
|---:|---|---:|---:|---:|---:|---:|
| 20 000 | RR + oracle variance | 21.11 | 39.37 | 95.5% | 95.2% | 1.000 |
| 20 000 | RR + OBM | 21.11 | 37.07 | 94.0% | 93.7% | 0.942 |
| 20 000 | RR + OBM-RR | 21.11 | 37.16 | 93.0% | 92.9% | 0.944 |
| 20 000 | RR + MSB | 21.11 | 36.71 | 94.0% | 93.3% | 0.933 |
| 50 000 | RR + oracle variance | 13.32 | 24.52 | 95.0% | 94.8% | 1.000 |
| 50 000 | RR + OBM | 13.32 | 23.71 | 94.0% | 93.7% | 0.967 |
| 50 000 | RR + OBM-RR | 13.32 | 23.48 | 93.0% | 93.1% | 0.958 |
| 50 000 | RR + MSB | 13.32 | 23.51 | 94.0% | 93.3% | 0.959 |
| 100 000 | RR + oracle variance | 9.39 | 17.25 | 95.0% | 94.7% | 1.000 |
| 100 000 | RR + OBM | 9.39 | 16.87 | 94.0% | 94.1% | 0.978 |
| 100 000 | RR + OBM-RR | 9.39 | 16.76 | 94.0% | 93.5% | 0.972 |
| 100 000 | RR + MSB | 9.39 | 16.66 | 94.0% | 93.6% | 0.966 |
| 300 000 | RR + oracle variance | 5.43 | 9.92 | 95.0% | 94.8% | 1.000 |
| 300 000 | RR + OBM | 5.43 | 9.82 | 94.0% | 94.5% | 0.989 |
| 300 000 | RR + OBM-RR | 5.43 | 9.82 | 94.0% | 94.2% | 0.989 |
| 300 000 | RR + MSB | 5.43 | 9.72 | 94.0% | 94.0% | 0.980 |
| 1 000 000 | RR + oracle variance | 2.97 | 5.43 | 95.0% | 94.8% | 1.000 |
| 1 000 000 | RR + OBM | 2.97 | 5.42 | 95.0% | 94.6% | 0.998 |
| 1 000 000 | RR + OBM-RR | 2.97 | 5.39 | 95.0% | 94.4% | 0.993 |
| 1 000 000 | RR + MSB | 2.97 | 5.36 | 95.0% | 94.4% | 0.987 |

The batch-means row from the same run is useful as a legacy baseline:

| T | RR + batch means coverage | Width / oracle |
|---:|---:|---:|
| 20 000 | 92.0% | 0.937 |
| 50 000 | 93.0% | 0.943 |
| 100 000 | 93.5% | 0.965 |
| 300 000 | 94.0% | 0.981 |
| 1 000 000 | 94.0% | 0.990 |

## Interpretation

The oracle row is stable across horizons.  Median oracle coverage is between
95.0% and 95.5%, and mean oracle coverage is between 94.7% and 95.2%.  Thus,
for this RR pair and problem class, the RR center and the normal approximation
are already adequate even at `T = 20_000`.

The practical variance estimators are the source of the remaining
undercoverage at shorter horizons.  At `T = 20_000`, OBM intervals are about
5.8% narrower than oracle intervals, and median coverage drops from 95.5% to
94.0%.  By `T = 300_000`, OBM width is within 1.1% of oracle width.  By
`T = 1_000_000`, OBM is essentially oracle-level.

With the default block size `b_n = floor(T^0.6)`, lugsail/OBM-RR does not
improve coverage over OBM in this coverage sweep.  It is neutral at large
`T`, and at smaller `T` it is usually slightly narrower than OBM and has
similar or lower coverage.  This does not contradict the separate
bias-variance report: that report optimizes over block sizes and shows that
lugsail can reduce MSE at shorter horizons.  The present sweep fixes the
production block-size rule and measures CI coverage, so it answers a different
question.

## Takeaways for the thesis

1. RR point-estimator and normal-approximation errors are not the main
   bottleneck in this sweep; the oracle row is already near nominal coverage.
2. The practical variance-estimator gap shrinks with `T`: OBM width/oracle
   moves from 0.942 at `T = 20_000` to 0.998 at `T = 1_000_000`.
3. Default lugsail is not a coverage improvement for this RR run.  Its value
   should be presented as a short-horizon variance-estimator bias-reduction
   tool whose effectiveness depends on block-size tuning, not as an automatic
   coverage fix.
4. The next diagnostic should be a block-size sweep for RR coverage, tracking
   OBM and OBM-RR widths, coverage, variance-estimator bias, and clamping or
   negative-estimate rates.


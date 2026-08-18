# NOBM vs OBM vs oracle for RR confidence intervals

**Date:** 2026-08-18
**Script:** `code/run_oracle_variance_comparison.py`
**Machine:** `train-4`
**Raw outputs:**

- `code/results/oracle_variance/oracle_rr_nobm_T20k_1M_pair0p20_0p10_w50.csv`
- `code/results/oracle_variance/oracle_rr_nobm_T20k_1M_pair0p20_0p10_w50_summary.csv`
- `code/results/oracle_variance/oracle_rr_nobm_T20k_1M_pair0p20_0p10_w50.log`

## Motivation

The PSTA article's confidence-interval section distinguishes three scale
normalizations: the analytic oracle variance, non-overlapping batch means
(NOBM), and overlapping batch means (OBM).  The earlier oracle comparison
(`reports/2026-05-26_oracle_variance_rr.md`,
`reports/2026-05-26_rr_coverage_T_sweep.md`) had no NOBM row at the OBM
block size: its only non-overlapping estimator was the legacy
`RR + batch means` with `K = floor(T^0.3)` large blocks.  This run adds a
NOBM estimator with the *same* block size as OBM, `b_n = floor(T^0.6)`,
isolating the effect of window overlap alone.

New estimator: `nobm_ci` in `code/lsa_inference/inference.py`.  It uses the
first `K = floor(T_eff / b_n)` disjoint blocks of the post-burn-in RR
projection, drops the tail shorter than a block, and applies the classical
normalization `sigma^2_NBM = b_n / (K - 1) * sum_j (bm_j - mean(bm))^2`
(Flegal & Jones 2010).  The interval center is the same full-sample RR
average as in every other row.

## Setup

Identical to the 2026-05-26 oracle runs (same seed, hence the same problems,
directions, and trajectories):

- 100 problems x 100 trajectories, `T in {20_000, 1_000_000}`, `d = 5`,
  `n_states = 10`.
- Problem generation: `eig_min = 0.25`, `eig_max = 0.60`,
  `noise_target = 0.35`, seed `42`.
- RR pair: `(0.20, 0.10)`; burn-in `1000`.
- Block sizes: OBM/NOBM/OBM-RR/MSB use `b_n = floor(T^0.6)`
  (380 at `T = 20_000`, 3981 at `T = 10^6`); legacy batch means use
  `K = floor(T^0.3)` blocks.
- MSB bootstrap replications: `500`.
- Runtime: 437 s with 50 workers.

All pre-existing rows reproduce the 2026-05-26 numbers exactly, confirming
that adding the estimator did not change the RNG stream.

## Results

`L2` and `Width` in units of `1e-3`; coverage target 95%; zero divergences.

| T | Method | L2 | Width | Cov median | Cov mean | Width / oracle |
|---:|---|---:|---:|---:|---:|---:|
| 20 000 | RR + batch means | 21.11 | 36.91 | 92.0% | 92.5% | 0.937 |
| 20 000 | RR + oracle variance | 21.11 | 39.37 | 95.5% | 95.2% | 1.000 |
| 20 000 | RR + NOBM | 21.11 | 37.43 | 94.0% | 93.8% | 0.951 |
| 20 000 | RR + OBM | 21.11 | 37.07 | 94.0% | 93.7% | 0.942 |
| 20 000 | RR + OBM-RR | 21.11 | 37.16 | 93.0% | 92.9% | 0.944 |
| 20 000 | RR + MSB | 21.11 | 36.71 | 94.0% | 93.3% | 0.933 |
| 1 000 000 | RR + batch means | 2.97 | 5.38 | 94.0% | 94.2% | 0.990 |
| 1 000 000 | RR + oracle variance | 2.97 | 5.43 | 95.0% | 94.8% | 1.000 |
| 1 000 000 | RR + NOBM | 2.97 | 5.43 | 95.0% | 94.7% | 1.001 |
| 1 000 000 | RR + OBM | 2.97 | 5.42 | 95.0% | 94.6% | 0.998 |
| 1 000 000 | RR + OBM-RR | 2.97 | 5.39 | 95.0% | 94.4% | 0.993 |
| 1 000 000 | RR + MSB | 2.97 | 5.36 | 95.0% | 94.4% | 0.987 |

Coverage percentiles across problems:

| T | Method | p10 | p25 | p50 | p75 | p90 |
|---:|---|---:|---:|---:|---:|---:|
| 20 000 | RR + NOBM | 90.9 | 92.0 | 94.0 | 95.0 | 96.0 |
| 20 000 | RR + OBM | 90.0 | 92.0 | 94.0 | 95.0 | 97.0 |
| 1 000 000 | RR + NOBM | 92.0 | 94.0 | 95.0 | 96.0 | 97.0 |
| 1 000 000 | RR + OBM | 92.0 | 93.8 | 95.0 | 96.0 | 97.0 |

## Interpretation

At the same block size `b_n = floor(T^0.6)`, window overlap barely matters
for RR intervals in this problem class.  At `T = 20_000` NOBM is slightly
wider than OBM (width/oracle 0.951 vs 0.942) with the same 94.0% median
coverage; at `T = 10^6` both are indistinguishable from the oracle
normalization (ratios 1.001 and 0.998, median coverage 95%).  The remaining
short-horizon undercoverage (about 1.5 pp below the oracle row) is a
block-estimator bias shared by both variants, not an artifact of
overlapping windows.

## Article impact

`article-psta/main.tex`, section "Оракульная и практическая дисперсия":
NOBM rows added to the oracle table, and the CI section now checks both NOBM
and OBM against the oracle normalization.

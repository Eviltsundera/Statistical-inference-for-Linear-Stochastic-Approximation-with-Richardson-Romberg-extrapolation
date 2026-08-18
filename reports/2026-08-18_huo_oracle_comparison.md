# Huo batch-mean estimator vs OBM and oracle (Table-3 setting)

**Date:** 2026-08-18
**Script:** `code/run_oracle_variance_comparison.py`
**Machine:** `train-4`
**Raw outputs:**

- `code/results/oracle_variance/oracle_rr_huo_T20k_1M_pair0p20_0p10_w50.csv`
- `code/results/oracle_variance/oracle_rr_huo_T20k_1M_pair0p20_0p10_w50_summary.csv`
- `code/results/oracle_variance/oracle_rr_huo_T20k_1M_pair0p20_0p10_w50.log`

## Motivation

Companion run to `reports/2026-08-18_huo_bm_sweep.md`: put the Huo et al.
(2023) estimator (mean of batch means with intra-batch burn-in `n0`,
variance from the same windows) into the exact setting of Table 3 of the
PSTA article (`article-psta/main.tex`, `tab:oracle`) and compare it with
the two methods of that table — the analytic oracle and OBM.  The legacy
`RR + batch means` configuration *is* the Huo estimator with `n0 = 0`;
the new rows turn on `n0 = n/4` and `n0 = n/2`.

Implementation: `run_rr_full` gained an `n0` pass-through to
`run_lsa_const_full` (which already supported intra-batch discard); the CI
is the existing `batch_mean_ci` with the same `n0`.  The extra engine calls
consume no RNG, so all pre-existing rows reproduce the 2026-05-26 /
2026-08-18 numbers exactly (verified).  Cross-check: the Huo widths agree
bit-for-bit with the independent projection-based implementation in
`run_huo_bm_sweep.py` on common configurations.  (The run also refreshed
the remaining bench rows — NOBM, OBM-RR, MSB — they are in the CSVs but
out of scope here.)

## Setup

Identical to the earlier oracle runs: 100 problems x 100 trajectories,
seed 42, `T in {20_000, 1_000_000}`, `d = 5`, `n_states = 10`, RR pair
`(0.20, 0.10)`, burn-in 1000.  Huo rows use `K = floor(T^0.3)` batches
(19 / 63), so the batch sizes are `n = 1000` / `15_857`; OBM uses
`b_n = floor(T^0.6)` (380 / 3981) with the plain post-burn-in average as
center.  Runtime: 366 s with 50 workers.

## Results

`L2` and `Width` in units of 1e-3; target 95%; zero divergences.
Medians over problems.

| T | Method | L2 | Width | Cov median | Cov mean | Width / oracle |
|---:|---|---:|---:|---:|---:|---:|
| 20 000 | RR + oracle variance | 21.11 | 39.37 | 95.5% | 95.2% | 1.000 |
| 20 000 | RR + OBM | 21.11 | 37.07 | 94.0% | 93.7% | 0.942 |
| 20 000 | RR + Huo BM (n0=0) | 21.11 | 36.91 | 92.0% | 92.5% | 0.937 |
| 20 000 | RR + Huo BM (n0=n/4) | 24.26 | 42.51 | 92.5% | 92.7% | 1.080 |
| 20 000 | RR + Huo BM (n0=n/2) | 29.49 | 52.02 | 93.0% | 92.7% | 1.321 |
| 1 000 000 | RR + oracle variance | 2.97 | 5.43 | 95.0% | 94.8% | 1.000 |
| 1 000 000 | RR + OBM | 2.97 | 5.42 | 95.0% | 94.6% | 0.998 |
| 1 000 000 | RR + Huo BM (n0=0) | 2.97 | 5.38 | 94.0% | 94.2% | 0.990 |
| 1 000 000 | RR + Huo BM (n0=n/4) | 3.41 | 6.22 | 94.0% | 94.2% | 1.146 |
| 1 000 000 | RR + Huo BM (n0=n/2) | 4.12 | 7.53 | 94.0% | 94.2% | 1.387 |

## Interpretation

At the Table-3 pair `(0.20, 0.10)` — fast iterate mixing — the Huo
construction never reaches OBM, and the intra-batch burn-in only makes it
worse:

1. **Center (L2).**  With `n0 = 0` the Huo center coincides with the
   tail average (L2 = 21.11 / 2.97, same as OBM and oracle).  Discarding
   `n0 = n/4` of every batch inflates the L2 error by 15%
   (21.11 -> 24.26 at `T = 20_000`, 2.97 -> 3.41 at `T = 10^6`);
   `n0 = n/2` by ~40%.  This is measured in full d-dim L2 (engine batch
   means), confirming the projection-based finding of the sweep report.
2. **Coverage.**  The burn-in buys +0.2 pp mean coverage at `T = 20_000`
   and exactly zero at `T = 10^6`, despite intervals 8–39% wider than
   the oracle: the widening merely compensates the noisier, worse center.
   Huo stays 1–2.5 pp below OBM at the short horizon.
3. **Bottom line.**  OBM with the tail-average center dominates every Huo
   variant at both horizons: higher coverage with narrower intervals, and
   at `T = 10^6` it is indistinguishable from the oracle (ratio 0.998)
   while Huo `n0 = n/2` is 39% wider with a 39% worse center.

Together with the sweep report: the intra-batch burn-in pays off only in
the slow-mixing short-horizon regime (pair `(0.02, 0.01)`, `T = 20_000`,
up to +4 pp coverage); in the article's Table-3 setting the Huo estimator
is strictly dominated by OBM.

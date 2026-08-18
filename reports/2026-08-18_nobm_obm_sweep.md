# NOBM vs OBM sweep over stepsize pairs and block sizes

**Date:** 2026-08-18
**Script:** `code/run_nobm_obm_sweep.py`
**Machine:** `train-4`
**Raw outputs:**

- `code/results/nobm_obm_sweep/nobm_obm_sweep_T20k_100k_1M_w50.csv`
- `code/results/nobm_obm_sweep/nobm_obm_sweep_T20k_100k_1M_w50_summary.csv`
- `code/results/nobm_obm_sweep/nobm_obm_sweep_T20k_100k_1M_w50.log`

## Motivation

`reports/2026-08-18_nobm_oracle_comparison.md` found NOBM and OBM nearly
indistinguishable at the production settings (pair `(0.20, 0.10)`,
`b_n = T^0.6`).  Hypothesis to test: a real difference between overlapping
and non-overlapping windows might appear at a different stepsize (slower
iterate mixing for small alpha) or a different block size in the variance
estimator (few blocks -> noisy NOBM).

## Setup

- 100 problems x 100 trajectories, seed 42 (same problems, directions, and
  trajectories as the oracle runs; RNG consumption order replicated).
- `T in {20_000, 100_000, 1_000_000}`, `d = 5`, `n_states = 10`,
  burn-in 1000.
- RR pairs `(2 alpha, alpha)`:
  `(0.02, 0.01), (0.04, 0.02), (0.10, 0.05), (0.20, 0.10)`.
  Trajectories are shared across pairs within a problem.
- Block sizes `b_n = floor(T^gamma)`, `gamma in {0.4, 0.5, 0.6, 0.7, 0.8}`.
  At `gamma = 0.8` NOBM has only 6 (T=2e4), 9 (1e5), 15 (1e6) blocks.
- Methods: analytic oracle, NOBM, OBM.  NOBM and OBM are evaluated on the
  same RR projection, so per-problem differences are exactly paired.
- Runtime: 499 s with 50 workers.

## Results

Full grid in the log/CSVs.  The essentials:

**1. The dominant effect is the shared block-size bias, identical for both
methods.**  Coverage collapses when `b_n` is small relative to the iterate
mixing time `~ 1/(alpha * lambda_min)`, and does so equally for NOBM and
OBM.  E.g. at `T = 20_000`, pair `(0.02, 0.01)`, `gamma = 0.4` both cover
37.5% (oracle row: 95.0%).  Smaller alpha shifts the required block size up
but never separates the two estimators.

**2. Paired coverage differences NOBM - OBM are within ~1 pp everywhere**
(60 configurations).  Pooled over T and pairs:

| gamma | mean paired cov diff, pp | SE | mean width diff |
|---:|---:|---:|---:|
| 0.4 | +0.04 | 0.01 | +0.07% |
| 0.5 | +0.09 | 0.02 | +0.20% |
| 0.6 | +0.13 | 0.02 | +0.56% |
| 0.7 | +0.21 | 0.04 | +1.54% |
| 0.8 | -0.24 | 0.06 | +4.34% |

The largest single-configuration effects:

- `T = 20_000`, pair `(0.02, 0.01)`, `gamma = 0.7`: NOBM covers
  **+1.03 pp** better (t = 7.2), being 2.9% wider.
- `T = 20_000`, pairs `(0.10, 0.05)` / `(0.20, 0.10)`, `gamma = 0.8`:
  NOBM covers **-0.6 ... -0.8 pp** worse despite being ~5.7% *wider*.

**3. Interpretation: two small opposing mechanisms.**  For
`gamma <= 0.7` OBM's extra downward bias (edge effects of relative order
`b/T`) makes it slightly narrower and slightly under-covering, so NOBM is
marginally better; the effect grows with `b/T` and with slower mixing.
At `gamma = 0.8` the number of non-overlapping blocks drops to 6-15 and the
chi-square noise of the NOBM variance estimate outweighs its smaller bias:
NOBM is clearly wider yet covers slightly *worse*, consistent with OBM's
~1.5x effective degrees of freedom at the same block size.

**4. No stepsize interaction beyond block size.**  At every pair the
NOBM-OBM gap follows the same gamma pattern; alpha only moves the shared
bias curve (through the mixing time), not the overlap effect.

## Takeaways

1. The hypothesis "NOBM and OBM may differ at other stepsizes" is not
   supported: the overlap choice never matters by more than ~1 pp of
   coverage, at any pair.
2. The hypothesis "may differ at other block sizes" holds only marginally
   and in both directions: NOBM slightly better for moderate blocks
   (`gamma <= 0.7`, up to +1 pp at slow mixing), OBM slightly better in the
   few-block regime (`gamma = 0.8`).
3. At the production rule `b_n = T^0.6` the difference is negligible at all
   stepsizes and horizons, confirming the article's claim that window
   overlap is immaterial there.
4. What actually matters for coverage is `b_n` vs the iterate mixing time
   `1/(alpha * lambda_min)`: for the smallest pair `(0.02, 0.01)` even the
   best block size under-covers at `T = 20_000` (88% vs oracle 95%).

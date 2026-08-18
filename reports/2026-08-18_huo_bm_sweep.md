# Huo batch-mean estimator: intra-batch burn-in and batch-count sweep

**Date:** 2026-08-18
**Script:** `code/run_huo_bm_sweep.py`
**Machine:** `train-4`
**Raw outputs:**

- `code/results/huo_bm_sweep/huo_bm_sweep_T20k_100k_1M_w50.csv`
- `code/results/huo_bm_sweep/huo_bm_sweep_T20k_100k_1M_w50_summary.csv`
- `code/results/huo_bm_sweep/huo_bm_sweep_T20k_100k_1M_w50.log`

## Motivation

Huo et al. (2023) estimate theta differently from the rest of our pipeline:
after the global burn-in the trajectory is split into `K` equal batches, the
first `n0` iterates of *every batch* are discarded (intra-batch burn-in),
theta is estimated per batch by the batch mean, and the point estimate is
the **average of the K batch means**.  The variance is estimated from the
same windows,

```
sigma^2 = ((n - n0) / K) * sum_k (bm_k - center)^2,
CI      = center +/- z_{0.975} * sqrt(sigma^2 / (K * (n - n0)))
```

(`code/docs/huo2023_experiment_spec.md`).  The legacy `RR + batch means`
rows in all earlier runs are exactly this estimator with `n0 = 0` and
`K = floor(T^0.3)`.  This sweep turns on the intra-batch burn-in
(`n0/n in {0, 0.1, 0.25, 0.5}`) and varies the batch count
(`K in {floor(T^0.3), 50, 100}`) to see what the mean-of-means center and
the per-batch burn-in actually buy.

## Setup

- 100 problems x 100 trajectories, seed 42 — same problems, directions,
  and trajectories as the oracle/NOBM runs (RNG consumption order
  replicated).
- `T in {20_000, 100_000, 1_000_000}`, `d = 5`, `n_states = 10`,
  global burn-in 1000.
- RR pairs `(0.02, 0.01), (0.04, 0.02), (0.10, 0.05), (0.20, 0.10)`;
  trajectories shared across pairs.
- Huo CI computed post-hoc from the RR projection: batch size
  `n = T_eff // K`, tail dropped, first `n0` points of each batch
  discarded; z-quantile as in the paper.
- Reference rows on the same projections: analytic oracle CI and
  production OBM (`b_n = T^0.6`), both centered at the plain
  post-burn-in average ("tail center").
- Center quality is measured by the absolute error along the CI direction
  (`err`), paired with the tail center's error (`err/tail`); the full
  d-dim L2 is not recoverable from the 1-d projection.
- Runtime: 501 s with 50 workers.

**Validation anchor:** the config `(K = floor(T^0.3), n0 = 0)` at pair
`(0.20, 0.10)` reproduces the legacy `RR + batch means` rows of
`reports/2026-08-18_nobm_oracle_comparison.md` exactly: width 36.91,
coverage 92.0/92.5 at `T = 20_000`; width 5.38, coverage 94.0/94.2 at
`T = 10^6`.

## Results

Median over problems; width and err in units of 1e-3; target 95%; zero
divergences.  Full grid in the log/CSVs.

**1. The mean-of-means center is never better than the plain tail
average.**  With `n0 = 0` it coincides with the tail average up to the
dropped tail (`err/tail = 1.000` everywhere).  With `n0 > 0` it is
strictly worse: discarding `n0 = n/2` inflates the directional error by
30–44% at fast pairs and long horizons (e.g. `T = 10^6`, any pair:
err/tail ~ 1.4), simply because half the data is thrown away.

**2. Intra-batch burn-in helps coverage only in the hard regime — short
horizon, slow mixing, few long batches.**  `T = 20_000`, pair
`(0.02, 0.01)`, `K = 19` (`n = 1000`):

| n0 | width | w/oracle | cov mean |
|---:|---:|---:|---:|
| 0 | 30.52 | 0.775 | 86.9% |
| 100 | 31.59 | 0.802 | 88.1% |
| 250 | 33.34 | 0.847 | 89.7% |
| 500 | 36.74 | 0.933 | 91.0% |

(oracle row: 39.37 / 95.3%; production OBM `b = 380`: 23.91 / 76.6%).
Dropping half of every batch buys +4.2 pp coverage for +20% width, by
decorrelating adjacent batch means and removing the downward bias of the
variance estimate.  The same pattern, weaker, holds for `(0.04, 0.02)`
(+2.2 pp) and fades as mixing gets faster.

**3. At `T = 10^6` the intra-batch burn-in is pure loss.**  All configs
cover 94–95% already at `n0 = 0`; increasing `n0` only widens the
interval (up to w/oracle ~ 1.4) and degrades the center (err/tail up to
1.44) with no coverage gain beyond ~0.5 pp.

**4. Batch *length* remains the dominant factor, as in the NOBM/OBM
sweep.**  At `T = 20_000` going from `K = 19` to `K = 100` collapses
coverage from 86.9% to 63.2% at the slow pair — batches of 190 steps are
far below the mixing time `~1/(alpha * lambda_min) = 400`.  No `n0` can
repair too-short batches (`K = 100`, `n0 = n/2`: 64.5%).  At `T = 10^6`
the choice of `K in {50, 63, 100}` is immaterial.

**5. Against the production pipeline.**  In the hard regime the Huo
construction with `K = floor(T^0.3)` beats production OBM (`b = T^0.6`)
mainly because its batches are longer (1000 vs 380 steps at
`T = 20_000`), with the intra-batch burn-in adding a further +4 pp on
top.  Where OBM already works (fast pairs, long horizons) the Huo
estimator with `n0 = 0` is equivalent, and with `n0 > 0` is strictly
wider with a worse center.

## Takeaways

1. The distinctive Huo center (average of batch means) adds nothing over
   the plain post-burn-in average: identical at `n0 = 0`, strictly worse
   once data is discarded.
2. The intra-batch burn-in is a variance-estimator fix, not a center fix:
   it trades width for coverage and pays off only when batches are long
   relative to mixing but the variance estimate is still biased (short
   `T`, small alpha).  There it recovers ~half of the coverage gap to the
   oracle (86.9% -> 91.0% vs 95.3%).
3. In the production regime (`T >= 10^5` or fast pairs) `n0 > 0` is
   harmful: wider CIs, worse point estimate, no coverage gain.
4. A cheaper route to the same coverage in the hard regime is simply
   longer batches (the `K = 19` column dominates `K = 50/100` uniformly),
   consistent with the NOBM/OBM sweep's conclusion that what matters is
   block length vs the mixing time `1/(alpha * lambda_min)`.

Remark: with `K = 19` batches the paper's z-quantile is optimistic; a
t-quantile with 18 dof would widen all `K = 19` intervals by a further
7.2% and lift coverage accordingly — worth keeping in mind before
attributing all of the residual gap to batch-mean bias.

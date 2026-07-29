# OBM / OBM-LW optimal block-size scaling plan

**Date:** 2026-06-08
**Purpose:** repeat the OBM and OBM-LW MSE experiments and estimate the
asymptotic scaling of the empirically optimal block size \(b^*\) within each
sample size \(T\).

No local experiment run was performed for this note.

## Scripts

- Experiment runner: `code/run_lugsail_decomposition.py`
- Existing fit plotter: `code/plot_lugsail_decomposition.py`
- New scaling analyzer: `code/analyze_lugsail_optimal_b.py`

The new analyzer accepts either:

- raw single-problem decomposition output from `run_lugsail_decomposition.py`;
- aggregated multi-problem output from `run_lugsail_bias_variance.py`.

It writes:

- `<prefix>_bstar.csv`: empirical \(b^*\) per \(T\) and method;
- `<prefix>_x_scan.csv`: stability metrics for \(b^*/T^x\);
- `<prefix>_scaling_summary.csv`: fitted \(b^* \approx C T^\eta\), fitted
  \(MSE^* \approx D T^\kappa\), and best candidate \(x\);
- `<prefix>_bstar_scaling.png`;
- `<prefix>_bstar_over_Tx.png`;
- `<prefix>_x_scan.png`.

## Recommended remote run

From the repository root on the remote machine:

```bash
cd code
uv run python run_lugsail_decomposition.py \
  --prob-seed 0 \
  --dir-seed 1 \
  --traj-seed 2 \
  --n-traj 1200 \
  --traj-chunk 200 \
  --T-list 10000 30000 100000 300000 1000000 \
  --lam-list 2 3 4 \
  --bn-points 80 \
  --bn-min-exp 0.15 \
  --bn-max-exp 0.90 \
  --out results/lugsail_decomp_dense_2026-06-08.csv \
  > results/lugsail_decomp_dense_2026-06-08.log 2>&1
```

Then regenerate the old MSE decomposition figures:

```bash
uv run python plot_lugsail_decomposition.py \
  --csv results/lugsail_decomp_dense_2026-06-08.csv \
  --outdir ../reports/figures/lugsail_decomp_dense_2026-06-08
```

Then run the new \(b^*/T^x\) analysis:

```bash
uv run python analyze_lugsail_optimal_b.py \
  results/lugsail_decomp_dense_2026-06-08.csv \
  --outdir ../reports/figures/lugsail_optimal_b_2026-06-08 \
  --prefix lugsail_dense \
  --x-grid 0.15:0.70:0.01 \
  --plot-x 0.20,0.25,0.333333,0.40,0.45,0.50,0.60
```

## Faster pilot run

Use this first if the remote queue is uncertain:

```bash
cd code
uv run python run_lugsail_decomposition.py \
  --n-traj 300 \
  --traj-chunk 100 \
  --T-list 10000 30000 100000 \
  --lam-list 2 3 4 \
  --bn-points 60 \
  --bn-min-exp 0.15 \
  --bn-max-exp 0.90 \
  --out results/lugsail_decomp_pilot_2026-06-08.csv \
  > results/lugsail_decomp_pilot_2026-06-08.log 2>&1

uv run python analyze_lugsail_optimal_b.py \
  results/lugsail_decomp_pilot_2026-06-08.csv \
  --outdir ../reports/figures/lugsail_optimal_b_pilot_2026-06-08 \
  --prefix lugsail_pilot \
  --x-grid 0.15:0.70:0.01
```

## Reading the output

For each method, compare:

- `eta_hat_bstar`: direct log-log slope from \(b^*\) versus \(T\);
- `best_x_by_cv`: exponent \(x\) making \(b^*/T^x\) most stable across the
  swept \(T\)'s;
- `best_x_by_slope`: exponent \(x\) making the log-log slope of
  \(b^*/T^x\) closest to zero;
- `mse_power_kappa`: empirical decay exponent for the minimum MSE.

The theory-motivated baselines are:

- OBM with leading \(1/b\) bias: \(b^* \asymp T^{1/3}\),
  \(MSE^* \asymp T^{-2/3}\);
- OBM-LW after leading \(1/b\) cancellation and residual \(1/b^2\) bias:
  \(b^* \asymp T^{1/5}\), \(MSE^* \asymp T^{-4/5}\).

The earlier saved run `code/results/lugsail_decomp_lab.csv` had empirical
optima closer to \(T^{0.55}\) for OBM and \(T^{0.40}\) for OBM-LW on the swept
grid. The dense rerun is meant to check whether this is a finite-\(T\) regime,
a grid artifact, or a signal that the effective finite-sample expansion is
dominated by additional terms.

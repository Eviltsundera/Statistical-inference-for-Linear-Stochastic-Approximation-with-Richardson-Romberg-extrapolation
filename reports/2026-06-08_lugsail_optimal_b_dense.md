# Dense OBM / OBM-LW optimal block-size scaling

**Date:** 2026-06-08
**Machine:** `beleriand`
**Runner:** `code/run_lugsail_decomposition.py`
**Analyzer:** `code/analyze_lugsail_optimal_b.py`

## Purpose

Repeat the OBM and OBM-LW variance-estimator experiment with a denser
block-size grid and estimate the empirical asymptotic scaling of the MSE-optimal
block size \(b^*\). The diagnostic target is stability of \(b^*/T^x\) over
several values of \(T\).

## Command

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
  --out results/lugsail_decomp_dense_2026-06-08.csv
```

Then:

```bash
uv run python plot_lugsail_decomposition.py \
  --csv results/lugsail_decomp_dense_2026-06-08.csv \
  --outdir ../reports/figures/lugsail_decomp_dense_2026-06-08

uv run python analyze_lugsail_optimal_b.py \
  results/lugsail_decomp_dense_2026-06-08.csv \
  --outdir ../reports/figures/lugsail_optimal_b_2026-06-08 \
  --prefix lugsail_dense \
  --x-grid 0.15:0.70:0.01 \
  --plot-x 0.20,0.25,0.333333,0.40,0.45,0.50,0.60
```

## Outputs

- Raw CSV: `code/results/lugsail_decomp_dense_2026-06-08.csv`
- Driver log: `code/results/lugsail_decomp_dense_2026-06-08.driver.log`
- MSE fit figures: `reports/figures/lugsail_decomp_dense_2026-06-08/`
- Scaling tables and figures: `reports/figures/lugsail_optimal_b_2026-06-08/`

The run produced 1484 rows.

## Best swept block sizes

| T | OBM \(b^*\) | OBM-LW \(\lambda=2\) \(b^*\) | OBM-LW \(\lambda=3\) \(b^*\) | OBM-LW \(\lambda=4\) \(b^*\) |
|---:|---:|---:|---:|---:|
| 10,000 | 170 | 35 | 29 | 24 |
| 30,000 | 286 | 49 | 40 | 36 |
| 100,000 | 554 | 96 | 77 | 62 |
| 300,000 | 1,141 | 189 | 149 | 132 |
| 1,000,000 | 2,905 | 406 | 356 | 312 |

At \(T=10^6\), the corresponding MSE values were:

- OBM: \(1.678 \cdot 10^{-1}\)
- OBM-LW \(\lambda=2\): \(7.100 \cdot 10^{-2}\)
- OBM-LW \(\lambda=3\): \(7.335 \cdot 10^{-2}\)
- OBM-LW \(\lambda=4\): \(7.742 \cdot 10^{-2}\)

## Scaling summary

The analyzer fits \(b^* \approx C T^\eta\) and scans normalizations
\(b^*/T^x\).

| Method | \(\hat\eta\) for \(b^*\) | \(C\) | log-scale \(R^2\) | best \(x\) by CV | MSE power |
|---|---:|---:|---:|---:|---:|
| OBM | 0.614 | 0.534 | 0.990 | 0.61 | -0.390 |
| OBM-LW \(\lambda=2\) | 0.544 | 0.203 | 0.987 | 0.54 | -0.460 |
| OBM-LW \(\lambda=3\) | 0.551 | 0.154 | 0.980 | 0.55 | -0.462 |
| OBM-LW \(\lambda=4\) | 0.559 | 0.120 | 0.980 | 0.56 | -0.460 |

## Interpretation

On this finite-\(T\) range, the empirical optimum is not in the simple
asymptotic \(T^{1/3}\) OBM / \(T^{1/5}\) OBM-LW regime. The dense rerun gives
stable effective exponents around \(0.61\) for OBM and \(0.54\)--\(0.56\) for
OBM-LW. This agrees qualitatively with the earlier observation that the
finite-sample optimum lies near larger block exponents than the leading-term
asymptotic prediction.

OBM-LW with \(\lambda=2\) remains the best MSE choice across the swept \(T\)'s
among the lugsail variants, with higher \(\lambda\) giving slightly larger MSE
at \(T=10^6\).

# OBM / OBM-LW MSE asymptotics by fixed block exponent

**Date:** 2026-06-09--2026-06-10
**Machine:** `beleriand`
**Runner:** `code/run_lugsail_decomposition.py`
**Analyzer:** `code/analyze_lugsail_mse_asymptotics.py`

## Purpose

Estimate the decay rate of

$$
\mathbb E[(\hat\sigma^2(T,b)-\sigma_\infty^2)^2]
$$

for OBM and OBM-LW when the block size is fixed as \(b \approx T^\eta\).
This is complementary to the previous \(b^*\)-scaling experiment: here the
diagnostic object is the whole rate curve \(r(\eta)\), where

$$
\mathrm{MSE}(T,T^\eta) \approx C(\eta) T^{-r(\eta)}.
$$

## Hypotheses

For OBM / Bartlett:

$$
\mathrm{MSE}_{OBM}(T,b) \approx C_1 b^{-2} + C_2 b/T,
$$

so for \(b=T^\eta\),

$$
r_{OBM}(\eta) = \min(2\eta,1-\eta).
$$

For OBM-LW, after leading \(1/b\) bias cancellation:

$$
\mathrm{MSE}_{LW}(T,b) \approx C_1 b^{-4} + C_2 b/T,
$$

so

$$
r_{LW}(\eta) = \min(4\eta,1-\eta).
$$

## Run

```bash
cd code
uv run python run_lugsail_decomposition.py \
  --prob-seed 0 \
  --dir-seed 1 \
  --traj-seed 2 \
  --n-traj 800 \
  --traj-chunk 200 \
  --T-list 10000 20000 30000 50000 100000 200000 300000 500000 1000000 \
  --lam-list 2 3 4 \
  --bn-points 90 \
  --bn-min-exp 0.15 \
  --bn-max-exp 0.90 \
  --out results/lugsail_mse_asymptotics_2026-06-09.csv

uv run python analyze_lugsail_mse_asymptotics.py \
  results/lugsail_mse_asymptotics_2026-06-09.csv \
  --outdir ../reports/figures/lugsail_mse_asymptotics_2026-06-09 \
  --prefix lugsail_mse_asymptotics \
  --eta-grid 0.15:0.75:0.025 \
  --plot-eta 0.20,0.25,0.333333,0.40,0.45,0.50,0.60
```

Runtime ended with `DONE 2026-06-09T23:56:48+00:00`.

## Outputs

- Raw CSV: `code/results/lugsail_mse_asymptotics_2026-06-09.csv`
- Driver log: `code/results/lugsail_mse_asymptotics_2026-06-09.driver.log`
- Analyzer tables and figures:
  `reports/figures/lugsail_mse_asymptotics_2026-06-09/`

The run produced 2992 raw rows.

## Best rate over fixed eta grid

| Method | eta | empirical rate | theory rate | difference | log-MSE R2 |
|---|---:|---:|---:|---:|---:|
| OBM | 0.600 | 0.3949 | 0.4000 | -0.0051 | 0.9900 |
| OBM-LW \(\lambda=2\) | 0.450 | 0.5489 | 0.5500 | -0.0011 | 0.9976 |
| OBM-LW \(\lambda=3\) | 0.425 | 0.5666 | 0.5750 | -0.0084 | 0.9981 |
| OBM-LW \(\lambda=4\) | 0.425 | 0.5684 | 0.5750 | -0.0066 | 0.9980 |

## Best swept block sizes

At \(T=10^6\):

| Method | \(b^*\) | effective exponent | MSE |
|---|---:|---:|---:|
| OBM | 2679 | 0.571 | \(1.701\cdot 10^{-1}\) |
| OBM-LW \(\lambda=2\) | 416 | 0.437 | \(6.751\cdot 10^{-2}\) |
| OBM-LW \(\lambda=3\) | 329 | 0.420 | \(7.131\cdot 10^{-2}\) |
| OBM-LW \(\lambda=4\) | 293 | 0.411 | \(7.479\cdot 10^{-2}\) |

## Interpretation

The fixed-\(\eta\) rate experiment supports the **right branch** of the simple
asymptotic templates very cleanly on this finite-\(T\) range. The best observed
fixed-\(\eta\) rates are almost exactly equal to the corresponding
\(\min(\cdot)\) predictions:

- OBM is best near \(\eta=0.60\), with rate \(0.395 \approx 1-\eta=0.40\).
- OBM-LW is best near \(\eta=0.425\)--\(0.45\), with rate
  \(0.55\)--\(0.57 \approx 1-\eta\).

This explains why the empirical \(b^*\) exponents from the previous run were
much larger than the formal \(1/3\) and \(1/5\): on the accessible \(T\)-range,
small-\(\eta\) choices are still dominated by finite-sample bias/transient
effects and do not yet show the nominal left-branch rates.

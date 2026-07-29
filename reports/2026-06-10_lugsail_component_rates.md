# Component-rate check for OBM / OBM-LW MSE

**Date:** 2026-06-10
**Input run:** `code/results/lugsail_mse_asymptotics_2026-06-09.csv`
**Analyzer:** `code/analyze_lugsail_component_rates.py`

## Purpose

Test the refined hypothesis from
`conversations/2026-06-10_obm-mse-hypothesis-refinement.md`: the MSE right
branch is already governed by the variance term \(b/T\), while the bias branch
for small block exponents is still pre-asymptotic.

The previous fixed-\(\eta\) analysis fitted only total MSE. This analysis fits
the components separately:

$$
|\mathrm{Bias}(T,T^\eta)|,\qquad
\mathrm{Bias}(T,T^\eta)^2,\qquad
\mathrm{Var}(\hat\sigma^2(T,T^\eta)),\qquad
\mathrm{MSE}(T,T^\eta).
$$

## Theory checked

For OBM:

$$
|\mathrm{Bias}| \asymp b^{-1},\qquad
\mathrm{Bias}^2 \asymp b^{-2},\qquad
\mathrm{Var}\asymp b/T.
$$

For \(b=T^\eta\), this predicts:

$$
q_{bias}(\eta)=\eta,\qquad q_{bias^2}(\eta)=2\eta,\qquad
q_{var}(\eta)=1-\eta.
$$

For OBM-LW:

$$
|\mathrm{Bias}| \asymp b^{-2},\qquad
\mathrm{Bias}^2 \asymp b^{-4},\qquad
\mathrm{Var}\asymp b/T.
$$

so:

$$
q_{bias}(\eta)=2\eta,\qquad q_{bias^2}(\eta)=4\eta,\qquad
q_{var}(\eta)=1-\eta.
$$

## Command

```bash
cd code
MPLCONFIGDIR=/tmp/matplotlib-codex .venv/bin/python \
  analyze_lugsail_component_rates.py \
  results/lugsail_mse_asymptotics_2026-06-09.csv \
  --outdir ../reports/figures/lugsail_component_rates_2026-06-10 \
  --prefix lugsail_components \
  --eta-grid 0.15:0.75:0.025 \
  --plot-eta 0.20,0.25,0.333333,0.40,0.45,0.50,0.60
```

## Outputs

- Component-rate table:
  `reports/figures/lugsail_component_rates_2026-06-10/lugsail_components_component_rates.csv`
- Selected \(T,\eta\) component values:
  `reports/figures/lugsail_component_rates_2026-06-10/lugsail_components_eta_values.csv`
- Figures:
  `reports/figures/lugsail_component_rates_2026-06-10/`

## Main result at empirical best MSE eta

| Method | eta | \(|bias|\) rate | theory \(|bias|\) rate | var rate | theory var rate | MSE rate |
|---|---:|---:|---:|---:|---:|---:|
| OBM | 0.600 | 0.210 | 0.600 | 0.392 | 0.400 | 0.395 |
| OBM-LW \(\lambda=2\) | 0.450 | -0.046 | 0.900 | 0.551 | 0.550 | 0.549 |
| OBM-LW \(\lambda=3\) | 0.425 | -0.437 | 0.850 | 0.576 | 0.575 | 0.567 |
| OBM-LW \(\lambda=4\) | 0.425 | -0.203 | 0.850 | 0.571 | 0.575 | 0.568 |

The variance branch matches theory almost exactly. The bias branch does not:
at the finite-\(T\) optimum, \(|bias|\) decays much more slowly than the
classical truncation-bias prediction, and for several OBM-LW cases it even
increases over the fitted \(T\)-range.

## Selected eta diagnostics

For OBM:

| eta | \(|bias|\) rate | theory | var rate | theory | MSE rate |
|---:|---:|---:|---:|---:|---:|
| 0.200 | -0.022 | 0.200 | 0.989 | 0.800 | -0.045 |
| 0.333 | -0.030 | 0.325 | 0.726 | 0.675 | -0.060 |
| 0.400 | -0.013 | 0.400 | 0.610 | 0.600 | -0.022 |
| 0.500 | 0.067 | 0.500 | 0.499 | 0.500 | 0.206 |
| 0.600 | 0.210 | 0.600 | 0.392 | 0.400 | 0.395 |

For OBM-LW \(\lambda=2\):

| eta | \(|bias|\) rate | theory | var rate | theory | MSE rate |
|---:|---:|---:|---:|---:|---:|
| 0.200 | -0.071 | 0.400 | 0.984 | 0.800 | -0.141 |
| 0.333 | -0.136 | 0.650 | 0.725 | 0.675 | -0.231 |
| 0.400 | -0.199 | 0.800 | 0.618 | 0.600 | 0.325 |
| 0.450 | -0.046 | 0.900 | 0.551 | 0.550 | 0.549 |
| 0.600 | 0.647 | 1.200 | 0.398 | 0.400 | 0.401 |

## Interpretation

This component check strongly supports the refined hypothesis.

1. The **variance component** is already asymptotic:

   $$
   \mathrm{Var}(\hat\sigma^2(T,T^\eta)) \approx C T^{-(1-\eta)}.
   $$

   This holds very cleanly near the empirically good eta values.

2. The **bias component** is not yet in the classical truncation regime:

   - OBM does not show \(|bias|\sim T^{-\eta}\) for small or moderate eta.
   - OBM-LW does not show \(|bias|\sim T^{-2\eta}\) near the predicted
     lugsail optimum \(\eta=0.2\).

3. Therefore the empirical best eta is pushed to the region where the variance
   branch dominates but the block size is large enough to escape the worst
   small-window bias saturation.

This explains why the previous best-eta experiment found \(\eta\approx 0.60\)
for OBM and \(\eta\approx 0.425\)--\(0.45\) for OBM-LW, rather than the formal
asymptotic optima \(1/3\) and \(1/5\).

## Next check

The most direct follow-up would be a high-\(T\), low-\(n_{traj}\) run focused on
small eta values \(0.15\)--\(0.35\). Its goal would not be precise MSE
minimization, but to see whether the bias rates eventually bend toward
\(\eta\) for OBM and \(2\eta\) for OBM-LW.

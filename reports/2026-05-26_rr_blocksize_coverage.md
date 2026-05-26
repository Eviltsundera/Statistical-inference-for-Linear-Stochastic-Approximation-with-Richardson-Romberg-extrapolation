# RR coverage sweep over OBM block size

**Date:** 2026-05-26
**Script:** `code/run_rr_blocksize_coverage.py`
**Machine:** `beleriand`
**Raw outputs:**

- `code/results/blocksize_coverage/rr_blocksize_T20k_100k_1M_pair0p20_0p10_w24.csv`
- `code/results/blocksize_coverage/rr_blocksize_T20k_100k_1M_pair0p20_0p10_w24_summary.csv`
- `code/results/blocksize_coverage/rr_blocksize_T20k_100k_1M_pair0p20_0p10_w24.log`

## Motivation

The preceding trajectory-length sweep used the production block-size rule
`b = floor(T^0.6)`.  At that rule, lugsail/OBM-RR did not improve coverage
over OBM.  This run asks whether the conclusion is intrinsic, or whether
lugsail helps after tuning the OBM block size.

The run evaluates

$$
b = \lfloor T^\eta \rfloor,\qquad
\eta \in \{0.3,0.4,0.5,0.6,0.7,0.8\},
$$

for OBM and OBM-RR.  The RR point estimator is fixed, so differences across
rows come only from long-run variance estimation.

## Setup

- 100 problems x 100 trajectories for each `T`.
- Horizons: `T in {20_000, 100_000, 1_000_000}`.
- Problem generation: `d = 5`, `n_states = 10`, `eig_min = 0.25`,
  `eig_max = 0.60`, `noise_target = 0.35`, seed `42`.
- RR pair: `(0.20, 0.10)`.
- Same problem seeds and projection directions are reused across horizons.
- OBM-RR/lugsail parameter: `lambda = 2`.
- The analytic finite-state long-run variance is used as the oracle row and
  as ground truth for variance-estimator bias and MSE.
- Runtime: 588 s, about 9.8 min with 24 workers.

## Results

`L2` and `Width` are reported in units of `1e-3`.  `Bias` is the median
relative bias of the raw variance estimator against the analytic
`sigma^2(u)`.  `Neg.` is the mean rate of negative raw lugsail estimates
before clamping.

| T | eta | b | Estimator | L2 | Width | Cov median | Width / oracle | Bias | Neg. |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 20 000 | - | 0 | Oracle | 21.11 | 39.37 | 95.5% | 1.000 | 0.000 | 0.00% |
| 20 000 | 0.3 | 19 | OBM | 21.11 | 19.17 | 66.0% | 0.487 | -0.768 | 0.00% |
| 20 000 | 0.3 | 19 | OBM-RR | 21.11 | 30.08 | 86.0% | 0.764 | -0.417 | 0.00% |
| 20 000 | 0.4 | 52 | OBM | 21.11 | 27.90 | 83.0% | 0.709 | -0.497 | 0.00% |
| 20 000 | 0.4 | 52 | OBM-RR | 21.11 | 37.46 | 94.0% | 0.951 | -0.089 | 0.00% |
| 20 000 | 0.5 | 141 | OBM | 21.11 | 34.58 | 91.5% | 0.878 | -0.222 | 0.00% |
| 20 000 | 0.5 | 141 | OBM-RR | 21.11 | 38.54 | 95.0% | 0.979 | -0.022 | 0.00% |
| 20 000 | 0.6 | 380 | OBM | 21.11 | 37.07 | 94.0% | 0.942 | -0.098 | 0.00% |
| 20 000 | 0.6 | 380 | OBM-RR | 21.11 | 37.16 | 93.0% | 0.944 | -0.056 | 0.00% |
| 20 000 | 0.8 | 2759 | OBM-RR | 21.11 | 24.54 | 65.5% | 0.623 | -0.434 | 14.77% |
| 100 000 | - | 0 | Oracle | 9.39 | 17.25 | 95.0% | 1.000 | 0.000 | 0.00% |
| 100 000 | 0.4 | 100 | OBM | 9.39 | 14.44 | 89.0% | 0.837 | -0.298 | 0.00% |
| 100 000 | 0.4 | 100 | OBM-RR | 9.39 | 17.12 | 95.0% | 0.993 | -0.010 | 0.00% |
| 100 000 | 0.5 | 316 | OBM | 9.39 | 16.35 | 93.5% | 0.948 | -0.097 | 0.00% |
| 100 000 | 0.5 | 316 | OBM-RR | 9.39 | 17.09 | 95.0% | 0.991 | -0.005 | 0.00% |
| 100 000 | 0.6 | 999 | OBM | 9.39 | 16.87 | 94.0% | 0.978 | -0.039 | 0.00% |
| 100 000 | 0.6 | 999 | OBM-RR | 9.39 | 16.76 | 94.0% | 0.972 | -0.029 | 0.00% |
| 100 000 | 0.8 | 10000 | OBM-RR | 9.39 | 12.41 | 75.0% | 0.720 | -0.305 | 6.71% |
| 1 000 000 | - | 0 | Oracle | 2.97 | 5.43 | 95.0% | 1.000 | 0.000 | 0.00% |
| 1 000 000 | 0.3 | 63 | OBM | 2.97 | 4.08 | 86.0% | 0.751 | -0.434 | 0.00% |
| 1 000 000 | 0.3 | 63 | OBM-RR | 2.97 | 5.31 | 94.0% | 0.979 | -0.047 | 0.00% |
| 1 000 000 | 0.4 | 251 | OBM | 2.97 | 5.10 | 93.0% | 0.940 | -0.121 | 0.00% |
| 1 000 000 | 0.4 | 251 | OBM-RR | 2.97 | 5.44 | 95.0% | 1.002 | 0.002 | 0.00% |
| 1 000 000 | 0.5 | 1000 | OBM | 2.97 | 5.36 | 95.0% | 0.987 | -0.028 | 0.00% |
| 1 000 000 | 0.5 | 1000 | OBM-RR | 2.97 | 5.43 | 95.0% | 1.001 | -0.001 | 0.00% |
| 1 000 000 | 0.6 | 3981 | OBM | 2.97 | 5.42 | 95.0% | 0.998 | -0.010 | 0.00% |
| 1 000 000 | 0.6 | 3981 | OBM-RR | 2.97 | 5.39 | 95.0% | 0.993 | -0.011 | 0.00% |
| 1 000 000 | 0.8 | 63095 | OBM-RR | 2.97 | 4.57 | 84.0% | 0.842 | -0.188 | 1.74% |

The full grid, including `eta = 0.7`, is in the summary CSV.

## Interpretation

The block-size sweep resolves the apparent contradiction between the
trajectory-length coverage sweep and the lugsail bias-variance diagnostic.
Lugsail helps when OBM uses a block size for which the Bartlett-window bias is
still large.  At `T = 20_000`, moving from OBM to OBM-RR at `eta = 0.5`
changes median relative bias from `-0.222` to `-0.022`, width/oracle from
`0.878` to `0.979`, and coverage from `91.5%` to `95.0%`.  At `T = 100_000`,
OBM-RR at `eta = 0.4` or `0.5` gives median coverage `95.0%`, while OBM at
the same block sizes is still too narrow.

The default rule `b = floor(T^0.6)` is already close to the oracle width for
this problem class.  At that rule, lugsail is neutral rather than beneficial:
its width and coverage are close to OBM.  This explains why the previous
`T`-sweep did not show a lugsail coverage gain.

Very large blocks are harmful.  At `eta = 0.8`, OBM-RR becomes too narrow and
can produce negative raw variance estimates: the negative rate is `14.77%` at
`T = 20_000`, `6.71%` at `T = 100_000`, and `1.74%` at `T = 1_000_000`.
The clamped CI widths therefore hide an important finite-sample instability,
and the raw negative/clamped rate should remain a reported diagnostic.

## Takeaways for the thesis

1. Lugsail is not an automatic coverage fix; it is useful when the selected
   block size leaves substantial negative OBM window bias.
2. The best lugsail rows occur at smaller block-size exponents than the
   default `0.6`: around `eta = 0.5` for `T = 20_000` and `eta = 0.4--0.5`
   for `T >= 100_000` in this run.
3. The default `eta = 0.6` remains a conservative production rule for OBM,
   but it can hide the benefit of lugsail because OBM is already close to
   oracle.
4. Large lugsail blocks should be treated cautiously because the signed
   OBM-RR estimator can be negative in finite samples.

# RR conditioning and matrix-noise stress test

**Date:** 2026-05-26
**Script:** `code/run_rr_conditioning_noise_stress.py`
**Machine:** `beleriand`
**Raw outputs:**

- `code/results/stress/rr_conditioning_noise_T100k_1M_pair0p20_0p10_w24.csv`
- `code/results/stress/rr_conditioning_noise_T100k_1M_pair0p20_0p10_w24_summary.csv`
- `code/results/stress/rr_conditioning_noise_T100k_1M_pair0p20_0p10_w24.log`

## Motivation

The block-size sweep showed that lugsail/OBM-RR can recover oracle-level
coverage when OBM still has negative Bartlett-window bias.  This stress test
checks whether that message survives moderate changes in the LSA problem
generator:

- weaker mean contraction of `Abar`;
- larger state-dependent matrix perturbations;
- a combined weaker-contraction / larger-noise setting.

The Markov-chain generator is unchanged in this run, so this is not yet a
mixing-time stress test.

## Setup

- 100 problems x 100 trajectories for each scenario and horizon.
- Horizons: `T in {100_000, 1_000_000}`.
- RR pair: `(0.20, 0.10)`.
- Block-size exponents: `eta in {0.4, 0.5, 0.6}`.
- OBM-RR/lugsail parameter: `lambda = 2`.
- Direction: one fixed random scalar direction per problem.
- Oracle row: analytic finite-state long-run variance `sigma^2(u)`.
- Runtime: 2124 s, about 35.4 min with 24 workers.

The scenarios are:

| Scenario | eig_min | eig_max | noise_target | median max_rho | unstable warning |
|---|---:|---:|---:|---:|---:|
| baseline | 0.25 | 0.60 | 0.35 | 0.972 | 0% |
| weak_mean | 0.12 | 0.30 | 0.18 | 0.987 | 0% |
| high_noise | 0.25 | 0.60 | 0.45 | 0.982 | 7% |
| weak_high_noise | 0.15 | 0.35 | 0.25 | 0.988 | 3% |

The warning rate is the fraction of generated problems for which the simple
one-step diagnostic `max_x rho(I + alpha A(x)) >= 1` holds at `alpha = 0.20`.
It is a finite-sample stability diagnostic, not a divergence count; no
trajectory divergences occurred in the reported rows.

## Main results at eta = 0.5

`L2` and `Width` are reported in units of `1e-3`.  `Bias` is the median
relative bias of the raw OBM-RR variance estimate against the analytic
`sigma^2(u)`.

| Scenario | T | Oracle cov. | OBM cov. | OBM-RR cov. | OBM-RR width/oracle | OBM-RR bias |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 100 000 | 95.0% | 93.5% | 95.0% | 0.991 | -0.005 |
| baseline | 1 000 000 | 95.0% | 95.0% | 95.0% | 1.001 | -0.001 |
| weak_mean | 100 000 | 95.0% | 92.0% | 95.0% | 0.993 | -0.008 |
| weak_mean | 1 000 000 | 95.0% | 94.0% | 95.0% | 1.000 | -0.002 |
| high_noise | 100 000 | 95.0% | 93.5% | 95.0% | 0.991 | -0.003 |
| high_noise | 1 000 000 | 95.0% | 95.0% | 95.0% | 0.996 | 0.001 |
| weak_high_noise | 100 000 | 95.0% | 92.0% | 95.0% | 0.990 | -0.006 |
| weak_high_noise | 1 000 000 | 95.0% | 94.0% | 95.0% | 0.998 | -0.001 |

At `eta = 0.5`, OBM-RR stays close to the oracle interval in all four
scenarios.  The weaker-contraction settings increase the absolute error and
oracle width, as expected, but the oracle coverage remains at 95%.  This
suggests that the RR center and the normal approximation are still adequate
for these stress levels.

## Useful eta = 0.4 diagnostic

The smaller block size `eta = 0.4` makes the OBM window bias more visible and
therefore shows the lugsail correction more clearly.

| Scenario | T | OBM cov. | OBM-RR cov. | OBM bias | OBM-RR bias | OBM-RR width/oracle |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 100 000 | 89.0% | 95.0% | -0.298 | -0.010 | 0.993 |
| baseline | 1 000 000 | 93.0% | 95.0% | -0.121 | 0.002 | 1.002 |
| weak_mean | 100 000 | 82.5% | 93.0% | -0.518 | -0.099 | 0.954 |
| weak_mean | 1 000 000 | 91.0% | 95.0% | -0.252 | -0.004 | 1.001 |
| weak_high_noise | 100 000 | 85.0% | 94.0% | -0.457 | -0.060 | 0.962 |
| weak_high_noise | 1 000 000 | 92.0% | 95.0% | -0.208 | 0.000 | 1.001 |

Under weaker mean contraction, the same block-size exponent leaves larger OBM
negative bias.  Lugsail reduces that bias substantially, but at the shorter
horizon `T = 100_000` and `eta = 0.4` it may still be slightly too narrow.
Moving to `eta = 0.5` fixes this in the present experiment.

## Interpretation

The stress test supports three conclusions.

1. The RR center remains reliable in these moderate stress settings: oracle
   median coverage is 95% in every scenario and horizon.
2. Weaker mean contraction mainly increases the long-run variance scale and
   makes OBM window bias more visible.  This is exactly the regime where
   lugsail is useful.
3. The tuned lugsail rows (`eta = 0.5` here) recover near-oracle width and
   nominal coverage without negative raw estimates.  The warning rates in
   `high_noise` and `weak_high_noise` should still be reported, because they
   indicate that a small fraction of generated problems is close to the local
   one-step stability boundary.

## Takeaways for the thesis

The earlier block-size conclusion is robust to these moderate changes in
conditioning and matrix noise.  Lugsail should be described as a block-size
sensitive correction for long-run variance bias: it helps when OBM is too
narrow, but the useful block-size range must be checked empirically.

The main remaining stress test is slower Markov-chain mixing.  That requires
changing the transition-matrix generator, so it is a separate diagnostic from
the conditioning/noise sweep reported here.

# RR mixing-rate stress test

**Date:** 2026-05-26
**Script:** `code/run_rr_mixing_stress.py`
**Machine:** `beleriand`
**Raw outputs:**

- `code/results/stress/rr_mixing_lazy_T100k_1M_pair0p20_0p10_w24.csv`
- `code/results/stress/rr_mixing_lazy_T100k_1M_pair0p20_0p10_w24_summary.csv`
- `code/results/stress/rr_mixing_lazy_T100k_1M_pair0p20_0p10_w24.log`

## Motivation

The conditioning/noise stress test kept the Markov-chain generator fixed.
This run changes the Markov dependence directly.  Starting from the usual
dense random transition matrix `P0`, it uses

$$
P_\rho = \rho I + (1-\rho) P_0.
$$

This preserves the stationary distribution but shrinks the spectral gap.  The
goal is to check whether the previous OBM/lugsail conclusions remain valid
when the chain mixes slowly.

## Setup

- 100 problems x 100 trajectories for each scenario and horizon.
- Horizons: `T in {100_000, 1_000_000}`.
- RR pair: `(0.20, 0.10)`.
- Problem generator: baseline `eig_min = 0.25`, `eig_max = 0.60`,
  `noise_target = 0.35`.
- Lazy probabilities:
  `rho in {0.0, 0.5, 0.8, 0.95}`.
- Block-size exponents: `eta in {0.4, 0.5, 0.6}`.
- OBM-RR/lugsail parameter: `lambda = 2`.
- Direction: one fixed random scalar direction per problem.
- Oracle row: analytic finite-state long-run variance `sigma^2(u)`.
- Runtime: 1985 s, about 33.1 min with 24 workers.

The median spectral gaps were:

| Scenario | rho | Median spectral gap | Median relaxation time |
|---|---:|---:|---:|
| baseline | 0.00 | 0.813 | 1.23 |
| lazy_0p50 | 0.50 | 0.426 | 2.35 |
| lazy_0p80 | 0.80 | 0.171 | 5.86 |
| lazy_0p95 | 0.95 | 0.043 | 23.42 |

## Main results at eta = 0.5

`L2` and `Width` are reported in units of `1e-3`.  The OBM-RR row uses
`eta = 0.5`, where the variance estimator is already close to the oracle in
the non-slow-mixing experiments.

| Scenario | T | L2 | Oracle width | Oracle cov. | OBM-RR cov. | OBM-RR width/oracle | OBM-RR bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 100 000 | 9.39 | 17.25 | 95.0% | 95.0% | 0.991 | -0.005 |
| baseline | 1 000 000 | 2.97 | 5.43 | 95.0% | 95.0% | 1.001 | -0.001 |
| lazy_0p50 | 100 000 | 16.35 | 30.03 | 95.0% | 95.0% | 0.989 | 0.002 |
| lazy_0p50 | 1 000 000 | 5.41 | 9.45 | 94.0% | 94.0% | 0.996 | 0.007 |
| lazy_0p80 | 100 000 | 32.48 | 51.78 | 92.0% | 92.0% | 0.999 | 0.022 |
| lazy_0p80 | 1 000 000 | 18.19 | 16.30 | 76.5% | 76.5% | 1.005 | 0.029 |
| lazy_0p95 | 100 000 | 124.79 | 107.45 | 77.0% | 80.0% | 1.055 | 0.124 |
| lazy_0p95 | 1 000 000 | 111.76 | 33.82 | 6.5% | 7.0% | 1.058 | 0.137 |

## Interpretation

For moderate laziness, `rho = 0.5`, the earlier conclusions still hold.  The
long-run variance scale grows, but the oracle row and the tuned OBM-RR row
remain close to nominal.

At `rho = 0.8`, the failure mode changes.  The practical variance estimator
is no longer the bottleneck: at `T = 10^6`, OBM-RR with `eta = 0.5` has
width/oracle `1.005`, yet both oracle and OBM-RR coverage are only `76.5%`.
Thus the center or the finite-sample normal approximation is failing under
slow mixing.

At `rho = 0.95`, the failure is severe.  Increasing `T` from `10^5` to
`10^6` shrinks the oracle width from `107.45e-3` to `33.82e-3`, but the
median L2 error only decreases from `124.79e-3` to `111.76e-3`.  Coverage
therefore collapses from `77.0%` to `6.5%` even with oracle variance.  OBM-RR
slightly widens the interval, but it cannot correct the center.

No negative raw lugsail estimates appeared in this run.  The problem is not
finite-sample non-PSD behavior of the variance estimator; it is the slow
decay of the point-estimator or normal-approximation error relative to the
`T^{-1/2}` confidence-interval width.

## Takeaways for the thesis

The mixing-rate stress test is the first experiment that finds a clear
finite-sample limitation of the current RR confidence intervals.  Under
moderate mixing slowdown, RR + tuned OBM-RR remains useful.  Under strong
slow mixing, oracle intervals already undercover, so OBM/lugsail cannot fix
the problem.

This should be presented as evidence that the theorem's dependence on
`t_mix` is not cosmetic.  Future experiments should vary horizon, burn-in,
and possibly stepsize as functions of the mixing time.

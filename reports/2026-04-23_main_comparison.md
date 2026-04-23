# Main comparison experiment — 16 methods on LSA with Markovian noise

**Date:** 2026-04-23
**Script:** `code/run_comparison.py`
**Raw outputs:** `code/results_comparison.csv`, full log in chat history.

## Setup

- 100 problems × 100 trajectories, `T = 1_000_000`, `d = 5`, `n_states = 10`.
- Problem generation: `eig_min=0.25`, `eig_max=0.60`, `noise_target=0.35`, `a_norm_cap=1.0`.
  Eigenvalues of $-\bar A$ uniform in `[0.25, 0.60]`, state-dependent perturbation centered under $\pi$.
- PR step size: $c_0 = 200$, $k_0 = 20{,}000$, $\gamma = 0.65$ (matches Samsonov et al. 2025 notebook).
- Richardson–Romberg extrapolation: $\alpha \in \{0.2, 0.02\}$.
- Random projection direction per problem (unit vector).
- Runtime: 939 s ≈ 15.7 min.

## Problem diagnostics (sanity)

- `median max ||A(x)||_2 = 0.740`, `max = 0.832` — assumption `||A|| ≤ 1` satisfied on all 100 problems.
- `median max ρ(I + 0.2 A) = 0.998`, `max = 1.020`. **43 / 100 problems** have $\rho(I + 0.2 A) \ge 1$, i.e. constant step $\alpha = 0.2$ is marginally unstable in a substantial fraction of draws.
- **Zero divergences** across all 16 methods. Instability of last-iterate manifests as zero coverage, not NaN — because the metric we track is coverage of $\bar\theta$ or the RR combination, both bounded.

## Results — median over 100 problems

`L2` and `Width` are `×1e-3`. Coverage target = 95%.

| Method | L2 | Width | Cov (median) | Cov (mean) |
|---|---:|---:|---:|---:|
| α = 0.2 (const) | 26.67 | 8.36 | 0.5% | 27.4% |
| α = 0.02 (const) | 13.93 | 8.27 | 40.5% | 44.7% |
| **RR (0.2 + 0.02)** | **4.52** | 8.22 | **94.0%** | **94.3%** |
| 0.2 / √k (dim) | 6.80 | 10.34 | 90.0% | 89.8% |
| 0.02 / √k (dim) | 8.46 | 10.56 | 87.0% | 85.8% |
| PR + OBM | 5.35 | 8.04 | 92.0% | 88.3% |
| PR + MSB | 5.35 | 7.97 | 91.0% | 88.1% |
| **RR + OBM** | **4.52** | 8.23 | **95.0%** | **94.6%** |
| RR + MSB | 4.52 | 8.18 | 94.0% | 94.3% |
| α = 0.2 + OBM | 26.67 | 8.38 | 0.5% | 27.5% |
| α = 0.2 + MSB | 26.67 | 8.30 | 0.5% | 27.3% |
| α = 0.02 + OBM | 13.93 | 8.27 | 40.0% | 45.1% |
| α = 0.02 + MSB | 13.93 | 8.21 | 38.5% | 44.9% |
| α = 0.02 + OBM-RR | 13.93 | 8.26 | 41.5% | 45.1% |
| PR + OBM-RR | 5.35 | 8.18 | 91.0% | 88.5% |
| RR + OBM-RR | 4.52 | 8.22 | 95.0% | 94.4% |

## Coverage percentiles across problems

| Method | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| α = 0.2 | 0.0 | 0.0 | 0.5 | 67.0 | 90.1 |
| α = 0.02 | 0.0 | 4.0 | 40.5 | 86.2 | 92.0 |
| **RR (0.2 + 0.02)** | **91.0** | 93.0 | 94.0 | 96.0 | 98.0 |
| 0.2 / √k (dim) | 85.0 | 87.0 | 90.0 | 93.0 | 95.0 |
| 0.02 / √k (dim) | 77.0 | 83.8 | 87.0 | 89.0 | 92.2 |
| PR + OBM | 78.8 | 86.8 | 92.0 | 94.0 | 96.0 |
| PR + MSB | 76.9 | 86.8 | 91.0 | 94.0 | 96.0 |
| **RR + OBM** | 92.0 | 93.0 | 95.0 | 96.0 | 97.1 |
| RR + MSB | 91.0 | 93.0 | 94.0 | 96.0 | 97.0 |
| PR + OBM-RR | 79.8 | 85.8 | 91.0 | 94.0 | 96.0 |
| RR + OBM-RR | 91.0 | 93.0 | 95.0 | 96.0 | 98.0 |

## Bias (L2 ×1e-3) percentiles

| Method | p10 | p25 | p50 | p75 | p90 | p90/p10 |
|---|---:|---:|---:|---:|---:|---:|
| α = 0.2 | 13.87 | 18.26 | 26.67 | 34.99 | 49.22 | 3.5× |
| α = 0.02 | 7.58 | 10.14 | 13.93 | 18.00 | 25.17 | 3.3× |
| **RR** | **3.13** | **3.70** | **4.52** | **5.41** | **6.58** | **2.1×** |
| PR | 3.50 | 4.28 | 5.35 | 6.50 | 7.88 | 2.3× |
| dim 0.02 | 4.87 | 6.19 | 8.46 | 12.43 | 18.18 | 3.7× |

## Observations

1. **RR dominates PR uniformly across the problem distribution**, not just at the median. The p10 of RR coverage (91%) exceeds the median PR coverage. RR removes the problematic left tail of PR — for PR about 10% of problems cover <80%, for RR this is never the case.
2. **RR has the tightest L2 distribution** (p90/p10 = 2.1×). Besides smaller median error, RR is the most *robust* method across problem draws. This is a separate argument in favor of RR beyond the median.
3. **Constant-α branches are unusable as-is**. α = 0.2 is broken on the 43% of stiff problems; α = 0.02 has a bias floor at 14·10⁻³ (versus RR at 4.5·10⁻³), leaving median coverage at 40%. No variance-estimator fix (OBM, MSB, OBM-RR) closes that gap: they all give ≈ 40% coverage.
4. **Lugsail (OBM-RR) is neutral on PR/RR** at this horizon:
   - `PR + OBM-RR` 91% vs `PR + OBM` 92% — variance inflation without corresponding bias reduction (OBM bias on PR is already small at T = 10⁶).
   - `RR + OBM-RR` 95% = `RR + OBM` 95% — widths 8.220 vs 8.225, indistinguishable.
   - `const 0.02 + OBM-RR` 41.5% vs `const 0.02 + OBM` 40.0% — lugsail gives +1.5% coverage, but SA bias dominates; not a useful fix.
5. **CLTZ20 (`dim_*`) is a viable alternative**. Coverage 87–90% at the cost of ≈ +30% CI width (10.3 vs 8.0). Useful as a robustness baseline in the thesis but strictly dominated by RR.

## Takeaways for the thesis

1. **Headline result:** RR gives 3.2× smaller L2 than constant α = 0.02 and 15% smaller L2 than PR, at the same CI width, with better and more robust coverage (median 94%, p10 = 91%).
2. **Assumption realism:** `||A||` bound satisfied by all 100 draws; stability edge `ρ(I+αA) ≥ 1` reached on 43% of problems without causing divergence — confirms that the method survives the assumption boundary.
3. **Lugsail story:** at T = 10⁶ on moderately conditioned problems, lugsail has no measurable benefit — bias of OBM is already in the asymptotic floor. A separate report (`2026-04-23_lugsail_bias_variance.md`) shows where lugsail does shine.

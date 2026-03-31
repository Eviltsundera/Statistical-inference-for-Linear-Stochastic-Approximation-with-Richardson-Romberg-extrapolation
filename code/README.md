# LSA Inference Comparison

Comparison of inference strategies for Linear Stochastic Approximation (LSA) with Markovian noise.

## Methods

| # | Method | Source |
|---|--------|--------|
| 1 | Constant step `alpha=0.2` + batch-mean CI | Huo et al. 2023 |
| 2 | Constant step `alpha=0.02` + batch-mean CI | Huo et al. 2023 |
| 3 | Richardson-Romberg (0.2+0.02) + batch-mean CI | Huo et al. 2023 |
| 4 | Diminishing `0.2/sqrt(k)` + CLTZ20 batch-mean CI | Huo et al. 2023 |
| 5 | Diminishing `0.02/sqrt(k)` + CLTZ20 batch-mean CI | Huo et al. 2023 |
| 6 | Polyak-Ruppert + OBM variance CI | Samsonov et al. 2025 |
| 7 | Polyak-Ruppert + multiplier subsample bootstrap CI | Samsonov et al. 2025 |

## Quick Start

```bash
uv sync
uv run python run_comparison.py
```

Default run: 10 problems, 50 trajectories, T=10000 (quick smoke test).

### Full-scale experiment

```bash
uv run python run_comparison.py --n-problems 100 --n-traj 100 --T 100000
```

### All options

```
--n-problems   Number of random LSA problems (default: 10)
--n-traj       Trajectories per problem (default: 50)
--T            Trajectory length (default: 10000)
--n-states     Markov chain states (default: 10)
--d            Dimension of theta (default: 5)
--seed         Random seed (default: 42)
```

## Metrics

- **L2 error** (bias): `||theta_bar - theta*||_2`
- **CI width**: width of 95% CI for 1st coordinate
- **Coverage**: fraction of runs where CI covers `theta*_1`

Output: mean table + percentile tables (10/25/50/75/90), saved to `results_comparison.csv`.

## Project Structure

```
run_comparison.py           — unified experiment runner
lsa_inference/
  markov_chain.py           — transition matrix P, stationary dist pi, chain simulation
  lsa_problem.py            — A(x), b(x) generation, theta* computation
  lsa_engine.py             — LSA iterations: constant, diminishing, Polyak-Ruppert, RR
  inference.py              — CI construction: batch-mean, OBM, MSB bootstrap
docs/
  huo2023_experiment_spec.md      — reproduction spec for Huo et al. 2023
  samsonov2025_experiment_spec.md — reproduction spec for Samsonov et al. 2025
```

## Problem Setup

Random LSA on finite-state Markov chain (Appendix C.1 of Huo et al.):

- Transition matrix `P` (n_states x n_states): random row-stochastic, irreducible, aperiodic
- Hurwitz mean matrix `Abar` (d x d): all eigenvalues have negative real parts
- State-dependent `A(x) = Abar + E(x)` with `E_pi[E(x)] = 0`
- State-dependent `b(x)` ~ U[-1,1]
- Target: `theta* = -Abar^{-1} bbar`

LSA iteration: `theta_{t+1} = theta_t + alpha_t (A(x_t) theta_t + b(x_t))`

## References

- Huo, Chen, Xie. *Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference.* arXiv:2312.10894, 2023.
- Samsonov, Sheshukova, Moulines, Naumov. *Statistical Inference for Linear Stochastic Approximation with Markovian Noise.* arXiv:2505.19102, 2025.

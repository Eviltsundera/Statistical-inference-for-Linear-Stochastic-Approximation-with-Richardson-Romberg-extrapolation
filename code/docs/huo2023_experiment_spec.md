# Huo et al. 2023 — Experiment Specification for Code Reproduction

## 1. LSA Iteration

$$\theta_{t+1}^{(\alpha)} = \theta_t^{(\alpha)} + \alpha\bigl(A(x_t)\theta_t^{(\alpha)} + b(x_t)\bigr), \quad t = 0, 1, \ldots$$

- `x_t` — Markov chain on finite state space `|X| = n` with transition matrix `P`
- `alpha` — constant stepsize
- Target: `theta* = -Abar^{-1} bbar`, where `Abar = E_pi[A(x)]`, `bbar = E_pi[b(x)]`

## 2. LSA Problem Generation (Appendix C.1)

### Transition matrix P (n x n, n=10)

1. Generate `M^(P) in [0,1]^{n x n}`, entries i.i.d. ~ U[0, 1]
2. Row-normalize: `P_ij = M_ij / sum_k M_ik`
3. Check aperiodicity and irreducibility. If not — regenerate
4. Compute stationary distribution `pi`

### Hurwitz matrix Abar (d x d, d=5)

1. Generate `M^(A) in R^{d x d}`, entries i.i.d. ~ N(0, 1)
2. If `Re(lambda_i(M^(A))) < 0` for all `i` -> `Abar = M^(A)`
3. Otherwise: `Abar = M^(A) - 2 * max(Re(lambda_i)) * I_d`

### Function A(x) (d x d for each x in X)

1. Generate noise matrix `E(x) in [-1,1]^{d x d}`, entries ~ U[-1, 1], for `x = 0, ..., n-2`
2. Set `E(n-1) = (Abar - sum_{x=0}^{n-2} pi_x * E(x)) * (... )` so that `E_pi[E(x)] = 0` — actually last state is set as:
   `A(n-1) = Abar - sum_{x=0}^{n-2} pi_x * E(x)` rewritten to keep `E_pi[A(x)] = Abar`
3. For x = 0, ..., n-2: `A(x) = Abar + E(x)`
4. For x = n-1: `A(n-1) = (Abar - sum_{x=0}^{n-2} pi_x * (Abar + E(x))) / pi_{n-1}`
   Simplified: we need `sum_x pi_x A(x) = Abar`, so
   `A(n-1) = (Abar - sum_{x=0}^{n-2} pi_x * A(x)) / pi_{n-1}`

### Function b(x) (d-vector for each x in X)

1. `b(x)_i` ~ i.i.d. U[-1, 1] for all x and i
2. `bbar = sum_x pi_x b(x)`
3. `b_max = max_x ||b(x)||`

### Target vector

`theta* = -Abar^{-1} bbar`

## 3. Inference Procedure (Section 4.2)

### Batching and point estimation

1. Run LSA for `T` iterations with stepsize `alpha`
2. First `b` iterates — burn-in (discarded). In main experiment: `b = n` (one batch length)
3. Remaining `T - b` iterates split into `K` equal batches of size `n = (T - b) / K`
4. Within each batch, discard first `n0` iterates (for decorrelation). Default: `n0 = 0`
5. Batch-mean estimate for k-th batch:

```
theta_bar_k = (1 / (n - n0)) * sum_{l = b + (n-1)*k + n0}^{b + n*k - 1} theta_l
```

More precisely, indexing batches k = 1, ..., K:
```
batch k spans iterates: b + (k-1)*n  to  b + k*n - 1
discard first n0 within batch
theta_bar_k = (1/(n - n0)) * sum_{l = b + (k-1)*n + n0}^{b + k*n - 1} theta_l
```

### Point estimate and covariance

```
theta_bar = (1/K) * sum_{k=1}^{K} theta_bar_k

Sigma_hat = ((n - n0) / K) * sum_{k=1}^{K} (theta_bar_k - theta_bar)(theta_bar_k - theta_bar)^T
```

### 95% Confidence Interval (coordinate-wise, for i-th coordinate)

```
CI_i = [ theta_bar_i - z_{0.975} * sqrt(Sigma_hat_{i,i} / (K * (n - n0))),
         theta_bar_i + z_{0.975} * sqrt(Sigma_hat_{i,i} / (K * (n - n0))) ]
```

where `z_{0.975} = 1.96`.

## 4. Richardson-Romberg Extrapolation (Section 4.3)

Run `M` LSA with different stepsizes `{alpha_1, ..., alpha_M}` on the **same** Markov chain trajectory `(x_k)`.

### RR coefficients via Vandermonde system

```
h_m = prod_{l=1, l != m}^{M} alpha_l / (alpha_l - alpha_m)
```

### RR extrapolated batch-mean

```
theta_tilde_k = sum_{m=1}^{M} h_m * theta_bar_k^{(alpha_m)}
```

Then inference uses the same batch-mean formulas but with `theta_tilde_k` instead of `theta_bar_k`:

```
theta_tilde = (1/K) * sum_{k=1}^{K} theta_tilde_k

Sigma_tilde = ((n - n0) / K) * sum_{k=1}^{K} (theta_tilde_k - theta_tilde)(theta_tilde_k - theta_tilde)^T

CI_i = theta_tilde_i +/- z_{0.975} * sqrt(Sigma_tilde_{i,i} / (K * (n - n0)))
```

### Stepsize schedules for RR

**Geometric decay:** `alpha_m = alpha_1 / c^{m-1}`, c >= 2.
In the main experiment: `alpha_1 = 0.2`, `alpha_2 = 0.02` (M=2, c=10).

**Equidistant decay:** `alpha_m = (a + b) - b*(m-1)/(M-1)` for m = 1, ..., M.

## 5. Diminishing Stepsize Baseline (Section C.2)

- Stepsize: `alpha_k = alpha_0 * k^{-0.5}`, two variants: `alpha_0 = 0.2` and `alpha_0 = 0.02`
- Batches have **unequal** (exponentially growing) lengths:
  ```
  e_k = ((k+1) * r)^{1/(1 - alpha)},  k = 0, ..., K
  r = T^{1-alpha} / (K + 1),  alpha = 0.5
  ```
  where `e_k` marks the end index of batch k
- First batch (k=0) is burn-in, discarded
- No RR extrapolation applied (diminishing stepsize converges a.s.)

## 6. Main Experiment Setup (Section 6.1.2)

### Parameters

| Parameter | Value |
|-----------|-------|
| Dimension d | 5 |
| State space size n = \|X\| | 10 |
| Number of LSA problems | 100 |
| Independent trajectories per problem | 100 |
| Trajectory length T | 10^5 |
| Number of batches K | 50 |
| Burn-in b | n = (T - b)/K, i.e. first batch discarded |
| Intra-batch discard n0 | 0 |

### Stepsize regimes (5 methods to compare)

| Label | Type | Details |
|-------|------|---------|
| `0.2` | Constant | alpha = 0.2 |
| `0.02` | Constant | alpha = 0.02 |
| `RR` | Constant + RR | alpha_1 = 0.2, alpha_2 = 0.02 (M=2) |
| `0.2/sqrt(k)` | Diminishing | alpha_k = 0.2 * k^{-0.5} |
| `0.02/sqrt(k)` | Diminishing | alpha_k = 0.02 * k^{-0.5} |

## 7. Metrics

For each LSA problem, across 100 independent runs compute:

1. **l2 error** = `||theta_bar - theta*||_2` — measures bias + variance of point estimate
2. **CI width** = width of 95% CI for 1st coordinate = `2 * z_{0.975} * sqrt(Sigma_hat_{1,1} / (K*(n-n0)))`
3. **Coverage probability** = fraction of 100 runs where CI covers `theta*_1` (1st coordinate only)

Then aggregate percentiles (10, 25, 50, 75, 90) across the 100 LSA problems.

## 8. Additional Experiments

### Batch number selection (Table 2)
- One fixed LSA problem, |X|=10, d=5
- T = 10^6, K in {50, 100, 500, 1000}
- 500 independent runs per setting
- Report mean and standard error

### Trajectory length (Table 3)
- One fixed LSA problem
- T in {10^3, 10^4, 10^5, 10^6}
- K = floor(T^{0.3})
- More stepsizes: {0.2, 0.15, 0.1, 0.05, 0.02} + RR + diminishing

### Nonlinear / logistic regression (Table 4)
- Features: x_t from 2-dimensional Gaussian AR(1) process
- Labels: y_t ~ Bernoulli(1 / (1 + e^{-w*^T x_t})), w* random unit vector in R^2
- SGD update (nonlinear SA, not LSA)
- Same inference procedure applied

## 9. Suggested Code Structure

```
generate_lsa_problem(n_states, d) -> (P, A, b, pi, Abar, theta_star)
sample_markov_chain(P, T, x0) -> x_trajectory
run_lsa(theta_0, alpha, A, b, x_trajectory, T) -> theta_iterates
run_lsa_diminishing(theta_0, alpha_0, A, b, x_trajectory, T) -> theta_iterates
batch_mean_inference(theta_iterates, K, burn_in, n0) -> (theta_bar, Sigma_hat, CI)
batch_mean_inference_diminishing(theta_iterates, K, T, alpha) -> (theta_bar, Sigma_hat, CI)
compute_rr_coefficients(alphas) -> h_m  (via Vandermonde)
rr_extrapolation(alphas, batch_means_per_alpha) -> (theta_tilde, Sigma_tilde, CI)
compute_metrics(theta_estimate, CI, theta_star) -> (l2_error, ci_width, coverage)
run_experiment(n_problems, n_runs, T, K, stepsize_regimes) -> results table
```

## 10. Key Points for Bias and Coverage Analysis

- **Bias** = `||E[theta_bar] - theta*||` ≈ average over runs of `||theta_bar - theta*||`
- **Coverage** = `mean(theta*_i in CI_i)` over runs — fraction of CIs that cover the true value
- Constant stepsize with alpha=0.2 has large bias -> poor coverage
- Constant stepsize with alpha=0.02 has small bias -> good coverage but slower convergence
- RR extrapolation achieves best of both worlds: removes bias, keeps fast convergence
- Diminishing stepsize 0.02/sqrt(k) is consistently worst due to slow convergence
- All CI widths and coverage reported for **1st coordinate only**

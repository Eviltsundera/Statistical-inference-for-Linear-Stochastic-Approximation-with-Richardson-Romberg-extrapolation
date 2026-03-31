# Samsonov et al. 2025 — Experiment Specification for Code Reproduction

## 1. LSA Iteration (Diminishing Stepsize, Polyak-Ruppert Averaging)

Standard LSA:
```
theta_k = theta_{k-1} - alpha_k * {A(Z_k) * theta_{k-1} - b(Z_k)},   k >= 1
```

Polyak-Ruppert averaged iterate:
```
theta_bar_n = (1/n) * sum_{k=0}^{n-1} theta_k
```

- `Z_k` — ergodic Markov chain on measurable space (Z, Z) with stationary distribution pi
- `alpha_k = c_0 / (k + k_0)^gamma` — diminishing stepsize, gamma in [1/2, 1)
- Target: `theta* = Abar^{-1} bbar`, where `Abar = E_pi[A(Z)]`, `bbar = E_pi[b(Z)]`
- CLT: `sqrt(n) * (theta_bar_n - theta*) -> N(0, Sigma_infty)` where `Sigma_infty = Abar^{-1} Sigma_eps Abar^{-T}`

Key difference from Huo et al.: this paper uses **diminishing** stepsizes (not constant), so there is **no asymptotic bias** — the averaged iterates converge to theta* directly.

## 2. Noise Covariance

```
Sigma_eps = E_pi[eps(Z_0) eps(Z_0)^T] + 2 * sum_{l=1}^{infty} E_pi[eps(Z_0) eps(Z_l)^T]
```
where `eps(z) = A_tilde(z) * theta* - b_tilde(z)`, `A_tilde(z) = A(z) - Abar`, `b_tilde(z) = b(z) - bbar`.

Asymptotic variance for projection on direction u:
```
sigma^2(u) = u^T * Abar^{-1} * Sigma_eps * Abar^{-T} * u
```

## 3. Multiplier Subsample Bootstrap (MSB) Procedure (Section 4)

This is the main inference method of the paper. It constructs bootstrap CIs without estimating the covariance matrix directly.

### 3.1 Overlapping Batch Mean (OBM) Estimator

Given `b_n` (block/lag size), for each starting point `t = 0, ..., n - b_n`, define:
```
theta_{b_n, t} = (1/b_n) * sum_{l=t}^{t + b_n - 1} theta_l
```
This is the "scale b_n" version of theta_bar_n (local average over a window of size b_n).

### 3.2 MSB Statistic

The bootstrap estimator of `theta_bar_n` is:
```
theta_{n,b_n}(u) = sqrt(b_n) / (n - b_n + 1) * sum_{t=0}^{n - b_n} w_t * (theta_{b_n,t} - theta_bar_n)^T * u
```

where:
- `w_t ~ N(0, 1)` are i.i.d. multiplier weights (independent of data)
- `u in S^{d-1}` is the projection direction

### 3.3 Bootstrap Variance Estimator

The OBM-based variance estimator (equivalent to bootstrap variance up to remainder):
```
sigma_theta_hat^2(u) = b_n / (n - b_n + 1) * sum_{t=0}^{n - b_n} ((theta_{b_n,t} - theta_bar_n)^T * u)^2
```

This estimates `sigma^2(u)`.

### 3.4 Confidence Interval Construction

Conditional on data, `theta_{n,b_n}(u) ~ N(0, sigma_theta_hat^2(u))` under the bootstrap probability P^b.

**CI for `u^T theta*`** at level `1 - q`:
```
CI = [ u^T theta_bar_n - z_{1-q/2} * sigma_theta_hat(u) / sqrt(n),
       u^T theta_bar_n + z_{1-q/2} * sigma_theta_hat(u) / sqrt(n) ]
```

Alternatively, using bootstrap quantiles directly:
```
CI = [ u^T theta_bar_n - Q_{1-q/2}^{boot} / sqrt(n),
       u^T theta_bar_n - Q_{q/2}^{boot} / sqrt(n) ]
```
where `Q_alpha^{boot}` is the alpha-quantile of `theta_{n,b_n}(u)` under bootstrap resampling.

## 4. Theoretical Results

### Berry-Esseen Bound (Theorem 1)
```
d_K(sqrt(n) * u^T * (theta_bar_n - theta*) / sigma_n(u), N(0,1)) <= B_n
```
where the dominant term is `O((log n)^{5/2} / n^{1/4})` when gamma = 3/4.

### Bootstrap Validity (Theorem 2)
With `b_n = ceil(n^{4/5})` and `alpha_k = c_0/(k_0 + k)^{3/5}`:
```
sup_x |P(sqrt(n) (theta_bar_n - theta*)^T u <= x) - P^b(theta_{n,b_n}(u) <= x)| <= O(n^{-1/10})
```
up to log factors. This guarantees the bootstrap CI has correct coverage asymptotically.

### OBM Variance Estimator Consistency (Corollary 2)
With `b_n = ceil(n^{3/4})`, `gamma = 1/2 + eps`, `alpha_k = c_0/(k_0 + k)^{1/2+eps}`:
```
|sigma_theta_hat^2(u) - sigma^2(u)| <= O(n^{-1/8+eps/2})
```
with high probability.

## 5. Numerical Experiments (Appendix G)

### 5.1 Problem: TD Learning on Garnet MDP

**Garnet problem** [Archibald et al. 1995] — a class of random MDPs parameterized by (N_s, a, b):
- `N_s = 6` states
- `a = 2` actions
- `b = 3` branching factor (number of reachable next states per (s,a) pair)
- `lambda = 0.8` discount factor
- Feature dimension `d = 2`

**Policy:**
```
pi(a|s) = U_a^{(s)} / sum_{i=1}^{|A|} U_i^{(s)}
```
where `U_a^{(s)} ~ U[0, 1]` i.i.d.

**Feature map:** Generate `Phi in R^{N_s x d}` with entries ~ N(0,1), then normalize each row:
```
phi(s) = Phi_s / ||Phi_s||
```

**TD learning update** (cast as LSA):
```
theta_k = theta_{k-1} - alpha_k * (A_k theta_{k-1} - b_k)
```
where:
```
A_k = phi(s_k) * {phi(s_k) - lambda * phi(s_{k+1})}^T
b_k = phi(s_k) * r(s_k, a_k)
```

**Markov chain:** `(s_k)` is the induced state process under policy pi, which is a geometrically ergodic Markov chain.

**Target:** `theta* = Abar^{-1} bbar` where:
```
Abar = E_{s~mu, s'~P_pi(.|s)} [phi(s) {phi(s) - lambda * phi(s')}^T]
bbar = E_{s~mu, a~pi(.|s)} [phi(s) r(s, a)]
```

### 5.2 Experiment Parameters

| Parameter | Value |
|-----------|-------|
| States N_s | 6 |
| Actions a | 2 |
| Branching factor b | 3 |
| Discount lambda | 0.8 |
| Feature dimension d | 2 |
| Stepsize gamma | 2/3 |
| Stepsize form | alpha_k = c_0 / (k_0 + k)^gamma |
| c_0, k_0 | "appropriately chosen" (not specified exactly) |

### 5.3 Trajectory Lengths and Block Sizes

| n (trajectory length) | b_n (block/lag size) |
|-----------------------|---------------------|
| 20480 | 250 |
| 204800 | 1200 |
| 1024000 | 3600 |

### 5.4 Inference Method

For each run:
1. Run LSA (TD learning) for n iterations with diminishing stepsize
2. Compute Polyak-Ruppert average theta_bar_n
3. Sample random unit vector u from S^{d-1}
4. Compute OBM variance estimator sigma_theta_hat^2(u) with block size b_n
5. Also compute the "true" asymptotic variance sigma^2(u) (for comparison)
6. Construct CIs at confidence levels {0.80, 0.90, 0.95}
7. Check coverage of u^T theta*

### 5.5 What is Reported (Table 2)

**Coverage probabilities** for the empirical distribution, using two variance estimates:
- `sigma_theta_hat^2(u)` — the OBM estimator (what you'd use in practice)
- `sigma^2(u)` — the true asymptotic variance (oracle, for comparison)

Also reported: **stddev** of the point estimate `u^T theta_bar_n` (x 10^3).

**Results from paper (Table 2):**

| n | b_n | Cov 0.95 (OBM) | Cov 0.95 (true) | Cov 0.90 (OBM) | Cov 0.90 (true) | Cov 0.80 (OBM) | Cov 0.80 (true) | stddev x10^3 |
|---|-----|----------------|-----------------|----------------|-----------------|----------------|-----------------|-------------|
| 20480 | 250 | 0.873 | 0.881 | 0.773 | 0.805 | 0.641 | 0.662 | 10.89 |
| 204800 | 1200 | 0.935 | 0.945 | 0.880 | 0.892 | 0.768 | 0.784 | 3.49 |
| 1024000 | 3600 | 0.942 | 0.948 | 0.887 | 0.897 | 0.769 | 0.788 | 1.56 |

### 5.6 Experimental Details

- Experiments follow the setup from [Fang, Xu, Yang 2018] (reference [29])
- Code available at: https://anonymous.4open.science/r/markov_lsa_normal_approximation
- Run on Google Colab: Intel Xeon 2.20GHz, 8 vCores, 53.5GB RAM, no GPU

## 6. Key Differences from Huo et al. 2023

| Aspect | Huo et al. 2023 | Samsonov et al. 2025 |
|--------|----------------|---------------------|
| **Stepsize** | Constant alpha | Diminishing alpha_k = c_0/(k+k_0)^gamma |
| **Bias** | Non-zero asymptotic bias (needs RR) | Zero bias (diminishing step) |
| **CLT type** | Asymptotic CLT | Non-asymptotic Berry-Esseen bound |
| **CI construction** | Batch-mean covariance estimator | Multiplier subsample bootstrap (OBM) |
| **Batching** | Non-overlapping K batches | Overlapping blocks of size b_n |
| **Variance estimator** | Non-overlapping batch-mean | Overlapping batch-mean (OBM) |
| **Rate guarantee** | Asymptotic only | O(n^{-1/10}) for coverage, O(n^{-1/8}) for variance |
| **Application** | Generic LSA + TD learning | Generic LSA + TD learning (Garnet MDP) |

## 7. Suggested Code Structure

```
# Problem generation (Garnet MDP for TD learning)
generate_garnet_mdp(N_s, n_actions, branching, discount) -> (P, r, pi_policy)
generate_feature_map(N_s, d) -> phi  # normalized random features
compute_td_matrices(phi, P_pi, r, pi_policy, discount) -> (Abar, bbar, theta_star)
compute_noise_covariance(phi, P_pi, r, theta_star, Abar, bbar) -> Sigma_eps
compute_asymptotic_variance(Abar, Sigma_eps, u) -> sigma_sq

# LSA / TD learning
run_td_learning(theta_0, stepsizes, phi, states_trajectory, rewards) -> theta_iterates
polyak_ruppert_average(theta_iterates) -> theta_bar_n

# Markov chain sampling
sample_mdp_trajectory(P, pi_policy, r, n, s0) -> (states, actions, rewards, next_states)

# Inference: Multiplier Subsample Bootstrap
compute_obm_variance(theta_iterates, theta_bar_n, b_n, u) -> sigma_theta_hat_sq
msb_confidence_interval(theta_bar_n, sigma_theta_hat, n, u, confidence_level) -> CI
# Alternative: full bootstrap with multiplier weights
msb_bootstrap_quantiles(theta_iterates, theta_bar_n, b_n, u, n_bootstrap) -> quantiles

# Metrics
compute_coverage(CI, true_value) -> bool
compute_bias(theta_bar_n, theta_star, u) -> bias

# Full experiment
run_experiment(n_values, b_n_values, n_runs, confidence_levels) -> results_table
```

## 8. Key Points for Bias and Coverage Analysis

- **Bias:** With diminishing stepsize, bias is zero asymptotically. Finite-sample bias decays with n. This is fundamentally different from constant stepsize (Huo et al.) where bias is O(alpha).
- **Coverage with OBM:** The OBM estimator sigma_theta_hat^2 slightly underestimates the true variance for small n, leading to under-coverage. As n grows, coverage approaches nominal levels.
- **Coverage with true variance:** Using the oracle sigma^2(u) gives better coverage, showing the gap is due to variance estimation, not the CLT approximation itself.
- **Block size b_n matters:** Too small b_n -> biased variance estimate (correlation not captured). Too large b_n -> high variance of the estimator. Theory suggests b_n ~ n^{4/5} for bootstrap, b_n ~ n^{3/4} for OBM.
- **For your comparison:** When comparing with Huo et al.'s constant-stepsize methods:
  - Diminishing stepsize has no bias but slower convergence (especially for small n)
  - Constant stepsize + RR trades off bias reduction vs variance inflation
  - MSB bootstrap is a different CI construction method than batch-mean covariance
  - Key metric: coverage probability at same trajectory length n

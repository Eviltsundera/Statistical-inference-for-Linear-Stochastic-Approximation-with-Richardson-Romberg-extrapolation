# Detailed Review: Ng & Perron (1996)
# "The Exact Error in Estimating the Spectral Density at the Origin"

**Journal:** Journal of Time Series Analysis, Vol. 17, No. 4, pp. 379–408  
**Authors:** Serena Ng (Université de Montréal), Pierre Perron (Université de Montréal and C.R.D.E.)  
**First version received:** November 1994  
**Keywords:** Kernel; persistence measures; bandwidth; lag window; non-parametric inference

---

## Overview and Significance

This paper provides **exact, closed-form analytical expressions** for the bias and variance of a general class of kernel-based spectral density estimators evaluated at frequency zero, for both the known-mean and unknown-mean cases. This is a significant departure from the standard asymptotic theory that was the only available tool at the time. The key contributions are:

1. Exact bias formula (eq. 3.2) and exact variance formula (eq. 3.3) for the **known-mean** estimator $\hat{h}_T$, extending the classic Neave (1971) result to frequency zero across 15 different lag windows.
2. Analogous exact bias (eq. 4.7) and variance (eq. 4.10) formulas for the **unknown-mean** estimator $\tilde{h}_T$, showing that the sample mean introduces additional non-trivial correction terms.
3. A systematic numerical study of 720 combinations of (window, bandwidth $M_T$, sample size $T$, DGP) revealing that asymptotic results are **poor guides to finite-sample behavior**, especially regarding the bandwidth-MSE relationship.
4. A **response surface analysis** that provides simple, practical regression-based formulas for predicting bias and variance as a function of $M_T$, $T$, $h(0)$, and the model parameters.

For anyone working on long-run variance estimation in stochastic approximation, MCMC, or econometrics, this paper is the authoritative finite-sample reference that shows exactly *why* bandwidth selection based on asymptotic plug-in rules can go badly wrong.

---

## 1. Problem Setup and Motivation

### 1.1 The spectral density at frequency zero

Consider a real-valued, weakly stationary linear process $X_t$ with finite mean $\mu = E(X_t)$ and autocovariance at lag $v$:

$$R(v) = E\{(X_t - \mu)(X_{t+v} - \mu)\}.$$

It is assumed that $R(v)$ is continuous at $v = 0$ and $\int_{-\infty}^{\infty}|R(v)|\,dv < \infty$. The power spectrum of $X_t$ is:

$$f(\omega) = \frac{1}{2\pi}\sum_{v=-\infty}^{\infty} R(v)\cos(v\omega) = \frac{1}{2\pi}\left\{R(0) + 2\sum_{v=1}^{\infty}R(v)\cos(v\omega)\right\}.$$

The **spectral density at the origin** (zero frequency) is:

$$h(0) = \sigma^2\psi(1)^2/(2\pi)$$

(in the ARMA representation), or equivalently $h(0) = \sum_{v=-\infty}^{\infty} R(v)$. This is the **long-run variance** up to the factor $2\pi$.

### 1.2 Why $h(0)$ matters in economics and statistics

1. **Persistence measurement.** For an integrated process $y_t$, its first difference $\Delta y_t$ has an MA representation $\Delta y_t = \psi(L)\varepsilon_t$. The cumulative impulse response $\psi(1) = B(1)/A(1)$ measures how permanent a unit shock is; Campbell and Mankiw's (1987) persistence measure is exactly $\psi(1) = \{2\pi f_{\Delta y}(0)/\sigma_\varepsilon^2\}^{1/2}$, so estimating $h(0)$ for the differenced series is equivalent to estimating $\psi(1)$.

2. **Unit root inference.** Phillips (1987) and Phillips–Perron (1988) tests require a consistent estimate of the spectral density at the origin for the residuals; these are used to correct the Dickey-Fuller statistic for serial correlation.

3. **GMM and HAC estimation.** The Newey-West (1987) HAC variance estimator — ubiquitous in econometrics — is precisely a kernel estimate of the spectral density at zero frequency applied to moment conditions. Its finite-sample properties are exactly what this paper studies.

4. **Cochrane's (1988) variance ratio** $V(k) = \text{var}(y_t - y_{t-k})/\{k\,\text{var}(y_t - y_{t-1})\}$ satisfies $\lim_{k\to\infty} V(k) = 2\pi f_{\Delta y}(0)$, so again controlled by $h(0)$.

---

## 2. The Estimator Class

### 2.1 Known-mean estimator

For $\mu = 0$ (WLOG), the class of lag-window estimators for $f(\omega)$ is:

$$f_T(\omega) = \frac{1}{2\pi}\sum_{v=-(T-1)}^{T-1} k_T^*\!\left(\frac{v}{M_T}\right)\hat{R}_T^*(v)\cos(v\omega), \tag{2.2}$$

where $\hat{R}_T^*(v) = T^{-1}\sum_{t=1}^{T-v} X_t X_{t+v}$ is the **biased** sample autocovariance, and $k_T^*(v/M_T)$ is a non-negative, bounded, even **lag window (kernel)** function. $M_T$ is the **bandwidth parameter**. Estimators of the form (2.2) are consistent for $f(\omega)$ if $M_T/T \to 0$ and $M_T \to \infty$ as $T \to \infty$.

At zero frequency, the estimator collapses to estimating $h(0) = \sigma^2\psi(1)^2 = \sum_{v=-\infty}^\infty R(v)$. Defining $\theta = v/M_T$ and $k_T(\theta) = (1 - v/T)k_T^*(v/M_T)$ with $k_T(0) = 0.5$, the formula simplifies to:

$$\hat{h}_T = 2\sum_{v=0}^{T-1} k_T(\theta)\hat{R}_T(v), \tag{3.1}$$

where $\hat{R}_T(v)$ is the unbiased estimator $\hat{R}_T(v) = (T-v)^{-1}\sum_{t=1}^{T-v} X_t X_{t+v}$.

### 2.2 Unknown-mean estimator

When the mean $\mu$ must be estimated by $\bar{X} = T^{-1}\sum_{t=1}^T X_t$, the demeaned sample autocovariance is:

$$\bar{R}_T^*(v) = T^{-1}\sum_{t=1}^{T-v}(X_t - \bar{X})(X_{t+v} - \bar{X}),$$

and the estimator is:

$$\tilde{h}_T = 2\sum_{v=0}^{T-1} k_T(\theta)\bar{R}_T(v), \tag{4.1}$$

with $\bar{R}_T(v) = (T-v)^{-1}\sum_{t=1}^{T-v}(X_t - \bar{X})(X_{t+v} - \bar{X})$.

**Critical observation:** Although at non-zero frequencies the periodogram and its mean-corrected version are identical, at zero frequency they differ fundamentally — the subtraction of $\bar{X}$ introduces a systematic distortion that does not vanish at any finite $T$.

### 2.3 The 15 lag windows studied

The paper evaluates the following lag windows (Table I), all satisfying $k^*(0) = 1$, $k^*(x) = k^*(-x)$, $\int_{-\infty}^\infty [k^*(x)]^2 < \infty$, and continuity at 0 and almost everywhere:

| No. | Name | $k^*(\theta)$ |
|-----|------|---------------|
| 1 | Truncated periodogram | $1$ |
| 2 | Bartlett (a) with $\hat{R}_T^*$ | $1 - \theta$ |
| 3 | Bartlett (b) with $\hat{R}_T$ | $(1-\theta)T/(T-v)$ |
| 4 | Parzen (a) | $1 - 6\theta^2 + 6\theta^3$ ($\theta \le 1/2$); $2(1-\theta)^3$ ($\theta > 1/2$) |
| 5 | Tukey–Hamming | $0.54 + 0.46\cos(\pi\theta)$ |
| 6 | Tukey–Hanning | $\frac{1}{2}(1 + \cos(\pi\theta))$ |
| 7 | Bohman | $(1-\theta)\cos(\pi\theta) + \sin(\pi\theta)/\pi$ |
| 8 | Daniell | $\sin(\pi\theta)/(\pi\theta)$ |
| 9 | Parzen (b) | $1 - \theta^2$ |
| 10 | Bartlett (c) | $\{1 - v/(M_T+1)\}T/(T-v)$ |
| 11 | Parzen (c) | $1/(1+\theta^2)$ |
| 12 | Tukey–Parzen | $0.436 + 0.564\cos(\pi\theta)$ |
| 13 | Normal | $\exp(-4.5\theta^2)$ |
| 14 | Quadratic (Andrews 1991 optimal) | $\frac{25}{12\pi^2\theta^2}\left\{\frac{\sin(6\pi\theta/5)}{6\pi\theta/5} - \cos(6\pi\theta/5)\right\}$ |
| 15 | Trapezoid | $1$ ($\theta \le 1/2$); $2(1-\theta)$ ($\theta > 1/2$) |

Windows 2 and 3 are both "Bartlett" but differ in whether they use $\hat{R}_T^*$ or $\hat{R}_T$; windows 3 and 10 are used by Campbell–Mankiw and Cochrane respectively.

**The Bartlett (a) and Bartlett (b) windows are always dominated** by other windows in MSE, in all experiments. The **trapezoid** window is a notable performer for nearly-integrated processes.

---

## 3. Asymptotic Theory (Background)

### 3.1 Characteristic exponent and bias

Let $r$ be the **characteristic exponent** of $k_T^*(\theta)$: the largest integer such that $k_T^{*(r)} = \lim_{\theta\to 0}[1 - k_T^*(\theta)]/|\theta|^r$ exists, is finite and non-zero. Let $h^{(r)}(0) = \sum_{v=-\infty}^\infty |v|^r |R(v)|$ be the Parzen "generalized $r$th derivative" of the spectral density at zero.

Under standard conditions (Priestley 1981):

**Asymptotic bias:**
$$\text{asymptotic bias}(\hat{h}_T) \approx (-M_T)^{-1}h^{(r)}(0)\,k_T^{*(r)}. \tag{2.5}$$

This holds when there exists $q \ge r$ with $h^{(q)}(0) < \infty$ and $T/(M_T)^r \to \infty$.

**Asymptotic variance:**
$$\lim_{T\to\infty}(T/M_T)\,\text{var}(\hat{h}_T) = 2h^2(0)\int_{-1}^1 [k_T^*(\theta)]^2\,d\theta. \tag{2.6}$$

This requires $M_T/T \to 0$ and $M_T \to \infty$.

**Key implications:**
- Bias decreases in $M_T$ (more lags = smoother estimate = less truncation bias).
- Variance increases in $M_T$ (more lags = more noisy autocovariance terms added).
- Optimal $M_T^{\text{opt}} \sim T^{1/(2r+1)}$ for the standard bias–variance tradeoff.
- Crucially, the asymptotic bias depends on the **curvature** $h^{(r)}(0)$, while the variance depends on $h(0)$ itself. These are different quantities, so the two components are driven by different features of the DGP.

---

## 4. Exact Bias and Variance: Known Mean

### 4.1 Exact bias formula

Since $\hat{R}_T(v)$ is unbiased for $R(v)$, we have $E(\hat{h}_T) = 2\sum_{v=0}^{T-1} k_T(v/M_T)R(v)$, and:

$$\text{bias}(\hat{h}_T) = E\{\hat{h}_T - h(0)\} = 2\sum_{v=1}^{T-1}\{k_T(v) - 1\}R(v) - 2\sum_{v=T}^{\infty}R(v). \tag{3.2}$$

The first sum captures the distortion from weighting autocovariances with $k_T(v) \ne 1$ (truncation inside the sample window). The second sum is the contribution of lags beyond $T-1$ (unavoidable truncation).

**Structure:** The bias depends on the entire autocovariance function $R(v)$, not just its curvature. For processes with slow autocovariance decay (nearly-integrated DGPs), $\sum_{v=T}^\infty R(v)$ can be very large relative to $h(0)$, making the bias enormous at small $M_T$.

### 4.2 Exact variance formula

The exact variance was derived using Neave's (1971) method, assuming $X_t$ is Gaussian (fourth cumulant = 0):

$$\text{var}(\hat{h}_T) = 8\left(\sum_{v=0}^{T-1}\frac{k_T^2(v)}{(T-v)^2}\left[\frac{T-v}{2}\{R^2(0) + R^2(v)\} + \sum_{x=1}^{T-v}(T-v-x)\{R^2(x) + R(x-v)R(x+v)\}\right]\right.$$
$$+ 2\sum_{0 \le u \le v \le T-1}\frac{k_T(v)k_T(u)}{(T-v)(T-u)}\left[\frac{T-u}{2}\sum_{x=0}^{u-v}\{R(x)R(x+v-u)\} + \sum_{x=u-v+1}^{T-v}(T-v-x)\{R(x)R(x+v-u) + R(x-u)R(x+v)\}\right]\!\!\left.\right). \tag{3.3}$$

This is a complicated double sum involving the full autocovariance structure of the process. It is computed numerically for each combination of (window, $M_T$, $T$, DGP).

**What (3.2) and (3.3) enable:** Given any ARMA model (so that $R(v)$ is known in closed form) and any lag window, the *exact* MSE = bias$^2$ + variance is computable for any finite $(T, M_T)$ pair. This is the main technical contribution.

---

## 5. Exact Bias and Variance: Unknown Mean

### 5.1 Structure of the problem

When $\bar{X}$ must be estimated, the sample autocovariance $\bar{R}_T(v)$ is no longer an unbiased estimator of $R(v)$. Define:

$$\text{var}(\bar{X}) = E(\bar{X} - \mu)^2 = T^{-1}R(0) + 2\{T^{-1}\sum_{r=1}^T(1 - r/T)R(r)\}, \tag{4.2}$$

$$\bar{R}(j) = T^{-1}\sum_{k=1}^T R(k-j). \tag{4.3}$$

Using these, one can show:

$$E\{(X_j - \mu)(\bar{X} - \mu)\} = \bar{R}(j), \tag{4.4}$$
$$E\{(X_j - \bar{X})(X_i - \bar{X})\} = R(j-i) + \text{var}(\bar{X}) - \bar{R}(j) - \bar{R}(i), \tag{4.5}$$
$$E\{\bar{R}_T(v)\} = R(v) + \text{var}(\bar{X}) - (T-v)^{-1}\sum_{t=1}^{T-v}\{\bar{R}(t+v) + \bar{R}(t)\}. \tag{4.6}$$

### 5.2 Exact bias of $\tilde{h}_T$

Using (4.6):

$$\text{bias}(\tilde{h}_T) = \text{bias}(\hat{h}_T) + 2\sum_{v=0}^{T-1}k_T(v)\left[\text{var}(\bar{X}) - (T-v)^{-1}\sum_{t=1}^{T-v}\{\bar{R}(t) + \bar{R}(t+v)\}\right]. \tag{4.7}$$

The second term is $O(M_T/T)$ and vanishes asymptotically. However, it shows:
- For **small $T$** and **large $M_T$**, the bias in the unknown-mean case can be substantially different from the known-mean case.
- For **nearly integrated processes** (large $R(v)$ for large $v$), the correction is especially important.

### 5.3 Exact variance of $\tilde{h}_T$

The variance formula takes the form:

$$\text{var}(\tilde{h}_T) = \text{var}(\hat{h}_T) + 4\left(\sum_{u,v=0}^{T-1}(T-v)^{-1}(T-u)^{-1}\sum_{t=1}^{T-v}\sum_{s=1}^{T-u}k_T(u)k_T(v)\right.$$
$$\times \big[2\,\text{var}(\bar{X})^2 + \text{var}(\bar{X})\{R(t-s) + R(t+v-s-u) + R(t-s-u) + R(t+v-s)\}$$
$$- 2\,\text{var}(\bar{X})\{\bar{R}(t+v) + \bar{R}(t) + \bar{R}(s) + \bar{R}(s+u)\}$$
$$+ 2\{\bar{R}(t)\bar{R}(t+v) + \bar{R}(s)\bar{R}(s+u)\}$$
$$+ \{\bar{R}(t+v) + \bar{R}(t)\}\{\bar{R}(s) + \bar{R}(s+u)\}$$
$$- \bar{R}(t+v)\{R(t-s) + R(t-s-u)\} + \bar{R}(t)\{R(t+v-s-u) + R(t+v-s)\}$$
$$- \bar{R}(s+u)\{R(t-s) + R(t+v-s)\} + \bar{R}(s)\{R(t+v-s-u) + R(t-s-u)\}\big]\bigg). \tag{4.10}$$

This expression is valid under normality of $X_t$ (or, more generally, when the fourth cumulant of the distribution is zero). The additional terms relative to $\text{var}(\hat{h}_T)$ are not necessarily positive — they can reduce or increase the variance depending on the DGP.

**Key insight from comparison of (4.7)/(4.10) with (3.2)/(3.3):** The extra terms in the unknown-mean formulas involve $\text{var}(\bar{X})$, $\bar{R}(j)$, and cross-products between these and $R(v)$. The exact sign depends on the direction of autocorrelation:
- For **positively autocorrelated processes**, the bias in the unknown-mean case has larger absolute value (more negative) and the variance can be larger or smaller.
- For **negatively autocorrelated processes**, the bias is positive in the known-mean case but can change sign as $M_T$ grows in the unknown-mean case.

---

## 6. Numerical Experiment Design

### 6.1 Parameter grid

For each of the 15 windows, the bias, variance, and MSE are computed over:
- **Sample sizes:** $T \in \{50, 100, 150\}$
- **Bandwidths:** 10 values of $M_T$ up to $T-1$ for each $T$
  - $T=50$: $M_T \in \{2,4,8,12,16,20,25,30,40,49\}$
  - $T=100$: $M_T \in \{2,4,8,14,20,30,40,50,70,99\}$
  - $T=150$: $M_T \in \{2,4,8,14,20,30,50,70,100,149\}$
- **DGPs:** 24 ARMA processes organized in 5 groups

Total: $15 \times 3 \times 10 \times 24 \times 2 = 21{,}600$ exact MSE evaluations (720 per mean treatment).

### 6.2 The 24 DGPs (5 groups)

**Group 1 — Nearly integrated (large $h(0)$):**
- AR1(0.9): $h(0) = 1/(1-0.9)^2 = 100$
- AR2(1.3, −0.35): near-unit-root, $h(0)$ large
- ARMA(0.9, 0.6): AR near unit root, positive MA
- ARMA(0.9, −0.5): AR near unit root, negative MA

These mimic the log of GDP, consumption, etc. that are hard to distinguish from unit root processes.

**Group 2 — Positively autocorrelated (moderate $h(0)$):**
- AR1(0.4), AR2(0.2, 0.4), ARMA(0.6, −0.5), ARMA(0.9, −0.8)

**Group 3 — Negatively autocorrelated:**
- AR1(−0.4), AR1(−0.9)

These arise when differencing near-unit-root data.

**Group 4 — Positive moving-average:**
- MA1(0), MA1(0.5), MA1(1.0), MA2(0.5, 0.5), ARMA(−0.3, 0.6), ARMA(0.3, 0.6)

**Group 5 — Negative moving-average (small $h(0)$, possibly near zero):**
- MA1(−0.5), MA1(−0.8), MA1(−1.0) (non-invertible boundary!), ARMA(0.9, −1.0), ARMA(0.6, −1.0), ARMA(0.6, −0.8), ARMA(−0.6, −1.0), MA2(−0.5, −0.5)

Group 5 is particularly important because the first-differenced series of a near-unit-root process often has a near-non-invertible MA component, making $h(0)$ close to zero. This is the case where the estimation problem is hardest and asymptotic approximations break down most severely.

---

## 7. Main Empirical Findings

### 7.1 No uniformly best window

The overall finding is that **no single lag window dominates in MSE** across all DGPs and bandwidths. The Bartlett (a) and Bartlett (b) windows are always dominated. The best window depends heavily on the process group.

**Table III (known mean) and Table IV (unknown mean)** show optimal (window, $M_T$) pairs for each DGP at $T \in \{50, 100, 150\}$. Key patterns:

- **Group 1 (nearly integrated):** Trapezoid window (15) best, with $M_T \approx T/7$ (e.g., $M_T = 14$ at $T = 100$). The trapezoid does well because its bias decreases quickly with $M_T$ for persistent processes.
- **Groups 2, 3, 4 (moderate autocorrelation):** Optimal $M_T \approx 4$ across all windows; performance is nearly identical at the optimum, so window choice matters less.
- **Group 5 (negative MA, small $h(0)$):** MSE is *monotonically decreasing* in $M_T$ — larger bandwidth is always better. At $M_T = T-1$ with unknown mean, windows 1 (truncated) and 10 (Bartlett c) give exactly zero by construction (since they cancel all autocovariances), which gives zero MSE when $h(0) = 0$ exactly.

### 7.2 The MSE–bandwidth relationship

**Figure 1** (10 panels) plots MSE vs. $M_T$ for representative DGPs:

- **(a)–(b): ARMA(0.9, ±0.5):** U-shaped MSE; minimum at large $M_T \approx 14$–$20$.
- **(c)–(d): AR(0.4) and AR(−0.4):** Minimum at small $M_T \approx 4$; sharp penalty for over-smoothing (Parzen curve rises steeply for large $M_T$).
- **(e): White noise MA1(0):** Minimum at $M_T = 2$; any smoothing is wasteful.
- **(f): ARMA(0.3, 0.6):** Moderate minimum around $M_T = 4$.
- **(g)–(j): Negative MA processes:** MSE monotonically decreasing. The trapezoid and Bartlett (c) reach near-zero MSE at large $M_T$.

The cost of using a **too-large $M_T$** is particularly severe for the trapezoid window (its MSE explodes). The cost of using a **too-small $M_T$** is most severe for Group 1 processes.

### 7.3 Bias and variance: known mean vs. unknown mean

The most theoretically important finding of the paper (Section 5.2):

**For positively autocorrelated processes (Groups 1, 2, 4):**
- In the **known mean** case: bias is negative and *monotonically decreasing* (more negative) in $M_T$; variance is monotonically increasing in $M_T$.
- In the **unknown mean** case: this monotonicity **breaks down**. The bias first becomes more negative, then *turns around* and increases (becomes less negative) at very large $M_T$. Variance is also non-monotone in $M_T$.
- The **larger the persistence** (closer to unit root), the larger the bandwidth that minimizes the bias in the unknown-mean case.

**For negatively autocorrelated processes (Group 3):**
- Known mean: bias is *positive* and decreasing in $M_T$.
- Unknown mean: bias is positive at small $M_T$, but becomes negative as $M_T$ grows, so the bias is non-monotone and changes sign.

**For processes with negative MA close to −1 (Group 5):**
- Both bias and variance are *lower* with negative serial correlation than with positive serial correlation — because $h(0) = B(1)^2/A(1)^2$ is small, and both bias and variance scale with $h(0)$.
- Variance is *decreasing* in $M_T$ for these processes (opposite to Groups 1–3), for both mean treatments.

**Mathematical explanation:** From (4.7), the difference in biases between the two mean treatments is:

$$\text{bias}(\tilde{h}_T) - \text{bias}(\hat{h}_T) = 2\sum_{v=0}^{T-1}k_T(v)\left[\text{var}(\bar{X}) - (T-v)^{-1}\sum_{t=1}^{T-v}\{\bar{R}(t) + \bar{R}(t+v)\}\right] = O(M_T/T).$$

For small $M_T$, the difference is negligible. For $M_T$ close to $T$, the correction term can be as large as the bias itself. This is why the asymptotic statement "the two biases differ by $O(M_T/T)$" is misleading in finite samples: $O(M_T/T)$ is $O(1)$ when $M_T \sim T$.

**The key practical implication:** Asymptotic theory says that as long as $M_T/T \to 0$, it does not matter whether the mean is known or unknown. But in practice, for $T = 50$ or $T = 100$ and $M_T = 20$ (which is not unusual in applied work), $M_T/T = 0.2$–$0.4$, and the difference can be very large.

### 7.4 Tables V and VI: per-window rankings

Tables V and VI report bias, variance, MSE and rankings for $T = 50$ for selected DGPs under known and unknown mean respectively. Key observations:

- **AR1(0.9):** Bias ≈ −18 to −31 (in thousands relative to $h(0)^2 = 10^4$). The trapezoid window gives the best (smallest absolute) bias. Variance ranges from 227 to 58,530 (thousands). Truncated periodogram is the worst.
- **AR1(0.4):** Bias ≈ −0.05 to −0.1; variance ≈ 0.14 to 0.50. All kernels perform similarly.
- **AR1(−0.4):** Bias is positive (0.01–0.02); variance is 0.003–0.011. Quadratic and Tukey–Hanning windows best.
- **MA1(0.5):** Bias ≈ −0.02; variance ≈ 0.14 to 0.85. Most windows perform similarly.
- **MA1(−0.5):** Bias ≈ 0.02; variance ≈ 0.02–0.07. Bartlett (a) and (b) are worst; most others are similar.

The rankings are **not stable across DGPs** and are not monotonically related to the asymptotic ordering of the kernels.

---

## 8. Response Surface Analysis

### 8.1 Motivation

With 720 observations (combinations of $M_T$, $T$, DGP) per mean treatment, it is possible to fit regression models for bias and variance as functions of the explanatory variables. This provides a compact summary of the finite-sample properties and can guide bandwidth selection.

### 8.2 Bias equation

Using asymptotic expression (2.5) as a baseline, and incorporating additional factors found to matter empirically:

$$\text{bias}(\hat{h}_T) = \alpha_0\frac{h''(0)}{M_T^2} + \alpha_1\frac{(1-\sum a_i)^{-2}}{M_T^3} + \alpha_2\frac{h(0)^3}{T^3} + \alpha_3\frac{h(0)^2/M_T}{T} + \alpha_4\frac{h(0)}{T}. \tag{6.1}$$

Where $\sum a_i$ is the sum of autoregressive coefficients (so $(1-\sum a_i)^{-1}$ blows up as the AR root approaches unity). The inclusion of $(1-\sum a_i)^{-2}/M_T^3$ captures the additional bias for nearly-integrated processes that is not captured by $h''(0)/M_T^2$ alone.

### 8.3 Variance equation

Using (2.6) as a baseline:

$$\text{var}(\hat{h}_T) = \beta_0\frac{h(0)^2 M_T}{T} + \beta_1\frac{1}{T^2} + \beta_2\frac{M_T}{T^2} + \beta_3\frac{h(0)}{T} + \beta_4\frac{h(0)^2}{T}. \tag{6.2}$$

The leading term $\beta_0 h(0)^2 M_T/T$ corresponds to the asymptotic result (2.6). Additional terms capture finite-sample corrections.

### 8.4 Results of the response surface regressions (Tables VII–X)

**Table VII (bias, known mean):** $R^2 \approx 0.86$–$0.90$ for all 15 windows.
- Coefficient on $h''(0)/M_T^2$: **0.005** for all windows (exactly the asymptotic coefficient, confirming the leading term).
- Coefficient on $(1-\sum a_i)^{-2}/M_T^3$: ranges from 2.45 (truncated) to 2.88 (Bartlett b). Highly significant for all windows.
- Coefficient on $h(0)/T$: ranges from −26 to −56, highly significant for all windows.

**Table VIII (bias, unknown mean):** $R^2 \approx 0.97$–$0.98$ (higher than known-mean case, suggesting the regression fits better when the sample mean is estimated, possibly because the variance-of-mean term adds a predictable systematic component).
- Coefficient on $h''(0)/M_T^2$: **0.002–0.005** (smaller than in Table VII).
- Coefficient on $h(0)/T$: ranges from −78 to −88, much larger in magnitude than in Table VII. This reflects the stronger finite-sample effect of the sample mean correction.

**Table IX (variance, known mean):** $R^2 \approx 0.96$–$0.98$.
- Coefficient $\hat{\beta}_0$ on $h(0)^2 M_T/T$: ranges from 0.68 (Parzen a) to 1.71 (truncated), compared to the asymptotic value $2\int_{-1}^1 [k^*(\theta)]^2 d\theta$. For Bartlett (a), the asymptotic value is 4/3 ≈ 1.33; the estimated $\hat{\beta}_0 = 0.82$, **substantially smaller**. This is the key evidence that asymptotic approximations overstate the variance in finite samples.

**Table X (variance, unknown mean):** $R^2 \approx 0.83$–$0.92$.
- Estimated $\hat{\beta}_0$ values are uniformly **smaller** than in Table IX (known mean case). The variance of $\tilde{h}_T$ is smaller than that of $\hat{h}_T$ at the same bandwidth for most processes, while the bias is larger. This creates a different optimal bandwidth in the two cases.

### 8.5 Comparison with asymptotic values

The critical result of the response surface analysis is quantitative confirmation that asymptotic approximations are poor guides:

- **Bias:** The asymptotic formula captures the *form* of the bias through $h''(0)/M_T^2$, but the effective coefficient in small samples differs from the theoretical one. More importantly, the term $(1-\sum a_i)^{-2}/M_T^3$ — absent from asymptotic theory — is highly significant and quantitatively important for nearly-integrated processes.

- **Variance:** The asymptotic proportionality constant $2\int_{-1}^1[k^*]^2 d\theta$ (e.g., 4/3 ≈ 1.33 for Bartlett) substantially **overestimates** the finite-sample variance. Including data from $T = 250$ (which requires over a day of computation on a 486/66 MHz machine) increases $\hat{\beta}_0$ by 10%, suggesting that asymptotic values are approached slowly.

- **Unknown mean:** No asymptotic values are available for $\alpha_0$ and $\beta_0$ in the unknown-mean case, since the literature has focused on the known-mean setting. The paper's response surfaces fill this gap empirically.

---

## 9. Conclusion

The paper's main findings (Section 7):

1. **Finite-sample behavior departs substantially from asymptotic predictions.** The bandwidth-MSE relationship can be non-monotone, the optimal bandwidth can differ dramatically between asymptotic recommendations and finite-sample optimality, and the Bartlett (a) window — widely used in persistence measurement (Campbell-Mankiw 1987) — is uniformly dominated.

2. **The treatment of the sample mean matters enormously in finite samples.** For nearly-integrated and positively autocorrelated processes, the unknown-mean bias is larger in magnitude and achieves its minimum at a larger $M_T$ than the known-mean bias. For large $M_T$, the biases can diverge substantially even when the asymptotic theory predicts $O(M_T/T)$ difference.

3. **Small bandwidths are sufficient for most processes,** except when the process is nearly integrated (Group 1) or has large negative moving-average coefficients (Group 5). In these cases, the optimal bandwidth can be as large as $T/5$ or even $T-1$.

4. **No single window dominates.** The trapezoid window is best for nearly-integrated processes; Tukey–Hanning and quadratic windows perform well for negatively autocorrelated processes; for generic moderate-autocorrelation processes, the specific window choice matters little at the optimal bandwidth.

5. **Asymptotic approximations are poor guides.** Plug-in bandwidth selection rules that use asymptotic proportionality constants (as in Andrews 1991) will systematically overestimate the required bandwidth and underestimate the variance, especially in samples of size $T \le 250$.

---

## 10. Connection to Stochastic Approximation and the LSA Context

This paper is directly relevant to any setting where one needs to estimate $h(0) = \sum_{v=-\infty}^\infty R(v)$ from a dependent sequence — which is precisely the problem arising in statistical inference for stochastic approximation (SA) iterates.

**Direct connections:**

1. **Long-run variance in SA/SGD:** In the Richardson-Romberg SA context, after establishing a CLT for the (possibly extrapolated) iterates, one needs an estimator of the asymptotic covariance matrix $\Sigma$. If the noise is Markovian, the relevant quantity is the spectral density at frequency zero of the noise sequence, which is exactly $h(0)$ in this paper's notation.

2. **Batch means vs. lag windows:** Batch mean estimators (e.g., as studied in Flegal & Jones 2010) and overlapping batch means are intimately related to the truncated periodogram and Bartlett lag window respectively. The finite-sample MSE analysis in this paper applies directly.

3. **Bandwidth selection in practice:** The finding that for nearly-integrated processes, large $M_T$ is needed (up to $T/5$), while for most processes small $M_T \approx 4$ suffices, informs the choice of batch size $b_n$ in the SA context. If the Markov chain driving the noise mixes slowly (analogous to a near-unit-root AR process), one needs larger batches.

4. **Known vs. unknown mean:** In SA, the mean of the noise is typically zero by assumption (the noise is a martingale difference or mean-zero process conditioned on the past). This means the **known-mean** formulas apply, and the simpler expressions (3.2)–(3.3) can be used. This is a favorable situation compared to MCMC or econometric settings where the mean must be estimated.

5. **Response surface formulas:** Equations (6.1) and (6.2) provide practical finite-sample approximations that could be used to develop adaptive bandwidth selection rules for SA variance estimation, parameterized by the observed persistence of the noise sequence.

---

## 11. Key Formulas Summary

| Formula | Content |
|---------|---------|
| (2.1) | Power spectrum definition |
| (2.2) | General lag-window estimator |
| (2.5) | Asymptotic bias |
| (2.6) | Asymptotic variance |
| (3.1) | Known-mean estimator $\hat{h}_T$ at origin |
| (3.2) | **Exact bias of $\hat{h}_T$** |
| (3.3) | **Exact variance of $\hat{h}_T$** (Neave extension) |
| (4.1) | Unknown-mean estimator $\tilde{h}_T$ at origin |
| (4.2)–(4.6) | Key auxiliary identities for demeaned autocovariances |
| (4.7) | **Exact bias of $\tilde{h}_T$** |
| (4.10) | **Exact variance of $\tilde{h}_T$** |
| (6.1) | Response surface regression for bias |
| (6.2) | Response surface regression for variance |
| (A5)–(A9) | Computational simplifications using $C(i,j) = \sum_{k=1}^j R(k-i)$ |

---

## References (key citations within paper)

- **Neave (1971):** Original exact bias and variance for $f_T(\omega)$ at $\omega \ne 0$ with known mean and Parzen/Tukey-Hamming windows.
- **Neave (1972):** Comparison of lag window generators; optimality properties.
- **Priestley (1981):** Standard reference for spectral analysis; asymptotic bias formula (2.5) and variance (2.6).
- **Andrews (1991):** Optimal QS (quadratic spectral) kernel; automatic bandwidth selection by plug-in.
- **Newey & West (1987):** HAC estimator using Bartlett weights, requiring exactly $h(0)$ at frequency zero.
- **Campbell & Mankiw (1987):** Persistence measure using $\psi(1)$, motivating the empirical exercise.
- **Cochrane (1988):** Variance ratio statistic; another persistence measure linked to $h(0)$.
- **Politis & Romano (1995):** Bias-corrected nonparametric spectral estimation; trapezoid window.
- **Isserlis (1918):** Moment formula for products of normal variates used in variance derivation.

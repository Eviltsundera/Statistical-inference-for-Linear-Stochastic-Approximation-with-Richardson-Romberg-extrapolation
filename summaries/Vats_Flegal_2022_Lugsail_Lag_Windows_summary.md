# Detailed Review: Vats & Flegal (2022)
# "Lugsail Lag Windows for Estimating Time-Average Covariance Matrices"

**Authors:** Dootika Vats (IIT Kanpur), James M. Flegal (UC Riverside)  
**arXiv:** arXiv:1809.04541v3 [stat.CO], submitted July 2021  
**Published:** Biometrika, 2022  

---

## Overview and Significance

This paper addresses a fundamental and pervasive problem in variance estimation for dependent data: **standard lag-window estimators of the time-average covariance matrix $\Sigma$ are systematically negatively biased under positive correlation**, and this negative bias leads to severely undersized confidence regions and premature stopping of simulations.

The authors propose a new family of **lugsail lag windows** — named after the fore-and-aft, four-cornered sail suspended from a spar — that can take values **above 1**, which is a deliberate departure from all classical lag windows. This seemingly simple modification has a mathematically precise effect: it offsets the negative first-order bias by an amount that can be tuned to the correlation strength of the process. Any existing lag window can be converted into its lugsail equivalent with no additional assumptions.

The paper makes contributions in two main directions:
1. **Spectral variance (SV) estimators**: lugsail versions are studied theoretically (bias, variance, consistency) and empirically in the HAC/time-series setting.
2. **Weighted batch means (BM) estimators**: a computationally efficient lugsail version is derived and analyzed, with significantly weakened mixing conditions (from $\phi$-mixing with 12 moments to $\alpha$-mixing with 4 moments).

---

## 1. The Problem: Negative Bias of Standard Estimators

### 1.1 Setup

Let $\{Y_t\}$ be a $p$-dimensional covariance-stationary stochastic process with mean $\mu$ and lag-$s$ autocovariance matrix
$$R(s) = \mathrm{E}[(Y_t - \mu)(Y_{t+s} - \mu)^T].$$

The object of interest is the **time-average covariance matrix** (also called the long-run covariance or asymptotic variance matrix):
$$\Sigma := \sum_{s=-\infty}^{\infty} R(s).$$

This matrix appears everywhere:
- In time series: estimation of spectra and long-run variance (Hannan 1970, Priestley 1981).
- In econometrics: HAC (heteroskedastic and autocorrelation consistent) covariance matrix estimation (Andrews 1991, Newey and West 1987).
- In steady-state simulation and MCMC: $\Sigma = \lim_{n \to \infty} n \cdot \mathrm{Var}_F(\bar{Y})$ is the limiting covariance of the Monte Carlo estimator (Glynn and Whitt 1992, Chan and Yau 2017a).

### 1.2 Classical Lag-Window (Spectral Variance) Estimators

The sample lag-$s$ covariance matrix is:
$$\hat{R}(s) = \frac{1}{n} \sum_{t=1}^{n-s} (Y_t - \bar{Y})(Y_{t+s} - \bar{Y})^T.$$

Standard estimators of $\Sigma$ downweight $\hat{R}(s)$ using a **lag window** $k: \mathbb{R} \to \mathbb{R}$ satisfying $k(0) = 1$, $k(x) = k(-x)$. The spectral variance (SV) estimator with bandwidth $b$ is:
$$\dot{\Sigma}_{k,b} = \sum_{s=-(n-1)}^{n-1} k\!\left(\frac{s}{b}\right) \hat{R}(s).$$

Three canonical examples:
- **Bartlett:** $k(x) = (1-|x|)\mathbf{1}_{|x|\le 1}$
- **Tukey-Hanning (TH):** $k(x) = (\tfrac{1}{2} + \tfrac{1}{2}\cos(\pi x))\mathbf{1}_{|x|\le 1}$
- **Quadratic Spectral (QS):** $k(x) = \frac{25}{12\pi^2 x^2}\!\left(\frac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5)\right)$

All three windows are **decreasing toward zero** and satisfy $|k(x)| \le 1$, which is the standard assumption in the literature (Andrews 1991). This is precisely the root cause of the negative bias problem.

### 1.3 Two Sources of Bias

From Theorem 2 (a classical result from Hannan 1970 and Andrews 1991), the bias of $\dot{\Sigma}_{k,b}$ has two components:

$$\mathrm{Bias}(\dot{\Sigma}_{k,b}) = \underbrace{\frac{k_q}{b^q} \Gamma^{(q)}}_{\text{first-order: }O(b^{-q})} + \underbrace{o(b^{-q})}_{\text{second-order}}$$

where:
- $q \ge 1$ is the order of the lag window, determined by $\lim_{x \to 0} \frac{1 - k(x)}{|x|^q} = k_q < \infty$.
- $\Gamma^{(q)} := -\sum_{s=1}^{\infty} s^q [R(s) + R(s)^T]$ is a matrix characterizing the correlation structure.

**Key observation:** For a positively correlated process, $\Gamma^{(q)}$ has **negative diagonal entries**, so the first-order bias $k_q \Gamma^{(q)} / b^q$ is **negative** regardless of the choice of $b$. Among the 17 standard estimators surveyed by Chan and Yau (2017b), 15 have negative first-order bias under positive correlation. This causes:
- Confidence regions that are too small (undercoverage).
- MCMC simulations terminated prematurely (since $\widehat{\mathrm{ESS}} = n \cdot |\hat{\mathrm{Var}}(Y_1)| / |\hat{\Sigma}|$ is overestimated when $\hat{\Sigma}$ is underestimated).

---

## 2. Lugsail Lag Windows: Construction and Intuition

### 2.1 The Core Idea

The key insight: if the standard window satisfies $k(x) \le 1$ and this causes negative bias, then one can construct a modified window that **takes values above 1** near zero to counteract the bias.

Formally, for $r \ge 1$ and a sequence $c_n \in [0,1)$ with $c_n \to c$ as $n \to \infty$, define the **lugsail version** of any lag window $k$ as:

$$k_L(x) = \frac{1}{1-c_n} k(x) - \frac{c_n}{1-c_n} k(rx). \tag{2}$$

**Intuition:** $k_L$ is a linear combination of the original window at scale 1 and the same window compressed by factor $r$. Near $x = 0$:
- $k(x) \approx 1$ and $k(rx) \approx 1$, so $k_L(0) = \frac{1}{1-c_n}(1) - \frac{c_n}{1-c_n}(1) = 1$ — the normalization condition is preserved.
- For small but nonzero $x$: $k(rx)$ decays faster than $k(x)$ (since $rx > x$), so $k_L(x) > k(x)$. This means the window **overweights the leading lag covariances** relative to the standard window.

Special cases:
- $c_n = 0$ or $r = 1$: recovers the original lag window $k$.
- $r = 1/c_n$ with the Bartlett window: gives the **flat-top Bartlett window** (Politis and Romano 1995, 1996).
- The lugsail window can therefore be seen as a generalization of the flat-top window idea.

### 2.2 Three Regimes of Lugsail Windows

The authors identify three practically important variants based on the correlation strength:

**Zero Lugsail** (moderate correlation, $\rho \in [0, 0.7)$ for AR(1)):
- $c_n = r^{-q}$ (so that $c_n r^q = 1$), yielding **zero first-order bias**.
- Recommended: $r = 2$, $c_n = r^{-q} = 2^{-q}$.
- The first-order term $\frac{1 - c_n r^q}{1 - c_n} \cdot \frac{k_q}{b^q} \Gamma^{(q)} = 0$.

**Adapt Lugsail** (moderate to high correlation, $\rho \in [0.7, 0.95)$):
- $c_n$ decays slowly to $r^{-q}$ as $n \to \infty$: specifically,
$$c_n^{\mathrm{a}} = \frac{\log(n) - \log(b) + 1}{r^q(\log(n) - \log(b)) + 1}.$$
- For small $n$: $c_n > r^{-q}$, so first-order bias is positive (helps offset the large second-order negative bias).
- For large $n$: $c_n \to r^{-q}$, converging to the zero lugsail as the second-order term becomes negligible.
- This is analogous in spirit to jackknifed estimators (Dingeç et al. 2015).

**Over Lugsail** (high to extreme correlation, $\rho \in [0.95, 1)$):
- $c_n = c^{\mathrm{o}} = \frac{2}{1 + r^q}$ (constant), deliberately inducing a **positive first-order bias of magnitude $+m$** where $-m$ is the first-order bias of the original estimator.
- Recommended: $r = 3$, $c^{\mathrm{o}} = 2/(1 + 3^q)$. For $q=1$: $c^{\mathrm{o}} = 1/2$; for $q=2$: $c^{\mathrm{o}} = 1/5$.
- Reasoning: in MCMC/steady-state simulation where correlation is extreme, it is better to overestimate $\Sigma$ (conservative inference) than to underestimate it (Simonoff 1993). Additional samples are cheap.

| Correlation | Lugsail type | $r$ | $c_n$ |
|---|---|---|---|
| Moderate $(\rho \in [0, 0.7))$ | Zero | 2 | $r^{-q}$ |
| Moderate to High $(\rho \in [0.7, 0.95))$ | Adapt | 2 | $c_n^{\mathrm{a}}$ |
| High to Extreme $(\rho \in [0.95, 1))$ | Over | 3 | $c^{\mathrm{o}} = 2/(1+r^q)$ |

### 2.3 The Lugsail SV Estimator

Substituting $k_L$ into the SV formula (3) gives:

$$\dot{\Sigma}_{k,L} = \frac{1}{1-c_n} \dot{\Sigma}_{k,b} - \frac{c_n}{1-c_n} \dot{\Sigma}_{k,b/r}. \tag{4}$$

This is a **linear combination of two standard SV estimators** — one with bandwidth $b$ and one with compressed bandwidth $b/r$. There are no new computational primitives required; the lugsail estimator is trivially computed from two runs of any existing SV code.

**Theorem 1** (Consistency): $\dot{\Sigma}_{k,L}$ inherits (strong) consistency from $\dot{\Sigma}_{k,b}$.

*Proof sketch:* Since $c_n \to c \in [0,1)$, the weights $\frac{1}{1-c_n}$ and $\frac{c_n}{1-c_n}$ converge to finite limits, so consistency follows from the consistency of $\dot{\Sigma}_{k,b}$ and $\dot{\Sigma}_{k,b/r}$.

---

## 3. Theoretical Results for SV Estimators

### 3.1 Mixing Assumption

**Assumption 1:** For some $\delta > 0$ and $q \ge 1$: $\mathrm{E}\|Y_1\|^{2+\delta} < \infty$ and there exists $\epsilon > 0$ such that $\{X_t\}$ is $\alpha$-mixing with $\alpha(n) = o(n^{-(q+1+\epsilon)(1+2/\delta)})$.

Here $\alpha$-mixing (strong mixing) is defined by
$$\alpha(n) = \sup_{s \ge 1} \sup_{A \in \mathcal{F}_1^s,\, B \in \mathcal{F}_{s+n}^\infty} |P(A \cap B) - P(A)P(B)|.$$

This is considerably weaker than $\phi$-mixing (uniform mixing), which is the traditional assumption in the batch means literature and which is satisfied only for **uniformly ergodic** Markov chains. The $\alpha$-mixing assumption encompasses **polynomially ergodic** Markov chains (relevant for many MCMC algorithms and for Markovian SA with polynomial step sizes).

### 3.2 Bias and Variance of Standard SV (Theorem 2)

Under Assumption 1 with $b^{q+1}/n \to 0$, $k$ continuous and bounded, and $\lim_{x\to 0} \frac{1-k(x)}{|x|^q} = k_q < \infty$:

$$\mathrm{Bias}(\dot{\Sigma}_{k,b}) = \frac{k_q}{b^q} \Gamma^{(q)} + o\!\left(\frac{1}{b^q}\right),$$

$$\frac{n}{b} \mathrm{Var}\!\left(\dot{\Sigma}_{k,b}^{ij}\right) = [\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2] \int_{-\infty}^{\infty} k^2(x)\,dx + o(1).$$

This is a classical result from Hannan (1970) and Andrews (1991), generalized here to the multivariate case with $\mu$ estimated rather than known. The condition $b^{q+1}/n \to 0$ (instead of $b^q/n \to 0$) accounts for centering by $\bar{Y}$.

**Remark on Andrews (1991):** The original proof assumes $|k(x)| \le 1$. The authors note this can be relaxed to $|k(x)| \le d$ for any $0 < d < \infty$. Since $|k_L(x)| \le (1-c_n)^{-1}$ (bounded uniformly in $n$), the theorem extends to lugsail windows.

### 3.3 Bias and Variance of Lugsail SV (Corollary 1)

Under the conditions of Theorem 2:

$$\mathrm{Bias}(\dot{\Sigma}_{k,L}) = \frac{1 - c_n r^q}{1 - c_n} \cdot \frac{k_q}{b^q} \Gamma^{(q)} + o\!\left(\frac{1}{b^q}\right),$$

$$\frac{n}{b} \mathrm{Var}\!\left(\dot{\Sigma}_{k,L}^{ij}\right) = [\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2] \int_{-\infty}^{\infty} k_L^2(x)\,dx + o(1).$$

**Bias analysis:** The multiplicative factor on the first-order bias term is $\frac{1 - c_n r^q}{1 - c_n}$:
- When $c_n r^q = 1$: zero first-order bias (zero lugsail).
- When $c_n r^q > 1$: positive first-order bias (over lugsail).
- When $c_n r^q < 1$: reduced negative bias relative to original.

**Variance analysis:** The variance increases by the ratio $\int k_L^2 / \int k^2$. This is the unavoidable bias-variance tradeoff. For the recommended settings, the variance increase is bounded and often modest (see Appendix A and Table 4 in the paper), especially for TH and QS windows.

**Why MSE is the wrong criterion:** Simonoff (1993) argues that for variance estimators used in hypothesis testing or CI construction, MSE conflates bias and variance in a misleading way. Negative bias causes undersized tests (which is bad), while positive bias causes oversized tests (conservative, less bad). Lugsail estimators may have slightly larger MSE but substantially better coverage.

### 3.4 Concrete Bias Expressions for Specific Windows (Appendix A)

**(a) Bartlett** ($q=1$, $k_1=1$): standard bias is $\Gamma/b$; lugsail bias is $\frac{\Gamma}{b} \cdot \frac{1-rc_n}{1-c_n}$.

Variance integral: $\int_{-\infty}^\infty k_L^2(x)\,dx = \frac{2}{3(1-c_n)^2}\!\left(1 + \frac{c_n^2}{r} - \frac{3c_n}{r} + \frac{c_n}{r^2}\right)$.

**(b) Tukey-Hanning** ($q=2$, $k_2=\pi^2/4$): standard bias is $\frac{\pi^2}{4}\frac{\Gamma^{(2)}}{b^2}$; lugsail bias is $\frac{\pi^2 \Gamma^{(2)}}{4b^2} \cdot \frac{1-c_n r^2}{1-c_n}$.

**(c) Quadratic Spectral** ($q=2$, $k_2=1.4212$): standard bias is $1.4212 \frac{\Gamma^{(2)}}{b^2}$; lugsail bias is $1.4212 \frac{\Gamma^{(2)}}{b^2} \cdot \frac{1-c_n r^2}{1-c_n}$.

Table 4 in the paper computes the normalized variance $\frac{n}{b} \mathrm{Var}(\hat{\Sigma}^{ij}) / (\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2)$ for all three windows under original, zero, adapt, and over lugsail settings. For the TH window, the zero lugsail variance is actually **smaller** than the original (0.964 vs. 0.750 for $a = b/n = 10^{-2}$), which is a notable exception to the usual variance-inflation narrative.

---

## 4. Weighted Batch Means Estimators

### 4.1 Motivation: Computational Infeasibility of SV

SV estimators require $O(nb)$ operations (sum over $s$ from $-(n-1)$ to $n-1$). With optimal bandwidth $b \propto n^\nu$ for $\nu > 0$, this is $O(n^{1+\nu})$, which is prohibitively slow for large MCMC outputs (e.g., $n = 10^5$, $p = 50$: the QS SV estimator takes **31877 seconds** per the paper's Table 3 vs. **0.078 seconds** for the BM estimator).

### 4.2 Standard Weighted BM Estimator

For $s = 1, \ldots, b < n$, define $\Delta_2(s) = k((s-1)/b) - 2k(s/b) + k((s+1)/b)$ (the second difference of $k$). Let $a_s = \lfloor n/s \rfloor$ and for $l = 0, \ldots, a_s - 1$, let $\bar{Y}_l(s) = s^{-1}\sum_{t=1}^s Y_{ls+t}$.

The **weighted BM estimator** is:
$$\hat{\Sigma}_{k,b} = \sum_{s=1}^{b} \frac{1}{a_s - 1} \sum_{l=0}^{a_s-1} s^2 \Delta_2(s)(\bar{Y}_l(s) - \bar{Y})(\bar{Y}_l(s) - \bar{Y})^T. \tag{5}$$

For piecewise linear $k$ (i.e., Bartlett), $\Delta_2 = 0$ almost everywhere, reducing (5) to the standard BM estimator:
$$\hat{\Sigma}_b = \frac{b}{a-1} \sum_{l=0}^{a-1} (\bar{Y}_l(b) - \bar{Y})(\bar{Y}_l(b) - \bar{Y})^T,$$
where $n = ab$, $a$ is the number of batches of size $b$, and $\bar{Y}_l(b)$ is the mean of batch $l$.

### 4.3 Lugsail Weighted BM Estimator

Plugging $k_L$ into (5) gives:
$$\hat{\Sigma}_{k,L} = \frac{1}{1-c_n} \hat{\Sigma}_{k,b} - \frac{c_n}{1-c_n} \hat{\Sigma}_{k,b/r}. \tag{6}$$

For the Bartlett case specifically, this yields:
$$\hat{\Sigma}_L = \frac{1}{1-c_n} \hat{\Sigma}_b - \frac{c_n}{1-c_n} \hat{\Sigma}_{b/r}. \tag{7}$$

Again: just run two BM estimators with batch sizes $b$ and $b/r$, combine linearly. Computationally trivial once the batch means are stored.

**Theorem 3** (Consistency): $\hat{\Sigma}_{k,L}$ inherits (strong) consistency from $\hat{\Sigma}_{k,b}$.

### 4.4 Bias of BM Estimator (Theorem 4)

**Assumption 2:** The integer sequence $b$ satisfies $b \to \infty$ and $n/b \to \infty$ as $n \to \infty$, with both $b$ and $n/b$ nondecreasing.

Under Assumption 1 with $q = 1$:

$$\mathrm{Bias}(\hat{\Sigma}_b) = \frac{\Gamma}{b} + o\!\left(\frac{1}{b}\right).$$

As a consequence, the **lugsail BM estimator** has bias:

$$\mathrm{Bias}(\hat{\Sigma}_L) = \frac{\Gamma}{b} \cdot \frac{1 - rc_n}{1-c_n} + o\!\left(\frac{1}{b}\right).$$

And under Assumption 2: $\lim_{n\to\infty} \mathrm{Bias}(\hat{\Sigma}_L) = 0$ (consistent).

**Two contributions of Theorem 4:**
1. **Multivariate:** Prior work (Chien et al. 1997, Flegal and Jones 2010, Song and Schmeiser 1995) only handled $p = 1$.
2. **Weaker mixing:** Prior results required $\phi$-mixing with 12 finite moments. Theorem 4 requires only $\alpha$-mixing with 4 finite moments ($q=1$, $\delta > 0$ from Assumption 1 requires $\mathrm{E}\|Y_1\|^{2+\delta} < \infty$, and with $q=1$ the moment condition becomes $4+\delta$ moments, so effectively 4 moments). This is critical because $\phi$-mixing implies **uniform ergodicity** of the Markov chain, while $\alpha$-mixing encompasses **polynomially ergodic** chains, which arise naturally with polynomial step sizes in SA.

The proof (Appendix B) proceeds by expressing $\mathrm{E}_F[\hat{\Sigma}_b^{ij}]$ in terms of $\mathrm{Cov}_F(\bar{Y}_1(b)^{(i)}, \bar{Y}_1(b)^{(j)})$ and $\mathrm{Cov}_F(\bar{Y}^{(i)}, \bar{Y}^{(j)})$, using a result from Song and Schmeiser (1995) connecting these to $\Sigma_{ij}$ and $\Gamma_{ij}$.

### 4.5 Variance of Lugsail BM Estimator (Theorem 6)

To establish variance, a **strong invariance principle** is needed.

**Theorem 5** (Kuelbs and Philipp 1980): Under Assumption 1 with $q = 1$, there exists a $p \times p$ lower triangular matrix $L$ with $LL^T = \Sigma$, a rate $\psi(n) = n^{1/2-\lambda}$ ($\lambda > 0$), and a finite random variable $D$ such that with probability 1:
$$\left\|\sum_{t=1}^n Y_t - n\mu_g - LB(n)\right\| < D(\omega)\psi(n),$$
where $B(n)$ is a $p$-dimensional standard Brownian motion. This strong approximation of partial sums by Brownian motion is the key technical tool.

**Theorem 6:** Under Theorem 5's assumptions, with $\mathrm{E}D^4 < \infty$, $\mathrm{E}_F\|Y_1\|^4 < \infty$, $b$ satisfying Assumption 2, and $b^{-1}\psi^2(n)\log n \to 0$:

$$\frac{n}{b} \mathrm{Var}(\hat{\Sigma}_L^{ij}) = \left[\frac{1}{r} + \frac{r-1}{r(1-c_n)^2}\right](\Sigma_{ij}^2 + \Sigma_{ii}\Sigma_{jj}) + o(1).$$

**Interpretation:**
- For $r = 1$ (original BM): variance coefficient is $1 \cdot (\Sigma_{ij}^2 + \Sigma_{ii}\Sigma_{jj})$.
- For $r > 1$ (lugsail BM): variance coefficient is $\frac{1}{r} + \frac{r-1}{r(1-c_n)^2} > 1$ — variance increases.
- As $c_n \to 0$: approaches $\frac{1}{r} + \frac{r-1}{r} = 1$ (variance unaffected).
- The zero lugsail with $c_n = r^{-q}$ achieves the bias elimination at the cost of a variance increase of $\frac{r-1}{r(1-r^{-q})^2}$.

The proof (Appendix C) uses the Brownian motion equivalent of $\hat{\Sigma}_L$ (replacing partial sums by Brownian paths), and applies Proposition 1 (moment formula for bivariate normals: $\mathrm{E}[X^2 Y^2] = 2l_{12}^2 + l_{11}l_{22}$) and Proposition 2 (Isserlis theorem for fourth moments of jointly normal vectors).

---

## 5. Positive Definiteness Adjustment

A practical concern: TH and lugsail estimators can have **negative eigenvalues** in finite samples (they are not positive semidefinite by construction, unlike Bartlett/QS/BM which are PSD). The authors provide a general adjustment:

Given any estimator $\hat{\Sigma}_n$:
1. Compute $\hat{V} = \mathrm{diag}(\hat{\Sigma}_n)$ (diagonal of univariate variance estimates).
2. Form the estimated correlation matrix $\hat{C}_n = \hat{V}^{-1/2} \hat{\Sigma}_n \hat{V}^{-1/2}$.
3. Eigendecompose: $\hat{C}_n = P\hat{D}_n P^T$.
4. Replace negative eigenvalues: $\hat{d}_i^+ = \max\{\hat{d}_i, \epsilon n^{-u}\}$ for $\epsilon > 0$, $u > 0$. As $n \to \infty$, $\epsilon n^{-u} \to 0$ and $\hat{d}_i \to d_i > 0$, so the adjustment vanishes asymptotically.
5. Return $\hat{\Sigma}_n^+ = \hat{V}^{1/2} P^T \hat{D}^+ P \hat{V}^{1/2}$.

This mirrors the approach of Jentsch and Politis (2015). Recommended defaults: $\epsilon = \sqrt{\log(n)/p}$, $u = 9/10$.

---

## 6. Numerical Examples

### 6.1 HAC Estimation (Linear Regression)

**Model:** $y_t = x_t^T \beta + u_t$ for $t = 1, \ldots, n$, where $\{u_t\}$ and $\{x_t\}$ are independent AR(1) processes with coefficients $\rho_u = \rho_x \in \{0.50, 0.70, 0.90\}$. Dimension $p = 5$.

**Metric:** Coverage probability of 90% asymptotic confidence regions over 1000 replications, for $n \in \{500, 1000\}$.

**Results** (Table 2): Lugsail windows consistently improve coverage.
- At $\rho = 0.50$, $n=500$: Bartlett 0.825 → Over 0.861; QS 0.836 → Over 0.864.
- At $\rho = 0.90$, $n=500$: Bartlett 0.553 → Over 0.573; QS 0.588 → Over 0.596.
- The improvement is monotone in correlation strength and the over lugsail wins in all cases.
- Larger $n$ helps but does not eliminate the benefit of lugsail adjustment for $\rho = 0.90$.

**Figure 3** shows average relative bias on diagonals. Lugsail versions have less negative bias for all three window types. The over lugsail slightly overestimates for moderate $\rho$ — consistent with the theory.

**Recommendation for HAC:** Use zero lugsail (moderate correlation is typical in econometric applications).

### 6.2 Vector Autoregression (Time Series)

**Model:** $Y_t = \Phi Y_{t-1} + \epsilon_t$ with $\Phi = \rho I_p$, $\rho = 0.95$ (high correlation), $\epsilon_t \sim N_p(0, \Omega)$. Dimension $p = 10$.

This model is geometrically ergodic (spectral norm of $\Phi < 1$), and the true $\Sigma$ is available in closed form (Dai and Jones 2017).

**Results** (Figure 4, Left): Relative bias on diagonals over 1000 replications.
- Original BM: strongly negative, especially for small $n$.
- Zero lugsail and adapt lugsail: converge to zero from below (still negatively biased for moderate $n$).
- **Over lugsail: converges to zero from above** — positive bias that shrinks with $n$.

**ESS analysis** (Figure 4, Right): Running plot of $\widehat{\mathrm{ESS}}/n = (|\widehat{\mathrm{Var}}(Y_1)| / |\hat{\Sigma}|)^{1/p}$ over $10^6$ MCMC iterations.
- BM, zero lugsail, adapt lugsail: all **overestimate** $\widehat{\mathrm{ESS}}/n$ (because $|\hat{\Sigma}|$ is underestimated when $\hat{\Sigma}$ has negative bias).
- **Over lugsail: converges to the truth from below** — underestimates ESS/n, which is conservative and prevents premature termination.

This is the critical result for MCMC diagnostics: a negatively biased $\hat{\Sigma}$ causes one to believe the chain has more effective samples than it does, potentially stopping too early.

**Compute time** (Table 3): At $n = 10^5$, $p = 50$:
- Over BM: **0.078 seconds**.
- Over Bartlett SV: 94.3 seconds.
- Over TH SV: 55.7 seconds.
- Over QS SV: **31,877 seconds** (≈ 8.8 hours!).

For MCMC/steady-state simulation, the lugsail BM is the only practical choice.

### 6.3 Bayesian Logistic Regression (Real Data)

**Data:** Framingham Heart Study (Kaggle), 4238 observations, binary response (10-year CHD risk), 15 covariates → 18-dimensional parameter $\beta$.

**Method:** Random walk Metropolis-Hastings with normal proposal from MCMCpack in R. Prior: $\beta \sim N_{18}(0, 100 I_{18})$.

**Results** (Figure 5): Running plot of $\widehat{\mathrm{ESS}}/n$ from $10^6$ MCMC iterations.
- The true value is unknown, but all four estimators appear to converge to the same quantity.
- Original BM, zero lugsail, adapt lugsail: converge from **above** (overestimate early → risk of premature termination).
- Over lugsail: converges from **below** (conservative → correct termination behavior).

---

## 7. Discussion and Open Problems

1. **Any lag window has a lugsail version**: The construction works for all lag windows, including those not analyzed here (see Anderson 1971 for a comprehensive list). Lugsail windows are also compatible with existing bias-adjustment methods (e.g., Kiefer and Vogelsang 2002b).

2. **MSE trade-off**: For fixed $b$, lugsail increases asymptotic variance, potentially increasing MSE. However, empirically lugsail estimators can use **smaller $b$**, which reduces variance enough to offset the theoretical increase. Optimal choice of $b$ for lugsail estimators is an open problem.

3. **First lag window to exceed 1**: The authors believe lugsail is the first systematic family of lag windows that can take values above 1. Berg and Politis (2009) explicitly claim "there is no benefit to allowing the window to have values larger than 1" — this paper directly refutes that claim.

4. **Scope**: The paper focuses on consistent nonparametric estimators. Inconsistent (fixed-$b$ asymptotics) and parametric estimators (den Haan and Levin 1996, 2000; Müller 2007, 2014) are a separate topic.

5. **Application domains**: Results are directly applicable wherever $\Sigma = \sum_{s=-\infty}^\infty R(s)$ appears: time series spectrum estimation, HAC econometrics, MCMC output analysis, steady-state simulation output analysis. The authors note potential applicability in signal processing (Boashash 1992) and genomic signal processing (Gunawan 2008).

---

## 8. Relevance to LSA / Richardson-Romberg Setting

The lugsail framework is directly applicable to variance estimation for **linear stochastic approximation (LSA) with Markovian noise**, as studied in the thesis context:

1. **Markov chain output from LSA iterates**: The sequence $\{\theta_n\}$ (or centered versions $\{\sqrt{n}(\theta_n - \theta^*)\}$) plays the role of $\{Y_t\}$. The batch means or SV estimator of the asymptotic variance $\Sigma$ of the CLT for LSA can be lugsail-adjusted.

2. **Weakened mixing conditions**: Theorem 4's $\alpha$-mixing assumption is particularly relevant. LSA iterates driven by geometrically ergodic Markov chains are $\alpha$-mixing, but not necessarily $\phi$-mixing (the chain may not be uniformly ergodic). The weaker condition directly applies.

3. **Richardson-Romberg (RR) extrapolation**: In the RR setting, one uses two iterate sequences with step sizes $\gamma$ and $\gamma/2$ and forms a linear combination $2\theta_n^{(\gamma/2)} - \theta_n^{(\gamma)}$. The time-average covariance of this combined sequence must be estimated. If both component sequences have $\hat{\Sigma}$ estimators with negative bias, the combined lugsail estimator (which is itself a linear combination of BM estimators) naturally addresses this.

4. **Practical recommendation for RR-MCMC output analysis**: Given the high correlation typical in MCMC (especially with small step sizes), the **over lugsail BM estimator** is the recommended choice for constructing confidence intervals and effective sample size estimates for RR iterates.

5. **The ESS connection**: If $\hat{\Sigma}$ is negatively biased (as standard BM/SV estimators are), then $\widehat{\mathrm{ESS}} = n \cdot |\widehat{\mathrm{Var}}(Y_1)| / |\hat{\Sigma}|$ is overestimated, causing premature stopping of MCMC. With the lugsail over estimator, ESS converges from below, which is the safe direction.

---

## Key Formulas Reference

| Object | Formula |
|---|---|
| Lugsail window | $k_L(x) = \frac{1}{1-c_n}k(x) - \frac{c_n}{1-c_n}k(rx)$ |
| Lugsail SV estimator | $\dot{\Sigma}_{k,L} = \frac{1}{1-c_n}\dot{\Sigma}_{k,b} - \frac{c_n}{1-c_n}\dot{\Sigma}_{k,b/r}$ |
| Lugsail BM estimator | $\hat{\Sigma}_L = \frac{1}{1-c_n}\hat{\Sigma}_b - \frac{c_n}{1-c_n}\hat{\Sigma}_{b/r}$ |
| First-order bias (SV) | $\frac{1-c_n r^q}{1-c_n} \cdot \frac{k_q}{b^q} \Gamma^{(q)}$ |
| First-order bias (BM) | $\frac{1-rc_n}{1-c_n} \cdot \frac{\Gamma}{b}$ |
| Variance (BM) | $\frac{b}{n}\!\left[\frac{1}{r}+\frac{r-1}{r(1-c_n)^2}\right](\Sigma_{ij}^2 + \Sigma_{ii}\Sigma_{jj})$ |
| Zero lugsail $c_n$ | $r^{-q}$ |
| Adapt lugsail $c_n$ | $\frac{\log n - \log b + 1}{r^q(\log n - \log b)+1}$ |
| Over lugsail $c_n$ | $\frac{2}{1+r^q}$ |
| PD adjustment | $\hat{\Sigma}_n^+ = \hat{V}^{1/2}P^T\hat{D}^+P\hat{V}^{1/2}$, $\hat{d}_i^+=\max\{\hat{d}_i,\epsilon n^{-u}\}$ |

# Detailed Review: Flegal & Jones (2010)
# "Batch Means and Spectral Variance Estimators in Markov Chain Monte Carlo"

**Journal:** The Annals of Statistics, Vol. 38, No. 2, pp. 1034–1070  
**DOI:** 10.1214/09-AOS735  
**Authors:** James M. Flegal (UC Riverside), Galin L. Jones (Univ. of Minnesota)  
**arXiv preprint:** arXiv:0811.1729

---

## Overview and Significance

This paper is a foundational reference for the theoretical analysis of long-run variance estimation in Markov chain Monte Carlo (MCMC). It provides the first rigorous treatment of strong consistency and mean-square consistency for batch means (BM), overlapping batch means (OBM), and spectral variance (SV) estimators under **geometric ergodicity** — a condition substantially weaker than the **uniform ergodicity** assumed in all prior work on these estimators. The paper also derives optimal batch sizes in terms of MSE and gives practical recommendations backed by empirical experiments.

For researchers working on statistical inference in stochastic approximation (SA) or SGD with Markovian noise, this paper is the essential starting point for any discussion of long-run covariance estimation, since the same estimators and the same bias-variance tradeoff structure appear when one replaces the MCMC chain output with SA iterates.

---

## 1. Problem Setup

### 1.1 The MCMC estimation problem

Let $\pi$ be a probability distribution on a state space $\mathsf{X}$, and let $g: \mathsf{X} \to \mathbb{R}$ be a $\pi$-integrable function. The goal is to compute

$$E_\pi g := \int_\mathsf{X} g(x)\,\pi(dx).$$

When $\pi$ is too complex for analytical or importance-sampling approaches, one simulates a Harris ergodic (aperiodic, $\pi$-irreducible, positive Harris recurrent) Markov chain $X = \{X_1, X_2, \ldots\}$ with invariant distribution $\pi$. By the ergodic theorem, with probability one and for any initial distribution,

$$\bar{g}_n := \frac{1}{n}\sum_{i=1}^n g(X_i) \to E_\pi g \quad \text{as } n \to \infty.$$

### 1.2 The Markov chain CLT and asymptotic variance

Under conditions ensuring a Markov chain central limit theorem (MCCLT), there exists a constant $\sigma_g^2 \in (0,\infty)$ such that

$$\sqrt{n}\,(\bar{g}_n - E_\pi g) \xrightarrow{d} \mathrm{N}(0, \sigma_g^2) \quad \text{as } n \to \infty.$$

The quantity $\sigma_g^2$ is the **asymptotic variance** (or long-run variance). It equals the spectral density of the process $\{g(X_i)\}$ at frequency zero:

$$\sigma_g^2 = \sum_{k=-\infty}^{\infty} \gamma(k),$$

where $\gamma(k) = E_\pi[Y_0 Y_k]$ with $Y_i = g(X_i) - E_\pi g$. Due to serial dependence in the chain, $\sigma_g^2 \neq \mathrm{Var}_\pi(g)$ in general, and the correction can be substantial when the chain mixes slowly.

### 1.3 Why estimating $\sigma_g^2$ matters

A consistent estimator $\hat{\sigma}_n^2 \approx \sigma_g^2$ is needed for two main purposes:

**1. Fixed-time confidence intervals.** After running $n$ iterations, one reports

$$\bar{g}_n \pm t_* \frac{\hat{\sigma}_n}{\sqrt{n}},$$

where $t_*$ is the appropriate Student-$t$ quantile. The width is an independent quality assessment of the point estimate.

**2. Fixed-width stopping rules.** Rather than fixing $n$ in advance, one runs the chain until the first $n$ satisfying

$$t_*\frac{\hat{\sigma}_n}{\sqrt{n}} + p(n) < \epsilon,$$

where $\epsilon > 0$ is the desired half-width and $p(n) = o(n^{-1/2})$ ensures the minimum effort $n^*$ for small $\epsilon$. Glynn and Whitt (1992) established that, when a functional CLT holds and $\hat{\sigma}_n^2 \to \sigma_g^2$ with probability 1, this rule produces asymptotically valid intervals as $\epsilon \to 0$.

### 1.4 Why naive estimators fail

The sample variance $\frac{1}{n}\sum_{i=1}^n (g(X_i) - \bar{g}_n)^2$ estimates $\mathrm{Var}_\pi(g)$, not $\sigma_g^2$. For a positively correlated chain, $\sigma_g^2 > \mathrm{Var}_\pi(g)$, so naive CIs are too narrow and have coverage below the nominal level. Specialized methods that account for the dependence structure are required.

---

## 2. Spectral Variance Estimators

### 2.1 Sample autocovariances

Define the lag-$s$ sample autocovariance as

$$\gamma_n(s) = \gamma_n(-s) := n^{-1}\sum_{t=1}^{n-s}(Y_t - \bar{Y}_n)(Y_{t+s} - \bar{Y}_n),$$

where $\bar{Y}_n = n^{-1}\sum_{i=1}^n Y_i$. If $E_\pi g^2 < \infty$, then $\gamma_n(s) \to \gamma(s)$ w.p.\ 1 as $n \to \infty$.

A naive estimator $\hat{\sigma}^2 = \sum_{s=-(n-1)}^{n-1} \gamma_n(s)$ is inconsistent (Anderson 1994, Bratley et al.\ 1987). Instead, one uses a truncated and weighted version.

### 2.2 The spectral variance estimator

$$\hat{\sigma}_S^2 := \sum_{s=-(b_n-1)}^{b_n-1} w_n(s)\,\gamma_n(s),$$

where:
- $w_n(\cdot)$ is the **lag window**, an even function satisfying $|w_n(s)| \le 1$, $w_n(0)=1$, $w_n(s)=0$ for $|s|\ge b_n$;
- $b_n$ is the **truncation point** (bandwidth), a sequence with $b_n \to \infty$ and $n/b_n \to \infty$.

This is a weighted sum of sample autocovariances out to lag $b_n - 1$, and it is precisely the estimator of the spectral density of $\{Y_i\}$ at frequency zero.

### 2.3 Lag windows in use

The paper analyzes several lag windows. The key regularity objects are the first and second discrete differences:

$$\Delta_1 w_n(k) := w_n(k-1) - w_n(k), \qquad \Delta_2 w_n(k) := w_n(k-1) - 2w_n(k) + w_n(k+1).$$

| Window name | Formula for $w_n(k)$, $|k| < b_n$ | Notes |
|---|---|---|
| Simple truncation | $1$ | Fails condition (d) of Theorem 1; inconsistent |
| **Bartlett (modified)** | $1 - |k|/b_n$ | $\Delta_2 w_n = 0$ for $k < b_n$; closely related to OBM |
| **Tukey–Hanning (TH)** | $\frac{1}{2}(1 + \cos(\pi k/b_n))$ | Twice differentiable; **recommended by authors** |
| Blackman–Tukey (gen.) | $1 - 2a + 2a\cos(\pi|k|/b_n)$, $a > 0$ | TH is the case $a = 1/4$ |
| Parzen ($q\ge 1$) | $[1 - |k|^q/b_n^q]$ | $q=1$ is modified Bartlett; $q\ge 2$ easier to verify |
| Scale-modified Bartlett | $[1 - \lambda|k|/b_n]$, $\lambda \ne 1$ | **Fails** condition (d); not recommended |

The Tukey–Hanning window is twice continuously differentiable on $[0,1]$, which is the key property invoked in Lemma 7 (Appendix B.2) to verify the technical conditions of Theorem 1.

---

## 3. Main Results for Spectral Variance Estimation

### 3.1 Standing assumptions

**Assumption 1** (Lag window): $w_n$ is even, $|w_n| \le 1$, $w_n(0)=1$, $w_n(s)=0$ for $|s|\ge b_n$.

**Assumption 2** (Bandwidth): $b_n \to \infty$, $n/b_n \to \infty$, both $b_n$ and $n/b_n$ nondecreasing.

**Assumption 3** (Minorization): There exists $s:\mathsf{X}\to[0,1]$ and a probability measure $Q$ such that $P(x,\cdot) \ge s(x)Q(\cdot)$ for all $x \in \mathsf{X}$.

Assumption 3 is technically needed for the proof via the split-chain construction, but the authors note (Remark 2) it is not required in practice: by fundamental Markov chain theory (Meyn–Tweedie, Ch.\ 5), geometric ergodicity implies an $n_0$-step minorization exists, so one can always work with $P^{n_0}$.

### 3.2 Theorem 1: Strong consistency of SV

> **Theorem 1.** Let $X$ be a geometrically ergodic Markov chain with invariant distribution $\pi$ and $g:\mathsf{X}\to\mathbb{R}$ a Borel function with $E_\pi|g|^{4+\delta+\epsilon}<\infty$ for some $\delta,\epsilon>0$. Suppose Assumptions 1, 2, 3 hold. Let $\alpha = 1/(4+\delta)$. Further suppose:
>
> (a) $b_n n^{-1}\sum_{k=1}^{b_n} k|\Delta_1 w_n(k)| \to 0$;  
> (b) $\sum_n (b_n/n)^c < \infty$ for some constant $c\ge 1$;  
> (c) $b_n n^{-1}\log n \to 0$;  
> (d) $b_n n^{2\alpha}(\log n)^3\!\left(\sum_{k=1}^{b_n}|\Delta_2 w_n(k)|\right)^2 \to 0$, and $n^{2\alpha}(\log n)^2 \sum_{k=1}^{b_n}|\Delta_2 w_n(k)| \to 0$;  
> (e) $b_n^{-1}n^{2\alpha}\log n \to 0$.  
>
> Then for any initial distribution, $\hat{\sigma}_S^2 \to \sigma_g^2$ with probability 1 as $n\to\infty$.

**Remark 1.** Taking $b_n = \lfloor n^\nu \rfloor$ for any $0 < \nu < 1$ automatically satisfies conditions (b) and (c).

**Remark 3.** For the Tukey–Hanning window with $b_n = \lfloor n^\nu \rfloor$:
- Condition (a) holds if $b_n^2/n \to 0$, i.e., $\nu < 1/2$.
- Condition (d) holds if $b_n^{-1}n^{2\alpha}(\log n)^3 \to 0$.

**Comparison with Damerdji (1991, 1994).** Prior work by Damerdji required:
- Uniform ergodicity (much stronger than geometric ergodicity);
- $E_\pi|g|^{12} < \infty$ (far stronger than $4+\delta$);
- A condition on $\Delta_2 w_n$ of the form $n^{1-2\alpha'}(\log n)\to 0$ for a specific $\alpha'$, which is incompatible with $b_n = \lfloor n^\nu \rfloor$ for any $\nu$.

The current paper thus represents a substantial weakening of all regularity conditions.

---

## 4. Batch Means Estimators

### 4.1 Non-overlapping batch means (BM)

Suppose the total run length is $n = a_n b_n$. Partition the output into $a_n$ non-overlapping batches of length $b_n$. Define batch $k$ average:

$$\bar{Y}_k := b_n^{-1}\sum_{i=1}^{b_n} Y_{kb_n+i}, \quad k = 0, 1, \ldots, a_n - 1.$$

The BM estimator of $\sigma_g^2$ is

$$\hat{\sigma}_{\mathrm{BM}}^2 := \frac{b_n}{a_n-1}\sum_{k=0}^{a_n-1}(\bar{Y}_k - \bar{Y}_n)^2.$$

**Key fact:** If $a_n$ and $b_n$ are both fixed as $n \to \infty$, then $\hat{\sigma}_{\mathrm{BM}}^2$ is **not** consistent (Glynn & Iglehart 1990). Consistency requires both $a_n \to \infty$ and $b_n \to \infty$. A classical choice is $a_n = b_n = \lfloor n^{1/2}\rfloor$, giving $n = a_n b_n \approx n$.

Jones et al.\ (2006) proved strong consistency of BM under geometric ergodicity with a moment condition similar to Theorem 2 below.

### 4.2 Overlapping batch means (OBM)

Use all possible overlapping batches of length $b_n$: there are $n - b_n + 1$ of them, indexed $j = 0, 1, \ldots, n-b_n$:

$$\bar{Y}_j(b_n) := b_n^{-1}\sum_{i=1}^{b_n} Y_{j+i}.$$

The OBM estimator is

$$\hat{\sigma}_{\mathrm{OBM}}^2 := \frac{nb_n}{(n-b_n)(n-b_n+1)}\sum_{j=0}^{n-b_n}\bigl(\bar{Y}_j(b_n) - \bar{Y}_n\bigr)^2.$$

**Fundamental connection.** OBM is asymptotically equivalent to the SV estimator with the **modified Bartlett lag window** $w_n(k) = [1 - |k|/b_n]I(|k|<b_n)$. Specifically:

$$\hat{\sigma}_{\mathrm{OBM}}^2 = \hat{\sigma}_{w,n}^2 + o_p(1) \quad \text{as } n \to \infty,$$

where $\hat{\sigma}_{w,n}^2$ is the SV estimator with the Bartlett window. For the Bartlett window, $\Delta_2 w_n(k) = 0$ for $k = 1, \ldots, b_n-1$ and $\Delta_2 w_n(b_n) = b_n^{-1}$, which simplifies many technical conditions.

### 4.3 Theorem 2: Strong consistency of OBM

> **Theorem 2.** Let $X$ be geometrically ergodic with invariant distribution $\pi$, and $g:\mathsf{X}\to\mathbb{R}$ a Borel function with $E_\pi|g|^{2+\delta+\epsilon}<\infty$ for some $\delta,\epsilon > 0$. Suppose Assumptions 2 and 3 hold. Further suppose:
>
> (a) $\sum_n (b_n/n)^c < \infty$ for some $c \ge 1$;  
> (b) $b_n n^{-1}\log n \to 0$;  
> (c) $n^{2\alpha}(\log n)^3/b_n \to 0$, where $\alpha = 1/(2+\delta)$;  
> (d) there exist $n_0 \in \mathbb{N}$ and $c_1 > 0$ such that $\log n / b_n \le c_1$ for all $n \ge n_0$;  
> (e) $b_n^4 n^{-2}\log\log n \to 0$ and $b_n^2 n^{-3}\log\log n \to 0$.  
>
> Then $\hat{\sigma}_{\mathrm{OBM}}^2 \to \sigma_g^2$ w.p.\ 1 as $n\to\infty$.

**Corollary 1.** If $b_n = \lfloor n^\nu \rfloor$ with $3/4 > \nu > (1+\delta/2)^{-1}$, then $\hat{\sigma}_{\mathrm{OBM}}^2 \to \sigma_g^2$ w.p.\ 1.

**Remark 5 (OBM vs. SV for OBM).** Theorem 2 requires a strictly weaker moment condition on $g$ than Theorem 1 ($2+\delta+\epsilon$ vs. $4+\delta+\epsilon$), because the Bartlett window structure simplifies the technical analysis. However, conditions (d) and (e) of Theorem 2 are not required for BM (Jones et al.\ 2006).

**Remark 7 (comparison with Jones et al.).** For BM, Jones et al.\ proved that with $b_n = \lfloor n^\nu \rfloor$, strong consistency holds for $\nu > (1 + \delta/2)^{-1}$, giving a slightly weaker lower bound on $\nu$ than Corollary 1 requires.

---

## 5. Mean-Square Consistency and Optimal Batch Size

### 5.1 MSE-consistency

Strong consistency and MSE-consistency ($E_\pi(\hat\sigma_n^2 - \sigma_g^2)^2 \to 0$) do not imply each other in general. The paper proves both for BM and OBM.

> **Theorem 3.** Let $\hat{\sigma}_n^2$ be either the BM or OBM estimator of $\sigma_g^2$, and let $X$ be a **stationary** geometrically ergodic chain with $E_\pi|g|^{4+\delta+\epsilon}<\infty$ and $E_\pi C^4 < \infty$ (where $C$ is defined via the strong invariance principle in eq.\ (14)). If $b_n^{-1}n^{2\alpha}(\log n)^3 \to 0$ with $\alpha = 1/(4+\delta)$, then $\mathrm{MSE}(\hat{\sigma}_n^2) \to 0$.

Note the stationarity requirement: unlike the strong consistency theorems, the MSE analysis requires the chain to have started from $\pi$. This is a standard restriction in MSE analysis for dependent processes.

### 5.2 Bias and variance asymptotics for BM

The following expressions are derived from the Brownian motion representation of BM and results of Chien, Goldsman & Melamed (1997). Under stationarity and the conditions of Theorem 3:

**Bias:**
$$b_n \cdot \mathrm{Bias}[\hat{\sigma}_{\mathrm{BM}}^2] = \Gamma + o(1) \quad \text{as } b_n \to \infty,$$

where
$$\Gamma := -2\sum_{s=1}^\infty s\,\gamma(s)$$
is a finite constant (the "bias coefficient") that encodes the long-range correlations of the process.

**Variance:**
$$\frac{n}{b_n}\,\mathrm{Var}(\hat{\sigma}_{\mathrm{BM}}^2) = 2\sigma_g^4 + o(1) \quad \text{as } n\to\infty, b_n\to\infty.$$

Combining:

$$\mathrm{MSE}(\hat{\sigma}_{\mathrm{BM}}^2) = \frac{\Gamma^2}{b_n^2} + \frac{2b_n\sigma_g^4}{n} + o\!\left(\frac{1}{b_n^2}\right) + o\!\left(\frac{b_n}{n}\right).$$

This is the fundamental **bias$^2$ + variance** decomposition with:
- Bias$^2$ term $\sim \Gamma^2/b_n^2$ (decreasing in $b_n$)
- Variance term $\sim 2\sigma_g^4 b_n/n$ (increasing in $b_n$)

### 5.3 Theorem 4: Asymptotic variance of BM and OBM

> **Theorem 4.** Under the conditions of Theorem 3, with the additional condition $b_n^{-1}n^{1/2+\alpha}(\log n)^{3/2} \to 0$:
>
> $$\frac{n}{b_n}\,\mathrm{Var}(\hat{\sigma}_n^2) = c\sigma_g^4 + o(1),$$
>
> where $c = 2$ for BM and $c = 4/3$ for OBM.

Consequently:

$$\frac{\mathrm{Var}(\hat{\sigma}_{\mathrm{OBM}}^2)}{\mathrm{Var}(\hat{\sigma}_{\mathrm{BM}}^2)} \to \frac{2}{3} \quad \text{as } n \to \infty.$$

OBM achieves the same asymptotic bias as BM but with $2/3$ of the asymptotic variance. This is the theoretical justification for preferring OBM over BM (see also Meketon & Schmeiser 1984 for the same result under different conditions).

For OBM the full MSE expansion is:

$$\mathrm{MSE}(\hat{\sigma}_{\mathrm{OBM}}^2) = \frac{\Gamma^2}{b_n^2} + \frac{4b_n\sigma_g^4}{3n} + o\!\left(\frac{1}{b_n^2}\right) + o\!\left(\frac{b_n}{n}\right).$$

### 5.4 Optimal batch size

Minimizing the leading terms of $\mathrm{MSE}(\hat{\sigma}_{\mathrm{BM}}^2)$ over $b_n$:

$$\frac{d}{db_n}\!\left(\frac{\Gamma^2}{b_n^2} + \frac{2b_n\sigma_g^4}{n}\right) = 0 \implies \hat{b}^*_{\mathrm{BM}} = \left(\frac{\Gamma^2 n}{\sigma_g^4}\right)^{1/3}.$$

For OBM:

$$\hat{b}^*_{\mathrm{OBM}} = \left(\frac{8\Gamma^2 n}{3\sigma_g^4}\right)^{1/3}.$$

Both are proportional to $n^{1/3}$. The proportionality constant $(\Gamma^2/\sigma_g^4)^{1/3}$ is problem-dependent and typically unknown — it involves the integrated correlation structure of the process. This is why in practice one often uses $b_n = \lfloor n^\nu \rfloor$ with $\nu \in \{1/3, 1/2, 2/3\}$ as a surrogate. The paper's experiments show that $\nu = 1/3$ (theoretically optimal exponent) is often too small in finite samples.

---

## 6. Comparison of Methods

### 6.1 OBM versus BM

| Criterion | BM | OBM |
|---|---|---|
| Strong consistency | Yes (Jones et al. 2006) | Yes (Theorem 2) |
| MSE-consistency | Yes (Theorem 3) | Yes (Theorem 3) |
| Moment condition on $g$ | $4+\delta$ (MSE) | $2+\delta$ (strong), $4+\delta$ (MSE) |
| Asymptotic variance | $2\sigma_g^4 b_n/n$ | $\frac{4}{3}\sigma_g^4 b_n/n$ |
| Bias (leading term) | $\Gamma/b_n$ | $\Gamma/b_n$ |
| Computational cost | $O(n)$ | $O(n b_n)$ in naive form; $O(n)$ with Welch trick |
| Stationarity for MSE | Required | Required |
| Conditions beyond BM | — | Extra log conditions (d), (e) |

The key practical tradeoff: OBM has lower variance but the same bias as BM. Welch (1987) argued that most of the variance reduction is already achieved with a small amount of overlapping; using $1/4$ overlaps (e.g., batches of length 64 starting at $X_1, X_{17}, X_{33}, X_{49}$) captures most of the benefit.

### 6.2 SV versus OBM

OBM with Bartlett window and SV with Tukey–Hanning window are asymptotically equivalent in bias structure, but differ in their finite-sample behavior:

- **Tukey–Hanning** is the recommended default: its twice-differentiable form of the window gives smoother spectral estimates and is observed to have better finite-sample coverage in the experiments.
- **Bartlett / OBM** are theoretically cleaner (simpler conditions in Theorem 2) and require a weaker moment condition.

### 6.3 Regenerative simulation (RS) versus BM/OBM/SV

RS uses the regeneration structure of the chain. It is theoretically rigorous but:
- Requires finding an accessible regeneration atom (practically, a minorization certificate);
- Suffers from very long tours in high dimensions or moderately-large state spaces;
- Is not well-suited to fixed-width stopping rules (the chain must be stored if a stopping criterion is applied globally).

In the experiments, RS achieves coverage comparable to BM/OBM/TH, but requires substantially more simulation effort ($7.12 \times 10^5$ mean iterations vs. $7 \times 10^5$ for others). All methods are within two standard errors of the nominal 0.95 level.

---

## 7. Numerical Experiments

### 7.1 AR(1) model

$$X_i = \rho X_{i-1} + \varepsilon_i, \quad \varepsilon_i \overset{\text{i.i.d.}}{\sim} \mathrm{N}(0,1), \quad |\rho| < 1.$$

True asymptotic variance: $\sigma_x^2 = 1/(1-\rho^2)$. Target: $E_\pi X = 0$.

**Setup:** 2000 independent replications; chain lengths $n \in \{10^3, 5\times10^3, 10^4, 5\times10^4, 10^5\}$; batch sizes $b_n = \lfloor n^\nu \rfloor$ for $\nu \in \{1/3, 1/2, 2/3\}$; nominal coverage $0.95$.

**Key findings:**

- For $\rho = 0.5$: all methods ($\nu \in \{1/3, 1/2\}$) achieve coverage within 2 SE of 0.95 at $n \ge 5\times10^3$. The choice $\nu = 2/3$ slightly undercovers for small $n$.
- For $\rho = 0.95$ (high correlation): $\nu = 1/3$ fails dramatically (coverage as low as 0.61 at $n = 10^3$). $\nu = 1/2$ is acceptable, $\nu = 2/3$ is best. This illustrates that batch sizes must grow faster when the chain is slowly mixing, because the bias $\Gamma/b_n$ is large.
- The **Tukey–Hanning** SV estimator consistently achieves coverage equal to or slightly above OBM and BM across all settings.

### 7.2 Bayesian probit regression (Lupus data)

**Model:** $\Pr(Y_i=1) = \Phi(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2})$ with flat prior over $\beta := (\beta_0,\beta_1,\beta_2)$, fitted to the Lupus nephritis data (van Dyk & Meng 2001, $n=55$ patients). Sampler: PX-DA algorithm (Liu & Wu 1999), geometrically ergodic (Roy & Hobert 2007). "True" posterior means estimated from a $10^8$-iteration run.

**Fixed-width experiment.** For each combination of method, $\nu$, and desired half-width $\epsilon \in \{0.1, 0.2, 0.3\}$, simulate until the criterion

$$\max_{j=0,1,2}\!\left\{t_*\frac{\hat{\sigma}_{\beta_j}}{\sqrt{n}}\right\} + \epsilon I(n \le n^*) + n^{-1} \le \epsilon$$

is satisfied (stopping every 10% of current $n$ if not). Based on 1000 replications.

**Key findings:**

- $\nu = 1/3$: all methods fail, coverage around 0.70–0.85 even for $\epsilon = 0.1$ (where simulations run past $10^5$ iterations).
- $\nu = 1/2$: coverage slightly below 0.95 for individual components. Tukey–Hanning achieves slightly better coverage with slightly higher simulation effort.
- **Bonferroni correction** (Table 5): using $98\tfrac{1}{3}\%$ individual CIs to achieve $95\%$ simultaneous coverage — at $\nu = 1/2$, all methods achieve simultaneous coverage $\ge 0.963$, exceeding the nominal $0.95$.

### 7.3 Summary and recommendations (Section 4.3)

The authors' recommendation (Section 4.3):

> **Use the Tukey–Hanning SV estimator with $b_n = \lfloor n^{1/2} \rfloor$ as the default.** If the moment conditions for SV are too strong, use OBM. Avoid $\nu = 1/3$ in practice despite its theoretical MSE-optimality: the batch size is too small to control bias for any realistic $n$.

---

## 8. Proof Techniques (Appendices A–C)

### 8.1 The strong invariance principle

The central technical tool is the **strong invariance principle** (Philipp & Stout 1975, Damerdji 1991). For a geometrically ergodic chain and $g$ with $E_\pi|g|^{2+\delta+\epsilon} < \infty$, there exists a Brownian motion $B(t)$ on a possibly enriched probability space and a constant $0 < \sigma_g < \infty$ such that (Lemma 3, using Bednorz–Latuszyński 2007 and Jones et al.\ 2006):

$$\left|\sum_{i=1}^n Y_i - nE_\pi g - \sigma_g B(n)\right| = O(n^\alpha\log n) \quad \text{w.p.\ 1},$$

where $\alpha = 1/(2+\delta)$. This approximation allows the analysis of $\hat{\sigma}_S^2$ and $\hat{\sigma}_{\mathrm{OBM}}^2$ to be reduced to analysis of their **Brownian motion analogues**, which are classical objects in probability theory.

The key Brownian motion lemmas used are:
- **Lemma 1** (Csörgő–Révész): $|B(n)| < (1+\varepsilon)\sqrt{2n\log\log n}$ for all large $n$, a.s.
- **Lemma 2** (Csörgő–Révész): $\sup_{0\le t\le n-b_n}\sup_{0\le s\le b_n}|B(t+s)-B(t)| \le (1+\varepsilon)\sqrt{2b_n(\log(n/b_n)+\log\log n)}$ for all large $n$, a.s.

### 8.2 Structure of the proof of Theorem 1 (Appendix B.1)

Define the "weighted SV-like" quantity:

$$\hat{\sigma}_{w,n}^2 := \frac{1}{n}\sum_{j=0}^{n-b_n}\sum_{k=1}^{b_n} k^2 \Delta_2 w_n(k)[\bar{Y}_j(k) - \bar{Y}_n]^2,$$

and its Brownian analogue $\tilde{\sigma}_*^2$ (with $Y$ replaced by $\sigma_g B$). The proof proceeds as:

1. **Lemma 4:** $\hat{\sigma}_{w,n}^2 - \sigma_g^2 \tilde{\sigma}_*^2 \to 0$ w.p.\ 1. This is proved by expanding the difference and controlling 7 cross terms using Lemmas 1 and 2 together with conditions (15)–(16) on $\Delta_2 w_n$. The key bound is $|A_k| \le 2Cn^\alpha\log n$ (from the strong invariance principle) combined with $|D_k| \le 2(1+\varepsilon)b_n^{1/2}(\log n)^{1/2}$ (from Lemma 2).

2. **Lemma 5:** Boundary terms involving $h(X_i) = [g(X_i)-E_\pi g]^2$ near the endpoints of the chain stay bounded w.p.\ 1. Proved using the ergodic theorem and the strong invariance principle for $h$.

3. **Lemma 6:** $\hat{\sigma}_{w,n}^2 + d_n = \hat{\sigma}_S^2$ for a sequence $d_n \to 0$ w.p.\ 1, and $\tilde{\sigma}_*^2 \to 1$ w.p.\ 1. This uses results of Damerdji (1991, 1994) and is where conditions (a)–(c) on the lag window are used.

Combining these three lemmas gives $\hat{\sigma}_S^2 \to \sigma_g^2$ w.p.\ 1.

### 8.3 Proof of Theorem 2 (Appendix B.3)

The proof exploits the specific structure of the Bartlett window: $\Delta_2 w_n(k) = 0$ for $k < b_n$. This means $\hat{\sigma}_{w,n}^2$ collapses to

$$\hat{\sigma}_{w,n}^2 = \frac{b_n}{n}\sum_{j=0}^{n-b_n}[\bar{Y}_j(b_n)-\bar{Y}_n]^2,$$

which is essentially $\hat{\sigma}_{\mathrm{OBM}}^2$ up to the normalizing constant. The proof then applies Lemma 8, which shows $\tilde{d}_n \to 0$ w.p.\ 1 for the corresponding Brownian quantity. Lemma 8 is proved via the Borel–Cantelli lemma using:
- **Lemma 9** (Kendall–Stuart): moment bounds for chi-squared random variables.
- **Lemma 10** (Whittle 1960): moment bounds for quadratic forms in i.i.d.\ standard normals.

### 8.4 Proof of Theorem 3 (MSE-consistency, Appendix C.2)

The MSE is split as $\mathrm{MSE} = \mathrm{Var} + \mathrm{Bias}^2$.

**Bias:** By Lemma 12, $|\hat{\sigma}_{\mathrm{BM}}^2 - \sigma_g^2\tilde{\sigma}_{\mathrm{BM}}^2| \le C^2 g_1(n) + Cg_2(n)$ where $g_1(n), g_2(n)\to 0$ under the condition $b_n^{-1}n^{2\alpha}(\log n)^3 \to 0$. Since $E\tilde{\sigma}_{\mathrm{BM}}^2 = 1$ (Lemma 11), Bias $\to 0$ follows from Lemma 14 and the generalized dominated convergence theorem.

**Variance:** From the Brownian analogue $\frac{n}{b_n}\mathrm{Var}[\tilde{\sigma}_{\mathrm{BM}}^2] = 2 + o(1)$ (Lemma 11), and the decomposition

$$\mathrm{Var}(\hat{\sigma}_{\mathrm{BM}}^2) = \sigma_g^4\,\mathrm{Var}(\tilde{\sigma}_{\mathrm{BM}}^2) + \eta(\hat{\sigma}_{\mathrm{BM}}^2,\tilde{\sigma}_{\mathrm{BM}}^2),$$

where $|\eta| = o(1)$ via Cauchy–Schwarz and Lemma 14. Thus $\mathrm{Var}(\hat{\sigma}_{\mathrm{BM}}^2) = 2\sigma_g^4 b_n/n + o(b_n/n) = o(1)$.

### 8.5 Proof of Theorem 4 (Asymptotic variance, Appendix C.3)

Directly from the decomposition in the proof of Theorem 3, using the Brownian moment result $\frac{n}{b_n}\mathrm{Var}[\tilde{\sigma}_{\mathrm{BM}}^2] = 2 + o(1)$ for BM and $\frac{n}{b_n}\mathrm{Var}[\tilde{\sigma}_{\mathrm{OBM}}^2] = 4/3 + o(1)$ for OBM (both from Lemma 11, attributed to Damerdji 1995).

---

## 9. Connection to LSA / Richardson–Romberg

This paper is a direct precursor to inference methods for stochastic approximation and SGD. The correspondences are:

| MCMC (Flegal–Jones) | LSA / SGD inference |
|---|---|
| $g(X_i)$: chain output | $\theta_n$: SA iterate |
| $\sigma_g^2$: long-run variance of $g(X_i)$ | $\Sigma$: long-run covariance of $\theta_n$ |
| $\hat\sigma_{\mathrm{BM}}^2$, $\hat\sigma_{\mathrm{OBM}}^2$ | BM / OBM estimator of $\Sigma$ for $\theta_n$ |
| $b_n \sim n^{1/3}$ (MSE-optimal exponent) | Same scaling for SA batch-size selection |
| Bias $= \Gamma/b_n$ in BM estimator | Bias $= C_1/b_n$ in OBM of $\Sigma$ for SA |
| Variance $= 2\sigma_g^4 b_n / n$ | Variance term $\sim b_n/n$ in OBM for SA |
| SV estimator = weighted autocovariance sum | Same construction applied to SA iterates |
| Fixed-width stopping rule | Sequential stopping criterion for SGD |

### 9.1 What this paper provides for the RR setting

The paper rigorously establishes the **bias-variance tradeoff structure** for OBM-type estimators:

$$\mathrm{MSE}(\hat\sigma_n^2) \asymp \frac{\Gamma^2}{b_n^2} + \frac{c\sigma_g^4 b_n}{n}.$$

This is precisely the structure that motivates **Richardson–Romberg in the batch size $b$**. If the estimator has expansion

$$E[\hat\Sigma_{n,b}] = \Sigma + \frac{C_1}{b} + \frac{C_2}{b^2} + O\!\left(\frac{b}{n}\right),$$

then the RR combination $\hat\Sigma_{n,b}^{\mathrm{RR}} := 2\hat\Sigma_{n,2b} - \hat\Sigma_{n,b}$ cancels the $C_1/b$ bias term, exactly as the **lugsail** construction of Vats & Flegal (2022) does.

Flegal–Jones (2010) is the standard citation for:
1. Justifying that OBM and BM are consistent estimators of $\sigma_g^2$ under geometric ergodicity.
2. Establishing the $\Gamma/b$ leading bias and $b/n$ variance structure.
3. Showing that the optimal batch size scales as $n^{1/3}$, but that $n^{1/2}$ is more robust in practice.

### 9.2 What this paper does NOT provide for the RR/LSA setting

- It does not address non-stationary chains or the transient bias of SA iterates ($O(\alpha)$ bias in constant-step SA).
- The MSE result requires stationarity; for LSA/SA one typically starts far from stationarity.
- It does not prove the refined scalar expansion $E[\hat\sigma_{n,b}^2(u)] = \sigma^2(u) + c_1(u)/b + c_2(u)/b^2 + c_3(u)b/n + \ldots$ in closed form.
- It does not address the matrix case or simultaneous inference (see Vats, Flegal & Jones 2018/2019 for multivariate extensions).

---

## 10. Critical Assessment

**Strengths:**
- Substantial weakening of regularity conditions relative to all prior work.
- Unified treatment of BM, OBM, and SV within a single technical framework (strong invariance principle).
- Practical empirical guidance with clear takeaways.
- The MSE expansion and optimal batch-size analysis (Section 3.2) are presented cleanly and match classical results from the simulation literature (Chien et al.\ 1997, Song & Schmeiser 1995).

**Limitations and open questions:**
- MSE-consistency requires stationarity; verifying this in practice requires knowing the burn-in is negligible.
- The condition $E_\pi C^4 < \infty$ (via the split-chain constant $C$) is not directly verifiable from the specification of $g$ and the chain.
- The optimal $b^* \propto n^{1/3}$ depends on the unknown constant $\Gamma/\sigma_g^4$; adaptive selection of $b_n$ is left to future work.
- The paper does not cover data-driven bandwidth selection or cross-validation methods for the truncation point.
- The theoretical gap between $\nu_{\mathrm{opt}} = 1/3$ and the practical recommendation of $\nu = 1/2$ motivates further work on finite-sample bias reduction — exactly the direction pursued by lugsail (Vats & Flegal 2022) and RR-in-$b$ methods.

---

## 11. Reference Chain for LSA/RR Work

The natural literature chain building on this paper in the direction of SGD/SA inference:

1. **Flegal & Jones (2010)** — foundation: BM/OBM/SV consistency and MSE structure under geometric ergodicity.
2. **Vats, Flegal & Jones (2018)** — multivariate SV estimators, strong consistency.
3. **Vats, Flegal & Jones (2019)** — multivariate BM, fixed-width stopping, effective sample size.
4. **Liu, Vats & Flegal (2022)** — asymptotic MSE for general multivariate BM class; batch-size selection.
5. **Vats & Flegal (2022)** — lugsail windows: first-order bias correction for SV and BM.
6. **Ng & Perron (1996)** — exact finite-sample bias/variance for spectral density at frequency zero; the $b/n$ finite-sample term.
7. **Singh, Shukla & Vats (2025)** — equal-batch inference for SGD/ASGD; Markovian SA bridge.

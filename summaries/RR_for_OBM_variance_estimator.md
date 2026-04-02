# Richardson–Romberg Extrapolation for the OBM Variance Estimator

## 1. Motivation

The Overlapping Batch Mean (OBM) variance estimator, used in Samsonov et al. (2025) for constructing confidence intervals for LSA iterates, is essentially a Bartlett-window long-run variance estimator. Like all kernel-based spectral density estimators, it suffers from a **truncation bias** that depends on the block size $b$: a finite window cannot capture all autocovariances, so the estimate is biased.

The idea is to apply Richardson–Romberg extrapolation **to the OBM estimator itself** (in the block size $b$), analogously to how RR is applied to the LSA step size $\alpha$ for bias reduction of point estimates.

This is a **distinct and complementary** application of RR:
- RR in $\alpha$: reduces bias of $\bar\theta_n$ as an estimator of $\theta^\star$.
- RR in $b$: reduces bias of $\hat\sigma^2_{n,b}$ as an estimator of the long-run variance $\sigma^2$.

---

## 2. OBM Variance Estimator: Setup

Given a sequence of iterate projections $\theta_1, \dots, \theta_T$ (e.g., $\theta_t = u^\top \theta_t^{(\alpha)}$ for some direction $u$), the overall mean $\bar\theta_T = T^{-1}\sum_{t=1}^T \theta_t$, and block size $b$, the OBM variance estimator is:

$$\hat\sigma^2_{n,b}(u) = \frac{b}{n - b + 1} \sum_{t=0}^{n-b} \left( \bar\theta_{b,t} - \bar\theta_n \right)^2,$$

where $\bar\theta_{b,t} = b^{-1}\sum_{k=t}^{t+b-1} \theta_k$ is the overlapping block average starting at time $t$.

In matrix form (for the full covariance):

$$\hat\Sigma_{n,b} = \frac{b}{n - b + 1} \sum_{t=0}^{n-b} (\bar\theta_{b,t} - \bar\theta_n)(\bar\theta_{b,t} - \bar\theta_n)^\top.$$

---

## 3. Bias Expansion of the OBM Estimator

Under standard Markov/weak-dependence conditions (uniform geometric ergodicity, boundedness of the noise), the OBM estimator admits a bias expansion:

$$\mathbb{E}[\hat\sigma^2_{n,b}(u)] = \sigma^2(u) + \frac{c_1(u)}{b} + \frac{c_2(u)}{b^2} + c_3(u)\frac{b}{n} + o\!\left(\frac{1}{b^2} + \frac{b}{n}\right),$$

where:
- $\sigma^2(u) = u^\top \Sigma_\infty u$ is the true long-run variance,
- $c_1(u)/b$ is the **leading truncation bias** from the finite Bartlett window (truncating autocovariances at lag $b$),
- $c_2(u)/b^2$ is the second-order truncation bias,
- $c_3(u) \cdot b/n$ is the **centering bias** from estimating the mean $\bar\theta_n$ (grows with $b/n$).

There may also be an additional SA transient term if the iterates have not fully reached their stationary regime.

### 3.1. Nature of the coefficients

The coefficient $c_1(u)$ is related to the derivative of the spectral density at frequency zero, or equivalently to the sum $\sum_{\ell} |\ell| \cdot \text{Cov}(\theta_0, \theta_\ell)$ — it encodes how fast the autocovariances decay. For geometrically ergodic chains, this sum is finite, and $c_1$ is well-defined.

The centering bias $c_3 \cdot b/n$ arises because $\bar\theta_n$ is used in place of $\theta^\star$; this introduces an $O(b/n)$ term that grows with block size.

---

## 4. RR-Corrected OBM Estimator

### 4.1. Construction

Use two block sizes $b$ and $\lambda b$ with $\lambda > 1$ (e.g., $\lambda = 2$). Define the RR-corrected estimator:

$$\hat\sigma^{2,\mathrm{RR}}_{n,b,\lambda}(u) = \frac{\lambda}{\lambda - 1} \hat\sigma^2_{n,\lambda b}(u) - \frac{1}{\lambda - 1} \hat\sigma^2_{n,b}(u).$$

### 4.2. Bias cancellation

Substituting the expansion:

$$\mathbb{E}[\hat\sigma^{2,\mathrm{RR}}_{n,b,\lambda}(u)] = \frac{\lambda}{\lambda-1}\left(\sigma^2 + \frac{c_1}{\lambda b} + \frac{c_2}{\lambda^2 b^2} + c_3\frac{\lambda b}{n}\right) - \frac{1}{\lambda-1}\left(\sigma^2 + \frac{c_1}{b} + \frac{c_2}{b^2} + c_3\frac{b}{n}\right)$$

$$= \sigma^2 + \underbrace{\left(\frac{\lambda}{\lambda-1}\cdot\frac{1}{\lambda} - \frac{1}{\lambda-1}\right)}_{= 0}\frac{c_1}{b} + O\!\left(\frac{1}{b^2}\right) + O\!\left(\frac{b}{n}\right).$$

The leading $c_1/b$ bias term **cancels exactly**. The residual bias is:

$$\text{Bias} = O\!\left(\frac{1}{b^2}\right) + O\!\left(\frac{b}{n}\right) + (\text{SA transient term}).$$

### 4.3. Concrete case $\lambda = 2$

$$\hat\sigma^{2,\mathrm{RR}}_{n,b,2}(u) = 2\,\hat\sigma^2_{n,2b}(u) - \hat\sigma^2_{n,b}(u).$$

This is the simplest and most natural choice. The weights $(2, -1)$ are the same as the two-level RR for step sizes.

---

## 5. Variance Inflation

### 5.1. The cost of bias reduction

The RR estimator is a linear combination of two OBM estimators. By the triangle inequality for standard deviations:

$$\text{Var}(\hat\sigma^{2,\mathrm{RR}}) \approx \left(\frac{\lambda}{\lambda-1}\right)^2 \text{Var}(\hat\sigma^2_{n,\lambda b}) + \left(\frac{1}{\lambda-1}\right)^2 \text{Var}(\hat\sigma^2_{n,b}).$$

For $\lambda = 2$: the weights are $2$ and $-1$, so variance roughly increases by a factor of $2^2 + 1^2 = 5$ compared to a single estimator (in the worst case, assuming independence; in practice the two estimators are correlated, so the inflation is smaller but still non-negligible).

### 5.2. Bias-variance tradeoff

The RR correction is beneficial when:
- The truncation bias $c_1/b$ is the **dominant source of error** (i.e., $b$ is not too large),
- The sample size $n$ is large enough that the variance inflation is tolerable.

If $b$ is already very large (close to optimal $b \sim n^{1/3}$ for MSE-optimal bandwidth), the truncation bias is already small and the variance inflation may not be worthwhile.

---

## 6. Important Caveats

### 6.1. SA transient bias is not corrected

The OBM estimator applied to LSA iterates has bias from two sources:
1. **Truncation bias** (finite window $b$) — corrected by RR in $b$.
2. **SA transient bias** (iterates have not reached stationarity) — **not corrected** by RR in $b$.

The total bias structure is:

$$\text{Bias}_{\text{total}} = \underbrace{\frac{c_1}{b} + \frac{c_2}{b^2}}_{\text{OBM truncation}} + \underbrace{c_3 \frac{b}{n}}_{\text{centering}} + \underbrace{\text{Bias}_{\mathrm{SA}}(n)}_{\text{SA dynamics}}.$$

RR in $b$ removes only the $c_1/b$ piece. If $\text{Bias}_{\mathrm{SA}}(n)$ dominates (e.g., for short trajectories or slow-mixing chains), the correction has limited impact.

To correct SA bias, one needs RR in the **step size** $\alpha$ — which is already done in the main thesis for the point estimator.

### 6.2. Two levels of RR can coexist

In principle, one can apply:
- **RR in $\alpha$** to debias the point estimator $\bar\theta_n^{(\alpha)}$,
- **RR in $b$** to debias the variance estimator $\hat\sigma^2_{n,b}$.

These are orthogonal corrections targeting different bias sources. The combined estimator would use RR iterates (correcting SA bias) with an RR-corrected OBM (correcting truncation bias).

### 6.3. The centering bias $c_3 \cdot b/n$ is not corrected

RR in $b$ with weights $\lambda/(\lambda-1)$ and $-1/(\lambda-1)$ cancels $c_1/b$ but does **not** cancel $c_3 \cdot b/n$. In fact:

$$\frac{\lambda}{\lambda-1} \cdot c_3 \frac{\lambda b}{n} - \frac{1}{\lambda-1} \cdot c_3 \frac{b}{n} = c_3 \frac{b}{n} \cdot \frac{\lambda^2 - 1}{\lambda - 1} = c_3 \frac{b(\lambda+1)}{n},$$

which is **larger** than the original $c_3 b/n$ by a factor of $(\lambda+1)/1$. For $\lambda = 2$ this is a factor of 3.

This means RR in $b$ **worsens** the centering bias. The regime where RR is helpful is therefore:
$$\frac{c_1}{b} \gg c_3 \frac{b}{n} \quad \Longleftrightarrow \quad b \ll \sqrt{\frac{c_1 n}{c_3}}.$$

---

## 7. Optimal Block Size Selection

### 7.1. Without RR

The MSE-optimal block size for OBM (balancing $O(1/b^2)$ truncation variance and $O(b^2/n^2)$ centering) is:

$$b_{\text{opt}} \sim n^{1/3}.$$

### 7.2. With RR

After RR removes the $1/b$ bias, the dominant bias terms are $1/b^2$ and $b/n$. The MSE-optimal block size shifts:

$$b_{\text{opt}}^{\mathrm{RR}} \sim n^{1/3} \quad \text{(same order, but the constant may change)}.$$

The practical implication is that RR allows using a **smaller block size** (better variance) without paying as much in bias.

---

## 8. Connection to Spectral Density Estimation

The OBM estimator $\hat\sigma^2_{n,b}$ is closely related to the Bartlett (triangular) kernel estimator of the spectral density at frequency zero:

$$\hat f(0) = \sum_{|\ell| \le b} w(\ell/b) \hat\gamma(\ell),$$

where $w$ is the Bartlett kernel $w(x) = (1 - |x|)_+$ and $\hat\gamma(\ell)$ is the sample autocovariance at lag $\ell$.

The bias expansion in $1/b$ corresponds to the familiar bias expansion for kernel spectral density estimators. RR in the bandwidth parameter is a well-known technique in nonparametric statistics (cf. higher-order kernels), and the RR-OBM estimator can be seen as an implicit higher-order kernel estimator.

This connection suggests that the theory of higher-order kernel bias correction (e.g., Andrews 1991, Politis 2011) could be leveraged to formalize the bias expansion and prove consistency of the RR-OBM estimator under Markovian noise.

---

## 9. Relevance to the Thesis

### What this adds

1. A **new inference strategy**: RR + RR-OBM — combining step-size RR (for point estimation bias) with block-size RR (for variance estimation bias). This could improve coverage closer to the nominal 95%.

2. The experimental results show RR+OBM achieves 94% coverage at nominal 95%. The missing 1% could partly be due to OBM truncation bias, which this correction addresses.

3. It provides a **theoretical direction**: proving consistency and deriving the bias expansion for OBM applied to RR-extrapolated LSA iterates under Markovian noise.

### What needs to be done

- Verify the bias expansion holds for RR-extrapolated iterates (the stationarity assumption may be more delicate since RR combines two non-stationary sequences).
- Implement the RR-OBM estimator in code and compare empirically.
- Study the variance inflation trade-off: is the bias reduction worth the variance cost for typical $T$ and $b_n$?

---

## 10. Summary

| Aspect | Standard OBM | RR-corrected OBM |
|--------|-------------|------------------|
| Truncation bias | $O(1/b)$ | $O(1/b^2)$ |
| Centering bias | $O(b/n)$ | $O(b/n)$ (slightly worse) |
| Variance | $\text{Var}_0$ | $\approx (2\lambda^2+2)/(\lambda-1)^2 \cdot \text{Var}_0$ |
| Optimal $b$ | $\sim n^{1/3}$ | $\sim n^{1/3}$ (smaller constant) |
| SA transient bias | Not corrected | Not corrected |

**Bottom line:** RR in the block size is a natural and theoretically grounded way to reduce OBM truncation bias. It is most useful when $b$ is moderate (not too large), $n$ is large, and the truncation bias is the dominant error source. It complements (but does not replace) RR in the step size for SA bias correction.

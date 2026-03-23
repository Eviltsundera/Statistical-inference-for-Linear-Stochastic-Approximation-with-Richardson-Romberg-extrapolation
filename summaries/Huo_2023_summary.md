# Detailed Summary: "Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference"

**Authors:** Dongyan (Lucy) Huo (Cornell), Yudong Chen (UW-Madison), Qiaomin Xie (UW-Madison)
**Year:** 2023, arXiv:2312.10894

---

## 1. Problem and Motivation

The paper studies **statistical inference** (constructing confidence intervals) using **linear stochastic approximation (LSA)** with a **constant stepsize** and **Markovian data**. The LSA iteration is:

$$\theta_{t+1} = \theta_t + \alpha\bigl(A(x_t)\,\theta_t + b(x_t)\bigr), \quad t = 0, 1, \dots$$

where $(x_t)_{t \ge 0}$ is a time-homogeneous, uniformly ergodic Markov chain on state space $\mathcal{X}$ with stationary distribution $\pi$, and $\alpha > 0$ vis a fixed stepsize. The target $\theta^*$ satisfies the steady-state equation $\mathbb{E}_\pi[A(x)]\theta^* + \mathbb{E}_\pi[b(x)] = 0$.

**Why this matters:** Classical SA theory requires a diminishing stepsize ($\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$). Constant stepsize is popular in practice because it offers faster initial convergence and easier hyperparameter tuning, but the iterates converge only in distribution (not almost surely), and the limiting expectation $\mathbb{E}[\theta_\infty^{(\alpha)}]$ is generally **biased** with respect to $\theta^*$. This bias has discouraged researchers from using constant-stepsize iterates for inference. This paper shows how to overcome this limitation.

---

## 2. Assumptions

**Assumption 1 (Uniform Ergodicity):** $(x_t)$ is a uniformly ergodic Markov chain with transition kernel $P$ and unique stationary distribution $\pi$. The initial state satisfies $x_0 \sim \pi$.

**Assumption 2 (Boundedness & Hurwitz Condition):** $A_{\max} := \sup_x \|A(x)\|_2 \le 1$, $b_{\max} := \sup_x \|b(x)\|_2 < \infty$, and $\bar{A} := \mathbb{E}_\pi[A(x)]$ is a **Hurwitz matrix** (all eigenvalues have strictly negative real parts). The Hurwitz condition ensures the stability of the dynamical system.

---

## 3. Central Limit Theorem (Theorem 3.1)

The paper's first main result is a **CLT for averaged constant-stepsize Markovian LSA iterates**.

**Theorem 3.1:** Under Assumptions 1--2, there exists $\alpha_0 \in (0,1)$ such that for all $\alpha \in (0, \alpha_0)$:

$$\sqrt{T}\bigl(\bar{\theta}_T - \mathbb{E}[\theta_\infty]\bigr) \xrightarrow{d} \mathcal{N}(0, \Sigma^*), \quad \text{as } T \to \infty,$$

where $\bar{\theta}_T = \frac{1}{T}\sum_{t=0}^{T-1}\theta_t$ is the Polyak-Ruppert average and $\Sigma^*$ is the asymptotic covariance.

**Key technical contribution:** Unlike i.i.d. settings, with Markovian data the iterates $(\theta_t)$ alone do not form a Markov chain. The authors must work with the **joint process** $(x_t, \theta_t)$, which is a time-homogeneous Markov chain thanks to the constant stepsize. The proof verifies the **Maxwell-Woodroofe condition** for this joint chain, which is nontrivial because $(x_t, \theta_t)$ does not enjoy a standard one-step contraction property in Wasserstein distance.

**Note:** The CLT centers at $\mathbb{E}[\theta_\infty]$, **not** at $\theta^*$. The gap $\mathbb{E}[\theta_\infty] - \theta^*$ is the asymptotic bias.

---

## 4. Asymptotic Bias and Richardson-Romberg Extrapolation

### 4.1 The Bias Structure

The asymptotic expectation admits a **power series expansion in $\alpha$** (from prior work [HCX23]):

$$\mathbb{E}[\theta_\infty^{(\alpha)}] - \theta^* = \sum_{i=1}^{\infty} \alpha^i B^{(i)},$$

where $B^{(i)}$ are vectors independent of $\alpha$. The leading bias term is $O(\alpha)$.

### 4.2 Richardson-Romberg (RR) Extrapolation

To eliminate the bias, the authors run LSA with $M$ distinct constant stepsizes $\mathcal{A} = \{\alpha_1, \dots, \alpha_M\}$ **on the same Markov chain trajectory** and form the linear combination:

$$\widetilde{\theta}_t^{\mathcal{A}} = \sum_{m=1}^{M} h_m \, \theta_t^{(\alpha_m)},$$

where the coefficients $\{h_m\}$ are determined by a **Vandermonde system**:

$$\sum_m h_m = 1; \quad \sum_m h_m \alpha_m^l = 0, \quad l = 1, \dots, M-1.$$

This eliminates the first $M-1$ terms in the bias expansion, reducing the bias from $O(\alpha)$ to $O(\alpha^M)$ --- an **exponential reduction** in the order of extrapolation.

The explicit formula for the coefficients is: $h_m = \prod_{l \ne m} \frac{\alpha_l}{\alpha_l - \alpha_m}$.

---

## 5. Inference Procedure

### 5.1 Point Estimation and Batching

Given a single Markov chain trajectory:
1. Run LSA with constant stepsize $\alpha$ and collect iterates $\theta_0, \theta_1, \dots$
2. Discard the first $b$ iterates as **burn-in**
3. Divide the remaining iterates into $K$ **batches** of size $n$
4. Within each batch, discard the first $n_0$ iterates (to reduce inter-batch correlation, at rate $\exp(-\alpha n_0)$)
5. Compute batch-mean estimators: $\hat{\theta}_k^{(\alpha)} = \frac{1}{n - n_0}\sum_{l \in \text{batch}_k} \theta_l^{(\alpha)}$

### 5.2 Covariance Estimation

The batch-mean covariance estimator:

$$\hat{\Sigma}^{(\alpha)} = \frac{n - n_0}{K} \sum_{k=1}^{K} \bigl(\hat{\theta}_k^{(\alpha)} - \bar{\theta}^{(\alpha)}\bigr)\bigl(\hat{\theta}_k^{(\alpha)} - \bar{\theta}^{(\alpha)}\bigr)^\top$$

This is consistent for $\Sigma^*$ as $n, K \to \infty$.

### 5.3 Confidence Interval Construction

For the $i$-th coordinate of $\mathbb{E}[\theta_\infty]$, the $(1-q) \times 100\%$ CI is:

$$\left[\hat{\theta}_i^{(\alpha)} - z_{1-q/2}\sqrt{\frac{\hat{\Sigma}_{i,i}^{(\alpha)}}{K(n-n_0)}}, \;\; \hat{\theta}_i^{(\alpha)} + z_{1-q/2}\sqrt{\frac{\hat{\Sigma}_{i,i}^{(\alpha)}}{K(n-n_0)}}\right]$$

### 5.4 Combining with RR Extrapolation

For each of the $M$ stepsizes, run the batching procedure to get $K$ batch-mean estimators. Then linearly combine: $\widetilde{\theta}_k^{\mathcal{A}} = \sum_m h_m \hat{\theta}_k^{(\alpha_m)}$. Conduct inference on these RR-extrapolated batch means.

**Key advantage of constant stepsize:** The fast geometric mixing rate $\exp(-\alpha n_0)$ means batches decorrelate quickly, making the covariance estimator robust to the number of batches $K$. In contrast, diminishing stepsizes produce increasingly correlated iterates, making the choice of $K$ critical.

---

## 6. Theoretical Guarantees for Stepsize Selection in RR

### 6.1 Geometric Decay Schedule (Proposition 5.1)

Stepsizes: $\alpha_m = \alpha_1 / c^{m-1}$, $c \ge 2$.

- **Coefficient bound:** $|h_k| \le h_{\max}(c) = \exp\!\bigl(\frac{2}{c-1}\bigr)$
- **Variance bound:** $\mathrm{Var}(\widetilde{\theta}_\infty^{\mathcal{A}}) = O\bigl(c \cdot \exp(16\, c^{-1/2})\bigr)$

The variance bound is **independent of $M$**, meaning adding more extrapolation levels does not blow up the variance. However, the smallest stepsize $\alpha_M \to 0$ as $M$ grows, causing slow mixing.

### 6.2 Equidistant Decay Schedule (Proposition 5.2)

Stepsizes: $\alpha_m = (a+b) - \frac{b(m-1)}{M-1}$, with $a+b < 1$.

- **Variance bound:** $\mathrm{Var}(\widetilde{\theta}_\infty^{\mathcal{A}}) = O\bigl((2M/b)^{2M}\bigr)$

Here the variance **grows as $M^M$**, which can blow up. The stepsizes don't decay to zero as fast, but the Vandermonde matrix becomes ill-conditioned when stepsizes are close together.

**Practical guidance:** Use geometric decay with moderate $c$ (e.g., $c=2$). The order of RR extrapolation around $M=5$ captures most of the bias reduction benefit. Stepsizes should not be too close to each other.

---

## 7. Zero Bias Special Cases

The paper identifies three important Markovian LSA scenarios where $\mathbb{E}[\theta_\infty] = \theta^*$ (no bias), so RR extrapolation is **unnecessary**:

### 7.1 Independent Multiplicative Noise

State $x_t = (s_t, \xi_t)$ with $\xi_t$ i.i.d. zero-mean noise independent of the Markov chain $s_t$:

$$\theta_{t+1} = \theta_t + \alpha\bigl((\bar{A} + \xi_t)\theta_t + b(s_t)\bigr)$$

Since $\xi$ is zero-mean and independent, $\mathbb{E}[\theta_\infty] = -\bar{A}^{-1}\bar{b} = \theta^*$.

### 7.2 Independent Additive Noise in Linear Regression

SGD for linear regression with $y_t = s_t^\top w^* + \epsilon_t$ where $s_t$ is Markovian and $\epsilon_t$ is i.i.d. noise. This has zero asymptotic bias (shown in [HCX23]).

### 7.3 Realizable Linear-TD Learning in RL

TD learning with linear function approximation under the **realizability assumption** ($V(s) = \phi(s)^\top v$ for some $v$) in the semi-simulator regime. The conditional expectation condition (Corollary A.5) is satisfied, yielding $\mathbb{E}[\theta_\infty] = \theta^*$.

The **sufficient condition** for zero bias (Corollary A.5) is:

$$\mathbb{E}\bigl[A(x_t)\theta^* + b(x_t) \mid x_{t+1} = x\bigr] = 0, \quad \forall x \in \mathcal{X}.$$

---

## 8. Numerical Experiments

### 8.1 Performance Comparison (Markovian LSA, d=5, |X|=10)

Across 100 random LSA problems:
- **Constant $\alpha = 0.2$:** Large bias, poor coverage (0% at low percentiles)
- **Constant $\alpha = 0.02$:** Smaller bias, better coverage (~90%)
- **RR extrapolation (combining $\alpha = 0.2$ and $0.02$):** Best overall --- smallest $\ell_2$ error, comparable CI widths, coverage around the target 95%
- **Diminishing stepsizes ($0.2/\sqrt{k}$, $0.02/\sqrt{k}$):** Competitive but generally worse than RR, especially with short trajectories

### 8.2 Batch Number Selection

Constant stepsizes are **robust to the choice of $K$** (the number of batches). Diminishing stepsizes degrade significantly when $K$ is too large, because the increasing correlation from the shrinking stepsize requires longer batches to decorrelate.

### 8.3 Trajectory Length

- At short trajectories ($10^3$): All methods struggle, but constant stepsize + RR is competitive
- At long trajectories ($10^6$): Diminishing stepsizes catch up, but constant stepsize + RR remains best
- Constant stepsize $\alpha=0.2$ without RR: Coverage $\to$ 0% as $T$ grows (converges to the biased $\mathbb{E}[\theta_\infty] \ne \theta^*$)

### 8.4 Comparison Against Bootstrapping

| Method | $\ell_2$ error | CI width | Coverage |
|--------|---------------|----------|----------|
| Constant stepsize + RR | $1.0 \times 10^{-5}$ | 0.00134 | 95.2% |
| Bootstrapping | $2.0 \times 10^{-3}$ | 0.00411 | 93.4% |

Bootstrapping requires $O(n^d)$ memory vs. $O(d)$ for the SA-based approach.

### 8.5 Nonlinear Extension (Logistic Regression)

Even for non-linear, non-Hurwitz settings (logistic regression with Gaussian AR(1) Markovian covariates), the proposed method performs well, suggesting robustness beyond the theoretical assumptions.

### 8.6 RR Stepsize Selection

- Geometric decay is generally better than equidistant decay
- Extrapolation order $M \approx 5$ captures most of the benefit
- Stepsizes that are too close together (small $b/(M-1)$ in equidistant schedule) hurt performance due to ill-conditioned Vandermonde systems

---

## 9. Key Proof Techniques

### CLT Proof (Appendix B.1)

1. **Removing the stationarity assumption:** The authors first prove (Theorem B.1) that $(x_k, \theta_k) \to \bar{\mu}$ in Wasserstein-2 distance for **any** initial distribution, not just $x_0 \sim \pi$. This uses a coupling argument between processes starting from different initial conditions.
2. **Maxwell-Woodroofe condition:** Define $h(x,\theta) = \theta - \mathbb{E}[\theta_\infty]$ and verify that $\sum_{n=1}^\infty n^{-3/2}\|\sum_{t=0}^{n-1} Q^t h\|_{L^2(\bar\mu)} < \infty$, where $Q$ is the transition kernel of the joint chain. They show this norm is $O(n^r)$ for $r < 1/2$ by bounding the mixing contribution using the geometric convergence rate.

### Variance Bounds for RR (Appendix B.2)

The key identity is $\mathrm{Var}(\theta_\infty^{(\alpha_m)}) = O(\alpha_m \tau_{\alpha_m}) = O(1)$. The variance of the extrapolated estimator is bounded by $(\sum |h_m| \sqrt{\mathrm{Var}(\theta_\infty^{(\alpha_m)})})^2$, which reduces to bounding $\sum |h_m|$. This is where the specific stepsize schedule matters.

---

## 10. Conclusions and Future Directions

**Main takeaways:**
1. Constant stepsize LSA iterates can be effectively used for statistical inference with Markovian data
2. RR extrapolation successfully reduces the asymptotic bias
3. The approach enjoys easy hyperparameter tuning, fast convergence, and robustness to batch number selection
4. Several important Markovian settings have zero bias, making the approach even simpler

**Open problems identified by the authors:**
- Prove **validity** (consistency) of the CIs constructed with constant stepsize and Markovian data (currently the CIs are not proven to be consistent)
- Develop an **anytime variance estimator** for fully online inference
- Given a fixed simulation budget, determine the **optimal order of RR extrapolation** and trajectory length allocation

---

## Relevance to Thesis (Richardson-Romberg Extrapolation for LSA)

This paper is directly relevant as it:
- Provides the theoretical framework (CLT + bias expansion) that justifies using RR extrapolation in the LSA/Markovian setting
- Gives explicit formulas for the extrapolation coefficients via the Vandermonde system
- Analyzes the bias-variance tradeoff in RR through stepsize schedule analysis (geometric vs. equidistant)
- Identifies the open problem of CI consistency, which could be a direction for further research

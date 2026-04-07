# Detailed Review: Singh, Shukla & Vats (2025)
# "On the Utility of Equal Batch Sizes for Inference in Stochastic Gradient Descent"

**Journal:** Journal of Machine Learning Research 26 (2025) 1–41  
**Submitted:** January 2024; Published: October 2025  
**Authors:** Rahul Singh (IIT Delhi / IIT Delhi Abu Dhabi), Abhinek Shukla (NUS), Dootika Vats (IIT Kanpur)  
**Code:** https://github.com/Abhinek-Shukla/SGD-EBS

---

## Overview and Significance

This paper addresses a central open problem in statistical inference for stochastic gradient descent (SGD): how to consistently estimate the asymptotic covariance matrix $\Sigma$ of the Polyak-Juditsky averaged SGD (ASGD) estimator, in a way that is memory-efficient, bias-reduced, and compatible with interpretable confidence regions.

The SGD iterate $\{\theta_i\}_{i \geq 1}$ is a **time-inhomogeneous, non-stationary Markov chain** converging to a Dirac mass at $\theta^*$ — fundamentally different from MCMC chains, which are stationary and ergodic. This difference means that classical MCMC output analysis theory (Flegal & Jones 2010, etc.) cannot be applied directly, and new theoretical machinery is needed. The paper provides this machinery for an **Equal Batch-Size (EBS)** estimator, proving mean-square consistency and obtaining explicit bounds, enabling both practical inference and a natural bias-reduction through the **lugsail technique** of Vats & Flegal (2022).

**Key contributions:**
1. **EBS estimator:** A memory-efficient batch-means estimator of $\Sigma$ using equal batch sizes that are powers of two, with explicit mean-square error bounds.
2. **Lugsail-EBS:** Free bias correction by exploiting the doubling structure — no additional memory cost compared to the standard EBS estimator.
3. **Marginal-friendly simultaneous confidence regions:** Hyper-rectangular confidence regions that maintain joint coverage and are interpretable per-coordinate.
4. **Prediction improvement:** Demonstrating that consistent estimators of $\Sigma$ can directly reduce misclassification rates in logistic regression.

---

## 1. Problem Setup

### 1.1 Statistical model

Data $\zeta_i \overset{\mathrm{iid}}{\sim} \Pi$ on $\mathbb{R}^r$. A loss function $f: \mathbb{R}^d \times \mathbb{R}^r \to \mathbb{R}$ defines expected risk $F(\theta) = \mathbb{E}_\Pi[f(\theta, \zeta)]$. The target is:
$$\theta^* = \arg\min_{\theta \in \mathbb{R}^d} F(\theta).$$

### 1.2 SGD iterate and Polyak-Juditsky averaging

The $i$-th SGD iterate:
$$\theta_i = \theta_{i-1} - \eta_i \nabla f(\theta_{i-1}, \zeta_i), \quad i = 1, 2, \ldots$$
with learning rate $\eta_i = \eta i^{-\alpha}$, $\alpha \in (0.5, 1)$.

The Polyak-Juditsky ASGD estimator:
$$\hat{\theta}_n := n^{-1} \sum_{i=1}^n \theta_i.$$

### 1.3 Asymptotic normality and the target matrix $\Sigma$

Polyak and Juditsky (1992) proved that under appropriate conditions on strongly convex $F$:
$$\sqrt{n}(\hat{\theta}_n - \theta^*) \xrightarrow{d} N(0, \Sigma), \quad \text{where } \Sigma = A^{-1} S A^{-1}. \tag{1}$$

Here:
- $A := \nabla^2 F(\theta^*)$ — the Hessian of $F$ at $\theta^*$
- $S := \mathbb{E}_\Pi\left[[\nabla f(\theta^*, \zeta)][\nabla f(\theta^*, \zeta)]^\top\right]$ — the gradient outer product matrix

Both $A$ and $S$ are unknown and must be estimated. The goal of the paper is to estimate $\Sigma$ directly, without separately estimating $A$ and $S$ (which would require computing or inverting a Hessian — expensive in high dimensions).

### 1.4 Why $\Sigma$ matters in practice

A consistent estimator $\hat{\Sigma}_n \to \Sigma$ allows:
- **Hypothesis testing:** Wald-type tests $n(\hat{\theta}_n - \theta_0)^\top \hat{\Sigma}_n^{-1} (\hat{\theta}_n - \theta_0) \approx \chi^2_d$
- **Confidence ellipsoids:** $\mathcal{E}_p = \{\theta : (\hat{\theta}_n - \theta)^\top \hat{\Sigma}_n^{-1} (\hat{\theta}_n - \theta) \leq \chi^2_{d, 1-p}\}$
- **Marginal confidence intervals:** $\hat{\theta}_{ni} \pm z_{1-p/2} \sqrt{\hat{\sigma}_{ii}/n}$
- **Prediction uncertainty quantification:** see Section 6.4

---

## 2. Existing Approaches and Their Limitations

### 2.1 Plug-in estimator (Chen et al. 2020)

Chen et al. (2020) proposed two estimators. The first is a plug-in estimator requiring repeated computation of $A^{-1}$, which is expensive ($O(d^3)$ per step). The second is a batch-means variant, which the current paper builds on.

### 2.2 Increasing Batch-Size (IBS) estimator (Zhu et al. 2021)

The motivation for large batch sizes is that $\theta_j$ and $\theta_k$ (with $k > j$) have correlation bounded by:
$$\prod_{i=j}^{k-1} \|I_d - \eta_{i+1} A\| \leq \exp\left(-\lambda_{\min}(A) \sum_{i=j}^{k-1} \eta_{i+1}\right). \tag{4}$$

So iterates become approximately uncorrelated when $\sum_{i=j}^{k-1} \eta_{i+1}$ is large. For $\eta_i = \eta i^{-\alpha}$, Zhu et al. (2021) chose batch sizes:
$$b_{n,k} \propto k^{\frac{1+\alpha}{1-\alpha}}.$$

**Problems with IBS:**
1. **Last batch quality:** As $n$ increases, the sample mean of all but the last batch improves, but the last (largest) batch cannot improve in quality since it contains all recent iterates.
2. **Finite-sample bias:** The estimator is systematically under-biased for finite $n$ (Markovian structure causes negative bias). This leads to overconfident inference.
3. **Lugsail incompatibility:** The IBS structure does not admit the lugsail bias correction, because combining IBS estimators of different batch sizes is non-trivial.
4. **QQ deviation:** The batch-mean vectors for IBS are far from Gaussian for moderate $n$, violating the key assumption that justifies batch-means estimators.

---

## 3. The Equal Batch-Size (EBS) Strategy

### 3.1 General batch-means estimator

After discarding a burn-in (or starting from $\theta_0$), the $n$ SGD iterates are divided into $K = a_n$ batches. For batch sizes $b_{n,1}, \ldots, b_{n,K}$ with $\tau_k = \sum_{j=1}^k b_{n,j}$, the $k$-th batch-mean is:
$$\bar{\theta}_k = b_{n,k}^{-1} \sum_{i=\tau_{k-1}+1}^{\tau_k} \theta_i.$$

The general batch-means estimator of $\Sigma$:
$$\hat{\Sigma}_{\mathrm{gen}} = \frac{1}{K} \sum_{k=1}^K b_{n,k} \left(\bar{\theta}_k - \hat{\theta}_n\right)\left(\bar{\theta}_k - \hat{\theta}_n\right)^\top. \tag{2}$$

### 3.2 Equal batch-size simplification

Under $b_{n,k} = b_n$ for all $k$, the number of batches is $a_n = \lfloor n/b_n \rfloor$, and the estimator simplifies to:
$$\hat{\Sigma}_{b_n} = a_n^{-1} \sum_{k=1}^{a_n} b_n (\bar{\theta}_k - \hat{\theta}_n)(\bar{\theta}_k - \hat{\theta}_n)^\top = \frac{b_n}{a_n} \sum_{k=1}^{a_n} \bar{\theta}_k \bar{\theta}_k^\top - b_n \hat{\theta}_n \hat{\theta}_n^\top. \tag{5}$$

The key intuition: when $b_n$ is large enough, each batch-mean $\bar{\theta}_k$ is approximately $N(\theta^*, \Sigma/b_n)$ (by local CLT), and $\sqrt{b_n}(\bar{\theta}_k - \theta^*)$ are approximately i.i.d. Hence their sample covariance estimates $\Sigma$.

### 3.3 Proposed memory-efficient batching: powers of two

The proposed batch sizes are:
$$b_n^* = \min\{2^\gamma : cn^\beta \leq 2^\gamma \text{ for } \gamma \in \mathbb{N}\}. \tag{6}$$

That is, $b_n^*$ is the **smallest power of 2 that is at least $cn^\beta$**. This ensures:
- $cn^\beta \leq b_n^* \leq 2cn^\beta$ (polynomial growth)
- $a_n^* = n/b_n^*$ satisfies $n^{1-\beta}/(2c) \leq a_n^* \leq n^{1-\beta}/c$ (polynomial decay)
- **Memory efficiency:** At any time $n$, only $a_n^* = O(n^{1-\beta})$ batch-mean vectors are stored, dramatically fewer than $n$.

**The doubling update:** When $n$ increases to the point where $b_n^*$ doubles, adjacent batch-mean vectors are averaged pairwise, halving the number of stored vectors. This is the key memory-saving mechanism.

The polynomial version used in experiments: $b_n = \lfloor cn^\beta \rfloor$.

---

## 4. Assumptions

### (A1) On the objective function $F$:
- (i) $F$ is **strongly convex** with parameter $M > 0$: $F(\theta_2) \geq F(\theta_1) + \nabla F(\theta_1)^\top (\theta_2 - \theta_1) + \frac{M}{2}\|\theta_2 - \theta_1\|^2$
- (ii) $\nabla F$ is **Lipschitz** with constant $L_F$
- (iii) The Hessian $A = \nabla^2 F(\theta^*)$ exists and $\|A(\theta - \theta^*) - \nabla F(\theta)\| \leq L_1 \|\theta - \theta^*\|^2$

Strong convexity implies $\lambda_{\min}(A) \geq M$, a critical ingredient for decorrelation across batches.

### (A2) On the stochastic gradients:
Let $D_i = \theta_i - \theta^*$, $\xi_i = \nabla f(\theta_{i-1}, \zeta_i) - \nabla F(\theta_{i-1})$.
- (i) $f(\cdot, \zeta)$ is continuously differentiable and $\|\nabla f(\theta, \zeta)\|$ is uniformly integrable, ensuring $\mathbb{E}_{i-1}[\xi_i] = 0$ (**martingale difference** structure)
- (ii) Conditional covariance expansion: $\mathbb{E}_{i-1}[\xi_i \xi_i^\top] = S + \mathcal{H}(D_{i-1})$, where $\|\mathcal{H}(D)\| \leq \sigma_1 \|D\| + \sigma_2 \|D\|^2$ and $|\mathrm{tr}(\mathcal{H}(D))| \leq \sigma_1 \|D\| + \sigma_2 \|D\|^2$
- (iii) Bounded 4th moment: $\mathbb{E}_{i-1}\|\xi_i\|^4 \leq \sigma_3 + \sigma_4 \|D_{i-1}\|^4$

These are standard conditions from Chen et al. (2020) and Zhu et al. (2021).

### (A3) On learning rate and batch size:
- (i) $\eta_i = \eta i^{-\alpha}$ with $\alpha \in (0.5, 1)$ (Polyak-Juditsky setup)
- (ii) $b_n n^{-\alpha} \to \infty$ and $b_n n^{-1} \to 0$ as $n \to \infty$

Condition (A3)(ii) ensures:
$$N := \sum_{i=\tau_{k-1}+1}^{\tau_k} \eta_i > \eta b_n \tau_k^{-\alpha} > \eta b_n n^{-\alpha} \to \infty, \tag{7}$$
guaranteeing fast decorrelation between batches. The chosen $b_n = cn^\beta$ satisfies (A3)(ii) for $\beta \in (\alpha, 1)$.

A key constant throughout the proofs:
$$C_d := \max\left\{L_F, L_1, \sigma_1^{2/3}, \sqrt{\sigma_2}, \sqrt{\sigma_3}, \sigma_4^{1/4}, \mathrm{tr}(S)\right\}.$$

---

## 5. Main Theoretical Results

### 5.1 Theorem 1: Consistency of EBS estimator

**Under Assumptions (A1), (A2), (A3), for sufficiently large $n$:**

$$\mathbb{E}\|\hat{\Sigma}_{b_n} - \Sigma\| \lesssim C_d^2 n^{-\alpha/2} a_n^{-1/4} + C_d^3 n^{-\alpha} + C_d a_n^{-1/2} + C_d b_n^{\alpha-1} + C_d b_n^{-1/2} n^{\alpha/2} + C_d a_n^{-1} + C_d^4 n^{-2\alpha} b_n. \tag{Thm 1}$$

Under (A3), each term $\to 0$, proving **mean-square consistency** (and hence consistency in probability and in $L^1$).

**Substituting $b_n = cn^\beta$:** the bound becomes:
$$\mathbb{E}\|\hat{\Sigma}_{b_n} - \Sigma\| \lesssim n^{-\alpha/2+(\beta-1)/4} + n^{(\beta-1)/2} + n^{-\beta(1-\alpha)} + n^{(\alpha-\beta)/2} + n^{\beta-1} + n^{\beta-2\alpha}. \tag{8}$$

**Analysis of dominant terms:**
- By (A3): $(\beta-1)/2 > \beta - 2\alpha$ and $-\alpha/2 + (\beta-1)/4 < (\beta-1)/2$, so the **second term** $n^{(\beta-1)/2}$ dominates among the 1st, 2nd, 5th, 6th.
- Also $-\beta(1-\alpha) < (\alpha-\beta)/2$, so the **fourth term** $n^{(\alpha-\beta)/2}$ dominates among the 3rd and 4th.

Keeping only dominant terms:
$$\mathbb{E}\|\hat{\Sigma}_{b_n} - \Sigma\| \lesssim n^{(\beta-1)/2} + n^{(\alpha-\beta)/2}.$$

**Optimal $\beta$:** Setting the two dominant exponents equal, $(\beta-1)/2 = (\alpha-\beta)/2$, gives $\beta^* = (1+\alpha)/2$.

At $\beta^* = (1+\alpha)/2$, the rate is $n^{(\alpha-1)/4}$, a very mild polynomial rate reflecting the non-stationarity of the SGD chain.

**Remark 2 (finite-$n$ correction):** For sample sizes in the thousands with $\alpha = 0.51$, the numerically optimal $\beta$ is near 0.66, approaching $\beta^* \approx 0.755$ as $n \to \infty$.

**Remark 4 (computational complexity):** At $\beta = (1+\alpha)/2$, computing EBS has complexity $\mathcal{O}(d^2 n^{(1+\alpha)/2} + dn)$, comparable to IBS.

### 5.2 Comparison with IBS bounds

Chen et al. (2020, Theorem 4.3) and Zhu et al. (2021, Theorem 3.1) prove similar mean-square bounds for different increasing batch-size strategies. The Zhu et al. (2021) bound is tighter than Chen et al. (2020). The current paper's EBS bound is of the same order at the optimal $\beta^*$.

**Key advantage of the explicit bound:** It allows principled selection of $b_n$ and reveals the bias-variance tradeoff — increasing $b_n$ reduces variance (fewer, better batches) but increases bias (term $n^{(\alpha-\beta)/2}$ gets worse when $\beta$ grows). The optimal $\beta^*$ balances these.

---

## 6. Bias-Reduced Estimation: Lugsail-EBS

### 6.1 The bias problem

For any finite $n$, the SGD iterates within a batch are positively correlated (Markovian structure). The batch-mean covariance $\mathrm{Cov}(\bar{\theta}_j, \bar{\theta}_k)$ for adjacent $j, k$ is positive, causing systematic **under-estimation** of $\Sigma$.

The bias is negative: $\mathbb{E}[\hat{\Sigma}_{b_n}] - \Sigma < 0$ (component-wise on diagonal). This leads to overconfident confidence intervals with **actual coverage below the nominal level** — a false sense of security.

**Example 1 (mean estimation):** For $y_i = \theta^* + \epsilon_i$, $f(\theta, \zeta) = (y - \theta)^2/2$, the bias is:
$$\mathrm{Bias}(\hat{\Sigma}_{b_n}) \approx \frac{-2C_1}{n} \sum_{1 \leq j < k \leq a_n} \sum_{p=\tau_{j-1}+1}^{\tau_j} \sum_{q=\tau_{k-1}+1}^{\tau_k} q^{-\alpha}(1 - q^{-\alpha})^{q-p}.$$

Even in this simple scalar case, the bias remains significant for moderate $n$ (Figure 2 in the paper).

### 6.2 The lugsail technique

Liu and Flegal (2018) and Vats and Flegal (2022) developed **lugsail estimators** as linear combinations of batch-means estimators at different batch sizes, using flat-top lag windows. The key idea is:

Using the flat-top lag window $w_n(m)$:
$$w_n(m) = \mathbb{1}(|m| \leq b_n/2) + 2\left(1 - \frac{|m|}{b_n}\right)\mathbb{1}(b_n/2 < |m| \leq b_n),$$
the **weighted batch-means estimator** (Liu & Flegal 2018):
$$\hat{\Sigma}_{\mathrm{wBM}} = \sum_{m=1}^{b_n} \frac{m^2 \Delta_2 w_n(m)}{a_m} \sum_{k=0}^{a_m - 1} (\bar{\theta}_{m,k} - \hat{\theta}_n)(\bar{\theta}_{m,k} - \hat{\theta}_n)^\top,$$
where $\Delta_2 w_n(m) = w_n(m-1) - 2w_n(m) + w_n(m+1)$.

Since $\Delta_2 w_n(m) = 0$ except at $m = b_n/2$ and $m = b_n$, employing (9) and renaming $b_n \to 2b_n$:

$$\hat{\Sigma}_{L, b_n} := 2\hat{\Sigma}_{2b_n} - \hat{\Sigma}_{b_n}. \tag{11}$$

### 6.3 Free implementation via the doubling structure

Since $b_n^*$ is always a power of 2, when the batch size is $b_n^*$, the batch size $2b_n^*$ can be obtained by **averaging adjacent batch-mean vectors**:
$$\tilde{\theta}_j = (\bar{\theta}_{2j-1} + \bar{\theta}_{2j}) / 2 \quad \text{(batch-mean at scale } 2b_n).$$

Define:
$$\hat{R}_{b_n} := \frac{b_n}{a_n} \sum_{j=1}^{a_n/2} \left[(\bar{\theta}_{2j-1} - \hat{\theta}_n)(\bar{\theta}_{2j} - \hat{\theta}_n)^\top + (\bar{\theta}_{2j} - \hat{\theta}_n)(\bar{\theta}_{2j-1} - \hat{\theta}_n)^\top\right]. \tag{12}$$

This captures the **cross-covariance between adjacent batches**. Then:
$$\hat{\Sigma}_{L, b_n} = \hat{\Sigma}_{b_n} + 2\hat{R}_{b_n}. \tag{13}$$

When batches are positively correlated ($\hat{R}_{b_n} > 0$), the lugsail correction adds to $\hat{\Sigma}_{b_n}$, reducing the systematic under-estimation. The crucial point: this requires **no additional memory** since $\hat{R}_{b_n}$ is computed from already-stored batch-mean vectors.

**This lugsail correction cannot be applied to the IBS estimator**, because IBS batch sizes are not doubling, and the required estimator at batch size $2b_{n,k}$ is not available for free.

### 6.4 Consistency of lugsail-EBS

**Proposition 5:** Under (A1), (A2), (A3):
$$\mathbb{E}\|\hat{R}_{b_n}\| \lesssim C_d^{1.25} n^{-\alpha/2} a_n^{-1/4} + C_d^2 n^{-\alpha/2} + C_d a_n^{-1/2} + C_d b_n^{\alpha-1} + C_d b_n^{-1/2} n^{\alpha/2} + C_d a_n^{-1} + C_d^4 n^{-2\alpha} b_n.$$

**Corollary 6:** Under the same assumptions:
$$\mathbb{E}\|\hat{\Sigma}_{L,b_n} - \Sigma\| \asymp \mathbb{E}\|\hat{\Sigma}_{b_n} - \Sigma\|.$$

The lugsail-EBS estimator achieves the **same asymptotic rate** as EBS, but substantially better finite-sample performance (Figure 2 and all numerical experiments confirm this).

**Proof sketch of Corollary 6:** The only different-order term between Proposition 5 and Theorem 1 is $n^{-\alpha/2}$. Under (A3)(ii): $b_n^{1/2} n^{-\alpha} < (b_n n^{-1})^\alpha < 1$, so $b_n^{1/2} n^{-\alpha/2} n^{-\alpha/2} < 1$, meaning $n^{-\alpha/2}$ decays faster than $b_n^{-1/2} n^{\alpha/2}$. Hence the bound on $\mathbb{E}\|\hat{R}_{b_n}\|$ is the same order as $\mathbb{E}\|\hat{\Sigma}_{b_n} - \Sigma\|$. Triangle inequality concludes.

**Remark 7 (PSD issue):** The lugsail-EBS estimator may fail to be positive semi-definite for small $n$. Users should choose $c$ large enough (e.g., $c = 0.1$ was sufficient in experiments) to ensure positive-definiteness.

---

## 7. Marginal-Friendly Simultaneous Confidence Regions

### 7.1 Why ellipsoids are not enough

The standard confidence ellipsoid from (1):
$$\mathcal{E}_p = \left\{\theta \in \mathbb{R}^d : (\hat{\theta}_n - \theta)^\top \hat{\Sigma}_n^{-1} (\hat{\theta}_n - \theta) \leq \chi^2_{d, 1-p}\right\}$$

has asymptotic coverage $1-p$ but **no marginal interpretation**: membership in $\mathcal{E}_p$ says nothing about the uncertainty in individual components $\theta^*_i$.

Simple uncorrected marginal CIs:
$$\hat{\theta}_{ni} \pm z_{1-p/2}\sqrt{\hat{\sigma}_{ii}/n}$$

have nominal $1-p$ marginal coverage for each coordinate separately, but the **joint coverage** (all $d$ intervals simultaneously contain their targets) can be as low as $(1-p)^d$ (if independent) or even lower. A naive hyper-rectangular region using these is:
$$C_{\mathrm{lb}}(z_{p/2}) = \prod_{i=1}^d \left[\hat{\theta}_{ni} - z_{1-p/2}\sqrt{\hat{\sigma}_{ii}/n},\ \hat{\theta}_{ni} + z_{1-p/2}\sqrt{\hat{\sigma}_{ii}/n}\right].$$

Bonferroni correction gives an at-least $1-p$ region but with widths inflated by using $z_{1-p/(2d)}$:
$$C_{\mathrm{ub}}(z_{p/2d}) = \prod_{i=1}^d \left[\hat{\theta}_{ni} - z_{1-p/(2d)}\sqrt{\hat{\sigma}_{ii}/n},\ \hat{\theta}_{ni} + z_{1-p/(2d)}\sqrt{\hat{\sigma}_{ii}/n}\right].$$

Clearly $C_{\mathrm{lb}} \subseteq C_{\mathrm{ub}}$, and Bonferroni can be too conservative.

### 7.2 Robertson et al. (2021) quasi-Monte Carlo approach

Robertson et al. (2021) developed an approach for Monte Carlo inference to find the optimal $z^*$ satisfying:
$$z_{1-p/2} < z^* < z_{1-p/(2d)} \quad \text{such that} \quad \mathbb{P}(\theta^* \in C(z^*)) \approx 1-p,$$
where:
$$C(z^*) = \prod_{i=1}^d \left[\hat{\theta}_{ni} - z^*\sqrt{\hat{\sigma}_{ii}/n},\ \hat{\theta}_{ni} + z^*\sqrt{\hat{\sigma}_{ii}/n}\right]. \tag{16}$$

The key insight: the joint probability $\mathbb{P}(\theta^* \in C(z^*))$ depends on all pairwise correlations through $\hat{\Sigma}_n$. Under $\hat{\theta}_n \approx N_d(\theta^*, \hat{\Sigma}_n/n)$, finding the right $z^*$ is a one-dimensional problem (bisection over $[z_{1-p/2}, z_{1-p/(2d)}]$), and utilizes the full correlation structure.

**Benefits:** 
- Maintains exactly $1-p$ joint coverage (vs. $<1-p$ for $C_{\mathrm{lb}}$)
- Much smaller volume than $C_{\mathrm{ub}}$ (Bonferroni is conservative)
- Every component has a marginal interpretation (unlike ellipsoid)
- Exploits off-diagonal elements of $\Sigma$ — richer dependence structure captured

The volume ratio $\left(\mathrm{Vol}(C(z^*))/\mathrm{Vol}(\mathcal{E}_p)\right)^{1/p}$ is reported: values near 1 are preferred; high values indicate overly conservative CIs.

### 7.3 Why IBS over-inflates volume

The IBS estimator tends to **over-estimate off-diagonal correlations** in $\Sigma$ (a consequence of non-Gaussian batch means). This inflates the $z^*$ needed to achieve $1-p$ coverage, producing hyper-rectangular regions that are far larger than necessary. The EBS estimators produce well-calibrated $z^*$ values with volumes close to the ellipsoid.

---

## 8. Proof Strategy and Technical Appendices

### 8.1 Auxiliary sequence decomposition

A central technique (following Chen et al. 2020) is the **auxiliary sequence** $\{U_i\}$:
$$U_i := (I_d - \eta_i A) U_{i-1} + \eta_i \xi_i, \quad U_0 = D_0,$$
where $D_i = \theta_i - \theta^*$. This replaces the nonlinear SGD recursion with a linear one. The difference $\delta_i = D_i - U_i$ satisfies:
$$\delta_i = (I - \eta_i A)\delta_{i-1} + \eta_i(AD_{i-1} - \nabla F(\theta_{i-1})),$$
which is smaller order due to the quadratic error in the Taylor expansion of $\nabla F$.

### 8.2 Key technical lemmas

**Result 1 (Chen et al. 2020, Lemma D.2):** Product matrix bounds:
$$\|Y_j^k\| \leq \exp\left(-\lambda_A \sum_{i=j+1}^k \eta_i\right), \quad \|S_j^k\| \lesssim k^\alpha, \quad \|Z_j^k\| \lesssim k^\alpha j^{-1} + \exp\left(-\lambda_A \sum_{i=j}^k \eta_i\right).$$

**Result 3 (Chen et al. 2020, Lemma 3.2):** Iterate error bounds:
$$\mathbb{E}\|D_n\|^2 \lesssim n^{-\alpha}(C_d + \|D_{n_0}\|^2), \quad \mathbb{E}\|D_n\|^4 \lesssim n^{-2\alpha}(C_d^2 + \|D_{n_0}\|^4).$$

**Lemma 8:** Batch endpoint asymptotics: $\tau_k \asymp k b_n$, $\sum_{i=\tau_{k-1}+1}^{\tau_k} \eta_i > \eta b_n n^{-\alpha} =: N$, etc.

**Lemma 9:** The estimator $\hat{S} = a_n^{-1} \sum_{k=1}^{a_n} b_n^{-1} \left(\sum_{i=\tau_{k-1}+1}^{\tau_k} \xi_i\right)\left(\sum_{i=\tau_{k-1}+1}^{\tau_k} \xi_i\right)^\top$ is consistent for $S$:
$$\mathbb{E}\|\hat{S} - S\|^2 \lesssim C_d^4 n^{-\alpha} a_n^{-1/2} + C_d^6 n^{-2\alpha} + C_d^2 a_n^{-1}.$$

**Lemma 10:** The core auxiliary sequence batch-means: $\mathbb{E}\left\|a_n^{-1}\sum_{k=1}^{a_n} b_n \bar{U}_k \bar{U}_k^\top - A^{-1}SA^{-1}\right\|$ is bounded by the terms in Theorem 1.

**Lemma 11:** The full batch-means (replacing $U_k$ with $\theta_k - \theta^*$) has the same bound.

### 8.3 Proof structure of Theorem 1 (Appendix C)

Define $\bar{D}_n = n^{-1}\sum_{i=1}^{\tau_{a_n}} D_i$. The proof decomposes:
$$a_n^{-1}\sum_{k=1}^{a_n} b_n (\bar{\theta}_k - \hat{\theta}_n)(\bar{\theta}_k - \hat{\theta}_n)^\top = a_n^{-1}\sum_{k=1}^{a_n} b_n \bar{D}_k \bar{D}_k^\top - b_n \bar{D}_n \bar{D}_n^\top$$
and writes $D_i = U_i + \delta_i$, then bounds cross-terms using Cauchy-Schwarz and the auxiliary sequence bounds. The $\delta_i$ terms are controlled by the nonlinear error in (A1)(iii).

### 8.4 Proof of lugsail consistency (Appendix D)

The alternate expression $\hat{\Sigma}_{L,b_n} = \hat{\Sigma}_{b_n} + 2\hat{R}_{b_n}$ is derived in Appendix D.2 by writing:
$$\hat{\Sigma}_{2b_n} = \frac{2b_n}{a_n/2} \sum_{j=1}^{a_n/2} \left(\frac{\bar{\theta}_{2j-1} + \bar{\theta}_{2j}}{2} - \hat{\theta}_n\right)\left(\frac{\bar{\theta}_{2j-1} + \bar{\theta}_{2j}}{2} - \hat{\theta}_n\right)^\top = \hat{\Sigma}_{b_n} + \hat{R}_{b_n},$$
so $\hat{\Sigma}_{L,b_n} = 2\hat{\Sigma}_{2b_n} - \hat{\Sigma}_{b_n} = \hat{\Sigma}_{b_n} + 2\hat{R}_{b_n}$.

Proposition 5 is proved via Lemmas 12 and 13 that control cross-covariance terms between adjacent batches in the auxiliary sequence decomposition.

---

## 9. Numerical Experiments

**Setup:** $\alpha = 0.51$, $\beta = (1+\alpha)/2 \approx 0.755$, $c = 0.1$ (for positive-definiteness). 1000 replications. Data sizes up to $5 \times 10^6$. Burn-in: first 1000 SGD iterates discarded.

**Estimators compared:**
| Name | Description |
|------|-------------|
| EBS | Doubling batch size $b_n^*$ |
| L-EBS | Lugsail-EBS (bias corrected) |
| EBS-poly | Polynomial batch size $b_n = \lfloor cn^\beta \rfloor$ |
| L-EBS-poly | Lugsail-EBS with polynomial batch size |
| IBS | Increasing batch size (Zhu et al. 2021) |
| Oracle | True $\Sigma$ (for reference) |

**Metrics:**
1. Relative Frobenius norm: $\|\hat{\Sigma} - \Sigma\|_F / \|\Sigma\|_F$
2. Multivariate ellipsoidal coverage probability (nominal 90%)
3. Simultaneous marginal-friendly coverage probability (nominal 90%)
4. Volume ratio: $\left(\mathrm{Vol}(C(z^*))/\mathrm{Vol}(\mathcal{E}_p)\right)^{1/p}$

### 9.1 Linear Regression ($d=5$, three correlation structures)

Model: $y_i = x_i^\top \beta^* + \epsilon_i$, $x_i \sim N(0, A)$, $\epsilon_i \sim N(0,1)$. True $\Sigma = A^{-1}$.

Three forms of $A$ (all with $\rho = 0.5$):
- Identity: $A = I_d$
- Toeplitz: $A_{ij} = \rho^{|i-j|}$
- Equicorrelation: $A_{ij} = \rho$ for $i \neq j$, 1 otherwise

**Results (Figure 4):**
- **Frobenius norm:** L-EBS and L-EBS-poly converge fastest; EBS and EBS-poly competitive with IBS or better. IBS starts with lower error for very small $n$ (its design targets large batches) but EBS-based estimators match or surpass it.
- **Multivariate ellipsoidal coverage:** L-EBS variants reach 90% coverage fastest.
- **Simultaneous marginal coverage (the critical metric):** EBS-based variants achieve target 90% coverage, while IBS drastically overshoots (90%+ simultaneous coverage) — but this is achieved through vastly inflated confidence regions, NOT through better estimation.
- **Volume ratio:** IBS regions are 10-100× larger than ellipsoids; EBS/L-EBS regions are near 1 (compact and well-calibrated).

### 9.2 Linear Regression ($d=20$, identity $A$)

The advantage of EBS becomes more pronounced in higher dimension (Figure 5). The volume inflation of IBS is even more dramatic, while L-EBS maintains near-oracle performance.

### 9.3 QQ plot analysis (Figure 6)

For $d=5$ with identity $A$: QQ plots of all components of all batch-mean vectors for IBS vs. EBS.
- **IBS:** Significant deviation from the theoretical Gaussian quantiles at $n = 50000$; still notable at $n = 10^6$.
- **EBS:** Follows Gaussian quantiles well at both sample sizes.

This confirms the heuristic: EBS batch-means are approximately Gaussian (as required for batch-means theory), while IBS batch-means are not, especially in finite samples.

### 9.4 LAD Regression ($d=20$)

Model: $y_i = x_i^\top \beta^* + \epsilon_i$, $\epsilon_i \sim \mathrm{DE}(0,1)$ (double exponential). Loss: $f(\beta, \zeta) = |y - x^\top \beta|$. True $\Sigma = A^{-1}$.

Results (Figure 7): Qualitatively identical conclusions. L-EBS significantly superior to IBS across all three correlation structures. IBS over-inflation of confidence regions is even more pronounced.

### 9.5 Prediction improvement via $\Sigma$ (Section 6.4)

**Setup:** Binary classification with logistic model $y_i \sim \mathrm{Bernoulli}(p_i)$, $p_i = \sigma(x_i^\top \beta^*)$.

**Standard prediction:** $\hat{y}_j = \mathbb{1}(\hat{p}_j > q)$ for threshold $q$.

**Improved prediction using EBS:** By the delta method:
$$\sqrt{n}(\hat{p}_j - p_j) \xrightarrow{d} N\left(0, (p_j(1-p_j))^2 x_j^\top \Sigma x_j\right).$$

A confidence interval for $p_j$: $\hat{p}_j \pm z_{0.975} \cdot \mathrm{se}_j$. Modified classification:
$$\tilde{y}_j = \mathbb{1}(\hat{p}_j - z_{0.975} \cdot \mathrm{se}_j > q),$$
classifying observation $j$ as positive only if the lower confidence bound exceeds $q$. This **reduces false positives** for appropriate $q$.

**Four real datasets:**
1. Santander Customer Transaction (Kaggle)
2. Covertype (Blackard 1998)
3. Spambase (Hopkins et al. 1999)
4. Diabetes Health Indicators (UCI ML Repository)

**Result (Figure 8):** The blue curve (with EBS confidence intervals) achieves **lower misclassification rate** than the black curve (without CIs) for small-to-moderate threshold values $q$. As $n \to \infty$ (large training set), $\mathrm{se}_j \to 0$ and both curves merge — consistent with theory.

---

## 10. Discussion and Extensions

### 10.1 $k$-neighboring batch averaging

The EBS strategy can be extended to average over $k$-neighboring batches for any fixed $k \geq 1$. All theoretical results continue to hold. However, larger $k$ reduces the effective number of batches $a_n/k$, decreasing efficiency of the covariance estimator.

### 10.2 Overlapping EBS

An overlapping batch-means estimator with EBS strategy would allow $O(n)$ batches (instead of $O(n^{1-\beta})$), achieving higher efficiency. However, the computational complexity would be $\mathcal{O}(d^2 n)$ (since overlapping batches share iterates). Similar theoretical results should hold, but their derivation requires handling the correlation between overlapping batch-means — an open direction.

### 10.3 Other SGD variants

The framework applies to any SGD variant for which Polyak-Juditsky asymptotic normality holds (1), including:
- Mini-batch SGD
- SGD with momentum (implicit SGD, Toulis & Airoldi 2017)
- Stochastic variance-reduced gradient methods (SVRG, SAGA, etc.)
- SGD with non-iid (Markovian) data — marginal-friendly CIs and delta-method predictions are directly applicable

### 10.4 Comparison with bootstrap approaches

Fang et al. (2018) and Xie et al. (2023) propose bootstrap techniques for estimating $\Sigma$. These are computationally expensive ($\mathcal{O}(d^2 n)$ per bootstrap replicate) and lack theoretical guarantees for the covariance estimator itself. The EBS approach is $\mathcal{O}(d^2 n^{(1+\alpha)/2})$ and has explicit consistency rates.

### 10.5 The Zhu & Dong (2021) approach

Zhu and Dong (2021) propose inference without consistent estimation of $\Sigma$ (using cancellation methods). While this yields valid confidence regions, it does not enable:
- Marginal inference (delta method requires $\Sigma$)
- Prediction uncertainty quantification
- Full uncertainty interpretation

### 10.6 Non-iid data

Li et al. (2023) and Liu et al. (2023) estimate the limiting covariance when the iid assumption on $\zeta_i$ is violated (Markovian observations). The EBS marginal-friendly CI construction and delta-method prediction improvement are directly applicable to these settings as well.

---

## 11. Relevance to Thesis (RR for LSA with Markovian Noise)

This paper is highly relevant to the thesis topic on the CLT and Berry-Esseen for Richardson-Romberg (RR) iterates in linear stochastic approximation (LSA) with Markovian noise:

1. **Same algebraic structure:** The ASGD iterate satisfies the LSA recursion $\theta_i = \theta_{i-1} - \eta_i \nabla f(\theta_{i-1}, \zeta_i)$, which linearizes to $D_i \approx (I - \eta_i A)D_{i-1} + \eta_i \xi_i$ — exactly the LSA with decreasing step sizes. The paper's auxiliary sequence technique is directly the linearization approach standard in SA analysis.

2. **Same sandwich formula $\Sigma = A^{-1}SA^{-1}$:** The asymptotic variance of ASGD has the same form as the CLT variance in general LSA. Estimating this matrix is precisely what's needed for inference after RR extrapolation.

3. **EBS for RR iterates:** The EBS batch-means estimator could be applied directly to RR iterates (which are linear combinations of two ASGD sequences) to obtain a consistent estimator of the RR asymptotic variance. The consistency proof would follow analogously, adapted for the two-timescale structure.

4. **Lugsail bias correction for RR:** The lugsail technique is applicable to any consistent batch-means estimator. For RR iterates, the same doubling trick would apply, potentially correcting the systematic under-estimation bias for finite-sample RR inference.

5. **Marginal-friendly CIs for RR:** The Robertson et al. (2021) CI construction (Section 5) applies immediately once a consistent estimator of the RR asymptotic covariance is available.

6. **Markovian noise extension:** The paper assumes iid $\zeta_i$, whereas the thesis considers Markovian $\zeta_i$. The main modification is that $\xi_i$ is no longer a martingale difference but has temporal correlations. The bias-correction (lugsail) becomes even more important in this case, as Markovian correlations compound the finite-sample under-estimation of $\Sigma$.

7. **MCMC-SGD parallel:** The paper explicitly draws the parallel between batch-means in MCMC (Flegal & Jones 2010) and SGD, noting that key differences (non-stationarity, time-inhomogeneity) require new proofs. For Markovian-noise LSA, the SGD chain itself has Markovian structure from both the noise AND the Markovian data — the decorrelation analysis becomes more complex (mixing time of the data Markov chain enters).

---

## 12. Summary of Key Formulas

| Symbol | Definition |
|--------|-----------|
| $\theta_i$ | $i$-th SGD iterate |
| $\hat{\theta}_n = n^{-1}\sum_{i=1}^n \theta_i$ | ASGD estimator |
| $\Sigma = A^{-1}SA^{-1}$ | Asymptotic covariance of $\sqrt{n}\hat{\theta}_n$ |
| $\bar{\theta}_k = b_n^{-1}\sum_{i=\tau_{k-1}+1}^{\tau_k} \theta_i$ | $k$-th batch-mean |
| $\hat{\Sigma}_{b_n} = a_n^{-1}\sum_k b_n(\bar{\theta}_k - \hat{\theta}_n)(\bar{\theta}_k - \hat{\theta}_n)^\top$ | EBS estimator |
| $b_n^* = \min\{2^\gamma : cn^\beta \leq 2^\gamma\}$ | Doubling batch size |
| $\hat{\Sigma}_{L,b_n} = 2\hat{\Sigma}_{2b_n} - \hat{\Sigma}_{b_n} = \hat{\Sigma}_{b_n} + 2\hat{R}_{b_n}$ | Lugsail-EBS estimator |
| $\hat{R}_{b_n} = \frac{b_n}{a_n}\sum_j [(\bar{\theta}_{2j-1}-\hat{\theta}_n)(\bar{\theta}_{2j}-\hat{\theta}_n)^\top + \text{sym}]$ | Adjacent-batch cross-covariance |
| $\beta^* = (1+\alpha)/2$ | Optimal batch growth exponent |
| Rate at $\beta^*$: $n^{(\alpha-1)/4}$ | Optimal MSE rate for EBS |
| $C(z^*) = \prod_i [\hat{\theta}_{ni} \pm z^*\sqrt{\hat{\sigma}_{ii}/n}]$ | Marginal-friendly simultaneous CI |

---

## 13. Limitations and Open Problems

1. **Online implementation:** The EBS estimator requires storing $a_n$ batch-mean vectors. An online update strategy (computing $\hat{\Sigma}$ without storing all past batch-means) remains open.

2. **Dependent data:** Extension to Markovian observations $\zeta_1, \zeta_2, \ldots$ (as in online RL, time series SGD) requires additional mixing-time arguments.

3. **Non-convex objectives:** The paper assumes strong convexity. Extension to non-convex $F$ (e.g., deep learning) would require different asymptotic normality results.

4. **Optimal bias-variance tradeoff in finite samples:** The optimal $\beta$ is approximately $(1+\alpha)/2$ asymptotically, but can differ significantly for $n$ in the thousands. Data-driven selection of $\beta$ is not addressed.

5. **High-dimensional regime:** All results assume $d$ is fixed. Rates likely degrade with $d$ (the constant $C_d$ grows with $d$). High-dimensional asymptotics ($d \to \infty$ with $n$) are not addressed.

6. **Overlapping EBS:** The combination of overlapping batch-means with equal batch sizes would use all $O(n)$ batches, dramatically improving efficiency, but requires new proof techniques for correlated batch-means.

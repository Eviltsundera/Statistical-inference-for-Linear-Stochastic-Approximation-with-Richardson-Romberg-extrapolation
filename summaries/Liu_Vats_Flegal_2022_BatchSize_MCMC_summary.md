# Deep Review: Liu, Vats, Flegal (2022)
# "Batch size selection for variance estimators in MCMC"

**Authors:** Ying Liu, Dootika Vats, James M. Flegal  
**arXiv:** 1804.05975v3 (July 2019), published in Stat. Comput. 2022  
**Keywords:** batch means, overlapping batch means, MSE-optimal batch size, AR(m)-fit, HAC, MCMC, long-run variance

---

## 0. Place in the Literature

Estimating the **long-run covariance matrix** $\Sigma$ in MCMC is the key ingredient for building confidence regions and stopping rules. Three prior works set the stage:

- **Andrews (1991)** — MSE-optimal bandwidth selection for spectral variance estimators in HAC regression; obtains $b \propto n^{1/5}$ for second-order kernels and $b \propto n^{1/3}$ for Bartlett. Results are computationally infeasible for high-dimensional MCMC.
- **Flegal & Jones (2010)** — strong consistency of BM and OBM for $\Sigma$ under $b \to \infty$, $n/b \to \infty$; proposed the heuristic $b = \lfloor n^{1/2} \rfloor$.
- **Vats et al. (2018)** — multivariate consistency of the generalized OBM (including all lag windows) for polynomially ergodic Markov chains; established a strong invariance principle.

**This paper** closes the gap: it moves from "the estimator is consistent for any $b \to \infty$" to "how to actually choose $b$ to minimize MSE." This is a fundamentally new step.

---

## 1. Setup and Notation

### 1.1 Probabilistic Model

Let $F$ be a probability measure with support $\mathsf{X} \subseteq \mathbb{R}^d$, and $g : \mathsf{X} \to \mathbb{R}^p$ be $F$-integrable.

**Goal:** estimate the $p$-dimensional vector
$$\theta := \int_{\mathsf{X}} g(x)\, dF.$$

**Tool:** a Harris $F$-ergodic Markov chain $\{X_t\}_{t \ge 1}$. Observation process: $Y_t = g(X_t)$.

By the ergodic theorem:
$$\bar{Y} = \frac{1}{n}\sum_{t=1}^{n} Y_t \xrightarrow{a.s.} \theta.$$

**Markov chain CLT** (Jones, 2004 et al.):
$$\sqrt{n}(\bar{Y} - \theta) \xrightarrow{d} N_p(0, \Sigma),$$
where the **long-run covariance matrix** is:
$$\Sigma = \sum_{k=-\infty}^{\infty} \mathrm{Cov}_F(Y_1, Y_{1+k}) = \mathrm{Var}_F(Y_1) + 2\sum_{k=1}^{\infty}\mathrm{Cov}_F(Y_1, Y_{1+k}).$$

Note that $\Sigma$ accounts for **all serial correlation** in $\{Y_t\}$ and equals the spectral density at frequency zero times $2\pi$.

### 1.2 Lag Autocovariances

Define the matrix lag autocovariance:
$$R(k) = E_F(Y_t - \theta)(Y_{t+k} - \theta)^T, \quad k \ge 0.$$

Then $\Sigma = \sum_{k=-\infty}^{\infty} R(k) = R(0) + \sum_{k=1}^{\infty}[R(k) + R(k)^T]$.

### 1.3 The Matrix $\Gamma$ — the Key Object

The authors introduce:
$$\Gamma = -\sum_{k=1}^{\infty} k\bigl[R(k) + R(k)^T\bigr],$$
with components $\Gamma_{ij} = -\sum_{k=1}^{\infty} k\bigl[R_{ij}(k) + R_{ji}(k)\bigr]$.

$\Gamma$ is a weighted sum of lag autocovariances with weights $-k$. It **controls the bias** of the estimator $\hat{\Sigma}$: the faster the correlations decay (fast mixing), the smaller $|\Gamma_{ij}|$, and the smaller the bias.

For an AR(1) process $Y_t = \phi Y_{t-1} + \epsilon_t$: $\Gamma = -2\phi \sigma^2/(1-\phi)^3$ (scalar for $p=1$).

---

## 2. The Generalized OBM Estimator: Construction

### 2.1 Motivation via Weighted Outer Products

The classical OBM estimator uses overlapping batches of length $b$:
$$\hat{\Sigma}_{OBM} = \frac{b}{n-b+1}\sum_{l=0}^{n-b}(\bar{Y}_l(b) - \bar{Y})(\bar{Y}_l(b) - \bar{Y})^T,$$
where $\bar{Y}_l(b) = b^{-1}\sum_{t=l+1}^{l+b} Y_t$.

The generalized estimator **extends** this by summing over all scales $k = 1, \ldots, b$ with weights defined by the lag window $w_n$:

$$\hat{\Sigma}_w = \frac{1}{n}\sum_{k=1}^{b}\sum_{l=0}^{n-k} k^2 \Delta_2 w_n(k)\,\bigl(\bar{Y}_l(k) - \bar{Y}\bigr)\bigl(\bar{Y}_l(k) - \bar{Y}\bigr)^T. \tag{1}$$

Here $\Delta_2 w_n(k) = w_n(k-1) - 2w_n(k) + w_n(k+1)$ is the **second difference** of the lag window.

### 2.2 Why Second Differences?

The key idea: the classical spectral estimator $\hat{\Sigma}_{SV} = \sum_{k} w_n(k)\hat{R}(k)$ can be rewritten via summation by parts:
$$\hat{\Sigma}_{SV} = \sum_{k=1}^{b}\Delta_2 w_n(k) \cdot \bigl[\text{sum of outer products of batch means at scale } k\bigr].$$

This is equivalent to (1), and this representation is **computationally more efficient**: outer products $(\bar{Y}_l(k) - \bar{Y})(\bar{Y}_l(k) - \bar{Y})^T$ are updated incrementally rather than computed from scratch.

### 2.3 Lag Windows: Formal Definitions

The window $w_n : \mathbb{Z} \to \mathbb{R}$ is an even function satisfying:
1. $|w_n(k)| \le 1$ for all $n, k$
2. $w_n(0) = 1$ for all $n$
3. $w_n(k) = 0$ for all $|k| \ge b$

**Bartlett:** $w_n(k) = (1 - |k|/b)\cdot I(|k| \le b)$

Second differences: $\Delta_2 w_n(b) = 1/b$ and $\Delta_2 w_n(k) = 0$ for $k < b$.
Therefore $C = b \cdot (1/b) = 1$ and estimator (1) reduces to:
$$\hat{\Sigma}_B(b) = \frac{b}{n}\sum_{l=0}^{n-b}(\bar{Y}_l(b) - \bar{Y})(\bar{Y}_l(b) - \bar{Y})^T,$$
which is the standard OBM estimator.

**Flat-top (Bartlett FT):**
$$w_n(k) = I(|k| \le b/2) + (2 - 2|k|/b)\cdot I(b/2 < |k| \le b).$$

Computing second differences: $\Delta_2 w_n(b/2) = -2/b$; $\Delta_2 w_n(b) = 2/b$; all others zero.
Therefore $C = b \cdot [(-2/b) + (2/b)] = 0$ — the key property.

The estimator reduces to: $\hat{\Sigma}_F(b) = 2\hat{\Sigma}_B(b) - \hat{\Sigma}_B(b/2)$.
This is a **linear combination** of two OBM estimators: it **corrects the bias** at the cost of higher variance ($S = 4/3$ vs $S = 2/3$).

**Interpretation of flat-top:** $\hat{\Sigma}_B(b)$ has bias $\propto b^{-1}$; the combination $2\hat{\Sigma}_B(b) - \hat{\Sigma}_B(b/2)$ cancels the leading bias term.

**Tukey-Hanning:** $w_n(k) = \frac{1+\cos(\pi k/b)}{2}\cdot I(|k| \le b)$.
Nonlinear window → nonlinear $S$ dependence on $b$ → no closed-form MSE-optimal $b$. Authors do not study it in detail.

### 2.4 Non-overlapping BM (Liu & Flegal, 2018)

For $n = ab$ (exactly $a$ batches of length $b$):
$$\hat{\Sigma}_{BM}(b) = \frac{b}{a-1}\sum_{l=0}^{a-1}(\bar{Y}_l - \bar{Y})(\bar{Y}_l - \bar{Y})^T,$$
where $\bar{Y}_l = b^{-1}\sum_{t=lb+1}^{(l+1)b}Y_t$.

BM is a special case of the generalized estimator with overlap parameter $r=1$ in Vats & Flegal (2018). MSE for BM coincides with (2) at $C=1$, $S=1$.

---

## 3. Asymptotic MSE: Detailed Analysis

### 3.1 The Invariance Principle (Lemma 1, Vats et al. 2018)

**Ergodicity assumption:** $\{X_t\}$ is a polynomially ergodic chain of order $\xi \ge (1+\epsilon)(1+2/\delta)$ for some $\epsilon, \delta > 0$; $E_F\|f(X)\|^{2+\delta} < \infty$.

Then for a lower-triangular $p \times p$ matrix $L$ with $\Sigma = LL^T$, some $\lambda > 0$, and a finite random variable $D$:
$$\left\|\sum_{t=1}^{n}f(X_t) - nE_F f - LB(n)\right\| < Dn^{1/2-\lambda} \quad \text{a.s.},$$
where $B(n)$ is a $p$-dimensional standard Brownian motion.

This is the **strong invariance principle**: the partial sums of the chain are approximated by Brownian motion at rate $n^{1/2-\lambda}$.

**Role of $\lambda$:** parameter related to the mixing rate of the chain. The closer $\lambda$ is to $1/2$, the faster the chain mixes. This constraint appears in the conditions on $b$ (Assumption 1).

### 3.2 Assumption 1

$b$ is an integer sequence such that:
- $b \to \infty$ and $n/b \to \infty$ monotonically (consistency condition),
- $b n^{1-2\lambda}(\sum_{k=1}^{b}|\Delta_2 w_n(k)|)^2 \log n \to 0$ (technical, controls approximation error),
- $n^{1-2\lambda}\sum_{k=1}^{b}|\Delta_2 w_n(k)| \to 0$.

For $b = \lfloor n^\nu \rfloor$: conditions hold when $1 - 2\lambda - \nu < 0$, i.e., $\nu > 1 - 2\lambda$.

### 3.3 Theorem 1: MSE (in Detail)

**Conditions:**
- Lemma 1 holds for $f = g$ and $f = g^2$ (elementwise).
- $E_F D^4 < \infty$, $E_F\|g\|^{4+\delta} < \infty$.
- Four technical conditions on the lag window (normalization, decay of second differences, compatibility with the invariance principle).

**MSE formula:**
$$\mathrm{MSE}\!\left(\hat{\Sigma}_{w,ij}\right) = \underbrace{\frac{C^2\Gamma_{ij}^2}{b^2}}_{\text{bias}^2} + \underbrace{[\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2]\cdot S\cdot\frac{b}{n}}_{\text{variance}} + o\!\left(\frac{b}{n}\right) + o\!\left(\frac{1}{b}\right). \tag{2}$$

#### Origin of the bias² term

From Theorem 2 (Appendix A):
$$\mathrm{Bias}(\hat{\Sigma}_{w,ij}) = \sum_{k=1}^{b}\Delta_2 w_n(k)\cdot\Gamma_{ij} + o(b/n) + o(1/b).$$

Leading term: $\Gamma_{ij}\sum_{k=1}^{b}\Delta_2 w_n(k)$.

Since $C = b\sum_{k=1}^{b}\Delta_2 w_n(k)$:
$$\mathrm{Bias}(\hat{\Sigma}_{w,ij}) \approx \frac{C\Gamma_{ij}}{b}.$$

Squaring: $\mathrm{Bias}^2 \approx C^2\Gamma_{ij}^2/b^2$. ✓

**Key step in the proof of Theorem 2:** uses the exact expression (from Vats & Flegal, 2018):
$$\mathrm{Cov}[\bar{Y}_l^{(i)}(k), \bar{Y}_l^{(j)}(k)] - \mathrm{Cov}[\bar{Y}^{(i)}, \bar{Y}^{(j)}] = \frac{n-k}{kn}\!\left(\Sigma_{ij} + \frac{n+k}{kn}\Gamma_{ij} + o(1/k^2)\right). \tag{4}$$

After summing over $k$ with weights $\Delta_2 w_n(k)$ and applying summation by parts, the theorem follows.

#### Origin of the variance term

From Theorem 3 (Appendix C), using the strong invariance principle:
$$\mathrm{Var}(\hat{\Sigma}_{w,jj}) = [\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2]\cdot S\cdot\frac{b}{n} + o(b/n),$$
where:
$$S = \frac{2}{3}\sum_{k=1}^{b}(\Delta_2 w_k)^2 k^3 \frac{1}{n} + 2\sum_{t=1}^{b-1}\sum_{u=1}^{b-t}\Delta_2 w_u\,\Delta_2 w_{t+u}\!\left(\frac{2}{3}u^3 + u^2 t\right)\frac{1}{n}.$$

In the limit as $b/n \to 0$:
- Bartlett: $S = 2/3$
- BM: $S = 1$
- Flat-top: $S = 4/3$

**Structure of the variance term:** $[\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2]$ is the 4th mixed moment of a centered normal vector $(Y^{(i)}, Y^{(j)})$ (Isserlis/Wick formula). It arises because the estimator is quadratic in the data (outer products of batch means).

**Lemma 2 (key intermediate result for Theorem 3):** For $0 < c_2 < c_1 < 1$:
$$A_2 = \left[\frac{2}{3}(\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2)\frac{c_1 b}{n} + \Sigma_{ij}^2\!\left(1 - 4\frac{c_1 b}{n}\right)\right] + o(b/n),$$
$$A_3 = \frac{(c_2 - 3c_1)c_2}{3c_1}(\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2)\frac{b}{n} + 2(c_1+c_2)\Sigma_{ij}^2\frac{b}{n} - \Sigma_{ij}^2 + o(b/n).$$

This lemma computes fourth moments of Brownian increments using **Proposition 1** (for $(X,Y)$ jointly normal: $E[X^2Y^2] = 2l_{12}^2 + l_{11}l_{22}$) and **Proposition 2** (Janssen & Stoica isomoment theorem for four jointly normal variables).

### 3.4 Minimizing MSE over $b$

Setting the derivative of (2) with respect to $b$ to zero:
$$\frac{d}{db}\mathrm{MSE} = -\frac{2C^2\Gamma_{ij}^2}{b^3} + [\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2]\cdot S\cdot\frac{1}{n} = 0.$$

Solution:
$$b_{\mathrm{opt},ij} = \left(\frac{2C^2\Gamma_{ij}^2}{\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2}\cdot\frac{n}{S}\right)^{1/3}. \tag{3}$$

**Scale:** $b_{\mathrm{opt},ij} \propto n^{1/3}$ — **substantially smaller** than $n^{1/2}$.

**Optimal MSE** at $b = b_{\mathrm{opt}}$:
$$\mathrm{MSE}(b_{\mathrm{opt}}) = \frac{3}{2}\left[\frac{(C^2\Gamma_{ij}^2)^{1/3}([\Sigma_{ii}\Sigma_{jj}+\Sigma_{ij}^2]S)^{2/3}}{n^{2/3}}\right] \propto n^{-2/3},$$
which is the standard MSE decay rate for nonparametric estimators.

### 3.5 Special Cases

**Bartlett ($C=1$, $S=2/3$):**
$$b_{\mathrm{opt},ij}^{(B)} = \left(\frac{3\Gamma_{ij}^2 n}{\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2}\right)^{1/3}.$$

This coincides with Andrews (1991) for the Bartlett kernel in the HAC context (though there $b$ is called bandwidth).

**BM ($C=1$, $S=1$):**
$$b_{\mathrm{opt},ij}^{(BM)} = \left(\frac{2\Gamma_{ij}^2 n}{\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2}\right)^{1/3}.$$

BM is slightly more conservative (requires smaller $b$) because it has higher variance ($S=1 > 2/3$).

**Flat-top ($C=0$, $S=4/3$):**
$$\mathrm{MSE}(\hat{\Sigma}_F) = 0 + [\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2]\cdot\frac{4}{3}\cdot\frac{b}{n} + o(b/n).$$

MSE is strictly decreasing in $b$ → formally $b_{\mathrm{opt}} = 0$, which is meaningless.

**Physical meaning:** flat-top corrects bias to zero (due to $C=0$), but at the cost of higher variance. The MSE criterion penalizes variance, so smaller $b$ looks "better" — but this is a false signal. In practice, flat-top works well at moderate $b$.

---

## 4. Practical Batch Size Estimation: Full Breakdown

### 4.1 The Aggregation Problem

Formula (3) gives $b_{\mathrm{opt},ij}$ for each $(i,j)$. For a $p \times p$ matrix, this is $p(p+1)/2$ different values. Using different $b$ for each element is:
- computationally infeasible,
- conceptually inconsistent (the matrix must be a single object).

The authors propose **aggregating over diagonal elements** (most important for confidence regions). Define:
$$b_{\mathrm{opt}} \propto \left(\frac{\sum_{i=1}^{p}\Gamma_{ii}^2}{\sum_{i=1}^{p}\Sigma_{ii}^2}\right)^{1/3}\cdot n^{1/3},$$
where the proportionality constant depends on $C$ and $S$.

This resembles the Andrews (1991) weighting system with a unit weight matrix, adapted for MCMC.

### 4.2 AR(m) Approximation: Step by Step

Directly estimating $\Gamma_{ii}$ and $\Sigma_{ii}$ from data is a nonparametric spectral density estimation problem requiring its own window choice. The authors propose a **parametric shortcut**: fitting an AR($m$) model.

**Step 1: Model**

Fit a stationary AR($m$) process to the $i$-th component $\{Y_t^{(i)}\}$:
$$W_t = \sum_{j=1}^{m}\phi_j W_{t-j} + \epsilon_t, \quad \epsilon_t \overset{iid}{\sim}(0, \sigma_\epsilon^2).$$

**Step 2: Order selection**

Order $m$ is chosen by **AIC**:
$$\mathrm{AIC}(m) = n\log\hat{\sigma}_\epsilon^2(m) + 2m.$$

In practice, $m$ is bounded above by $\lfloor 10\log_{10}n \rfloor$ (R default for `ar()`).

**Why not AR(1)?** Components $Y_t^{(i)} = g_i(X_t)$ of a Markov chain are **not necessarily** first-order Markovian. For slowly mixing chains, $g(X_t)$ can have long autocorrelation functions that AR(1) cannot capture. AR($m$) for large enough $m$ can approximate any ARMA structure.

**Step 3: Yule-Walker equations**

Sample autocovariances $\hat{\gamma}(k)$ satisfy the Yule-Walker system:
$$\begin{pmatrix}\hat{\gamma}(0) & \cdots & \hat{\gamma}(m-1) \\ \vdots & \ddots & \vdots \\ \hat{\gamma}(m-1) & \cdots & \hat{\gamma}(0)\end{pmatrix}\begin{pmatrix}\hat{\phi}_1 \\ \vdots \\ \hat{\phi}_m\end{pmatrix} = \begin{pmatrix}\hat{\gamma}(1) \\ \vdots \\ \hat{\gamma}(m)\end{pmatrix}, \quad \hat{\sigma}_\epsilon^2 = \hat{\gamma}(0) - \sum_{j=1}^{m}\hat{\phi}_j\hat{\gamma}(j).$$

Solved via the Durbin-Levinson algorithm in $O(m^2)$ operations.

**Step 4: Closed-form formulas for $\Sigma_{p,i}$ and $\Gamma_{p,i}$**

For AR($m$), it is known that:
$$\sum_{k=-\infty}^{\infty}\gamma(k) = \frac{\sigma_\epsilon^2}{(1 - \sum_{j=1}^{m}\phi_j)^2}.$$
Therefore:
$$\Sigma_{p,i} = \frac{\hat{\sigma}_\epsilon^2}{(1 - \sum_{j=1}^{m}\hat{\phi}_j)^2}.$$

For $\Gamma_{p,i}$, the authors use a decomposition of $\sum_{k=1}^{\infty}k\gamma(k)$ via Yule-Walker equations (following Taylor, 2018). Expanding $\gamma(k) = \sum_{j=1}^{m}\phi_j\gamma(k-j)$ for $k > 0$ and weighting by $k$:

$$\sum_{k=1}^{\infty}k\gamma(k) = \left[\sum_{j=1}^{m}\phi_j\sum_{k=1}^{j}k\gamma(k-j)\right] + \frac{\sigma_\epsilon^2 - \gamma(0)}{2}\left(\sum_{j=1}^{m}j\phi_j\right)\left(\frac{1}{1 - \sum_{j=1}^{m}\phi_j}\right).$$

Substituting estimates:
$$\Gamma_{p,i} = -2\!\left[\!\left(\sum_{j=1}^{m}\hat{\phi}_j\sum_{k=1}^{j}k\hat{\gamma}(k-j)\right) + \frac{\hat{\sigma}_\epsilon^2 - \hat{\gamma}(0)}{2}\cdot\frac{\sum_{j=1}^{m}j\hat{\phi}_j}{1 - \sum_{j=1}^{m}\hat{\phi}_j}\right]\!.$$

This is a **closed-form expression** — computed in $O(m)$ operations after obtaining $\hat{\phi}$ and $\hat{\gamma}$.

**Step 5: Pilot run**

The authors recommend:
1. Run the chain for $n_0 = 10^4$ iterations (pilot run).
2. Fit AR($m$) to each component; compute $\Sigma_{p,i}$ and $\Gamma_{p,i}$.
3. Compute $\hat{b}_{\mathrm{opt}}$ via the aggregation formula.
4. Run the main chain of length $n$ and estimate $\Sigma$ with batch size $\hat{b}_{\mathrm{opt}}$.

The authors also test the scenario with no separate pilot (estimating $b$ and $\Sigma$ from the same data) — coverage improves further (theoretical analysis of this "random $b$" remains open).

### 4.3 Lag-based Method (Politis, 2003)

**Algorithm:**
1. Compute sample autocorrelations $\hat{\rho}_i(k)$ for each component.
2. Define $\hat{\rho}(k) = \max_i |\hat{\rho}_i(k)|$ — the maximum $k$-lag correlation.
3. Find the smallest $r$ such that $|\hat{\rho}(r+s)| < 2\sqrt{\log(n)/n}$ for $s = 1, 2, 3, 4, 5$.
4. Set $b = 2r$.

**Theoretical justification:** the threshold $2\sqrt{\log(n)/n} = O(\sqrt{\log n/n})$ is the standard significance threshold for sample autocorrelations.

**Drawbacks:**
- Requires computing $\hat{\rho}(k)$ for all $k$ — $O(n^2)$ or $O(n\log n)$ via FFT.
- Does not guarantee Assumption 1 (consistency issue).
- High variance — in slowly mixing chain examples, severely underestimates $b$.

---

## 5. Numerical Experiments: Detailed Breakdown

### 5.1 VAR(1): Theoretically Known Optimal $b$

**Model:** $X_t = \Phi X_{t-1} + \epsilon_t$, $\epsilon_t \sim N_p(0, I_p)$, $p = 3$.

Matrix $\Phi = \rho\Phi_0$, where $\Phi_0 = B/(\gamma+0.001)$, $B = AA^T$, $A$ has i.i.d. $N(0,1)$ entries, $\gamma$ is the largest eigenvalue of $B$. Series: $\rho \in \{0.80, 0.82, \ldots, 0.90\}$.

**Why VAR(1)?** For this model:
$$\Sigma = (I_p - \Phi)^{-1}V + V(I_p - \Phi)^{-T} - V, \quad V = \mathrm{Var}(X_1),$$
$$\Gamma = -[(I_p-\Phi)^{-2}\Phi V + V\Phi^T(I_p - \Phi^T)^{-2}].$$

So the **true** $b_{\mathrm{opt},ii}$ is analytically computable.

**Quality metric:** MSE of the pilot estimate $\hat{C} = (\sum_i\hat{\Gamma}_{ii}^2 / \sum_i\hat{\Sigma}_{ii}^2)^{1/3}$ relative to the true $C$, averaged over 1000 replications.

**Results (Figure 2, left):**
- AR($m$)-MSE of $\hat{C}$ is roughly 5–30× smaller than NP, especially at $\rho \to 0.90$.
- At $\rho = 0.90$: NP-MSE $\approx 0.4$, AR-MSE $\approx 0.05$.

**Results (Figure 2, right):** mean $\hat{b}$ over 1000 replications for 7 methods:
- $\lfloor n^{1/3}\rfloor \approx 46$ (for $n=10^5$, very small)
- $\lfloor n^{1/2}\rfloor \approx 316$ (too large)
- AR.BM, NP.BM, AR.OBM, NP.OBM: between 100 and 300
- Lag: $\approx 0$–$30$ (systematically underestimates!)

**Results (Figure 3):** log MSE for $\hat{\Sigma}_{BM}$, $\hat{\Sigma}_{OBM}$, $\hat{\Sigma}_{BM-FT}$, $\hat{\Sigma}_{OBM-FT}$:
- Bartlett windows: AR and NP give smallest MSE (better than $\lfloor n^{1/3}\rfloor$, $\lfloor n^{1/2}\rfloor$, Lag).
- Flat-top windows: Lag and $\lfloor n^{1/3}\rfloor$ give smallest MSE (flat-top benefits from small $b$).
- At $\rho = 0.90$, the difference reaches $e^2 \approx 7$-fold reduction in MSE.

### 5.2 Bayesian Logistic Regression: Realistic Case

**Data:** *Anguilla australis* from `dismo` R package (Elith et al., 2008) — presence/absence of short-finned eel at 1000 New Zealand sites. 6 covariates (5 continuous + 1 categorical with 5 levels), 9-dimensional $\beta$.

**Model:**
$$Y_i | x_i, \beta \sim \mathrm{Bernoulli}\!\left(\frac{1}{1 + \exp(-x_i^T\beta)}\right), \quad \beta \sim N(0, \sigma_\beta^2 I_9), \quad \sigma_\beta^2 = 100.$$

Sampled via random walk Metropolis-Hastings (geometrically ergodic, Vats et al., 2019).

**Ground truth:** average over 1000 chains of length $10^6$.

**Coefficient note:** the true coefficient in $b_{\mathrm{opt}} \propto n^{1/6}$ is significantly greater than 1, meaning $\lfloor n^{1/2}\rfloor$ happens to be close to optimal for this specific example.

**Results (Figure 4, boxplot):**
- AR: median $\approx 7$ (for coefficient $C^*$), narrow IQR
- NP: median $\approx 7$, wide IQR with outliers reaching 25

**Results (Table 1):** 90% CR coverage for $n \in \{10^4, 5\cdot10^4, 10^5\}$:

| $n$ | BM: $\lfloor n^{1/3}\rfloor$ | BM: $\lfloor n^{1/2}\rfloor$ | BM: AR | BM: NP | BM: Lag |
|---|---|---|---|---|---|
| $10^4$ | 0.279 | 0.722 | 0.731 | 0.709 | 0.703 |
| $5\times10^4$ | 0.499 | 0.826 | 0.823 | 0.831 | 0.808 |
| $10^5$ | 0.615 | 0.861 | 0.860 | 0.849 | 0.823 |

Similarly for OBM (slightly better). Flat-top versions give intermediate results.

**Observations:**
- $\lfloor n^{1/3}\rfloor$ — catastrophically small (bias dominates).
- $\lfloor n^{1/2}\rfloor$ — good for this specific example (coincidence with optimal).
- AR($m$) and NP methods — comparable coverage, AR more stable.
- Lag — slightly worse than AR/NP, especially for BM.

### 5.3 Bayesian Dynamic Space-Time Model: Hard Case

**Data:** temperatures at 10 stations, 12 months, year 2000.

**Model (Finley et al., 2012):**
$$y_t(s) = \mathbf{x}_t(s)^T\boldsymbol{\beta}_t + u_t(s) + \epsilon_t(s), \quad \epsilon_t(s) \sim N(0, \tau_t^2),$$
$$\boldsymbol{\beta}_t = \boldsymbol{\beta}_{t-1} + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim N_p(0, \Sigma_\eta),$$
$$u_t(s) = u_{t-1}(s) + w_t(s), \quad w_t(s) \sim GP(0, C_t(\cdot,\cdot;\sigma_t^2,\phi_t)),$$
where $C_t(s_1,s_2;\sigma_t^2,\phi_t) = \sigma_t^2\rho(s_1,s_2;\phi_t)$ is a spatial covariance function.

Sampled via Metropolis-within-Gibbs.

**Task:** estimate $\beta_1^{(1)}$ and $\beta_2^{(1)}$ (elevation coefficients for first two months).

**Results (Figure 5, boxplot):**
- AR: median $\approx 20$, compact IQR
- NP: median $\approx 30$, wide IQR with extreme outliers to 80+

**Results (Table 2):** 90% CR coverage for $n \in \{10^4, 5\cdot10^4, 10^5, 2\cdot10^5\}$:

| $n$ | BM: $\lfloor n^{1/3}\rfloor$ | BM: $\lfloor n^{1/2}\rfloor$ | BM: AR | BM: NP | BM: Lag |
|---|---|---|---|---|---|
| $10^4$ | 0.389 | 0.612 | 0.736 | 0.775 | 0.775 |
| $5\times10^4$ | 0.439 | 0.732 | 0.804 | 0.842 | 0.816 |
| $10^5$ | 0.477 | 0.764 | 0.820 | 0.841 | 0.810 |
| $2\times10^5$ | 0.553 | 0.807 | 0.838 | 0.861 | 0.823 |

**Observations:**
- Both $\lfloor n^{1/3}\rfloor$ and $\lfloor n^{1/2}\rfloor$ are poor (slowly mixing chain → large $b$ needed).
- NP method slightly better than AR here due to overestimated coefficient (larger $b$).
- Lag method gives higher coverage than $n^{1/2}$ but lower than AR/NP.
- Poor lag performance is explained by the **short pilot run**: $10^4$ iterations are insufficient for reliable autocorrelation estimation in slowly mixing chains.

---

## 6. Discussion Section: Detailed Analysis

### 6.1 Three Competing Methods

| | AR($m$)-fit | NP (empirical rule) | Lag-based |
|---|---|---|---|
| **Computational cost** | $O(m^2 n)$ Yule-Walker | $O(n\log n)$ FFT correlations | $O(n\log n)$ + iterations |
| **Variance of $b$ estimate** | Low | High | High |
| **Assumption 1 guarantee** | Yes | Yes | No |
| **Consistency of $\hat{\Sigma}$** | Guaranteed | Guaranteed | Not guaranteed |
| **Robustness to $m$** | Depends on AIC | — | — |

### 6.2 No-Pilot Scenario

The authors note that in practice many users do not do pilot runs. They repeat experiments estimating $b$ from the same chain used for $\Sigma$ estimation (random batch size). Coverage improves, but theoretical analysis (asymptotics of random $b$) remains an open problem.

### 6.3 Flat-top: Why Still Useful?

Despite the MSE-optimal $b$ pathology, flat-top has an important property: at fixed $b$ it is less sensitive to changes in $b$ than Bartlett. This is especially valuable for short chains where $b$ estimation is unreliable. Therefore for **short chains**, the authors recommend flat-top with Bartlett-optimal $b$ or lag-based $b$.

---

## 7. Mathematical Appendix: Proof Structure

### 7.1 Proof of Theorem 2 (Bias)

**Key relation** (Vats & Flegal, 2018, Theorem 2):
$$\mathrm{Cov}[\bar{Y}_l^{(i)}(k), \bar{Y}_l^{(j)}(k)] - \mathrm{Cov}[\bar{Y}^{(i)}, \bar{Y}^{(j)}] = \frac{n-k}{kn}\!\left(\Sigma_{ij} + \frac{n+k}{kn}\Gamma_{ij} + o(1/k^2)\right). \tag{4}$$

Since $\sum_{k=1}^{b}k\Delta_2 w_n(k) = 1$ (normalization condition of Theorem 1), compute:
$$E[\hat{\Sigma}_{w,ij}] = \Sigma_{ij} + \sum_{k=1}^{b}\Delta_2 w_n(k)\cdot\Gamma_{ij} + o(b/n) + o(1/b).$$

Therefore:
$$\mathrm{Bias} = \sum_{k=1}^{b}\Delta_2 w_n(k)\cdot\Gamma_{ij} + o(b/n) + o(1/b).$$

For Bartlett: $\sum_{k=1}^b \Delta_2 w_n(k) = 1/b$, Bias $= \Gamma_{ij}/b = C\Gamma_{ij}/b$. ✓  
For flat-top: $\sum_{k=1}^b \Delta_2 w_n(k) = 0$, Bias $= 0$ (to leading order). ✓

### 7.2 Proof of Theorem 3 (Variance) — Key Steps

**Transition to Brownian motion:** from the strong invariance principle (Lemma 1), replacing $\sum_{t=1}^n Y_t \approx \theta n + LB(n)$:
$$\bar{Y}_l(k) \approx \theta + k^{-1}L[B(l+k) - B(l)].$$

Denote $\bar{C}_l(k) = k^{-1}L[B(l+k) - B(l)]$ — the Brownian batch mean.

**Variance decomposition:**
$$\mathrm{Var}(\hat{\Sigma}_{w,jj}) = \frac{1}{n^2}\sum_{k,k'}\sum_{l,l'}k^2{k'}^2\Delta_2 w_n(k)\Delta_2 w_n(k')\,\mathrm{Cov}\!\left[(\bar{C}^{(j)}_l(k) - \bar{C}^{(j)})^2,\, (\bar{C}^{(j)}_{l'}(k') - \bar{C}^{(j)})^2\right].$$

**Applying Propositions 1 and 2:**

**Proposition 1:** if $(X,Y) \sim N\!\left(0, \begin{pmatrix}l_{11} & l_{12} \\ l_{12} & l_{22}\end{pmatrix}\right)$, then $E[X^2 Y^2] = 2l_{12}^2 + l_{11}l_{22}$.

**Proposition 2** (Janssen & Stoica, 1987): for jointly normal $(X_1,X_2,X_3,X_4)$ with zero mean:
$$E[X_1 X_2 X_3 X_4] = E[X_1 X_2]E[X_3 X_4] + E[X_1 X_3]E[X_2 X_4] + E[X_1 X_4]E[X_2 X_3].$$

These two results allow computing all fourth mixed moments of Brownian increments in closed form.

After summing over all $k, k'$ with weights $\Delta_2 w_n(k)$ and applying the window normalization conditions:
$$\mathrm{Var}(\hat{\Sigma}_{w,jj}) = \left([\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2]S + o(1)\right)\frac{b}{n}.$$

---

## 8. Relation to HAC Estimators and Literature

### 8.1 Andrews (1991): HAC Context

Andrews considers estimating $\Omega = \sum_{k=-\infty}^{\infty}E[u_t u_{t+k}^T]$ in regression $y_t = x_t\beta + u_t$. The spectral estimator:
$$\hat{\Omega} = \sum_{k=-(T-1)}^{T-1}w_T(k)\hat{R}(k), \quad \hat{R}(k) = T^{-1}\sum_t u_t u_{t+k}^T.$$

Andrews (1991, Theorem 1) obtains identical MSE formulas: $\mathrm{MSE} = O(b^{-2}) + O(b/T)$. For the Parzen kernel (analogous to Bartlett): $b_{\mathrm{opt}} \propto T^{1/3}$.

**Differences from this paper:**
1. Andrews works in an i.i.d. or specific mixing context, not with polynomially ergodic Markov chains.
2. Andrews' spectral estimator requires computing $\hat{R}(k)$ for all $k = 0,\ldots,T-1$ — $O(T^2p^2)$ operations, infeasible for high-dimensional MCMC.
3. The OBM formulation of this paper runs in $O(n b p^2)$ — much more efficient.

### 8.2 Politis & Romano (1995, 1996, 1999): Flat-top and Lag-based

- Flat-top kernels proposed for **bias reduction** in spectral estimators.
- Lag-based selection: cut off at the last significant lag.

### 8.3 Damerdji (1991): Original Generalized OBM

For $p=1$, proposed representation (1) as a way to unify OBM and spectral estimators. Connection: $\hat{\Sigma}_w = \sum_{k} w_n(k)\hat{\gamma}(k)$ via summation by parts.

### 8.4 Vats et al. (2018): Multivariate Generalization

- Extension to $p > 1$.
- Strong consistency of $\hat{\Sigma}_w$ under Assumption 1 and polynomial ergodicity conditions.
- Established the strong invariance principle (Lemma 1 of this paper).
- Application to sequential MCMC stopping rules.

### 8.5 Vats & Flegal (2018): MSE for Non-overlapping BM

For BM, the full MSE formula (Theorems 5 and 6 in Vats & Flegal, 2018):
$$\mathrm{MSE}(\hat{\Sigma}_{BM,ij}) = \frac{\Gamma_{ij}^2}{b^2} + [\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2]\frac{b}{n} + o(b/n) + o(1/b),$$
which coincides with Theorem 1 of this paper at $C=1$, $S=1$.

---

## 9. Relevance to Thesis (LSA + RR + OBM): Deep Analysis

### 9.1 Thesis Problem Structure

In LSA of the form $\theta_{n+1} = \theta_n - \gamma_{n+1}(A\theta_n - b + \epsilon_{n+1})$ with Markovian noise $\{\epsilon_n\}$, the CLT takes the form:
$$\sqrt{n}(\theta_n - \theta^*) \xrightarrow{d} N(0, V),$$
where $V$ is determined by the spectral properties of $A$ and the noise statistics.

**Task:** estimate $V$ (or $V$ for the RR extrapolant) from the trajectory $\{\theta_k\}_{k=1}^n$.

### 9.2 OBM Estimator for LSA Iterates

Directly applicable:
$$\hat{V}_{OBM} = \frac{b}{n}\sum_{l=0}^{n-b}(\bar{\theta}_l(b) - \bar{\theta})(\bar{\theta}_l(b) - \bar{\theta})^T.$$

If $\sqrt{n}(\theta_n - \theta^*) \to N(0, V)$ and $\{\theta_k - \theta^*\}$ forms a "quasi-stationary" dependent process (in the sense that the CLT holds), then OBM theory applies.

**Important caveat:** LSA iterates $\theta_k$ are **non-stationary** (they converge to $\theta^*$). OBM is applied to centered iterates $\theta_k - \theta^*$, but $\theta^*$ is unknown. In practice, $\bar{\theta}$ is used as an estimate, introducing additional bias $O(1/n)$, negligible for large $n$.

### 9.3 RR Extrapolation and the Structure of $V^{RR}$

In Richardson-Romberg extrapolation:
$$\theta_n^{RR} = 2\theta_n^{(\gamma)} - \theta_n^{(\gamma/2)},$$
where $\theta_n^{(\gamma)}$ are iterates with step $\gamma_k = \gamma/k$ and $\theta_n^{(\gamma/2)}$ with step $\gamma_k/2$.

CLT for the RR estimator:
$$\sqrt{n}(\theta_n^{RR} - \theta^*) \xrightarrow{d} N(0, V^{RR}),$$
where $V^{RR} = 4V^{(\gamma)} + V^{(\gamma/2)} - 4\mathrm{Cov}$ (with cross-covariance term if the two streams are correlated).

If the streams are **independent** (separate chains), OBM is applied to each separately. If a **single chain** drives both (shared noise), then $V^{RR}$ is a function of the joint process $(\theta_n^{(\gamma)}, \theta_n^{(\gamma/2)})$.

**Estimator:**
$$\hat{V}^{RR}_{OBM} = 4\hat{V}^{(\gamma)}_{OBM} + \hat{V}^{(\gamma/2)}_{OBM} - 4\hat{C}_{OBM},$$
where $\hat{C}_{OBM}$ is the cross-OBM covariance estimator.

**Optimal $b$ for $\hat{V}^{RR}_{OBM}$:** MSE optimization is performed separately for each component. The structure is the same: $b_{\mathrm{opt}} \propto n^{1/3}$ with a constant determined by the matrices $\Gamma$ and $\Sigma$ for the corresponding stream. In practice, the AR($m$) approach is applied to each marginal.

### 9.4 Batch Size Choice under Markovian Noise

In LSA with Markovian noise, $\{\epsilon_n\}$ is a Markov chain, and $\{Y_n\} = \{A\theta_n + b - \epsilon_n\}$ is also a dependent process. The mixing rate determines how fast $R_{ij}(k)$ decays, and hence how significant $\Gamma_{ij}$ is.

**Practical recommendation for the thesis:**
1. Run LSA for $n_0 = 10^4$ steps (pilot).
2. Fit AR($m$) with AIC to each component $\{(\theta_k)_i\}$.
3. Compute $\hat{b}_{\mathrm{opt}}$ with $C=1$, $S=2/3$ (Bartlett OBM) or $C=1$, $S=1$ (BM).
4. Run the main loop and compute $\hat{V}_{OBM}$ with this $b$.

### 9.5 Open Questions Relevant to the Thesis

1. **Non-stationarity of LSA iterates:** rigorous justification of OBM applicability to non-stationary $\{\theta_k\}$ (analogous to the "non-stationary" theorem for MCMC, where non-stationarity is also present in the transient phase).

2. **Rate of decay of $\Gamma_{ij}$ for RR:** does $\Gamma^{RR}$ depend on the step difference $\gamma$ vs $\gamma/2$ in a non-trivial way?

3. **Optimal $b$ for the cross-OBM term $\hat{C}_{OBM}$:** is formula (3) applicable to covariance matrices (not only variances)?

4. **Berry-Esseen for OBM estimator:** the CLT for $\hat{V}_{OBM}$ is known (via continuous mapping theorem), but the convergence rate (Berry-Esseen analogue) is not.

---

## 10. Summary

### 10.1 Main Theoretical Contributions

1. **Multivariate MSE formula** for the generalized OBM under polynomial ergodicity (Theorem 1).
2. **Explicit formula** for the optimal $b_{\mathrm{opt}} \propto n^{1/3}$ with precise dependence on $\Gamma$ and $\Sigma$.
3. **Diagnosis of flat-top pathology**: $C=0$ → MSE decreases in $b$ → MSE criterion inapplicable.
4. **AR($m$)-estimator** for the proportionality constant with closed-form Yule-Walker formulas.

### 10.2 Main Practical Contributions

1. $b = \lfloor n^{1/2}\rfloor$ is suboptimal — it inflates the variance of the $\Sigma$ estimator.
2. AR($m$)-estimator is substantially more stable than nonparametric alternatives.
3. For slowly mixing chains (typical in Bayesian analysis), the coverage gap between optimal and standard methods is significant even at $n = 10^4$–$10^5$.
4. Flat-top estimators are more robust to errors in $b$ → preferred for short chains.

### 10.3 Extended Method Comparison Table

| Method | $b$ | $C$ | $S$ | bias² | variance | Recommended when |
|---|---|---|---|---|---|---|
| BM, Bartlett | $\lfloor n^{1/3}\rfloor$ fixed | 1 | 1 | large | small | Never (constant wrong) |
| OBM, Bartlett | $\lfloor n^{1/2}\rfloor$ fixed | 1 | 2/3 | small | large $\propto n^{-1/2}$ | Never for accuracy |
| OBM, Bartlett | AR($m$)-opt $b$ | 1 | 2/3 | optimal | optimal | **Long chains** |
| BM | AR($m$)-opt $b$ | 1 | 1 | optimal | optimal | Long chains |
| OBM, Flat-top | Bartlett-opt $b$ | 0 | 4/3 | $\approx 0$ | larger | Short chains |
| OBM, Flat-top | Lag-based $b$ | 0 | 4/3 | $\approx 0$ | moderate | Short chains |

### 10.4 Recommendations for Limited $n$

When $n < 10^4$ and the chain mixes slowly, no method guarantees good coverage:
- Use flat-top with any moderate $b$ (least sensitive to $b$ misspecification).
- Run several independent short chains and average estimates.
- Apply stopping rules (Vats et al., 2019) rather than fixed $n$.

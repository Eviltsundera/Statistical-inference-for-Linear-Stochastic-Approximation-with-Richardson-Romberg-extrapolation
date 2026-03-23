# Detailed Summary: "Statistical Inference for Linear Stochastic Approximation with Markovian Noise"

**Authors:** Sergey Samsonov (HSE), Marina Sheshukova (HSE), Eric Moulines (Ecole Polytechnique, MBUZAI), Alexey Naumov (HSE, Steklov Mathematical Institute)
**Year:** 2025, arXiv:2505.19102

---

## 1. Problem and Motivation

The paper studies **non-asymptotic** Berry--Esseen type bounds and bootstrap-based confidence intervals for the **Polyak--Ruppert averaged iterates** of the **linear stochastic approximation (LSA)** algorithm driven by **Markovian noise**.

The LSA iteration is:

$$\theta_k = \theta_{k-1} - \alpha_k \{\mathbf{A}(Z_k)\theta_{k-1} - \mathbf{b}(Z_k)\}, \quad k \ge 1,$$

with Polyak--Ruppert averaging:

$$\bar{\theta}_n = n^{-1}\sum_{k=0}^{n-1}\theta_k, \quad n \ge 1.$$

Here $\mathbf{A}: \mathsf{Z} \to \mathbb{R}^{d \times d}$ and $\mathbf{b}: \mathsf{Z} \to \mathbb{R}^d$ are measurable functions, and $(Z_k)_{k \in \mathbb{N}}$ is an ergodic Markov chain with stationary distribution $\pi$ satisfying $\mathbb{E}_\pi[\mathbf{A}(Z_k)] = \bar{\mathbf{A}}$ and $\mathbb{E}_\pi[\mathbf{b}(Z_k)] = \bar{\mathbf{b}}$. The target is $\theta^\star = \bar{\mathbf{A}}^{-1}\bar{\mathbf{b}}$ (the unique solution of $\bar{\mathbf{A}}\theta = \bar{\mathbf{b}}$).

**Why this matters:** Under appropriate conditions, the CLT holds: $\sqrt{n}(\bar{\theta}_n - \theta^\star) \xrightarrow{d} \mathcal{N}(0, \Sigma_\infty)$. However, most existing non-asymptotic results focus on i.i.d. noise. For Markovian noise, which arises naturally in reinforcement learning (TD learning, Q-learning, actor-critic), deriving precise Berry--Esseen rates and valid bootstrap procedures is substantially more challenging due to the dependence structure. This paper provides the **first non-asymptotic guarantees** on the rate of convergence of bootstrap-based confidence intervals for SA with Markov noise.

---

## 2. Setting and Assumptions

### 2.1 Error Decomposition

Starting from the LSA recursion, the error satisfies:

$$\theta_n - \theta^\star = (\mathbf{I} - \alpha_n \bar{\mathbf{A}}_n)(\theta_{n-1} - \theta^\star) - \alpha_n \varepsilon_n,$$

where $\varepsilon(z) = \tilde{\mathbf{A}}(z)\theta^\star - \tilde{\mathbf{b}}(z)$ with $\tilde{\mathbf{A}}(z) = \mathbf{A}(z) - \bar{\mathbf{A}}$ and $\tilde{\mathbf{b}}(z) = \mathbf{b}(z) - \bar{\mathbf{b}}$. The noise covariance under stationarity is:

$$\Sigma_\varepsilon = \mathbb{E}_\pi[\varepsilon(Z_0)\varepsilon(Z_0)^\top] + 2\sum_{\ell=1}^{\infty}\mathbb{E}_\pi[\varepsilon(Z_0)\varepsilon(Z_\ell)^\top].$$

### 2.2 Assumptions

**A1 (Markov chain with geometric ergodicity):** $(Z_k)_{k \in \mathbb{N}}$ is a Markov chain on a Polish space $(\mathsf{Z}, \mathcal{Z})$ with a unique invariant distribution $\pi$ and is **uniformly geometrically ergodic**: there exists $t_{\mathrm{mix}} \in \mathbb{N}$ such that

$$\Delta(\mathsf{P}^k) := \sup_{z,z' \in \mathsf{Z}} \mathrm{d_{TV}}(\mathsf{P}^k(z,\cdot), \mathsf{P}^k(z',\cdot)) \le (1/4)^{\lceil k/t_{\mathrm{mix}} \rceil}.$$

The parameter $t_{\mathrm{mix}}$ is the **mixing time**.

**A2 (Hurwitz and boundedness):** $\int \mathbf{A}(z)\mathrm{d}\pi(z) = \bar{\mathbf{A}}$ with $-\bar{\mathbf{A}}$ being **Hurwitz** (all eigenvalues have strictly negative real parts). The noise is bounded: $\|\varepsilon\|_\infty = \sup_z \|\varepsilon(z)\| < +\infty$, and $\mathrm{C_A} = \sup_z \|\mathbf{A}(z)\| \vee \sup_z \|\tilde{\mathbf{A}}(z)\| < \infty$. Also $\lambda_{\min}(\Sigma_\varepsilon) > 0$ (non-degeneracy).

**A3 (Polynomial step sizes):** $\alpha_k = c_0/(k + k_0)^\gamma$ with $\gamma \in [1/2, 1)$ and $c_0 \le 1/(2a)$, where $a = \lambda_{\min}(P)/(2\|Q\|)$ from the Lyapunov equation $\bar{\mathbf{A}}^\top Q + Q\bar{\mathbf{A}} = P$ (Proposition 1). Additionally, $n$ must be large enough: $k_0 > g(a, t_{\mathrm{mix}}, c_0, \mathrm{C_A}, \kappa_Q, \alpha_\infty)(\log n)^{1/\gamma}$.

### 2.3 Key Structural Properties

**Proposition 1 (Lyapunov equation and contraction):** For any $P = P^\top \succ 0$, there exists a unique $Q = Q^\top \succ 0$ satisfying $\bar{\mathbf{A}}^\top Q + Q\bar{\mathbf{A}} = P$. Setting $a = \lambda_{\min}(P)/(2\|Q\|)$, for any $\alpha \in [0, \alpha_\infty]$:

$$\|\mathbf{I} - \alpha \bar{\mathbf{A}}\|_Q^2 \le 1 - \alpha a.$$

This is the key contraction inequality in the $Q$-norm that drives the entire analysis.

**Asymptotic covariance:** Under A1--A3, the covariance matrix is:

$$\Sigma_\infty = \bar{\mathbf{A}}^{-1}\Sigma_\varepsilon \bar{\mathbf{A}}^{-\top}.$$

---

## 3. Main Result 1: Berry--Esseen Bound (Theorem 1)

### 3.1 Statement

For a fixed unit vector $u \in \mathbb{S}^{d-1}$, the paper studies the rate of convergence to normality of the projected, rescaled average $\sqrt{n}u^\top(\bar{\theta}_n - \theta^\star)/\sigma_n(u)$, where $\sigma_n^2(u) = u^\top \Sigma_n u$ is a finite-sample variance proxy (defined via $\Sigma_n = n^{-1}\sum_{\ell=2}^{n-1} Q_\ell \Sigma_\varepsilon Q_\ell^\top$ with $Q_\ell = \alpha_\ell \sum_{k=\ell}^{n-1} G_{\ell+1:k}$).

**Theorem 1.** Under A1, A2, A3, for any $u \in \mathbb{S}^{d-1}$, $\theta_0 \in \mathbb{R}^d$, and any initial distribution $\xi$ on $(\mathsf{Z},\mathcal{Z})$:

$$\mathrm{d_K}\!\left(\frac{\sqrt{n}u^\top(\bar{\theta}_n - \theta^\star)}{\sigma_n(u)}, \mathcal{N}(0,1)\right) \le \mathrm{B}_n,$$

where

$$\mathrm{B}_n = \frac{\mathrm{C_{K,1}}\log^{3/4}n}{n^{1/4}} + \frac{\mathrm{C_{K,2}}\log n}{n^{1/2}} + \frac{\mathrm{C_1^D}\|\theta_0 - \theta^\star\| + \mathrm{C_2^D}}{\sqrt{n}} + \mathrm{C_3^D}\frac{(\log n)^2}{n^{\gamma-1/2}} + \mathrm{C_4^D}\frac{(\log n)^{5/2}}{n^{\gamma-1/2}}.$$

**Corollary 1.** Converting from the finite-sample variance $\sigma_n(u)$ to the asymptotic variance $\sigma(u) = \sqrt{u^\top \bar{\mathbf{A}}^{-1}\Sigma_\varepsilon \bar{\mathbf{A}}^{-\top} u}$:

$$\mathrm{d_K}\!\left(\frac{\sqrt{n}u^\top(\bar{\theta}_n - \theta^\star)}{\sigma(u)}, \mathcal{N}(0,1)\right) \le \mathrm{B}_n + C_\infty n^{\gamma - 1}.$$

**Remark 1 (Optimal rate):** Setting $\gamma = 3/4$ yields the optimal bound:

$$\mathrm{d_K}\!\left(\frac{\sqrt{n}u^\top(\bar{\theta}_n - \theta^\star)}{\sigma(u)}, \mathcal{N}(0,1)\right) \lesssim_{pr} \frac{(\log n)^{5/2}}{n^{1/4}},$$

which matches (up to log factors) the $O(n^{-1/4})$ rate from the i.i.d. setting and the recent result of [Wu, Wei, Rinaldo 2025] for TD learning.

### 3.2 Proof Strategy for Theorem 1: Full Decomposition Chain

The proof is the paper's main technical contribution. It proceeds through a carefully structured **hierarchy of decompositions**, each introducing a Poisson-equation-based martingale conversion. We trace the entire chain below.

#### Step 1: First-Level Decomposition -- Transient vs. Fluctuation

Starting from the error recursion (3): $\theta_n - \theta^\star = (\mathbf{I} - \alpha_n\mathbf{A}_n)(\theta_{n-1} - \theta^\star) - \alpha_n\varepsilon_n$, iterating backwards yields:

$$\theta_k - \theta^\star = \Gamma_{1:k}^{(\alpha)}\{\theta_0 - \theta^\star\} - \sum_{j=1}^{k}\Gamma_{j+1:k}^{(\alpha)}\alpha_j\varepsilon(Z_j),$$

where $\Gamma_{m:k}^{(\alpha)} = \prod_{i=m}^{k}(\mathbf{I} - \alpha_i \mathbf{A}(Z_i))$ is the **random** matrix product (with $\Gamma_{m:k}^{(\alpha)} = \mathbf{I}$ for $m > k$). This gives the **first-level decomposition** (eq. (27)):

$$u^\top(\theta_k - \theta^\star) = \underbrace{u^\top \Gamma_{1:k}^{(\alpha)}\{\theta_0 - \theta^\star\}}_{\tilde{\theta}_k^{(\mathrm{tr})} \text{ -- transient}} + \underbrace{u^\top\!\left(-\sum_{j=1}^{k}\Gamma_{j+1:k}^{(\alpha)}\alpha_j\varepsilon(Z_j)\right)}_{\tilde{\theta}_k^{(\mathrm{fl})} \text{ -- fluctuation}}.$$

**Why this matters:** The transient term decays exponentially (controlled by Proposition 10 on the stability of random matrix products). The fluctuation term is the main object of study.

#### Step 2: Second-Level Decomposition -- Deterministic vs. Random Matrix Products

The key idea is to **replace the random matrix products** $\Gamma_{j+1:k}^{(\alpha)}$ in the fluctuation term by their **deterministic counterparts** $G_{m:k} = \prod_{i=m}^{k}(\mathbf{I} - \alpha_i\bar{\mathbf{A}})$. This yields the decomposition (28):

$$u^\top \tilde{\theta}_k^{(\mathrm{fl})} = u^\top J_k^{(0)} + u^\top H_k^{(0)},$$

where $J_k^{(0)}$ and $H_k^{(0)}$ are defined by the pair of recursions:

$$J_k^{(0)} = (\mathbf{I} - \alpha_k\bar{\mathbf{A}})\,J_{k-1}^{(0)} - \alpha_k\varepsilon(Z_k), \qquad J_0^{(0)} = 0,$$

$$H_k^{(0)} = (\mathbf{I} - \alpha_k\mathbf{A}(Z_k))\,H_{k-1}^{(0)} - \alpha_k\tilde{\mathbf{A}}(Z_k)J_{k-1}^{(0)}, \qquad H_0^{(0)} = 0.$$

Solving these recursions gives the **explicit representations** (eq. (29)):

$$J_k^{(0)} = -\sum_{j=1}^{k}\alpha_j G_{j+1:k}\,\varepsilon(Z_j), \qquad H_k^{(0)} = -\sum_{j=1}^{k}\alpha_j \Gamma_{j+1:k}\tilde{\mathbf{A}}(Z_j)\,J_{j-1}^{(0)}.$$

**Interpretation:**
- $J_k^{(0)}$ uses only the **deterministic** products $G_{j+1:k}$, making it a **weighted linear statistic** of the Markov chain $\{Z_j\}$ -- amenable to Poisson-equation techniques.
- $H_k^{(0)}$ is the **misadjustment error** between the random and deterministic products. It is driven by $\tilde{\mathbf{A}}(Z_k) = \mathbf{A}(Z_k) - \bar{\mathbf{A}}$ (the centered random matrix) acting on $J_{k-1}^{(0)}$, making it a **second-order** term.

#### Step 3: Polyak--Ruppert Averaging and the $W + D_1$ Decomposition

Taking the Polyak--Ruppert average $\bar{\theta}_n = n^{-1}\sum_{k=0}^{n-1}\theta_k$ and using the decompositions above:

$$\sqrt{n}(\bar{\theta}_n - \theta^\star) = W + D_1,$$

where:

$$W = -\frac{1}{\sqrt{n}}\sum_{\ell=1}^{n-1}Q_\ell\,\varepsilon(Z_\ell), \qquad D_1 = \frac{1}{\sqrt{n}}\sum_{k=0}^{n-1}\Gamma_{1:k}(\theta_0 - \theta^\star) + \frac{1}{\sqrt{n}}\sum_{k=1}^{n-1}H_k^{(0)}.$$

Here $Q_\ell = \alpha_\ell\sum_{k=\ell}^{n-1}G_{\ell+1:k}$ are the **time-varying aggregation weights** (eq. (8)). The term $W$ is the **main Gaussian term** and $D_1$ is a remainder. The crucial observation is that $W$ is a **weighted additive functional** of the Markov chain $(Z_\ell)$ -- the weights $Q_\ell$ are deterministic matrices that depend only on the step sizes.

#### Step 4: The Poisson Equation and Martingale Extraction from $W$

This is the **central technical step**. The term $W = -n^{-1/2}\sum_{\ell=1}^{n-1}Q_\ell\varepsilon(Z_\ell)$ is a weighted sum of the dependent random variables $\varepsilon(Z_\ell)$. To apply Berry--Esseen bounds, we need to convert it into a martingale. The tool is the **Poisson equation**.

**The Poisson equation.** For any function $f$ with $\pi(f) = 0$ (zero mean under stationarity), the Poisson equation is:

$$g(z) - \mathsf{P}g(z) = f(z), \qquad \text{i.e.,} \qquad (\mathrm{Id} - \mathsf{P})g = f,$$

where $\mathsf{P}$ is the transition kernel of the Markov chain. Under geometric ergodicity (A1), the solution is given by the **Neumann series**:

$$\hat{\varepsilon}(z) := \sum_{k=0}^{\infty}\mathsf{P}^k\varepsilon(z),$$

which converges because $\|\mathsf{P}^k\varepsilon\|_\infty \le \|\varepsilon\|_\infty\cdot\Delta(\mathsf{P}^k) \le \|\varepsilon\|_\infty(1/4)^{\lfloor k/t_{\mathrm{mix}}\rfloor}$. The solution satisfies $\hat{\varepsilon}(z) - \mathsf{P}\hat{\varepsilon}(z) = \varepsilon(z)$ and is bounded: $\|\hat{\varepsilon}\|_\infty \le (8/3)t_{\mathrm{mix}}\|\varepsilon\|_\infty$.

**Why the Poisson equation is needed.** The random variables $\varepsilon(Z_\ell)$ are **not** independent -- they are driven by a Markov chain. The Poisson equation provides a **telescoping identity** that converts an additive functional of a Markov chain into a martingale plus boundary terms:

$$\varepsilon(Z_\ell) = \hat{\varepsilon}(Z_\ell) - \mathsf{P}\hat{\varepsilon}(Z_\ell) = \underbrace{\left[\hat{\varepsilon}(Z_\ell) - \mathsf{P}\hat{\varepsilon}(Z_{\ell-1})\right]}_{\text{martingale difference}} + \underbrace{\left[\mathsf{P}\hat{\varepsilon}(Z_{\ell-1}) - \mathsf{P}\hat{\varepsilon}(Z_\ell)\right]}_{\text{telescoping remainder}}.$$

**Applying the Poisson decomposition to $W$.** Substituting $\varepsilon(Z_\ell) = \hat{\varepsilon}(Z_\ell) - \mathsf{P}\hat{\varepsilon}(Z_\ell)$ into $W$:

$$W = -\frac{1}{\sqrt{n}}\sum_{\ell=1}^{n-1}Q_\ell\left[\hat{\varepsilon}(Z_\ell) - \mathsf{P}\hat{\varepsilon}(Z_\ell)\right].$$

Now, adding and subtracting $Q_\ell\mathsf{P}\hat{\varepsilon}(Z_{\ell-1})$ inside the sum and rearranging (using Abel summation / summation by parts for the time-varying weights $Q_\ell$):

$$W = n^{-1/2}M + D_2,$$

where the **martingale** $M$ and the **remainder** $D_2$ are:

$$M = -\sum_{\ell=2}^{n-1}\Delta M_\ell, \qquad \Delta M_\ell := Q_\ell\!\left(\hat{\varepsilon}(Z_\ell) - \mathsf{P}\hat{\varepsilon}(Z_{\ell-1})\right).$$

$$D_2 = -\frac{1}{\sqrt{n}}Q_1\hat{\varepsilon}(Z_1) + \frac{1}{\sqrt{n}}Q_{n-1}\mathsf{P}\hat{\varepsilon}(Z_{n-1}) + \sum_{\ell=1}^{n-2}(Q_\ell - Q_{\ell+1})\mathsf{P}\hat{\varepsilon}(Z_\ell).$$

**Why $\Delta M_\ell$ is a martingale difference.** By the Markov property:

$$\mathbb{E}[\hat{\varepsilon}(Z_\ell) \mid \mathcal{F}_{\ell-1}] = \mathbb{E}[\hat{\varepsilon}(Z_\ell) \mid Z_{\ell-1}] = \mathsf{P}\hat{\varepsilon}(Z_{\ell-1}),$$

so $\mathbb{E}[\Delta M_\ell \mid \mathcal{F}_{\ell-1}] = Q_\ell(\mathsf{P}\hat{\varepsilon}(Z_{\ell-1}) - \mathsf{P}\hat{\varepsilon}(Z_{\ell-1})) = 0$. Since $Q_\ell$ is a deterministic matrix, the martingale property is preserved.

**Structure of $D_2$ -- the Poisson remainder.** The three terms in $D_2$ have clear origins:
1. $-n^{-1/2}Q_1\hat{\varepsilon}(Z_1)$: **left boundary** -- the Poisson solution evaluated at the first step, weighted by $Q_1$.
2. $n^{-1/2}Q_{n-1}\mathsf{P}\hat{\varepsilon}(Z_{n-1})$: **right boundary** -- the conditional expectation of $\hat{\varepsilon}$ at the last step.
3. $\sum_{\ell=1}^{n-2}(Q_\ell - Q_{\ell+1})\mathsf{P}\hat{\varepsilon}(Z_\ell)$: **time-variation correction** -- arises because the weights $Q_\ell$ change with $\ell$. If the weights were constant ($Q_\ell \equiv Q$), this term would vanish and we would get a pure telescoping. The non-stationarity of the step sizes prevents this.

The remainder $D_2$ is controlled using bounds on $\|Q_\ell\|$ (Lemma 3: $\|Q_\ell\| \le \mathcal{L}_Q = \kappa_Q^{1/2}(c_0 + 4/(a(1-\gamma)))$), $\|S_\ell\| := \|Q_\ell - \bar{\mathbf{A}}^{-1}\|$ (Lemma 5), and $\|Q_{\ell+1} - Q_\ell\|$ (Lemma 6: $\|Q_{\ell+1} - Q_\ell\| \le \mathcal{L}_{Q,2}\alpha_{\ell+1}$). These lemmas establish that the boundary terms are $O(n^{-1/2})$ and the time-variation correction is $O(n^{1/2-\gamma})$.

#### Step 5: The Quadratic Characteristic of $M$ and Its Poisson-Based Concentration

The **quadratic characteristic** (predictable variation) of the martingale $M$ is:

$$\langle M\rangle_n = \sum_{\ell=2}^{n-1}\mathbb{E}^{\mathcal{F}_{\ell-1}}[\Delta M_\ell \Delta M_\ell^\top] = \sum_{\ell=2}^{n-1}Q_\ell\,\bar{\varepsilon}(Z_{\ell-1})\,Q_\ell^\top,$$

where $\bar{\varepsilon}(z) := \mathbb{E}[\hat{\varepsilon}(Z_1)\hat{\varepsilon}(Z_1)^\top \mid Z_0 = z] - \mathsf{P}\hat{\varepsilon}(z)\mathsf{P}\hat{\varepsilon}(z)^\top = \mathsf{P}(\hat{\varepsilon}\hat{\varepsilon}^\top)(z) - (\mathsf{P}\hat{\varepsilon})(z)(\mathsf{P}\hat{\varepsilon})^\top(z)$ (eq. (10)). Note $\pi(\bar{\varepsilon}) = \Sigma_\varepsilon$ (the noise covariance).

To apply the Berry--Esseen theorem for martingales, we need to show that $u^\top\langle M\rangle_n u$ concentrates around $n\sigma_n^2(u)$. This requires bounding:

$$|u^\top\langle M\rangle_n u - n\sigma_n^2(u)| = \left|\sum_{\ell=1}^{n-2}u^\top Q_{\ell+1}\{\bar{\varepsilon}(Z_\ell) - \pi(\bar{\varepsilon})\}Q_{\ell+1}^\top u\right| + \text{small corrections}.$$

This is again an **additive functional of the Markov chain** (with time-varying weights), but now it is a **quadratic** functional (involving $\bar{\varepsilon}(Z_\ell)$). The Poisson equation is applied again, this time to the function $\bar{\varepsilon}(z) - \pi(\bar{\varepsilon})$. Specifically:

**Lemma 22.** $\mathbb{E}_\xi^{1/p}[|u^\top\langle M\rangle_n u - n\sigma^2(u)|^p] \le 13^2\sigma_\infty^2(u)\,p^{1/2}\,t_{\mathrm{mix}}^{5/2}\,n^{1/2}.$

**Proof idea:** The function $h_\ell(z) = u^\top Q_{\ell+1}\bar{\varepsilon}(z)Q_{\ell+1}^\top u$ has $\pi(h_\ell) = u^\top Q_{\ell+1}\Sigma_\varepsilon Q_{\ell+1}^\top u$. One applies the Poisson equation to $h_\ell(z) - \pi(h_\ell)$ and uses the concentration inequality from [Paulin 2015, Corollary 2.11] (McDiarmid's inequality for Markov chains). The bound on $\sup_z|h_\ell(z) - h_\ell(z')| \le 2\sigma_\infty(u)t_{\mathrm{mix}}^2\|\varepsilon\|_\infty^2$ follows from bounding $\|\bar{\varepsilon}(z)\|$ via the Poisson solution norm.

**Lemma 23** provides a sharper bound: $\mathbb{E}_\xi^{1/p}[|u^\top\langle M\rangle_n u - n\sigma_n^2(u)|^p] \le 32n^{1/2}p^{1/2}\mathcal{L}_Q^2\|\varepsilon\|_\infty^2 t_{\mathrm{mix}}^{5/2}n^{1/2}.$ This uses:
1. The representation $h_\ell(z) = u^\top Q_{\ell+1}\hat{\varepsilon}(z)Q_{\ell+1}^\top u$ where $\hat{\varepsilon}$ is the Poisson solution.
2. The bound $\sup_{z,z'}|h_\ell(z) - h_\ell(z')| \le (8/3)^2\sigma_\infty^2(u)t_{\mathrm{mix}}^2$.
3. Application of the Markov chain concentration bound [Paulin 2015, Corollary 2.11].

#### Step 6: Berry--Esseen for the Martingale $M$ (Proposition 13)

With the quadratic characteristic under control, the paper applies a **quantitative martingale CLT**. This is based on a chain of results:

**Theorem 3 (Bolthausen [12]).** For a martingale $(S_n, s_n)$ with $\|X\|_\infty \le \varkappa$ and $V_n^2 = 1$ a.s., one has $\mathrm{d_K}(S_n/s_n) \le L(\varkappa)n\log n/s_n^3$.

**Lemma 21 (Extension, building on [12] and [28]).** For martingale-differences $X_i$ with $\|X\|_\infty \le \varkappa$, any $p \ge 1$:

$$\mathrm{d_K}(S_n/s_n, \mathcal{N}(0,1)) \le \frac{L(\varkappa)(2n+1)\log(2n+1)}{s_n^3} + C_1\sqrt{p}\,s_n^{-2p/(2p+1)}\left(\mathbb{E}|\sum \sigma_i^2 - s_n^2|^p\right)^{1/(2p+1)} + C_2 s_n^{-2p/(2p+1)}p\varkappa^{2p/(2p+1)}.$$

The first term is the **classical Berry--Esseen rate** for bounded martingales. The second term accounts for the **non-stationarity of the conditional variances** -- this is where the concentration bound on $|u^\top\langle M\rangle_n u - n\sigma_n^2(u)|$ enters. The third term is a Lindeberg-type correction.

**Proposition 13** applies Lemma 21 with $s_n^2 = n\sigma_n^2(u)$, $\varkappa = (16/3)\|\varepsilon\|_\infty\mathcal{L}_Q t_{\mathrm{mix}}$, and $p = \log n$. The conditional variance concentration from Lemma 22 and Lemma 23 enters the second term, yielding:

$$\mathrm{d_K}\!\left(\frac{u^\top M}{\sqrt{n}\sigma_n(u)}, \mathcal{N}(0,1)\right) \le \frac{\mathrm{C_{K,1}}\log^{3/4}n}{n^{1/4}} + \frac{\mathrm{C_{K,2}}\log n}{n^{1/2}},$$

where:
- $\mathrm{C_{K,1}} = 4\sqrt{2}\,\mathrm{e}^{1/8}C_1 t_{\mathrm{mix}}^{5/4}\mathcal{L}_Q\|\varepsilon\|_\infty C_\Sigma$ (from the conditional variance term)
- $\mathrm{C_{K,2}} = \frac{16\mathrm{e}^{1/4}t_{\mathrm{mix}}C_2\mathcal{L}_Q\|\varepsilon\|_\infty C_\Sigma}{3} + 3L(\varkappa)C_\Sigma^3$ (from the classical term and Lindeberg correction)

#### Step 7: Controlling the Remainder $D$ via $L^p$ Bounds

The overall decomposition is now:

$$\frac{\sqrt{n}u^\top(\bar{\theta}_n - \theta^\star)}{\sigma_n(u)} = \underbrace{\frac{u^\top M}{\sqrt{n}\sigma_n(u)}}_{\text{main Gaussian term}} + \underbrace{\frac{u^\top D}{\sigma_n(u)}}_{\text{remainder}},$$

where $D = D_1 + D_2$ with $D_1 = n^{-1/2}\sum_{k=0}^{n-1}\Gamma_{1:k}(\theta_0 - \theta^\star) + n^{-1/2}\sum_{k=1}^{n-1}H_k^{(0)}$ and $D_2$ from Step 4. The **smoothing inequality** (Proposition 12) gives:

$$\mathrm{d_K}\!\left(\frac{\sqrt{n}u^\top(\bar{\theta}_n - \theta^\star)}{\sigma_n(u)}, \mathcal{N}(0,1)\right) \le \mathrm{d_K}\!\left(\frac{u^\top M}{\sqrt{n}\sigma_n(u)}, \mathcal{N}(0,1)\right) + 2\left\{\mathbb{E}\left[\left|\frac{u^\top D}{\sigma_n(u)}\right|^p\right]\right\}^{1/(p+1)}.$$

Setting $p = \log n$ provides an optimal balance. The remainder $D$ splits into sub-terms:

$$D_1 = D_{11} + D_{12}, \qquad D_2 = D_{21} + D_{22} + D_{23},$$

where:
- $D_{11} = n^{-1/2}\sum_{k=0}^{n-1}\Gamma_{1:k}(\theta_0 - \theta^\star) + n^{-1/2}\sum_{k=1}^{n-1}u^\top H_k^{(\mathrm{init})}$: **transient + initial condition remainder**
- $D_{12} = n^{-1/2}\sum_{k=1}^{n-1}u^\top H_k^{(0)}$: **misadjustment remainder** (random vs. deterministic products)
- $D_{21} = -n^{-1/2}Q_1\hat{\varepsilon}(Z_1)$: **left Poisson boundary**
- $D_{22} = n^{-1/2}Q_{n-1}\mathsf{P}\hat{\varepsilon}(Z_{n-1})$: **right Poisson boundary**
- $D_{23} = \sum_{\ell=1}^{n-2}(Q_\ell - Q_{\ell+1})\mathsf{P}\hat{\varepsilon}(Z_\ell)$: **time-variation correction**

Each is bounded in $L^p$ (Proposition 5):
- $D_{21}, D_{22}$: bounded by Lemma 3 ($\|Q_\ell\| \le \mathcal{L}_Q$) and $\|\hat{\varepsilon}\|_\infty \le (8/3)t_{\mathrm{mix}}\|\varepsilon\|_\infty$, giving $O(n^{-1/2})$.
- $D_{23}$: uses Lemma 6 ($\|Q_\ell - Q_{\ell+1}\| \le \mathcal{L}_{Q,2}\alpha_{\ell+1}$) and the weighted Burkholder inequality (Lemma 25), giving $O(n^{1/2-\gamma})$.
- $D_{11}$: uses Proposition 10 (exponential decay of $\|\Gamma_{1:k}^{(\alpha)}\|$), giving $O(\|\theta_0 - \theta^\star\|/\sqrt{n})$.
- $D_{12}$: requires the **third-level decomposition** below.

#### Step 8: Third-Level Decomposition -- The Perturbation Expansion of $H_k^{(0)}$

The misadjustment $H_k^{(0)}$ is the most challenging remainder. It is handled by a **depth-$L$ perturbation expansion** (eq. (32)):

$$H_k^{(0)} = \sum_{\ell=1}^{L}J_k^{(\ell)} + H_k^{(L)},$$

where $J_k^{(\ell)}$ and $H_k^{(\ell)}$ satisfy the recursions (eq. (33)):

$$J_k^{(\ell)} = (\mathbf{I} - \alpha_k\bar{\mathbf{A}})\,J_{k-1}^{(\ell)} - \alpha_k\tilde{\mathbf{A}}(Z_k)\,J_{k-1}^{(\ell-1)}, \qquad J_0^{(\ell)} = 0,$$

$$H_k^{(\ell)} = (\mathbf{I} - \alpha_k\mathbf{A}(Z_k))\,H_{k-1}^{(\ell)} - \alpha_k\tilde{\mathbf{A}}(Z_k)\,J_{k-1}^{(\ell)}, \qquad H_0^{(\ell)} = 0.$$

**Interpretation:** At each level, $J_k^{(\ell)}$ uses only deterministic products (amenable to analysis) and $H_k^{(\ell)}$ is the remaining random-product error that is pushed to higher order. The paper takes $L = 1$, giving:

$$H_k^{(0)} = J_k^{(1)} + H_k^{(1)}.$$

**Explicit form of $J_k^{(1)}$.** Solving the recursion:

$$J_k^{(1)} = -\sum_{j=1}^{k}\alpha_j G_{j+1:k}\tilde{\mathbf{A}}(Z_j)J_{j-1}^{(0)} = \sum_{j=1}^{k-1}\alpha_j\left\{\sum_{\ell=j+1}^{k}\alpha_\ell G_{\ell+1:k}\tilde{\mathbf{A}}(Z_\ell)G_{j+1:\ell-1}\right\}\varepsilon(Z_j).$$

Changing the order of summation and introducing $S_{j:k} := \sum_{\ell=j}^{k}\alpha_\ell G_{\ell+1:k}\tilde{\mathbf{A}}(Z_\ell)G_{j+1:\ell-1}$:

$$J_k^{(1)} = \sum_{j=1}^{k-1}\alpha_j S_{j+1:k}\varepsilon(Z_j).$$

This is a **double sum** (bilinear in the Markov chain), making it substantially harder to bound than $J_k^{(0)}$. The analysis (Proposition 9, D.2--D.3 in the appendix) proceeds by:

1. **Block decomposition:** Split the time axis into blocks of size $m$ (to be optimized), writing $N = \lfloor k/m\rfloor$ and decomposing $J_k^{(1)}$ into three sums $T_1 + T_2 + T_3$ (within-block, cross-block, and tail).

2. **Berbee coupling for the cross-block term $T_2$:** The term $T_2$ involves $\varepsilon(Z_{jm+i})$ evaluated at times separated by $m$ steps. To exploit the near-independence of distant observations, introduce **independent copies** $Z_{jm+i}^\star$ (via Berbee's lemma, [62, Lemma 5.1]):
   - $Z_{jm+i}^\star$ is independent of $\mathfrak{F}_{(j+1)m+i}^k := \sigma(Z_{(j+1)m+i}, \ldots, Z_k)$
   - $\mathbb{P}_\xi(Z_{jm+i}^\star \ne Z_{jm+i}) \le 2(1/4)^{\lceil m/t_{\mathrm{mix}}\rceil}$ (eq. (38))
   - $Z_{jm+i}^\star$ and $Z_{jm+i}$ have the same distribution

   This splits $T_2 = T_{21} + T_{22}$, where $T_{21}$ uses the original $Z$'s (now approximately a martingale thanks to the coupling gap) and $T_{22}$ captures the coupling error.

3. **Burkholder inequality for $T_{21}$** (via Minkowski + Burkholder [54, Theorem 8.6]):

   $$\mathbb{E}_\xi^{1/p}[|u^\top T_{21}|^p] \lesssim \kappa_Q^{3/2}\mathrm{C_A}\|\varepsilon\|_\infty t_{\mathrm{mix}}^{1/2}p^{3/2}\sqrt{m}\left(\sum \alpha_j^2(\sum \alpha_\ell^2)\prod\sqrt{1-\alpha_\ell a}\right)^{1/2}.$$

4. **Coupling error for $T_{22}$:** Uses the bound $\|\varepsilon(Z_{km+i}) - \varepsilon(Z_{km+i}^\star)\|^p \le \|\varepsilon\|_\infty^p \cdot \mathbf{1}\{Z \ne Z^\star\}$ and the coupling probability $(1/4)^{m/(2pt_{\mathrm{mix}})}$.

5. **McDiarmid's inequality for $T_1$ and $T_3$** (Lemma 11): The within-block terms $u^\top S_{\ell+1:\ell+m}\varepsilon(Z_\ell)$ satisfy the bounded differences property:

   $$|h_r(z) - h_r(z')| \le 2^{3/2}\kappa_Q\mathrm{C_A}\alpha_r\prod_{k=\ell+1,k\ne r}^{\ell+m}\sqrt{1-a\alpha_k},$$

   and the concentration bound from [Paulin 2015]:

   $$\mathbb{P}_\xi(|u^\top S_{j+1:j+m}v| \ge t) \le 2\exp\left\{-\frac{t^2}{49t_{\mathrm{mix}}\kappa_Q^2\mathrm{C_A}^2(\sum \alpha_r^2)\prod(1-a\alpha_k)}\right\}.$$

6. **Block size optimization:** $m = \lceil 2pt_{\mathrm{mix}}\log(1/\alpha_k)/\log 4\rceil$ balances the coupling error $(1/4)^{m/(2pt_{\mathrm{mix}})} \le \sqrt{\alpha_k}$ against the block overhead.

The final bound (Proposition 9): $\mathbb{E}_\xi^{1/p}[|u^\top J_k^{(1)}|^p] \lesssim \mathrm{D}_4^{(\mathrm{M})}t_{\mathrm{mix}}p^2\alpha_k\sqrt{\log(1/\alpha_k)}$.

**Bound on $H_k^{(1)}$** (eq. (36)): $\mathbb{E}_\xi^{1/p}[|u^\top H_k^{(1)}|^p] \lesssim \mathrm{D}_5^{(\mathrm{M})}t_{\mathrm{mix}}p^2\alpha_k\sqrt{\log(1/\alpha_k)}$. This uses similar arguments as for $H_k^{(0)}$ but the additional factor of $\alpha_k$ makes it a higher-order correction.

#### Step 9: The Poisson Equation in the Burkholder/Rosenthal Inequalities

The Poisson equation appears at yet another level in the probability inequalities used throughout the proof. **Lemma 25** (weighted Burkholder for Markov chains) uses the following mechanism:

For a bounded function $f: \mathsf{Z} \to \mathbb{R}^d$ with $\pi(f) = 0$ and deterministic weights $A_k$:

$$\sum_{k=1}^n u^\top A_k(f(Z_k) - \pi(f)) = \underbrace{\sum_{k=2}^n u^\top A_k(g(Z_k) - \mathsf{P}g(Z_{k-1}))}_{T_1 \text{ -- martingale-differences}} + \underbrace{\sum_{k=1}^{n-1}u^\top(A_{k+1}-A_k)\mathsf{P}g(Z_k) + \ldots}_{T_2 \text{ -- boundary + Abel summation}},$$

where $g$ is the **Poisson solution** of $f$. The martingale term $T_1$ is bounded by Burkholder's inequality. The term $T_2$ is bounded using $\|\mathsf{P}g\|_\infty \le \|g\|_\infty \le (8/3)t_{\mathrm{mix}}\|f\|_\infty$.

Similarly, **Lemma 24** (Rosenthal inequality for Markov chains) uses the Poisson equation to write $\bar{S}_n = \sum(f(Z_k) - \pi(f))$ as a martingale plus boundary terms, then applies the Pinelis version of Rosenthal's inequality [58].

#### Step 10: Variance Comparison ($\sigma_n^2(u)$ vs. $\sigma^2(u)$)

**Lemma 9** shows $|\sigma_n^2(u) - \sigma^2(u)| \le C'_\infty n^{\gamma-1}$. The proof decomposes $\Sigma_n - \Sigma_\infty$ as:

$$\Sigma_n - \Sigma_\infty = \underbrace{\frac{1}{n}\sum_{t=2}^{n-1}(Q_t - \bar{\mathbf{A}}^{-1})\Sigma_\varepsilon\bar{\mathbf{A}}^{-\top} + \frac{1}{n}\sum_{t=2}^{n-1}\bar{\mathbf{A}}^{-1}\Sigma_\varepsilon(Q_t - \bar{\mathbf{A}}^{-1})^\top}_{R_1 \text{ -- first-order deviation}} + \underbrace{\frac{1}{n}\sum_{t=2}^{n-1}(Q_t - \bar{\mathbf{A}}^{-1})\Sigma_\varepsilon(Q_t - \bar{\mathbf{A}}^{-1})^\top - \frac{2}{n}\Sigma_\infty}_{R_2 \text{ -- second-order + boundary}}.$$

Using **Lemma 4** ($Q_\ell - \bar{\mathbf{A}}^{-1} = S_\ell - \bar{\mathbf{A}}^{-1}G_{\ell:n-1}$), **Lemma 5** ($\|S_\ell\| \le \sqrt{\kappa_Q}C_{\gamma,a}^{(S)}(\ell+k_0)^{\gamma-1}$), and **Lemma 8** ($\sum\|G_{t:n-1}\|^m \le \kappa_Q^{m/2}/(1 - (1-(a/2)\alpha_{n-2})^m)$), the four sub-terms of $R_2$ are each bounded to give the $n^{\gamma-1}$ rate.

The passage from Theorem 1 to Corollary 1 then uses the **Gaussian comparison lemma** (Lemma 20, from [20]): if $\xi_i \sim \mathcal{N}(0, \sigma_i^2)$ with $|\sigma_1^2/\sigma_2^2 - 1| \le \delta$, then $\sup_x|\mathbb{P}(\xi_1 \le x) - \mathbb{P}(\xi_2 \le x)| \le \frac{3}{2}\delta$.

#### Summary of the Decomposition Hierarchy

The complete chain of decompositions is:

$$\sqrt{n}(\bar{\theta}_n - \theta^\star) = \underbrace{W}_{\text{weighted linear statistic}} + \underbrace{D_1}_{\text{transient + misadjustment}}$$

$$\downarrow \text{Poisson eq.}$$

$$W = \underbrace{n^{-1/2}M}_{\text{martingale}} + \underbrace{D_2}_{\text{Poisson boundary + Abel correction}}$$

$$\downarrow \text{Perturbation expansion}$$

$$D_1 \ni H_k^{(0)} = \underbrace{J_k^{(1)}}_{\text{deterministic-product bilinear form}} + \underbrace{H_k^{(1)}}_{\text{higher-order misadjustment}}$$

$$\downarrow \text{Block decomposition + Berbee coupling}$$

$$J_k^{(1)} = \underbrace{T_{21}}_{\text{martingale-like (Burkholder)}} + \underbrace{T_{22}}_{\text{coupling error}} + \underbrace{T_1 + T_3}_{\text{within-block (McDiarmid/Poisson)}}$$

At each level, the Poisson equation converts dependent sums into martingales plus controllable remainders.

### 3.3 Comparison with Prior Work

| Paper | Setting | Metric | Rate |
|-------|---------|--------|------|
| [69] Samsonov et al. 2024 | i.i.d. noise, LSA | Convex distance | $O(n^{-1/4})$ |
| [83] Wu et al. 2024 | i.i.d. noise, TD | Convex distance | $O(n^{-1/3})$ |
| [84] Wu et al. 2025 | Markov noise, TD | Convex distance | $\log n / n^{1/4}$ |
| [74] Srikant 2024 | Markov noise, general LSA | 1-Wasserstein | $\sqrt{\log n}/n^{1/8}$ (implies $O(n^{-1/12})$ in Kolmogorov) |
| **This paper** | **Markov noise, general LSA** | **Kolmogorov** | $(\log n)^{5/2}/n^{1/4}$ |

Key improvements: (1) General LSA (not just TD); (2) Kolmogorov distance (stronger than convex distance); (3) Matching the i.i.d. rate up to log factors; (4) Sharper than [74]'s result which gave $O(n^{-1/12})$ in Kolmogorov distance.

---

## 4. Main Result 2: Multiplier Subsample Bootstrap (MSB)

### 4.1 The MSB Procedure

The paper adopts the multiplier subsample bootstrap (MSB) of [Ma, Zhang 2024]. For block size $b_n$ and each $t = 0, \ldots, n - b_n$, define:

- **Block average:** $\bar{\theta}_{b_n,t} = b_n^{-1}\sum_{\ell=t}^{t+b_n-1}\theta_\ell$
- **MSB estimator:**

$$\hat{\theta}_{n,b_n}(u) = \frac{\sqrt{b_n}}{\sqrt{n - b_n + 1}}\sum_{t=0}^{n-b_n} w_t (\bar{\theta}_{b_n,t} - \bar{\theta}_n)^\top u,$$

where $(w_t)$ are i.i.d. $\mathcal{N}(0,1)$ multiplier weights, independent of the data $\Xi_n = \{Z_\ell\}_{\ell=1}^n$.

The key idea: the **bootstrap world** distribution $\mathbb{P}^{\mathrm{b}}(\hat{\theta}_{n,b_n}(u) \le x)$ should approximate the **real world** distribution $\mathbb{P}(\sqrt{n}(\bar{\theta}_n - \theta^\star)^\top u \le x)$.

### 4.2 Connection to Overlapping Batch Means (OBM)

The bootstrap variance estimator is:

$$\hat{\sigma}_\theta^2(u) = \frac{b_n}{n - b_n + 1}\sum_{t=0}^{n-b_n}\left((\bar{\theta}_{b_n,t} - \bar{\theta}_n)^\top u\right)^2.$$

Up to a multiplicative factor tending to 1, this coincides with the **overlapping batch mean (OBM)** estimator for the asymptotic variance, originally proposed by [Meketon and Schmeiser 1984].

**Proposition 2** establishes that $\hat{\sigma}_\theta^2(u) = \hat{\sigma}_\varepsilon^2(u) + \mathcal{R}_{\mathrm{var}}(u)$, where $\hat{\sigma}_\varepsilon^2(u)$ is the OBM estimator applied to the non-observable noise variables $\{\varepsilon(Z_\ell)\}$, and $\mathcal{R}_{\mathrm{var}}(u)$ is a remainder bounded in $L^p$.

This is important because the LSA iterates $\{\theta_\ell\}$ alone do not form a Markov chain (even when $\{Z_k\}$ does), so classical OBM consistency results do not apply directly. The proposition shows that the block bootstrap for $\{\theta_\ell\}$ is equivalent (up to controllable corrections) to the block bootstrap for the unobservable noise $\{\varepsilon(Z_\ell)\}$.

### 4.3 Variance Estimation: Concentration of OBM (Proposition 3, Corollary 2)

**Proposition 3.** Under A1 and A2, for any $p \ge 2$, $n \ge 2b_n + 1$:

$$\mathbb{E}_\xi^{1/p}\!\left[|\hat{\sigma}_\varepsilon^2(u) - \sigma^2(u)|^p\right] \lesssim \frac{pt_{\mathrm{mix}}^3\|\varepsilon\|_\infty^2}{\sqrt{n}} + \frac{p^2 t_{\mathrm{mix}}^2\sqrt{b_n}\|\varepsilon\|_\infty}{\sqrt{n}} + \frac{pt_{\mathrm{mix}}^2\|\varepsilon\|_\infty^2}{\sqrt{b_n}}.$$

This result relies on **martingale decomposition** for **quadratic forms** of Markov chains, using the framework of [Atchade and Cattaneo 2014].

**Corollary 2.** Setting $b_n = \lceil n^{3/4}\rceil$, $\varepsilon \in (0, 1/\log n)$, $\alpha_k = c_0/(k_0+k)^{1/2+\varepsilon}$: with probability $\ge 1 - n^{-1}$,

$$|\hat{\sigma}_\theta^2(u) - \sigma^2(u)| \lesssim_{\log n} n^{-1/8+\varepsilon/2}.$$

This recovers (up to log factors) the classical rate $n^{-1/8}$ for OBM estimation of the asymptotic variance, previously obtained in [Roy and Balasubramanian 2023] and [Chen et al. 2020].

### 4.4 Proof Ideas for Proposition 3

The proof applies the concentration inequality for quadratic forms of Markov chains from [Atchade and Cattaneo 2014] (see also [Moulines, Naumov, Samsonov 2025]). The key steps:

1. Represent $\hat{\sigma}_\varepsilon^2(u)$ as a quadratic functional of the centered noise process $\bar{W}_{b_n,t} = b_n^{-1}\sum_{\ell=t+1}^{t+b_n}\varepsilon(Z_\ell)$.
2. Decompose the quadratic form into a martingale part and a remainder, using the Poisson equation.
3. Apply the Rosenthal-type inequality (Lemma 24) for additive functionals of Markov chains and concentration bounds for quadratic forms.

---

## 5. Main Result 3: Bootstrap Validity (Theorem 2)

### 5.1 Statement

**Theorem 2.** Under A1, A2, A3, set $b_n = \lceil n^{4/5}\rceil$ and $\alpha_k = c_0/(k_0+k)^{3/5}$. Then for $n$ large enough, with $\mathbb{P}$-probability $\ge 1 - 1/n$:

$$\sup_{x \in \mathbb{R}}|\mathbb{P}(\sqrt{n}(\bar{\theta}_n - \theta^\star)^\top u \le x) - \mathbb{P}^{\mathrm{b}}(\hat{\theta}_{n,b_n}(u) \le x)| \lesssim_{\log n} n^{-1/10}.$$

This is the **first non-asymptotic bound** on the accuracy of bootstrap-based confidence intervals for SA with Markovian noise.

### 5.2 Proof Strategy (Triangle Inequality Scheme)

The proof uses a clever **triangle inequality** linking three distributions:

$$\underbrace{\sqrt{n}u^\top(\bar{\theta}_n - \theta^\star)}_{\text{Real world}} \xleftarrow{\text{Gaussian approx., Cor. 1}} \xi \sim \mathcal{N}(0, \sigma^2(u)) \xrightarrow{\text{Gaussian comparison, Lemma 20}} \xi^{\mathrm{b}} \sim \mathcal{N}(0, \hat{\sigma}_\theta^2(u)) \xleftarrow{\text{exact match}} \underbrace{\hat{\theta}_{n,b_n}(u)}_{\text{Bootstrap world}}$$

**Step 1 (Real world $\to$ Gaussian):** By Corollary 1:

$$\mathrm{d_K}(\sqrt{n}u^\top(\bar{\theta}_n - \theta^\star), \xi) \lesssim_{\log n} n^{-1/4} + n^{1/2-\gamma} + n^{\gamma - 1}.$$

**Step 2 (Gaussian comparison):** Since $\hat{\theta}_{n,b_n}(u)$ is exactly $\mathcal{N}(0, \hat{\sigma}_\theta^2(u))$ under $\mathbb{P}^{\mathrm{b}}$, we need to compare $\mathcal{N}(0, \sigma^2(u))$ and $\mathcal{N}(0, \hat{\sigma}_\theta^2(u))$. Using Lemma 20 (Gaussian comparison from [Devroye, Mehrabian, Reddad 2018]):

$$\mathrm{d_K}(\mathcal{N}(0,\sigma^2(u)), \mathcal{N}(0,\hat{\sigma}_\theta^2(u))) \le \frac{3}{2}\left|\frac{\hat{\sigma}_\theta^2(u)}{\sigma^2(u)} - 1\right|.$$

This requires a **high-probability** bound on $|\hat{\sigma}_\theta^2(u) - \sigma^2(u)|/\sigma^2(u)$, obtained by combining Proposition 2, Proposition 3, and Markov's inequality.

**Step 3 (Optimization):** The final rate is obtained by optimizing $\gamma$ and $b_n$:

$$\mathrm{d_K} \lesssim_{\log n} \frac{b_n^{1/2}}{n^{1-\gamma/2}} + \frac{b_n^{1/2}}{n^{\gamma/2}} + \frac{b_n^{1/2}}{n^{1/2}} + \frac{1}{n^{1/4}} + \frac{1}{n^{\gamma-1/2}} + \frac{1}{n^{1-\gamma}}.$$

Setting $b_n = \lceil n^{4/5}\rceil$ and $\gamma = 3/5$ yields the rate $n^{-1/10}$.

### 5.3 Discussion

The $n^{-1/10}$ rate arises from a **three-way trade-off**:
1. **Berry--Esseen rate** (Theorem 1): improves with aggressive step sizes ($\gamma$ close to $3/4$)
2. **OBM variance estimation** (Corollary 2): improves with conservative step sizes ($\gamma$ close to $1/2$)
3. **Block size** $b_n$: larger blocks improve variance estimation but worsen the effective sample size

The choice $\gamma = 3/5$ balances the Berry--Esseen contribution ($n^{-1/4}$ optimal at $\gamma = 3/4$) against the variance estimation contribution.

**Comparison with i.i.d. bootstrap:** In the i.i.d. noise setting, [Samsonov et al. 2024] and [Sheshukova et al. 2025] achieved $O(n^{-1/2})$ for the coverage probability error. The $n^{-1/10}$ rate here is slower, reflecting the additional cost of dealing with Markovian dependence.

---

## 6. Application to TD Learning (Section 5)

### 6.1 Setup

The TD learning algorithm for policy evaluation in a discounted MDP $(\mathcal{S}, \mathcal{A}, \mathsf{P}, r, \lambda)$ with linear function approximation $V_\theta^\pi(s) = \varphi(s)^\top\theta$ is an instance of LSA with:

$$\mathbf{A}_k = \varphi(S_k)[\varphi(S_k) - \lambda\varphi(S_{k+1})]^\top, \quad \mathbf{b}_k = \varphi(S_k)r(S_k, A_k).$$

### 6.2 Verification of Assumptions

Under two standard assumptions:
- **TD 1:** The state-process Markov chain $\mathsf{P}_\pi$ is uniformly geometrically ergodic with mixing time $\tau$
- **TD 2:** The design matrix $\Sigma_\varphi = \mathbb{E}_\mu[\varphi(s)\varphi(s)^\top]$ is non-degenerate and $\sup_s \|\varphi(s)\| \le 1$

**Proposition 4** verifies that the TD updates satisfy assumption A2 with explicit constants:
- $\mathrm{C_A} = 2(1+\lambda)$
- $\|\varepsilon\|_\infty = 2(1+\lambda)(\|\theta_\star\| + 1)$
- $\alpha_\infty = (1-\lambda)/((1+\lambda)^2)$
- $a = (1-\lambda)\lambda_{\min}(\Sigma_\varphi)$

This ensures that both Theorem 1 and Theorem 2 apply directly to the TD scheme.

---

## 7. Numerical Experiments (Appendix G)

The experiments use a **Garnet problem** (random MDP) with $N_s = 6$ states, $a = 2$ actions, branching factor $b = 3$, feature dimension $d = 2$, discount $\lambda = 0.8$.

**Table 2** shows coverage probabilities of OBM estimation for the empirical distribution at confidence levels $\{0.8, 0.9, 0.95\}$:

| $n$ | $b_n$ | 0.95 ($\hat{\sigma}_\theta^2$ / $\sigma^2$) | 0.9 ($\hat{\sigma}_\theta^2$ / $\sigma^2$) | 0.8 ($\hat{\sigma}_\theta^2$ / $\sigma^2$) | stddev $\times 10^3$ |
|-----|-------|------|------|------|-------|
| 20480 | 250 | 0.873 / 0.881 | 0.773 / 0.805 | 0.641 / 0.662 | 10.89 |
| 204800 | 1200 | 0.935 / 0.945 | 0.880 / 0.892 | 0.768 / 0.784 | 3.49 |
| 1024000 | 3600 | 0.942 / 0.948 | 0.887 / 0.897 | 0.769 / 0.788 | 1.56 |

The results show that: (1) coverage improves with $n$; (2) using the true variance $\sigma^2(u)$ gives slightly better coverage than OBM $\hat{\sigma}_\theta^2(u)$, confirming that variance estimation is a bottleneck; (3) even at moderate sample sizes, the bootstrap provides reasonable coverage.

---

## 8. Key Technical Tools in the Appendix

### 8.1 Stability of Random Matrix Products (Appendix E, Proposition 10)

This is a central tool. The result bounds the $L^p$ norm of the random matrix product:

$$\mathbb{E}_\xi^{1/p}\!\left[\|\Gamma_{j:n}^{(\alpha)}\|^p\right] \le C_\Gamma d^{1/\log n}\exp\!\left\{-(a/12)\sum_{k=j}^{n}\alpha_k\right\},$$

where $C_\Gamma = \sqrt{\kappa_Q}\,\mathrm{e}^2$.

**Proof idea:** The proof uses a **block decomposition** of the product $\Gamma_{j:n}^{(\alpha)} = \prod_{\ell=1}^N \mathbf{Y}_\ell$ where each block $\mathbf{Y}_\ell$ corresponds to $h$ consecutive time steps. The key ingredients are:

1. **Proposition 11** (from [Durmus et al. 2021]): A matrix concentration result for products of random matrices adapted to a filtration. If $\|\mathbb{E}^{\mathfrak{F}_{\ell-1}}[\mathbf{Y}_\ell]\|_Q^2 \le 1 - \mathfrak{m}_\ell$ and $\|\mathbf{Y}_\ell - \mathbb{E}^{\mathfrak{F}_{\ell-1}}[\mathbf{Y}_\ell]\|_Q \le \sigma_\ell$, then $\|\mathbf{Z}_n\|_{q,p}^2 \le \kappa_P \prod_{\ell=1}^n (1 - \mathfrak{m}_\ell + (q-1)\sigma_\ell^2)$.

2. **Lemma 12** (contraction per block): $\|\mathbb{E}_\xi[\mathbf{Y}_\ell]\|_Q \le 1 - (\sum_{k=j_{\ell-1}}^{j_\ell - 1}\alpha_k)a/6$. This is proved by decomposing $\mathbf{Y}_\ell = \mathbf{I} - (\sum \alpha_k)\bar{\mathbf{A}} - \mathbf{S}_\ell + \mathbf{R}_\ell$, where $\mathbf{S}_\ell$ is a linear statistic of the Markov chain and $\mathbf{R}_\ell$ is a higher-order remainder.

3. **Lemma 14** (deviation per block): $\|\mathbf{Y}_\ell - \mathbb{E}_\xi[\mathbf{Y}_\ell]\|_Q \le C_\sigma(\sum \alpha_k)$, using Markov chain concentration.

4. The block size $h$ is chosen (equation (18)) as $h = \lceil 16 t_{\mathrm{mix}}\kappa_Q^{1/2}\mathrm{C_A}/a\rceil$ to ensure that the contraction dominates the deviation within each block.

### 8.2 Last Iterate Bound (Appendix D, Proposition 6)

Bounds the $p$-th moment of $|u^\top(\theta_k - \theta^\star)|$:

$$\mathbb{E}_\xi^{1/p}\!\left[|u^\top(\theta_k - \theta^\star)|^p\right] \lesssim \mathrm{D}_1^{(\mathrm{M})}t_{\mathrm{mix}}\sqrt{p\alpha_k} + C_\Gamma d^{1/\log n}\exp\!\left\{-(a/12)\sum_{\ell=1}^k\alpha_\ell\right\}\|\theta_0 - \theta^\star\|.$$

This is used both in controlling the transient term and in the bootstrap proofs.

### 8.3 Depth-1 Perturbation Expansion (Appendix D.2, Propositions 8--9)

The term $H_k^{(0)}$ (misadjustment between random and deterministic matrix products) is further expanded as $H_k^{(0)} = J_k^{(1)} + H_k^{(1)}$ where:

$$J_k^{(\ell)} = (\mathbf{I} - \alpha_k\bar{\mathbf{A}})J_{k-1}^{(\ell)} - \alpha_k\tilde{\mathbf{A}}(Z_k)J_{k-1}^{(\ell-1)}, \quad H_k^{(\ell)} = (\mathbf{I} - \alpha_k\mathbf{A}(Z_k))H_{k-1}^{(\ell)} - \alpha_k\tilde{\mathbf{A}}(Z_k)J_{k-1}^{(\ell)}.$$

The key technical challenge for $J_k^{(1)}$ is that it is a **double sum** $J_k^{(1)} = \sum_{j=1}^{k-1}\alpha_j S_{j+1:k}\varepsilon(Z_j)$ where $S_{j:k} = \sum_{\ell=j}^{k}\alpha_\ell G_{\ell+1:k}\tilde{\mathbf{A}}(Z_\ell)G_{j+1:\ell-1}$. The authors handle this by:

1. Splitting the sum into blocks of size $m$ to create a structure amenable to martingale techniques.
2. Introducing **independent copies** $Z_{jm+i}^\star$ of the Markov chain (Berbee's coupling lemma) to decouple the dependence between blocks.
3. Applying **Burkholder's inequality** to the resulting martingale-like structure.
4. Using **McDiarmid's bounded differences inequality** for the within-block terms.

The block size $m$ is chosen to balance the coupling error (which decays as $(1/4)^{m/(2pt_{\mathrm{mix}})}$) against the block-size overhead, yielding $m = \lceil 2pt_{\mathrm{mix}}\log(1/\alpha_k)/\log 4\rceil$.

### 8.4 Rosenthal and Burkholder Inequalities for Markov Chains (Appendix H)

**Lemma 24 (Rosenthal inequality):** For bounded $f$ and $\bar{S}_n = \sum_{k=0}^{n-1}\{f(Z_k) - \pi(f)\}$:

$$\mathbb{E}_\pi^{1/p}[|\bar{S}_n|^p] \le \mathrm{D}_{24,1}\,n^{1/2}t_{\mathrm{mix}}^{1/2}p^{1/2}\|f\|_\infty + \mathrm{D}_{24,2}\,t_{\mathrm{mix}}p\|f\|_\infty.$$

**Lemma 25 (Weighted Burkholder):** For weighted sums $\sum u^\top A_k(f(Z_k) - \pi(f))$ with time-varying weights $A_k$:

$$\mathbb{E}_\xi^{1/p}\!\left[|\sum u^\top A_k(f(Z_k)-\pi(f))|^p\right] \le (16/3)t_{\mathrm{mix}}p^{1/2}\|f\|_\infty(\sum \|A_k\|^2)^{1/2} + (8/3)t_{\mathrm{mix}}(\|A_1\| + \|A_n\| + \sum\|A_{k+1}-A_k\|)\|f\|_\infty.$$

Both results use the **Poisson equation** $g(z) - \mathsf{P}g(z) = f(z) - \pi(f)$ to convert the additive functional into a martingale plus boundary terms.

---

## 9. Conclusions and Open Directions

**Main contributions:**
1. $O(n^{-1/4})$ Berry--Esseen rate (up to log factors) for Polyak--Ruppert averaged LSA with Markov noise -- matching the i.i.d. rate
2. First non-asymptotic guarantees for bootstrap-based CI coverage in the Markov noise setting: $O(n^{-1/10})$
3. Recovery of the $n^{-1/8}$ rate (up to logs) for OBM variance estimation
4. Direct application to TD learning for policy evaluation

**Open problems (identified by the authors):**
- Generalize Theorem 1 to **non-linear SA** algorithms
- Obtain **multivariate** (non-projected) Berry--Esseen bounds
- Improve the $n^{-1/10}$ bootstrap rate -- can it match the $O(n^{-1/2})$ rate from the i.i.d. setting?
- Can the $d$-dimensional Berry--Esseen analysis (controlling the remainder $D$ in $\mathbb{R}^d$, not just projections) yield better rates?

---

## Relevance to Thesis (Richardson-Romberg Extrapolation for LSA)

This paper is relevant because:
- It provides **non-asymptotic Berry--Esseen bounds** for averaged LSA with Markov noise, which form the distributional approximation foundation that any inference procedure (including RR-based ones) must build upon
- The **multiplier subsample bootstrap** procedure analyzed here is an alternative to the RR extrapolation approach of [Huo et al. 2023] for constructing confidence intervals -- comparing the two approaches (bootstrap vs. RR) is a natural research direction
- The **OBM variance estimation** rate ($n^{-1/8}$) establishes the cost of estimating the asymptotic covariance non-parametrically, which is relevant for any CI construction method
- The **proof techniques** (Poisson decomposition, random matrix product stability, martingale Berry--Esseen) provide a toolbox applicable to analyzing RR extrapolation in the Markov noise setting
- Unlike [Huo et al. 2023] which uses **constant** stepsize, this paper uses **diminishing** stepsizes -- the two approaches offer complementary perspectives on the same problem

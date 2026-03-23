# Detailed Summary: "High-Order Error Bounds for Markovian LSA with Richardson-Romberg Extrapolation"

**Authors:** Ilya Levin, Alexey Naumov, Sergey Samsonov (HSE University)
**Year:** 2025, arXiv:2508.05570

---

## 1. Problem and Motivation

The paper studies the **bias** and **high-order error bounds** of the Linear Stochastic Approximation (LSA) algorithm with **Polyak–Ruppert (PR) averaging** under **Markovian noise**. The focus is on the **constant step size** regime $\alpha_k = \alpha > 0$.

The LSA iteration with constant step size $\alpha$ is:

$$\theta_k^{(\alpha)} = \theta_{k-1}^{(\alpha)} - \alpha\{\mathbf{A}(Z_k)\theta_{k-1}^{(\alpha)} - \mathbf{b}(Z_k)\}, \quad k \ge 1,$$

with Polyak–Ruppert averaging (burn-in $n_0$):

$$\bar{\theta}_n^{(\alpha)} = (n - n_0)^{-1} \sum_{k=n_0}^{n-1} \theta_k^{(\alpha)}.$$

Here $\{Z_k\}_{k \in \mathbb{N}}$ is a **Markov chain** on a measurable space $(\mathsf{Z}, \mathcal{Z})$, $\mathbf{A}(Z_k)$ and $\mathbf{b}(Z_k)$ are stochastic estimates of $\bar{\mathbf{A}}$ and $\bar{\mathbf{b}}$, and the target is $\theta^\star = \bar{\mathbf{A}}^{-1}\bar{\mathbf{b}}$.

### Why constant step size?

Constant step sizes are attractive because they enable **geometrically fast forgetting** of initialization (Dieuleveut, Durmus, and Bach, 2020) and are easier to use in practice. However, constant step sizes produce an **unavoidable bias** that cannot be eliminated by PR averaging alone. This bias is **linear in $\alpha$** and arises from:
1. **Non-linearity** in the SA iteration (even for linear SA), and
2. **Markovian noise** structure (dependence between observations).

### Key question

The Richardson-Romberg (RR) extrapolation procedure:

$$\bar{\theta}_n^{(\alpha,\mathrm{RR})} = 2\bar{\theta}_n^{(\alpha)} - \bar{\theta}_n^{(2\alpha)}$$

cancels the leading $O(\alpha)$ bias term by running two sequences with step sizes $\alpha$ and $2\alpha$ on the **same noise sequence** $\{Z_k\}$. The paper asks: **What is the leading error term of the RR iterate, and does it achieve the asymptotically optimal covariance $\Sigma_\infty$?**

### Main contributions

1. **Novel bias decomposition technique:** The authors propose a new linearization-based approach to quantify the asymptotic bias of $\bar{\theta}_n^{(\alpha)}$. They study the limiting distribution $\Pi_\alpha$ of the joint Markov chain $\{(\theta_k^{(\alpha)}, Z_{k+1})\}$ and analyze the bias $\Pi_\alpha(\theta_0) - \theta^\star$. Using the perturbation-expansion framework of Aguech, Moulines, and Priouret (2000), they decompose the components into terms **ordered by powers of $\alpha$**.

2. **High-order moment error bounds for RR iterates:** The leading error term of $\bar{\theta}_n^{(\alpha,\mathrm{RR})} - \theta^\star$ scales with $\{\mathrm{Tr}\,\Sigma_\varepsilon^{(\mathrm{M})}\}^{1/2}$, which is the **asymptotically minimax optimal** covariance for Polyak–Ruppert iterates, aligning with $\Sigma_\infty = \bar{\mathbf{A}}^{-1}\Sigma_\varepsilon^{(\mathrm{M})}(\bar{\mathbf{A}}^{-1})^\top$.

---

## 2. Setting and Assumptions

### 2.1 Noise term and error recursion

From the LSA recursion, the error satisfies:

$$\theta_k^{(\alpha)} - \theta^\star = (\mathbf{I} - \alpha\mathbf{A}(Z_k))(\theta_{k-1}^{(\alpha)} - \theta^\star) - \alpha\varepsilon(Z_k),$$

where:
- $\varepsilon(z) = \tilde{\mathbf{A}}(z)\theta^\star - \tilde{\mathbf{b}}(z)$ is the noise term
- $\tilde{\mathbf{A}}(z) = \mathbf{A}(z) - \bar{\mathbf{A}}$, $\tilde{\mathbf{b}}(z) = \mathbf{b}(z) - \bar{\mathbf{b}}$ are the centered versions

### 2.2 Assumptions

**UGE 1 (Uniform Geometric Ergodicity).** $\{Z_k\}_{k \in \mathbb{N}}$ is a Markov chain with kernel $\mathsf{Q}$ on a complete separable metric space $(\mathsf{Z}, \mathcal{Z})$. $\mathsf{Q}$ admits $\pi$ as its unique invariant distribution and is **uniformly geometrically ergodic**: there exists $t_{\mathrm{mix}} \in \mathbb{N}^\star$ such that for all $k \in \mathbb{N}^\star$:

$$\Delta(\mathsf{Q}^k) \le (1/4)^{\lfloor k/t_{\mathrm{mix}} \rfloor},$$

where $\Delta(\mathsf{Q}^k) = \sup_{z,z' \in \mathsf{Z}} (1/2)\|\mathsf{Q}^k(z,\cdot) - \mathsf{Q}^k(z',\cdot)\|_{\mathrm{TV}}$ is the **Dobrushin coefficient**. Equivalently, there exist $\zeta > 0$ and $\rho \in (0,1)$ such that $\sup_z \|\mathsf{Q}^k(z,\cdot) - \pi\|_{\mathrm{TV}} \le \zeta\rho^k$.

**A1 (Boundedness).** $\mathrm{C_A} = \sup_{z \in \mathsf{Z}} \|\mathbf{A}(z)\| \vee \sup_{z \in \mathsf{Z}} \|\tilde{\mathbf{A}}(z)\| < \infty$ and the matrix $-\bar{\mathbf{A}}$ is **Hurwitz** (all eigenvalues have strictly negative real parts). This ensures $\bar{\mathbf{A}}\theta = \bar{\mathbf{b}}$ has a unique solution $\theta^\star$.

**A2 (Noise regularity).** $\int_\mathsf{Z} \mathbf{A}(z)\mathrm{d}\pi(z) = \bar{\mathbf{A}}$ and $\int_\mathsf{Z} \mathbf{b}(z)\mathrm{d}\pi(z) = \bar{\mathbf{b}}$. Moreover, $\|\varepsilon\|_\infty = \sup_{z \in \mathsf{Z}} \|\varepsilon(z)\| < +\infty$.

### 2.3 Key quantities

**Noise covariance matrix (Markovian):**

$$\Sigma_\varepsilon^{(\mathrm{M})} = \mathbb{E}_\pi[\varepsilon(Z_0)\varepsilon(Z_0)^\top] + 2\sum_{\ell=1}^{\infty} \mathbb{E}_\pi[\varepsilon(Z_0)\varepsilon(Z_\ell)^\top].$$

This is the limiting covariance for $n^{-1/2}\sum_{t=0}^{n-1}\varepsilon(Z_t)$ (cf. Douc et al. 2018, Theorem 21.2.10).

**Asymptotically optimal covariance matrix:**

$$\Sigma_\infty = \bar{\mathbf{A}}^{-1}\Sigma_\varepsilon^{(\mathrm{M})}(\bar{\mathbf{A}}^{-1})^\top.$$

This $\Sigma_\infty$ is optimal in the Rao–Cramér sense and corresponds to the last iterate of the preconditioned process $\bar{A}^{-1}\theta_k$.

**Joint Markov kernel $\bar{\mathsf{P}}_\alpha$:** Since $\theta_k^{(\alpha)}$ alone may not be Markov (due to Markovian noise), the authors consider the joint process $(\theta_k^{(\alpha)}, Z_{k+1})$ with kernel:

$$\bar{\mathsf{P}}_\alpha f(\theta, z) = \int_\mathsf{Z} \mathsf{Q}(z, \mathrm{d}z') f(\mathbf{F}_{z'}(\theta), z'),$$

where $\mathbf{F}_z(\theta) = (\mathbf{I} - \alpha\mathbf{A}(z))\theta + \alpha\mathbf{b}(z)$.

**Step size thresholds.** Various upper bounds on $\alpha$ are required, captured by:

$$\alpha_{q,\infty}^{(\mathrm{M})} = \left[\alpha_\infty \wedge \kappa_Q^{-1/2}\mathrm{C_A}^{-1} \wedge a/(6e\kappa_Q\mathrm{C_A})\right] \times \lceil 8\kappa_Q^{1/2}\mathrm{C_A}/a\rceil^{-1},$$

$$\alpha_{p,\infty}^{(\mathrm{b})} = \alpha_{q,\infty}^{(\mathrm{M})} \wedge \mathrm{c_A^{(M)}}/q,$$

where $a$ is the real part of the minimum eigenvalue of $\bar{\mathbf{A}}$, $\kappa_Q = \lambda_{\max}(Q)/\lambda_{\min}(Q)$ from the Lyapunov equation, and $\mathrm{c_A^{(M)}} = a/\{12\,\mathrm{C_\Gamma}\}$.

---

## 3. Bias of the LSA Iterates (Section 4)

### 3.1 Perturbation-expansion decomposition

The key technical tool is the **perturbation-expansion framework** from Aguech, Moulines, and Priouret (2000). Define the product of random matrices:

$$\Gamma_{m:n}^{(\alpha)} = \prod_{i=m}^{n}(\mathbf{I} - \alpha\mathbf{A}(Z_i)), \quad m \le n.$$

The error decomposes into **transient** and **fluctuation** terms:

$$\theta_n^{(\alpha)} - \theta^\star = \underbrace{\Gamma_{1:n}^{(\alpha)}\{\theta_0 - \theta^\star\}}_{\tilde{\theta}_n^{(\mathrm{tr})}} + \underbrace{\left(-\alpha\sum_{j=1}^{n}\Gamma_{j+1:n}^{(\alpha)}\varepsilon(Z_j)\right)}_{\tilde{\theta}_n^{(\mathrm{fl})}}.$$

### 3.2 Higher-order decomposition of the fluctuation term

The fluctuation term $\tilde{\theta}_n^{(\mathrm{fl})}$ is further decomposed using the recurrence relations. Define:

$$J_n^{(0,\alpha)} = (\mathbf{I} - \alpha\bar{\mathbf{A}})\,J_{n-1}^{(0,\alpha)} - \alpha\varepsilon(Z_n), \quad J_0^{(0,\alpha)} = 0,$$
$$H_n^{(0,\alpha)} = (\mathbf{I} - \alpha\mathbf{A}(Z_n))\,H_{n-1}^{(0,\alpha)} - \alpha\tilde{\mathbf{A}}(Z_n)J_{n-1}^{(0,\alpha)}, \quad H_0^{(0,\alpha)} = 0,$$

so that $\tilde{\theta}_n^{(\mathrm{fl})} = J_n^{(0,\alpha)} + H_n^{(0,\alpha)}$.

More generally, for $L \in \mathbb{N}^\star$ and $\ell \in \{1, \ldots, L\}$:

$$J_n^{(\ell,\alpha)} = (\mathbf{I} - \alpha\bar{\mathbf{A}})\,J_{n-1}^{(\ell,\alpha)} - \alpha\tilde{\mathbf{A}}(Z_n)J_{n-1}^{(\ell-1,\alpha)},$$
$$H_n^{(\ell,\alpha)} = (\mathbf{I} - \alpha\mathbf{A}(Z_n))\,H_{n-1}^{(\ell,\alpha)} - \alpha\tilde{\mathbf{A}}(Z_n)J_{n-1}^{(L,\alpha)}.$$

This gives the **cornerstone decomposition** (with $L = 2$):

$$\theta_n^{(\alpha)} - \theta^\star = \tilde{\theta}_n^{(\mathrm{tr})} + J_n^{(0,\alpha)} + J_n^{(1,\alpha)} + J_n^{(2,\alpha)} + H_n^{(2,\alpha)}.$$

**Key insight:** Each successive term $J_n^{(k,\alpha)}$ is of **higher order in $\alpha$**. Specifically:
- $J_n^{(0,\alpha)}$ — the "linearized" fluctuation, driven by $\varepsilon(Z_n)$, no dependence on $\tilde{\mathbf{A}}$
- $J_n^{(1,\alpha)}$ — first-order correction, involves one factor of $\tilde{\mathbf{A}}$, contributes the **leading bias** $O(\alpha)$
- $J_n^{(2,\alpha)}$ — second-order correction, contributes bias $O(\alpha^2)$
- $H_n^{(2,\alpha)}$ — remainder, bounded at $O(\alpha^{5/2})$

### 3.3 Bounding the transient and fluctuation terms

**Transient term:** Bounded using exponential stability of the random matrix product from Durmus et al. (2025, Proposition 7):

$$R_{n,p,\alpha}^{(\mathrm{tr})} \lesssim (\alpha n)^{-1}.$$

**Fluctuation term $\tilde{\theta}_n^{(\mathrm{fl})}$:** Uses the perturbation-expansion technique. For any $\ell \ge 0$, the vectors $J_n^{(\ell,\alpha)}$ and $H_n^{(\ell,\alpha)}$ satisfy recurrences that can be bounded via Rosenthal-type inequalities for Markov chains.

### 3.4 Bias expansion for LSA (Proposition 2 and Corollary 1)

Instead of analyzing $\{J_k^{(\ell,\alpha)}\}$ alone (which may not be Markov), the authors consider the **joint process**:

$$Y_t = (Z_{t+1}, J_t^{(0,\alpha)}, J_t^{(1,\alpha)})$$

with Markov kernel $\mathsf{Q}_{J^{(1)}}$. Under **A1**, **A2**, and **UGE 1**, for $\alpha \in (0, \alpha_{p,\infty}^{(\mathrm{b})})$, this process admits a unique stationary distribution $\Pi_{J^{(1)},\alpha}$.

**Proposition 2.** Under A1, A2, UGE 1, for $\alpha \in (0, \alpha_\infty^{(\mathrm{b})})$:

$$\lim_{n \to \infty} \mathbb{E}[J_n^{(1,\alpha)}] = \mathbb{E}[J_\infty^{(1,\alpha)}] = \alpha\Delta + R(\alpha),$$

where:

$$\Delta = \bar{\mathbf{A}}^{-1}\sum_{k=1}^{\infty}\mathbb{E}[\{\mathsf{Q}^k\tilde{\mathbf{A}}(Z_\infty)\}\varepsilon(Z_\infty)],$$

and $R(\alpha)$ is a remainder bounded as $\|R(\alpha)\| \le 12\|\bar{\mathbf{A}}^{-1}\|\,\mathrm{C_A^2}\,t_{\mathrm{mix}}^2\,\alpha^2\|\varepsilon\|_\infty$.

**Corollary 1 (Asymptotic bias expansion):**

$$\lim_{n \to \infty} \mathbb{E}[\theta_n] = \Pi_\alpha(\theta_0) = \theta^\star + \alpha\Delta + O(\alpha^{3/2}).$$

**Key point:** The bias is **linear in $\alpha$** with coefficient $\Delta$ determined by the correlation structure of the Markov chain. The $\Delta$ term matches the representation in Lauand and Meyn (2023a, Theorem 2.5).

**Remark 1 (Sharper expansion):** By analyzing $J_n^{(2,\alpha)}$ more carefully (Proposition 6), the bias can be expanded as a power series:

$$\lim_{n \to \infty} \mathbb{E}[J_n^{(2,\alpha)}] = \alpha^2\Delta_2 + R_2(\alpha), \quad \|R_2(\alpha)\| \lesssim \alpha^{5/2}.$$

where $\Delta_2 = -\sum_{k=1}^{\infty}\sum_{i=0}^{\infty}\mathbb{E}[\tilde{\mathbf{A}}(Z_{\infty+k+i+1})\tilde{\mathbf{A}}(Z_{\infty+i+1})\varepsilon(Z_{\infty-k-i})]$. This suggests the remainder in Corollary 1 could potentially be improved to $O(\alpha^2)$.

### 3.5 Discussion: connection to adjoint kernel representation

The coefficient $\Delta$ can be reformulated using the **adjoint kernel** $\mathsf{Q}^\star$ such that $\pi \otimes \mathsf{Q}(A \times B) = \pi \otimes \mathsf{Q}^\star(B \times A)$, and the **independent kernel** $\Pi$ with $\Pi(z, A) = \pi(A)$. The authors in Huo, Chen, and Xie (2023a) considered the Neumann series expansion for the operator $(\mathbf{I} - \mathsf{Q}^\star + \Pi)^{-1}(\mathsf{Q}^\star - \Pi)$. The result in Proposition 2 can be reformulated in terms of $\mathsf{Q}^\star$, which is more desirable as it requires **reversibility** of the Markov kernel $\mathsf{Q}$.

---

## 4. Analysis of Richardson-Romberg Procedure (Section 5)

### 4.1 The RR estimator

The Richardson-Romberg extrapolation (Hildebrand, 1987) cancels the leading $O(\alpha)$ bias:

$$\bar{\theta}_n^{(\alpha,\mathrm{RR})} = 2\bar{\theta}_n^{(\alpha)} - \bar{\theta}_n^{(2\alpha)}.$$

After this procedure, the **remainder** in the bias has order $O(\alpha^{3/2})$ (from Corollary 1).

### 4.2 Augmented Markov chain for the RR iterate

To analyze the RR iterate, the authors consider the joint Markov chain:

$$V_t = (J_t, Z_{t+1}),$$

where $J_t$ collects the relevant components. This chain has kernel $\mathsf{Q}_J$ and, by Corollary 2, admits a unique stationary distribution under $\alpha \in (0, \alpha_\infty^{(\mathrm{b})})$.

**Corollary 2.** Under A1, A2, UGE 1, for $\alpha \in (0, \alpha_\infty^{(\mathrm{b})})$, the process $\{V_t\}_{t \in \mathbb{N}}$ is a Markov chain with a unique stationary distribution $\Pi_{J,\alpha}$.

The invariant distribution $\Pi_{J,\alpha}$ coincides with the distribution of $(J_\infty^{(0,\alpha)}, Z_{\infty+1})$.

### 4.3 The key centered quantity

Define the function $\psi(J, z) = \tilde{\mathbf{A}}(z)J$ for $J \in \mathbb{R}^d$, $z \in \mathsf{Z}$, and its centered version:

$$\tilde{\psi}(J, z) = \psi(J, z) - \mathbb{E}_{\Pi_{J,\alpha}}[\psi_0],$$
$$\psi_t = \tilde{\psi}(J_t^{(0,\alpha)}, Z_{t+1}).$$

**Proposition 3.** Under A1, A2, UGE 1, for any probability measure $\xi$ on $\mathbb{R}^d \times \mathsf{Z}$, $2 \le p < \infty$, $\alpha \in (0, \alpha_{p,\infty}^{(\mathrm{b})})$:

$$\mathbb{E}_\xi^{1/p}\Big[\Big\|\sum_{t=0}^{n-1}\tilde{\psi}_t\Big\|^p\Big] \le c_{W,1}^{(2)}\,p^{3/2}(\alpha n)^{1/2} + c_{W,2}^{(2)}\,p^3\alpha^{-1/2}\sqrt{\log(1/\alpha\alpha)}.$$

This bound shows that $\sum \tilde{\psi}_t$ has a **non-zero bias** (it is not automatically centered) but after centering it can be effectively estimated. This provides theoretical justification for the numerical experiments.

### 4.4 Error decomposition for RR iterates

Using the definition of the RR iterate and the bias decomposition:

$$\bar{\mathbf{A}}(\bar{\theta}_n^{(\alpha,\mathrm{RR})} - \theta^\star) = \{2\alpha(n - n_0)\}^{-1}(4\theta_{n_0}^{(\alpha)} - \theta_{n_0}^{(2\alpha)}) - (4\theta_n^{(\alpha)} - \theta_n^{(2\alpha)}))$$

$$+ \{n - n_0\}^{-1}\sum_{t=n_0}^{n-1}\{\mathsf{e}(\theta_t^{(2\alpha)}, Z_{t+1}) - 2\mathsf{e}(\theta_t^{(\alpha)}, Z_{t+1})\},$$

where $\mathsf{e}(\theta, z) = \varepsilon(z) + \tilde{\mathbf{A}}(z)\theta$ is the noise function evaluated at $\theta$.

The first term is the **transient** contribution; the second term involves the **noise + bias correction**. The sum $\sum \mathsf{e}(\theta_t^{(2\alpha)}, Z_{t+1}) - 2\mathsf{e}(\theta_t^{(\alpha)}, Z_{t+1})$ is further decomposed using:

$$\sum_{t=n_0}^{n-1}\mathsf{e}(\theta_t^{(\alpha)}, Z_{t+1}) = \underbrace{E_n^{(\mathrm{tr},\alpha)}}_{\text{transient}} + \underbrace{E_n^{(\mathrm{fl},\alpha)}}_{\text{fluctuation}},$$

where:
- $E_n^{(\mathrm{tr},\alpha)} = \sum_{t=n_0}^{n-1}\tilde{\mathbf{A}}(Z_{t+1})\Gamma_{1:t}^{(\alpha)}\{\theta_0 - \theta^\star\}$ — linear statistics of the Markov chain, bounded via Rosenthal inequality for Markov chains (Durmus et al., 2023)
- $E_n^{(\mathrm{fl},\alpha)}$ — further decomposed into terms involving $J_t^{(\ell,\alpha)}$ and $H_t^{(2,\alpha)}$

### 4.5 Main Result: High-order moment bounds for RR iterates (Theorem 2)

**Theorem 2.** Assume A1, A2, and UGE 1. Fix $2 \le p < \infty$, then for any $n \ge t_{\mathrm{mix}}$, $\alpha \in (0, \alpha_{p,\infty}^{(\mathrm{b})})$ and initial probability measure $\xi$ on $(\mathsf{Z}, \mathcal{Z})$:

$$\mathbb{E}_\xi^{1/p}[\|\bar{\mathbf{A}}(\bar{\theta}_n^{(\alpha,\mathrm{RR})} - \theta^\star)\|^p]$$
$$\le 2\,\mathrm{C_{Rm,1}}\{\mathrm{Tr}\,\Sigma_\varepsilon^{(\mathrm{M})}\}^{1/2}p^{1/2}n^{-1/2} + R_{n,p,\alpha}^{(\mathrm{fl})} + R_{n,p,\alpha}^{(\mathrm{tr})}\|\theta_0 - \theta^\star\|\exp\{-\alpha a n/24\},$$

where:
- $\mathrm{C_{Rm,1}} = 60\mathrm{e}$ (constant from Rosenthal's martingale inequality)
- $R_{n,p,\alpha}^{(\mathrm{fl})}$ — the fluctuation remainder, specified in equation (25)
- $R_{n,p,\alpha}^{(\mathrm{tr})}$ — the transient remainder

The **fluctuation remainder** has the following structure:

$$R_{n,p,\alpha}^{(\mathrm{fl})} \lesssim p\,n^{-3/4} + (p^{3/2}(\alpha n)^{-1/2} + \alpha^{1/2})p^{3/2}n^{-1/2} + p^{7/2}\alpha^{3/2}\log^{3/2}(1/\alpha\alpha),$$

$$R_{n,p,\alpha}^{(\mathrm{tr})} \lesssim (\alpha n)^{-1}.$$

### 4.6 Corollary 3: Optimal step size choice

**Corollary 3.** Under A1, A2, UGE 1, for $2 \le p < \infty$ and any $n \ge t_{\mathrm{mix}}$, choosing the step size:

$$\alpha(n, d, t_{\mathrm{mix}}, p) = \alpha_{p,\infty}^{(\mathrm{b})}\,n^{-1/2},$$

it holds with probability at least $1 - \delta$ (with $p = \ln(3\mathrm{e}/\delta)$) that:

$$\|\bar{\mathbf{A}}(\bar{\theta}_n^{(\alpha,\mathrm{RR})} - \theta^\star)\| \lesssim \sqrt{\mathrm{Tr}\,\Sigma_\varepsilon^{(\mathrm{M})}}\cdot\sqrt{\frac{\log(1/\delta)}{n}} + (1 + \log^{3/2}(n)\log^{5/2}(1/\delta))\frac{\log(1/\delta)}{n^{3/4}}$$

$$+ n^{-1/2}\log(1/\delta)\|\theta_0 - \theta^\star\|\exp\left\{-\alpha_{1+\log d,\infty}^{(\mathrm{b})}\,n^{1/2}\right\}.$$

**Interpretation:**
- The **leading term** scales as $\sqrt{\mathrm{Tr}\,\Sigma_\varepsilon^{(\mathrm{M})}}\,n^{-1/2}$, matching the asymptotically minimax optimal rate for PR averaging
- With $\alpha = O(n^{-1/2})$, this aligns with the rate observed in the i.i.d. case (Sheshukova et al., 2024)
- The term with $p^3$ could potentially be improved to $p^2$ through a more careful analysis of the Rosenthal inequality (via Lemma 5)
- The remainder $O(\alpha^{3/2})$ in the bias could potentially be improved to $O(\alpha^2)$ using the second-order expansion of $J_n^{(3,\alpha)}$

---

## 5. Proof Ideas and Techniques

### 5.1 Coupling construction (Appendix B.1)

A central proof technique is the **coupling of Markov chains**. For two initial distributions $\xi, \xi'$ on $(\mathsf{Z}, \mathcal{Z})$, there exists a **maximal exact coupling** $(\Omega, \mathcal{F}, \tilde{\mathbb{P}}_{\xi,\xi'}, Z, Z', T)$ of $\mathbb{P}_\xi^{\mathsf{Q}}$ and $\mathbb{P}_{\xi'}^{\mathsf{Q}}$ such that:

$$\|\xi\mathsf{Q}^n - \xi'\mathsf{Q}^n\|_{\mathrm{TV}} = 2\mathbb{P}(T > n).$$

Two LSA sequences $\theta_n^{(\alpha)}$ and $\tilde{\theta}_n^{(\alpha)}$ are run with $Z_n$ and $\tilde{Z}_n$ respectively, where the noise sequences **couple** (become identical) after a random time $T$. The key bound is:

$$\tilde{\mathbb{E}}_{z,\tilde{z}}[\mathsf{c}_0((\theta_n^{(\alpha)}, Z_n), (\tilde{\theta}_n^{(\alpha)}, \tilde{Z}_n))] \lesssim \rho_\alpha^n\,\mathsf{c}_0((z, \theta), (\tilde{z}, \tilde{\theta})),$$

where $\rho_\alpha = \mathrm{e}^{-\alpha a/24}$ and $\mathsf{c}_0$ is a carefully designed cost function:

$$\mathsf{c}_0((\theta, z), (\theta', z')) = (\|\theta - \theta'\| + \mathbf{1}_{\{z \ne z'\}})(1 + \|\theta - \theta^\star\| + \|\theta' - \theta^\star\|).$$

This cost function is **not a metric** but is **symmetric, lower semi-continuous, and distance-like** (Douc et al. 2018, Chapter 20.1).

### 5.2 Contraction of Wasserstein semimetric (Appendix B.2)

For the analysis of each component $J_n^{(\ell,\alpha)}$, the authors prove contraction results for Wasserstein semimetrics with increasingly complex cost functions.

**Lemma 1 (Contraction for $J^{(0,\alpha)}$).** For pairs $y = (J, z)$ and $y' = (J', z')$ with $y \ne y'$:

$$\tilde{\mathbb{E}}_{y,y'}^{1/p}[\|J_n^{(0,\alpha)} - \tilde{J}_n^{(0,\alpha)}\|^p] \le \mathrm{c}_{W,1}\,t_{\mathrm{mix}}^{1/2}\,p^{1/2}\,\rho_{1,\alpha}^{n/p}(\|J\| + \|J'\| + \sqrt{\alpha a}\|\varepsilon\|_\infty),$$

where $\rho_{1,\alpha} = \mathrm{e}^{-\alpha a/12}$.

**Lemma 2 (Contraction for $J^{(1,\alpha)}$).** An analogous but more complex bound involving $\sqrt{\log(1/\alpha a)}$ factors:

$$\tilde{\mathbb{E}}_{y,y'}^{1/p}[\|J_n^{(1,\alpha)} - \tilde{J}_n^{(1,\alpha)}\|^p] \le \mathrm{c}_{W,1}^{(1)}\,p^2\,t_{\mathrm{mix}}^{3/2}\,\rho_{1,\alpha}^{n/p}\sqrt{\log(1/\alpha a)}(\|J^{(0)}\| + \|\tilde{J}^{(0)}\| + \|J^{(1)}\| + \|\tilde{J}^{(1)}\| + \sqrt{\alpha a}\|\varepsilon\|_\infty).$$

**Proposition 5 / Lemma 3 (Contraction for $J^{(2,\alpha)}$).** Similar structure with $p^{7/2}t_{\mathrm{mix}}^{5/2}(\log(1/\alpha a))^{3/2}$.

The proofs follow a recursive pattern:
1. Decompose $J_n^{(\ell,\alpha)} - \tilde{J}_n^{(\ell,\alpha)}$ using the coupling construction
2. Split into a term depending on $J_{n \wedge T}^{(\ell,\alpha)}$ (pre-coupling) and a sum involving $\tilde{\mathbf{A}}(Z_{n-k+1})(J_{n-k}^{(\ell-1,\alpha)} - \tilde{J}_{n-k}^{(\ell-1,\alpha)})$ (post-coupling interaction)
3. Use Hölder's and Minkowski's inequalities to separate the random matrix product from the initial conditions
4. Apply the bound on the coupling time $\tilde{\mathbb{P}}_{z,z'}^{1/2}(T \ge k) \le \zeta^{1/2}\rho^{k/2}$

### 5.3 Proof of Theorem 1 (existence of invariant distribution $\Pi_\alpha$)

The proof uses:
1. The coupling construction to show contraction of $\bar{\mathsf{P}}_\alpha$ in the Wasserstein semimetric associated with $\mathsf{c}_0$
2. Douc et al. (2018, Theorem 20.3.4) to conclude existence and uniqueness of $\Pi_\alpha$
3. Villani (2009, Theorem 6.9) to conclude $\Pi_\alpha(\|\theta_0 - \theta^\star\|) < \infty$

### 5.4 Proof of Proposition 2 (bias coefficient $\Delta$)

The proof studies the stationary distribution of $J_\infty^{(1,\alpha)}$ under $\Pi_{J^{(1)},\alpha}$:

1. From the recursion for $J_n^{(1,\alpha)}$ (eq. 14), derive $\mathbb{E}[J_{\infty+1}^{(1)}] = \mathbb{E}[J_\infty^{(1)}] - \alpha\bar{\mathbf{A}}\mathbb{E}[J_\infty^{(1)}] - \alpha\mathbb{E}[\tilde{\mathbf{A}}(Z_{\infty+1})J_\infty^{(0)}]$
2. Since $\mathbb{E}[J_\infty^{(0)}] = 0$ (linear statistics of $\varepsilon(Z_k)$), expand $\bar{\mathbf{A}}\mathbb{E}[J_\infty^{(1)}] = -\mathbb{E}[\tilde{\mathbf{A}}(Z_{\infty+1})J_\infty^{(0)}]$
3. Express $\mathbb{E}[\tilde{\mathbf{A}}(Z_{\infty+1})J_\infty^{(0)}]$ using the Neumann series expansion $\sum_{k=1}^{\infty}\tilde{\mathbf{A}}(Z_{\infty+1})(\mathbf{I} - \alpha\bar{\mathbf{A}})^{k-1}\varepsilon(Z_{\infty-k+1})$
4. Separate the leading term $\alpha\sum_{k=1}^{\infty}\mathbb{E}[\tilde{\mathbf{A}}(Z_{\infty+k+1})\varepsilon(Z_{\infty-k+1})]$ from the remainder $R(\alpha)$ involving higher-order $\alpha$ terms
5. Use the $\sigma$-algebra $\mathcal{F}_t^- = \sigma(Z_{\infty-t}, Z_{\infty-t-1}, \ldots)$ and the Markov property to simplify expectations to $\mathbb{E}[\mathsf{Q}^k\tilde{\mathbf{A}}(Z_\infty)\varepsilon(Z_\infty)]$

### 5.5 Rosenthal-type inequality (Appendix C)

**Theorem 3 (Rosenthal inequality for Markov chains).** Under A1, A2, UGE 1, for the Markov chain $((J_t^{(0,\alpha)}, Z_{t+1}), t \ge 0)$ with the function $\psi(J, z) = \tilde{\mathbf{A}}(z)J$:

$$\mathbb{E}_\xi^{1/p}\Big[\Big\|\sum_{t=0}^{n-1}\tilde{\psi}_t\Big\|^p\Big] \le \mathrm{C_{Ros,1}}\,p^{1/2}(\alpha n)^{1/2}\mathrm{v} + \mathrm{C_{Ros,2}}\,p\,b\,\mathrm{v},$$

where $\mathrm{v}$ and $b$ are variance and tail parameters. This extends results of Durmus et al. (2023) to the current augmented chain setting.

The proof relies on:
- The cost function $\mathsf{c}_J$ (eq. 60) designed for the chain $(J_n^{(0,\alpha)}, Z_{n+1})$
- Establishing Lipschitz property: $\|\psi(J, z) - \psi(\tilde{J}, \tilde{z})\| \le 2\mathrm{C_A}\,\mathsf{c}_J((J, z), (\tilde{J}, \tilde{z}))$
- Using Lemma 7 (from Durmus et al. 2025) to convert Wasserstein contraction into variance bounds

### 5.6 Proof sketch for Theorem 2

1. **Decompose** $\bar{\theta}_n^{(\alpha,\mathrm{RR})} - \theta^\star$ using equation (26) into transient and fluctuation components
2. **Leading term:** The sum $\sum_{t=n_0}^{n-1}\varepsilon(Z_t)$ is a linear statistic of the Markov chain $\{Z_k\}$, bounded via the Rosenthal inequality (Durmus et al. 2023), yielding the $\sqrt{\mathrm{Tr}\,\Sigma_\varepsilon^{(\mathrm{M})}}\,n^{-1/2}$ rate
3. **Term involving $J_t^{(0,\alpha)}$:** For the statistic $\sum_{t=n_0}^{n-1}\tilde{\mathbf{A}}(Z_{t+1})J_t^{(0,\alpha)}$, express in terms of $J_n^{(1,\alpha)}$ via the expansion (14), then use the expansion from (88) to obtain a centered random variable plus a bias term. This yields the bound $O((\alpha/n)^{1/2} + \alpha^{-1/2}n^{-1})$
4. **Term involving $J_t^{(1,\alpha)}$ and $H_t^{(2,\alpha)}$:** The analogous term involving $J_n^{(2,\alpha)}$ is bounded via Proposition 9, contributing $O(\alpha^2)$
5. **Combining:** With the optimal choice $\alpha = \alpha_{p,\infty}^{(\mathrm{b})}\,n^{-1/2}$, all remainder terms are of order $n^{-3/4}$ or smaller

---

## 6. Experiments (Section 6)

The authors validate their bounds on a 2D example from Lauand (2024):
- **Markov chain:** $\{Z_k\}$ on $\mathsf{Z} = \{0, 1\}$ with transition matrix $P = \begin{pmatrix} a & 1-a \\ 1-a & a \end{pmatrix}$, $a = 0.3$
- **Noisy observations:** $\mathbf{A}(z) = z \cdot A^{(1)} + (1-z) \cdot A^{(0)}$, $\mathbf{b}(z) = z \cdot b^{(1)} + (1-z) \cdot b^{(0)}$
- **Setup:** $\bar{\mathbf{A}} = \mathbf{I}$, $\bar{\mathbf{b}} = (1/2)b^{(1)}$, $\theta_0 = \theta^\star$, $N_{\mathrm{traj}} = 400$

### Results (Figure 1):

**(a) RR error vs. $n$:** For step sizes $\alpha = n^{-\beta}$ with $\beta \in \{1/2, 2/3, 3/4, 5/6\}$, the error of RR iterates decreases faster than PR iterates.

**(b) PR error vs. $n$:** Scaled by the leading term from the bound (25), confirming the predicted rate.

**(c) Rescaled RR error:** For $\beta \ge 1/2$, the term $R_{n,p,\alpha}^{(\mathrm{fl})}$ scales as $n^{\beta-2}$ (confirming theoretical prediction).

**(d) MSE comparison:** The key practical result — after a few iterations, the RR procedure error **decreases faster** than PR averaging error. The optimal step size for RR is $\alpha = n^{-1/2}$, while for PR averaging this choice introduces a large bias.

---

## 7. Comparison with Prior Work (Section 4, Discussion)

| Aspect | Huo et al. (2024) | Huo, Chen, Xie (2023a) | This paper |
|---|---|---|---|
| Setting | General SA | LSA | LSA |
| Bias decomposition | Up to linear term in $\alpha$ | Infinite series in $\alpha$ | Power series, ordered by $\alpha$ |
| Leading term coefficient | Not explicit | Not explicit | **Explicit $\Delta$** |
| Higher-order terms | MSE only | MSE under RR | **High-order moment bounds** |
| Markov chain assumption | General | **Reversibility** required | **UGE (no reversibility)** |
| Covariance optimality | Leading MSE with $\Sigma_\infty$ | Leading MSE with $\Sigma_\infty$ | **Leading moment with $\Sigma_\infty$** |
| RR analysis | — | Eliminates leading bias | Full high-order bounds |

---

## 8. Conclusion and Future Directions

The paper establishes:
1. **Explicit bias expansion** for constant step-size LSA with Markovian noise, ordered by powers of $\alpha$
2. **High-order moment bounds** for the RR iterate achieving the optimal leading term $\sqrt{\mathrm{Tr}\,\Sigma_\varepsilon^{(\mathrm{M})}}\,n^{-1/2}$
3. The bounds are **non-asymptotic** with explicit dependence on $n$, $\alpha$, $p$, and $t_{\mathrm{mix}}$

**Open directions:**
- Generalization to **non-linear** Markovian SA
- SA with **state-dependent noise**
- Improving the remainder from $O(\alpha^{3/2})$ to $O(\alpha^2)$ in the bias (requires bounding $J_n^{(3,\alpha)}$)
- Tightening the $p^3$ term to $p^2$ in the fluctuation bound via a refined Rosenthal inequality

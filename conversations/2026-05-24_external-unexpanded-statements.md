# Внешние statements, которые в тексте не выписаны полностью

## Вопрос

Составить список всех мест, где текст ссылается на леммы, propositions,
theorems или схемы из других статей, но соответствующее утверждение не
выписано явно в дипломе.

## Критерий

Я включал место в список, если внешний result используется или упоминается как
statement, но в текущем тексте не выписаны одно или несколько из:

- точная формула оценки;
- условия применимости;
- admissibility threshold;
- зависимость constants;
- объект, к которому применяется theorem;
- полный statement, если в тексте дана только фраза вида "by Proposition X".

Не включал места, где внешний statement уже локально выписан как рабочая
лемма с формулой. Например, `Input A`, `Input B`, Levin Proposition 2,
Levin Corollary 6, Levin Propositions 8--9 в `src/pr_weights.typ` уже
выписаны достаточно явно как working forms.

## Формулировки внешних утверждений

Ниже я добавляю рабочие формулировки для всех ссылок из списка. Это не
дословные цитаты, а математические statements, переписанные в обозначениях
диплома. Для Levin, Samsonov и Huo они сверены с локальными PDF в `papers/`.
Для Douc et al. и Dieuleveut--Durmus--Bach локального PDF в репозитории нет,
поэтому ниже дана стандартная форма ровно того результата, который нужен в
дипломе.

### 1. Levin admissibility threshold $\alpha_*(q,t_{\mathrm{mix}})$

В Levin et al. используются несколько малых шаговых потолков, а не один
универсальный $\alpha_*$. Удобная формулировка для диплома:

> Assume A1, A2 and UGE 1. For each $q \ge 2$ there is a threshold
> $\alpha_{q,\infty}^{(M)}$, defined in Levin et al. by their
> Eqs. (30)--(31), such that random-product stability estimates hold for
> $\alpha \lesssim t_{\mathrm{mix}}^{-1}\alpha_{q,\infty}^{(M)}$ and moment
> order up to $q$. The threshold is a minimum of a deterministic Lyapunov
> ceiling, bounded-noise ceilings involving $C_A,\kappa_Q,a$, and the factor
> $c_A^{(M)}/q$.

For the bias and invariant-chain results Levin et al. introduce
$\alpha_{p,\infty}^{(b)}$, a smaller threshold built from
$\alpha_{p(1+\log d),\infty}^{(M)}$, additional boundedness/stability ceilings,
and a $t_{\mathrm{mix}}^{-1}$ factor. Thus in the diploma
$\alpha_*(q,t_{\mathrm{mix}})$ should be read as the minimum of the Levin
ceilings needed for:

$$
\text{Proposition 2},\quad
\text{Corollary 6},\quad
\text{Proposition 8},\quad
\text{Proposition 9},
$$

applied at both $w=\alpha$ and $w=2\alpha$.

### 2. Startup threshold $\alpha_{\mathrm{st}}(p)$

Levin et al. Proposition 5 states a coupling contraction under the restriction

$$
\alpha \in \left(0,\alpha_\infty \wedge (ap)^{-1}\log \rho^{-1}\right),
\qquad
\rho_{1,\alpha}=e^{-\alpha a/12}.
$$

The thesis threshold $\alpha_{\mathrm{st}}(p)$ is not a named Levin constant.
It should be formulated as a local minimum:

> $\alpha_{\mathrm{st}}(p)$ is the minimum of the Levin Proposition-5
> startup-contraction ceiling at moment order $p$, the random-product
> stability ceiling used in Levin Appendix D.1 at moment order $2p$, and the
> small-step conditions needed to apply these estimates uniformly for
> $w\in\{\alpha,2\alpha\}$.

So the condition in the thesis should be $2\alpha\le
\alpha_{\mathrm{st}}(p)$.

### 3. Random-product stability behind Levin Appendix D.1, Proposition 9

The specific product estimate used inside Levin Proposition 9 is:

> Under A1, A2 and UGE 1, for $2\le p\le q/2$ and admissible $\alpha$,
> the random product satisfies
> $$
> \left\|\Gamma_{\ell+1:n}^{(\alpha)}
>       \widetilde A(Z_\ell)\right\|_{L_{2p}}
> \le
> 2\kappa_Q^{1/2} C_A e^2 d^{1/q}
> e^{-\alpha a(n-\ell)/12}.
> $$

Levin et al. use this with Proposition 8 to prove the one-trajectory bound for
$H_n^{(2,\alpha)}$. The stronger thesis form

$$
\|\Gamma_{s+1:k}^{(w)}V_s\|_{L_p}
\le C_{\mathrm{prod}}e^{-c_{\mathrm{prod}}w a(k-s)/p}
   \|V_s\|_{L_{2p}}
$$

is a conditional/adapted-vector extension of the same product-stability
argument; it is not literally stated as Levin Proposition 9.

### 4. Levin Appendix B.2 Proposition 5 and Eq. (55)

Let

$$
Y=(z,J^{(0)},J^{(1)},J^{(2)}),\qquad
\widetilde Y=(\widetilde z,\widetilde J^{(0)},\widetilde J^{(1)},
\widetilde J^{(2)}),
$$

and define the cost from Levin Eq. (49):

$$
\begin{aligned}
c_{J,2}(Y,\widetilde Y)
&=\|J^{(0)}-\widetilde J^{(0)}\|
 +\|J^{(1)}-\widetilde J^{(1)}\|
 +\|J^{(2)}-\widetilde J^{(2)}\|\\
&\quad+
\left(
\|J^{(0)}\|+\|\widetilde J^{(0)}\|
+\|J^{(1)}\|+\|\widetilde J^{(1)}\|
+\|J^{(2)}\|+\|\widetilde J^{(2)}\|
+\sqrt{\alpha a}\|\varepsilon\|_\infty
\right)\mathbf 1_{\{z\ne \widetilde z\}} .
\end{aligned}
$$

Levin Proposition 5 says:

> Under A1, A2 and UGE 1, for any $n\ge1$, $p\ge1$, and
> $\alpha\in(0,\alpha_\infty\wedge(ap)^{-1}\log\rho^{-1})$,
> $$
> W_{c_{J,2},p}^{1/p}
> \left(\delta_y Q_{J^{(2)}}^n,\delta_{\widetilde y}Q_{J^{(2)}}^n\right)
> \le
> c_{W,3}^{(2)}p^{7/2}t_{\mathrm{mix}}^{5/2}
> \rho_{1,\alpha}^{n/p}
> \log^{3/2}(1/(\alpha a))\,c_{J,2}(y,\widetilde y),
> $$
> where $\rho_{1,\alpha}=e^{-\alpha a/12}$ and
> $c_{W,3}^{(2)}=c_{W,1}^{(2)}+c_{W,2}^{(2)}$ is defined in Levin Eq. (55).

The componentwise $L_p$ estimates in the diploma follow by using this coupling
and projecting the cost onto the coordinates $J^{(0)},J^{(1)},J^{(2)}$.

### 5. Levin Corollary 4: invariant law for the depth-two augmented chain

Levin Corollary 4 states:

> Under A1, A2 and UGE 1, if $\alpha\in(0,\alpha_\infty^{(b)})$, then the
> Markov chain
> $$
> Y_t=(Z_{t+1},J_t^{(0,\alpha)},J_t^{(1,\alpha)},J_t^{(2,\alpha)})
> $$
> has a unique invariant distribution $\Pi_{J^{(2)},\alpha}$.

This corollary covers the $J$-coordinates only. It does not include
$H^{(2,\alpha)}$ in the state.

### 6. Levin Proposition 9: one-trajectory $H^{(2)}$ bound

Levin Appendix D.1 Proposition 9 states:

> Under A1, A2 and UGE 1, for $2\le p\le q/2$, admissible $\alpha$, and any
> initial distribution $\xi$ of the base chain,
> $$
> \|H_n^{(2,\alpha)}\|_{L_p(\xi)}
> \le
> D_H d^{1/q}t_{\mathrm{mix}}^{5/2}p^{7/2}
> \alpha^{3/2}\log^{3/2}(1/(\alpha a)),
> $$
> where $D_H=384\kappa_Q^{1/2}C_Aa^{-1}e^2D_J$ and $D_J$ is the constant from
> Levin Proposition 8.

The proof uses

$$
H_n^{(2,\alpha)}
=-\alpha\sum_{\ell=1}^n
\Gamma_{\ell+1:n}^{(\alpha)}\widetilde A(Z_\ell)J_{\ell-1}^{(2,\alpha)}.
$$

### 7. Finite-past Cauchy construction for stationary $H^{(2)}$

This is not a named Levin theorem. The working statement needed in the thesis
is the following consequence of Levin Proposition 5, Levin Proposition 9, and
the product-stability estimate:

> On a two-sided stationary copy of the Markov chain, start the recursions for
> $J^{(0,w)},J^{(1,w)},J^{(2,w)},H^{(2,w)}$ from zero at time $-m$ and evaluate
> at time $0$. Then for every admissible $p$ the sequence
> $$
> (J_{0,m}^{(0,w)},J_{0,m}^{(1,w)},J_{0,m}^{(2,w)},H_{0,m}^{(2,w)})
> $$
> is Cauchy in $L_p$ as $m\to\infty$. Its limit defines the stationary
> full augmented state.

A useful explicit bound to state locally is:

$$
\|H_{0,m}^{(2,w)}-H_{0,m'}^{(2,w)}\|_{L_p}
\le C p^{7/2}t_{\mathrm{mix}}^{5/2}
w^{3/2}\log^{3/2}(1/(w a))e^{-c w a\min(m,m')/p},
$$

up to harmless changes in the polynomial/logarithmic prefactor. This is a
derived statement, not a verbatim proposition from Levin.

### 8. Conditional product stability at a random coupling time

There is no separate external statement in Levin with a random coupling time
$T$. The statement used in the thesis is a local corollary of the conditional
proof of product stability:

> If $T$ is an exact-coupling time with
> $\mathbb P(T>r)\le C_Te^{-c_Tr}$ and the deterministic-start product
> estimate of item 3 holds conditionally on the past, then for adapted
> $(V_s)$ with $\sup_s\|V_s\|_{L_{2p}}\le B$,
> $$
> \|\Gamma_{T+1:k}^{(w)}V_T\mathbf 1_{\{T\le k\}}\|_{L_p}
> \le C\frac{p}{w a}B e^{-c w a k/p}.
> $$
> Also, if $\|U_\ell\|_{L_{2p}}\le Be^{-c_0w a\ell/p}$, then
> $$
> w\sum_{\ell=1}^k
> \|\Gamma_{\ell+1:k}^{(w)}U_\ell\|_{L_p}
> \le C\frac{p}{a}B e^{-c w a k/p}.
> $$

This should be labelled as a local lemma, not as a direct citation.

### 9. Full-state startup contraction extension

No Levin statement directly contracts the full state including $H^{(2)}$.
The exact external content is:

1. Levin Proposition 5 contracts
   $(Z,J^{(0)},J^{(1)},J^{(2)})$ in the cost $c_{J,2}$.
2. Levin Proposition 8 gives a one-trajectory moment bound for $J^{(2)}$.
3. Levin Proposition 9 gives a one-trajectory moment bound for $H^{(2)}$.
4. The proof of Levin Proposition 9 uses product stability for the random
   products.

The thesis statement is therefore a local extension:

> The finite-start remainder
> $R_{k,\mathrm{fin}}^{(w)}=J_{k,\mathrm{fin}}^{(1,w)}
> +J_{k,\mathrm{fin}}^{(2,w)}+H_{k,\mathrm{fin}}^{(2,w)}$
> and a stationary augmented copy can be coupled so that
> $$
> \|R_{k,\mathrm{fin}}^{(w)}-R_{k,\mathrm{aug}}^{(w)}\|_{L_p}
> \le A_{\mathrm{st}}(p,q,w)e^{-c_{\mathrm{st}}w a k/p}.
> $$

This is not a direct Levin corollary; it must be proved in the diploma or
declared as an additional imported/technical input.

### 10. Levin Proposition 8 for the initial cost

Levin Proposition 8 states:

> Under A1, A2 and UGE 1, for $2\le p<\infty$, admissible $\alpha$, and every
> initial distribution $\xi$ of the base chain,
> $$
> \|J_n^{(2,\alpha)}\|_{L_p(\xi)}
> \le
> D_J t_{\mathrm{mix}}^{5/2}p^{7/2}
> \alpha^{3/2}\log^{3/2}(1/(\alpha a)).
> $$

Together with the elementary bounds for $J^{(0)}$ and $J^{(1)}$, this controls
the invariant initial cost in $c_{J,2}$ after passing to the finite-past
stationary limit.

### 11. Generic step-size restrictions in the burned-in misadjustment theorem

The phrase "step-size restrictions of the Levin depth-two and
startup-contraction bounds" should expand to:

$$
2\alpha\le \alpha_*(q,t_{\mathrm{mix}}),\qquad
2\alpha\le \alpha_{\mathrm{st}}(p),\qquad
2\alpha\le\alpha_{\mathrm{inv}},\qquad
\alpha a\le 1/4,
$$

with $p\le q/2$. Here $\alpha_*$ collects the stationary Levin inputs in items
1, 4, 6, 10, and $\alpha_{\mathrm{st}}$ collects the startup/product inputs in
items 2, 3, 8, 9.

### 12. Final theorem imported inputs and thresholds

The final theorem should explicitly assume:

1. Assumptions 1--3 of the thesis: UGE, boundedness/Hurwitz/Lyapunov
   contraction, bounded centered noise.
2. The Levin stationary depth-two inputs: Proposition 2, Corollary 6,
   Proposition 8, Proposition 9, and Proposition 5/Corollary 4 for the
   stationary augmented chain.
3. The Samsonov martingale normal-approximation input: Lemma 21, plus the
   Poisson/martingale decomposition used to control the bracket.
4. The local startup extension for the full augmented state including
   $H^{(2)}$.
5. Non-degeneracy $\sigma^2(u)>0$.

If the theorem is meant to be deterministic-start, it should also say whether
the constants are uniform over the initial law of $Z_0$ or $Z_1$. Levin
Proposition 8/9 and Samsonov Theorem 1 are stated for arbitrary initial
distribution of the base chain, so this uniformity is plausible if all
intermediate startup bounds keep the same property.

### 13. Samsonov Proposition 9: depth-one/depth-two perturbation bound

Samsonov et al. Proposition 9 states, in their decreasing-step notation:

> Under A1, A2 and A3, for any $p\ge2$, any initial distribution $\xi$, and
> every $k\ge1$,
> $$
> \|u^\top J_k^{(1)}\|_{L_p}
> \lesssim
> D_4^{(M)}t_{\mathrm{mix}}p^2\alpha_k\log(1/\alpha_k),
> $$
> and
> $$
> \|u^\top H_k^{(1)}\|_{L_p}
> \lesssim
> D_5^{(M)}t_{\mathrm{mix}}p^2\alpha_k\log(1/\alpha_k).
> $$

It is based on the perturbation expansion

$$
H_k^{(0)}=\sum_{\ell=1}^L J_k^{(\ell)}+H_k^{(L)},
$$

with recursions

$$
J_k^{(\ell)}
=(I-\alpha_k\overline A)J_{k-1}^{(\ell)}
-\alpha_k\widetilde A(Z_k)J_{k-1}^{(\ell-1)},
$$

$$
H_k^{(\ell)}
=(I-\alpha_k A(Z_k))H_{k-1}^{(\ell)}
-\alpha_k\widetilde A(Z_k)J_{k-1}^{(\ell)}.
$$

The constant-step thesis uses the decomposition idea, not this proposition
verbatim.

### 14. Samsonov Step (S8)

The local PDF does not contain a literal named "Step (S8)". The statement
apparently meant by this reference is the Samsonov perturbation step:

> Replace the first misadjustment
> $$
> H_k^{(0)}
> =-\sum_{j=1}^k\alpha_j\Gamma_{j+1:k}\widetilde A(Z_j)J_{j-1}^{(0)}
> $$
> by the depth expansion
> $H_k^{(0)}=J_k^{(1)}+H_k^{(1)}$ for $L=1$, and then control
> $J_k^{(1)}$ and $H_k^{(1)}$ by Proposition 9.

Thus the diploma should not cite "Step (S8)" unless that step is defined
locally.

### 15. Levin Proposition 2 in the exploratory depth-one subsection

Levin Proposition 2 states:

> Under A1, A2 and UGE 1, for $\alpha\in(0,\alpha_\infty^{(b)})$,
> $$
> \lim_{n\to\infty}\mathbb E[J_n^{(1,\alpha)}]
> =
> \mathbb E[J_\infty^{(1,\alpha)}]
> =
> \alpha\Delta+R(\alpha),
> $$
> where
> $$
> \Delta=\overline A^{-1}
> \sum_{k=1}^{\infty}
> \mathbb E[\widetilde A(Z_{\infty+k})\varepsilon(Z_\infty)]
> $$
> and
> $$
> \|R(\alpha)\|
> \le
> 12\|\overline A^{-1}\|C_A^2t_{\mathrm{mix}}^2
> \alpha^2\|\varepsilon\|_\infty.
> $$

### 16. Poisson covariance identity / Markov-chain CLT identity

For a centered bounded function $f$ and a uniformly geometrically ergodic
chain, the Poisson solution

$$
\widehat f(z)=\sum_{k=0}^\infty P^k f(z)
$$

satisfies $\widehat f-P\widehat f=f$. Define the martingale increment

$$
\zeta_k=\widehat f(Z_k)-P\widehat f(Z_{k-1})
$$

and its conditional covariance

$$
\mathcal V_f(z)
=P(\widehat f\widehat f^\top)(z)-P\widehat f(z)P\widehat f(z)^\top.
$$

Then

$$
\pi(\mathcal V_f)
=
\mathbb E_\pi[f(Z_0)f(Z_0)^\top]
+\sum_{\ell\ge1}\mathbb E_\pi[f(Z_0)f(Z_\ell)^\top]
+\sum_{\ell\ge1}\mathbb E_\pi[f(Z_\ell)f(Z_0)^\top].
$$

For $f=\varepsilon$, this is $\Sigma_\varepsilon^{(M)}$. The associated Markov
chain CLT gives

$$
n^{-1/2}\sum_{k=1}^n f(Z_k)
\Rightarrow N(0,\pi(\mathcal V_f)).
$$

### 17. Geometric forgetting of constant-step algorithms

The form needed in the thesis is:

> For sufficiently small constant step size $\alpha$, the joint data-iterate
> chain has a unique invariant law and the dependence on the initial condition
> decays geometrically, typically
> $$
> W_2^2(\mathcal L(Z_t,\theta_t),\bar\mu)
> \le C(1-c\alpha)^t
> $$
> after the Markov-chain mixing transient, with constants independent of $t$.

This is the "geometric forgetting" statement cited in the introduction. The
same form appears explicitly in Huo et al. Theorem A.3/HCX23 as
$W_2^2(\mathcal L(x_t,\theta_t),\bar\mu)=O((1-c\alpha)^t)$ for
$t\ge \tau_\alpha$.

### 18. Huo higher-order power-series bias expansion

Huo et al. Theorem A.4 states:

> Under Assumptions 1--2 and sufficiently small $\alpha$, the asymptotic bias
> has the infinite expansion
> $$
> \mathbb E[\theta_\infty^{(\alpha)}]-\theta^*
> =
> \sum_{i=1}^{\infty}\alpha^i B^{(i)},
> $$
> where the vectors $B^{(i)}$ do not depend on $\alpha$.

Their RR coefficients $h_m$ are chosen by

$$
\sum_m h_m=1,\qquad
\sum_m h_m\alpha_m^\ell=0,\quad \ell=1,\dots,M-1,
$$

so the first $M-1$ powers are cancelled.

### 19. Levin residual RR bias order

Levin Corollary 1 gives

$$
\Pi_\alpha(\theta_0)=\theta^*+\alpha\Delta+O(\alpha^{3/2}).
$$

Therefore for the two-level RR combination,

$$
2\Pi_\alpha(\theta_0)-\Pi_{2\alpha}(\theta_0)-\theta^*
=
2O(\alpha^{3/2})-O((2\alpha)^{3/2})
=O(\alpha^{3/2}),
$$

because the linear terms $2\alpha\Delta-2\alpha\Delta$ cancel.

### 20. Levin high-order moment bounds for PR-averaged RR

Levin Theorem 2 states:

> Under A1, A2 and UGE 1, for $2\le p<\infty$, $n\ge t_{\mathrm{mix}}$,
> admissible $\alpha$, and any initial law $\xi$ of the base chain,
> $$
> \|\overline A(\overline\theta_n^{(\alpha,\mathrm{RR})}-\theta^*)\|_{L_p}
> \le
> 2C_{\mathrm{Rm},1}
> \{\operatorname{Tr}\Sigma_\varepsilon^{(M)}\}^{1/2}
> p^{1/2}n^{-1/2}
> +R_{n,p,\alpha}^{(\mathrm{fl})}
> +R_{n,p,\alpha}^{(\mathrm{tr})}
> \|\theta_0-\theta^*\|e^{-\alpha an/24}.
> $$

The fluctuation remainder contains terms of order
$(\alpha n)^{-1/2}n^{-1/2}$, $\alpha^{1/2}n^{-1/2}$, and
$\alpha^{3/2}\log^{3/2}(1/(\alpha a))$ up to polynomial factors in $p$ and
$t_{\mathrm{mix}}$.

### 21. Samsonov Berry--Esseen and bootstrap inference for standard PR

Samsonov Theorem 1 states:

> Under A1--A3, for every unit vector $u$, every $\theta_0$, and every initial
> law $\xi$,
> $$
> d_K\left(
> \frac{\sqrt n\,u^\top(\overline\theta_n-\theta^*)}{\sigma_n(u)},
> N(0,1)
> \right)
> \le B_n,
> $$
> where
> $$
> B_n=
> C_{K,1}\frac{\log^{3/4}n}{n^{1/4}}
> +C_{K,2}\frac{\log n}{n^{1/2}}
> +\frac{C_1^D\|\theta_0-\theta^*\|+C_2^D}{\sqrt n}
> +C_3^D\frac{(\log n)^2}{n^{\gamma-1/2}}
> +C_4^D\frac{(\log n)^{5/2}}{n^{\gamma-1/2}}.
> $$

Their Corollary 1 replaces $\sigma_n(u)$ by the asymptotic
$\sigma(u)$ at additional cost $C_\infty n^{\gamma-1}$.

For bootstrap, Samsonov Theorem 2 states, for
$b_n=\lceil n^{4/5}\rceil$ and
$\alpha_k=c_0/(k_0+k)^{3/5}$:

$$
\sup_x\left|
\mathbb P(\sqrt n(\overline\theta_n-\theta^*)^\top u\le x)
-\mathbb P_b(\overline\theta_{n,b_n}(u)\le x)
\right|
\lesssim_{\log n} n^{-1/10}
$$

with probability at least $1-1/n$.

### 22. Markov-chain CLT theorem

The Douc et al. CLT form used here is:

> If $(Z_k)$ is stationary and geometrically ergodic and $f$ is centered with
> a bounded Poisson solution, then
> $$
> n^{-1/2}\sum_{k=1}^n f(Z_k)
> \Rightarrow N(0,\Sigma_f),
> $$
> where
> $$
> \Sigma_f
> =
> \mathbb E_\pi[f(Z_0)f(Z_0)^\top]
> +\sum_{\ell\ge1}\mathbb E_\pi[f(Z_0)f(Z_\ell)^\top]
> +\sum_{\ell\ge1}\mathbb E_\pi[f(Z_\ell)f(Z_0)^\top].
> $$

For reversible/scalar notation this is often written with the middle two sums
as $2\sum_{\ell\ge1}\mathbb E[f(Z_0)f(Z_\ell)]$.

### 23. Levin invariant distribution of the joint LSA-data chain

Levin Theorem 1 states:

> Assume A1, A2 and UGE 1. For $2\le p\le q$ and admissible
> $\alpha$, the Markov kernel $\overline P_\alpha$ of
> $(\theta_k^{(\alpha)},Z_{k+1})$ has a unique invariant distribution
> $\Pi_\alpha$, and
> $$
> \Pi_\alpha(\|\theta_0-\theta^*\|)<\infty.
> $$

The proof uses exact coupling of the base chains and Wasserstein contraction
with cost

$$
c_0((\theta,z),(\theta',z'))
=
(\|\theta-\theta'\|+\mathbf 1_{\{z\ne z'\}})
(1+\|\theta-\theta^*\|+\|\theta'-\theta^*\|).
$$

### A. Already-explicit imported inputs mentioned at the end of the note

These were marked as "already explicit enough" in the original checklist, but
for completeness the external statements are:

**Levin Lemma 11 / Markov concentration.** Under UGE 1, for centered bounded
time-inhomogeneous functions $g_i$ with $\|g_i\|_\infty\le c_i$,

$$
\mathbb P_\xi\left(
\left\|\sum_{i=1}^n g_i(Z_i)\right\|\ge t
\right)
\le
2\exp\left(
-\frac{t^2}{2u_n^2}
\right),
\qquad
u_n=8\left(\sum_{i=1}^n c_i^2\right)^{1/2}t_{\mathrm{mix}}^{1/2}.
$$

This is stated for arbitrary initial law $\xi$.

**Samsonov Lemma 21 / Bolthausen--Fan martingale bound.** For bounded scalar
martingale differences $X_i$, $S_n=\sum_iX_i$, deterministic scale $s_n>0$,
and conditional variances $\sigma_i^2$, if $\|X_i\|_\infty\le\kappa$, then

$$
d_K(S_n/s_n,N(0,1))
\le
\frac{L(\kappa)(2n+1)\log(2n+1)}{s_n^3}
+C_1\sqrt p\,s_n^{-2p/(2p+1)}
\left(\mathbb E\left|\sum_i\sigma_i^2-s_n^2\right|^p\right)^{1/(2p+1)}
+C_2p\,s_n^{-2p/(2p+1)}\kappa^{2p/(2p+1)}.
$$

**Levin Corollary 6 / centered bilinear bound.** For
$\overline\psi_\alpha(j,z)=\widetilde A(z)j-
\Pi_{J,\alpha}[\widetilde A(Z_1)J_0^{(0,\alpha)}]$,

$$
\left\|
\sum_{t=0}^{r-1}
\overline\psi_\alpha(J_t^{(0,\alpha)},Z_{t+1})
\right\|_{L_p}
\le
c_{W,1}^{(2)}p^{3/2}(\alpha r)^{1/2}
+c_{W,2}^{(2)}p^3\alpha^{-1/2}\log^{1/p}(1/(\alpha a)),
$$

under A1, A2, UGE 1 and the Levin small-step condition.

**Future-centered bilinear estimate used in `src/last_iterate.typ`.** The
external Samsonov/Durmus block-coupling input used there is the following
scalar form: for deterministic centered future kernels $g_{k,\ell}$ with
$\|g_{k,\ell}\|_\infty\le\beta_{k,\ell}$ and adapted bounded vectors $\xi_k$,

$$
\left\|
\sum_{k=1}^{n-1}
\left[
\sum_{\ell=1}^{n-k}g_{k,\ell}(Z_{k+\ell})^\top\xi_k
-\mathbb E\left(
\sum_{\ell=1}^{n-k}g_{k,\ell}(Z_{k+\ell})^\top\xi_k
\mid \mathcal F_k
\right)
\right]
\right\|_{L_p}
\le
C p^{3/2}t_{\mathrm{mix}}^{1/2}
\sup_k\|\xi_k\|_\infty
\left(\sum_{k,\ell}\beta_{k,\ell}^2\right)^{1/2}.
$$

This is the form extracted from the block-decomposition/Berbee-coupling
argument behind Samsonov Appendix D.2, Proposition 9 and Lemma 11.

## Proof-Critical Places

### 1. Hidden Levin admissibility threshold $\alpha_*(q,t_{\mathrm{mix}})$

Где: `src/pr_weights.typ:42`.

Текст говорит, что существует ceiling $\alpha_*(q,t_{\mathrm{mix}})$, при
котором работают stationary bias, centered bilinear и depth-two moment
estimates Levin et al. (2025). Рабочие оценки потом выписаны, но сам
admissibility statement не раскрыт.

Что не выписано:

- точные условия из Levin et al.;
- от чего реально зависит $\alpha_*(q,t_{\mathrm{mix}})$;
- является ли threshold общим для Proposition 2, Corollary 6, Propositions
  8--9 или это минимум нескольких thresholds;
- какие constants разрешено считать fixed problem constants.

### 2. Hidden startup threshold $\alpha_{\mathrm{st}}(p)$

Где: `src/pr_weights.typ:54`.

Текст вводит $\alpha_{\mathrm{st}}(p)$ как threshold, под которым работают
product-stability и full-state startup contractions, но сам внешний statement
не выписан в этом месте. Позже есть working lemmas, но состав threshold и
условия остаются скрыты.

Что не выписано:

- точная формулировка startup/product stability input из Levin;
- почему моментный порядок именно $2p$;
- какие assumptions входят в "product-stability ceiling";
- зависит ли $\alpha_{\mathrm{st}}(p)$ от $q,d,t_{\mathrm{mix}},a,C_A$.

### 3. Random-product stability from Levin Appendix D.1, Proposition 9

Где: `src/burn_in_transfer.typ:318` and `src/burn_in_transfer.typ:323`.

В тексте есть локальная lemma `Imported random-product stability` с формулой,
но перед ней стоит ссылка на "the stability and bounded-noise assumptions used
in Levin et al. (2025, Appendix D.1, Proposition 9)".

Что не выписано:

- точный statement Levin Proposition 9, из которого берется product stability;
- какие stability/bounded-noise assumptions нужны именно для
  $\Gamma_{s+1:k}^{(w)}V_s$;
- почему estimate можно применять к произвольному
  $\mathcal F_s$-measurable vector $V_s$;
- точная зависимость $C_{\mathrm{prod}},c_{\mathrm{prod}}$.

### 4. Levin Appendix B.2 Proposition 5 and Eq. (55)

Где: `src/burn_in_transfer.typ:652` and `src/burn_in_transfer.typ:660`.

Cost $c_{J,2}^{(w)}$ выписан, и выписана componentwise оценка для
$J^{(0)},J^{(1)},J^{(2)}$. Но сам Wasserstein contraction theorem из Levin
Appendix B.2, Proposition 5, и constants from Eq. (55) не выписаны.

Что не выписано:

- полный Wasserstein contraction statement;
- метрика/пространство, где живет augmented chain;
- exact Proposition-5 step-size restriction;
- constants from Levin Eq. (55);
- как из Wasserstein contraction выводится componentwise $L_p$ coupling bound.

### 5. Levin Corollary 4 invariant law for depth-two augmented chain

Где: `src/burn_in_transfer.typ:672`.

Текст пишет, что Corollary 4 gives invariant law
$\Pi_{J,2,w}$ for $Y_k^{(w)}$, но statement не выписан.

Что не выписано:

- условия существования и единственности invariant law;
- какой именно augmented state покрывает Corollary 4;
- moment bounds under the invariant law.

### 6. Levin Proposition 9 for the $H^{(2)}$ one-trajectory bound

Где: `src/burn_in_transfer.typ:675`.

Текст говорит, что Levin Appendix D.1 Proposition 9 proves the required
one-trajectory moment bound for $H^{(2,w)}$. Representation for $H^{(2,w)}$
выписана, а позже используется bound $B_H(p,q,w)$, но внешний proposition
как самостоятельный input не сформулирован.

Что не выписано:

- точная one-trajectory estimate for $H^{(2,w)}$;
- applies it to finite-start and finite-past stationary copies uniformly;
- какие moment orders и $q$ нужны;
- точный logarithmic factor and constants.

### 7. Finite-past Cauchy construction for stationary $H^{(2)}$

Где: `src/burn_in_transfer.typ:683` and `src/burn_in_transfer.typ:694`.

Текст утверждает, что Levin Proposition 9 plus random-product stability make
the finite-past sequence $H_{0,m}^{(2,w)}$ Cauchy in $L_p$.

Что не выписано:

- сама Cauchy estimate for
  $\|H_{0,m}^{(2,w)}-H_{0,m'}^{(2,w)}\|_{L_p}$;
- uniform-in-truncation bound needed to pass to the stationary limit;
- почему предел имеет invariant law для full augmented state including
  $H^{(2)}$.

### 8. Conditional-on-past product stability at random coupling time

Где: `src/burn_in_transfer.typ:725`.

В proof of `Conditional product stability at a coupling time` сказано, что
the proof of Levin's product-stability estimate is conditional on the past,
hence it may be applied on each event $T=s$.

Что не выписано:

- внешний conditional version of product stability;
- measurability assumptions at random time $T$;
- treatment of empty products at $T=k$;
- proof that constants are unchanged under conditioning on $T=s$.

Это лучше либо доказать полностью локально, либо вынести как imported input.

### 9. Full-state startup contraction extension

Где: `src/burn_in_transfer.typ:752` and `src/burn_in_transfer.typ:774`.

Лемма `Full-state startup contraction for the depth-two augmented remainder`
сама выписана, но ее proof relies on several external statements that are
only named:

- Levin Proposition-5 coupling for the $J$ coordinates;
- Levin Proposition 9 for $H^{(2)}$;
- Levin Proposition 8 for the invariant $J^{(2)}$ initial cost;
- random-product stability inside Levin Appendix D.1.

Что не выписано:

- full imported statements behind these references;
- exact moment/cost assumptions;
- exact dependence of $A_{\mathrm{st}}(p,q,w)$ on imported constants;
- a self-contained derivation of the $H^{(2)}$ part.

Это главное proof-critical место.

### 10. Levin Proposition 8 used in initial cost estimate

Где: `src/burn_in_transfer.typ:810`.

Текст говорит, что elementary estimates and Levin Proposition 8 for
$J^{(2,w)}$ give the initial cost bound.

Что не выписано:

- точный Proposition 8 statement being used for invariant/finite-past
  $J^{(2,w)}$;
- why it controls the cost $c_{J,2}^{(w)}(Y_0^{fin},Y_0^{aug})$;
- uniformity in the stationary finite-past limit.

### 11. Generic "step-size restrictions of Levin depth-two and startup-contraction bounds"

Где: `src/burn_in_transfer.typ:993`.

Theorem `Burned-in PR-averaged RR misadjustment bound` assumes these
restrictions by name, but does not restate them.

Что не выписано:

- exact restrictions;
- whether they are exactly $2\alpha\le\alpha_*(q,t_{\mathrm{mix}})$ and
  $2\alpha\le\alpha_{\mathrm{st}}(p)$, or include more constraints;
- all conditions needed for both $w=\alpha$ and $w=2\alpha$.

### 12. Final theorem imported inputs and thresholds

Где: `src/burn_in_transfer.typ:1335` and `src/burn_in_transfer.typ:1359`.

Final theorem assumes "the imported Levin and Samsonov inputs summarized in
@sec:imported-inputs" and defines

$$
\alpha_{\mathrm{adm}}(p,q)
  =
  \min\{\alpha_\infty,\alpha_{\mathrm{inv}},(2a)^{-1},
          \alpha_*(q,t_{\mathrm{mix}}),\alpha_{\mathrm{st}}(p)\}.
$$

Что не выписано:

- complete list of imported assumptions in the theorem statement;
- exact content of $\alpha_*(q,t_{\mathrm{mix}})$;
- exact content of $\alpha_{\mathrm{st}}(p)$;
- whether the theorem is uniform over initial law of $Z_0/Z_1$.

## Partially Written / Lower Priority Places

### 13. Samsonov Proposition 9 depth-one decomposition

Где: `src/pr_weights.typ:90` and `src/last_iterate.typ:7`.

The decomposition formula is mostly written locally, so this is not a major
gap. But the external Proposition 9 itself is not stated.

Что не выписано:

- exact assumptions of Samsonov Proposition 9;
- full Step S8/decomposition scheme;
- relation between Samsonov notation and current $J,H,R$ notation.

### 14. Samsonov Step (S8)

Где: `src/last_iterate.typ:329`.

Текст says "after Step (S8) of the Samsonov scheme", but Step (S8) is not
described.

Что не выписано:

- what Step (S8) states;
- how it produces the displayed depth-one misadjustment term;
- assumptions under which it is valid.

This subsection is exploratory and not used in the final assembly, so priority
is lower.

### 15. Levin Proposition 2 in exploratory depth-one subsection

Где: `src/last_iterate.typ:352`.

Only the shorthand

$$
\mathbb E_\pi[J_\infty^{(1,\alpha)}]=\alpha\Delta+O(\alpha^2)
$$

is given. The fully explicit version appears later in `src/pr_weights.typ`,
so this is mostly duplication/exposition.

Что не выписано здесь:

- the explicit remainder constant;
- assumptions;
- definition of $\Delta$.

### 16. Poisson covariance identity / Markov-chain CLT identity

Где: `src/pr_weights.typ:648`.

The identity

$$
\pi(\mathcal V_\epsilon)=\Sigma_\epsilon^{(M)}
$$

is written, but the external statement from Samsonov Eq. (10) / Douc et al.
Theorem 21.2.5 is not.

Что не выписано:

- exact Poisson-equation identity;
- conditions under which the long-run covariance equals the mean conditional
  covariance of the Poisson martingale increment;
- relation to the Markov-chain CLT theorem.

The calculation is standard and probably acceptable, but it is an external
statement.

## Background / Introduction Claims

These are not proof blockers, but they are external claims whose statements are
not written out.

### 17. Geometric forgetting of constant-stepsize algorithms

Где: `src/introduction.typ:20`.

Dieuleveut--Durmus--Bach (2020) is cited for geometrically fast forgetting of
the initial condition. No theorem statement or rate is written.

### 18. Huo higher-order power-series bias expansion

Где: `src/introduction.typ:28`.

The text says Huo--Chen--Xie (2023) gives higher-order bias expansions in
integer powers of $\alpha$, but does not state assumptions or expansion form.

### 19. Levin residual RR bias order

Где: `src/introduction.typ:38`.

The text says the RR residual bias is $O(\alpha^{3/2})$ or higher by Levin et
al. (2025). The exact theorem and conditions are not stated.

### 20. Levin high-order moment bounds for PR-averaged RR

Где: `src/introduction.typ:47`.

The text says high-order moment bounds show the leading error scales as
$\sqrt{\operatorname{Tr}\Sigma_\epsilon^{(M)}}n^{-1/2}$. The exact theorem,
norm, constants, and assumptions are not written.

### 21. Samsonov Berry--Esseen and bootstrap inference for standard PR

Где: `src/introduction.typ:53`.

The text says these results have been obtained, but gives no statement,
rate, or assumptions.

### 22. Markov-chain CLT theorem

Где: `src/introduction.typ:174`.

Douc et al. (2018, Theorem 21.2.10) is cited for the Markov-chain CLT. The
covariance formula is written, but the theorem statement and assumptions are
not.

### 23. Invariant distribution of the joint LSA-data chain

Где: `src/introduction.typ:185`.

Levin et al. (2025) is cited for existence of invariant distribution
$\Pi_\alpha$ for the joint process. The exact proposition and small-step
condition are not stated.

## Already Explicit Enough

I would not list the following as missing, because the working expressions are
already displayed:

- `src/pr_weights.typ:14`: Markov concentration input from Levin Lemma 11.
- `src/pr_weights.typ:26`: Bolthausen--Fan / Samsonov Lemma 21 input.
- `src/pr_weights.typ:992`: Levin Proposition 2 working form.
- `src/pr_weights.typ:1005`: Levin Corollary 6 working form.
- `src/pr_weights.typ:1020`: Levin Propositions 8--9 working forms.
- `src/last_iterate.typ:93`: imported future-centered bilinear estimate.
- `src/zeroth_order_rr.typ:130`: scalar concentration input from Levin Lemma
  11.

## Recommended Fix Order

1. Expand the startup block first: items 3--10 are the only serious
   proof-critical missing external statements.
2. Add a compact table defining $\alpha_*(q,t_{\mathrm{mix}})$ and
   $\alpha_{\mathrm{st}}(p)$ as minima of named imported thresholds.
3. In the final theorem, say explicitly whether the result is uniform over the
   initial law of the Markov chain.
4. For introduction/background citations, either leave as prose or move exact
   theorem statements to a short literature appendix; they are not central
   proof blockers.

# Burn-in transition readiness

## Question

Проверить, готова ли текущая stationary часть к переходу к burn-in, и решить,
лучше ли делать burn-in отдельной главой.

## Short answer

Да, переходить можно, но burn-in надо делать отдельной главой. Текущая
Theorem 3 теперь честно является stationary $n_0=0$ theorem for the
augmented-chain statistic

$$
S_{n,\mathrm{stat}}^{\mathrm{RR}}(u),
$$

а не theorem для deterministic-start average. Это достаточно чистая точка
разделения: stationary chapter закрывает reference theorem, новая chapter
должна доказать transfer.

## What is ready

- `typst compile main.typ` проходит.
- Theorem-like references now compile through real Typst labels.
- В PDF не найдено пустых ссылок вида `Lemma )`, `Theorem )`, `Corollary )`,
  `Eq. )`, `equation )`.
- В `src/pr_weights.typ` Theorem 3 называется
  `Stationary n_0 = 0 PR-averaged RR Berry--Esseen bound`.
- В конце главы явно сказано, что это statement for
  $S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)$, not the deterministic-start RR
  average itself.
- Burn-in уже не притворяется малой правкой к old weights: текст явно требует
  burned-in weights $Q_{\ell,n_0}^{(\alpha)}$ and corresponding Poisson,
  variance-comparison, and misadjustment bounds.

## Progress

Closed on 2026-05-17:

- [x] deterministic transient after burn-in. Added
  `src/burn_in_transfer.typ`, lemma
  `lem:burn-deterministic-transient` and corollary `cor:burn-log-transient`.
  The bound is
  $$
  |D_{\mathrm{tr},n,n_0}^{\mathrm{RR}}(u)|
  \le
  \frac{C\|u\|\|\theta_0-\theta^\star\|}
       {\alpha a\sqrt{m}}
  (1-\alpha a)^{n_0/2},
  \qquad m=n-n_0.
  $$
  Under $n_0\ge 2\beta(\alpha a)^{-1}\log n$, this becomes
  $C(\alpha a\sqrt m)^{-1}n^{-\beta}$, hence it is negligible at the
  balanced scale $\alpha=c n^{-1/2}$ when $m\asymp n$.

- [x] burned-in RR weight bounds for
  $Q_{\ell;n_0,n}^{\mathrm{RR}}$. Added lemma
  `lem:burn-rr-weight-bounds`. The key split is:
  $$
  \sum_{\ell<n_0}\|Q_{\ell;n_0,n}^{\mathrm{RR}}\|^2
  +
  \sum_{\ell\ge n_0}
  \|Q_{\ell;n_0,n}^{\mathrm{RR}}-\bar A^{-1}\|^2
  \le
  \frac{C}{\alpha a},
  $$
  and
  $$
  \sum_{\ell=1}^{n-2}
  \|Q_{\ell+1;n_0,n}^{\mathrm{RR}}
    -Q_{\ell;n_0,n}^{\mathrm{RR}}\|
  \le
  \frac{C}{a^2}.
  $$
  This avoids the incorrect comparison
  $Q_{\ell;n_0,n}^{\mathrm{RR}}\approx\bar A^{-1}$ for pre-burn-in
  indices $\ell<n_0$.

- [x] Poisson decomposition and predictable-variance comparison for the
  burned-in depth-zero sum. Added:
  `lem:burn-variance-comparison`, `lem:burn-poisson-decomp`,
  `lem:burn-bracket-conc`, and `cor:burn-bracket-asymp`.
  The variance proxy is
  $$
  \Sigma_{n,n_0}^{\mathrm{bRR}}
  =
  \frac1m\sum_{\ell=2}^{n-1}
  Q_{\ell;n_0,n}^{\mathrm{RR}}
  \Sigma_\epsilon^{(M)}
  (Q_{\ell;n_0,n}^{\mathrm{RR}})^\top,
  \qquad m=n-n_0,
  $$
  with
  $$
  \|\Sigma_{n,n_0}^{\mathrm{bRR}}-\Sigma_\infty\|
  \le
  \frac{C}{m\alpha a}.
  $$
  The Poisson remainder satisfies
  $\|D_{2,n,n_0}^{\mathrm{bRR}}\|_\infty
  \le C t_{\mathrm{mix}}\|\epsilon\|_\infty/\sqrt m$ up to the same
  deterministic weight constants.

- [x] startup transfer from finite-start remainders to stationary
  augmented-chain remainders. Added `lem:burn-startup-contraction`,
  `lem:burn-startup-transfer`, and `cor:burn-log-startup`. The transfer is
  phrased through the Levin-type Wasserstein contraction of the augmented
  chain and then summed over the post-burn-in PR window:
  $$
  \|\mathcal U_{n,n_0}^{\mathrm{start,RR}}\|_{L_p}
  \le
  \frac{C p A_{\mathrm{st}}(p,q,\alpha)}
       {\alpha a\sqrt m}
  \exp(-c_{\mathrm{st}}\alpha a n_0/p).
  $$
  Under
  $n_0\ge \beta p(c_{\mathrm{st}}\alpha a)^{-1}\log n$ and
  $\alpha=c n^{-1/2}$, $m\asymp n$, this becomes
  $\mathrm{polylog}(n)n^{-1/4-\beta}$.
  With the Berry--Esseen choice $p\asymp\log n$, this startup condition is
  $n_0\gtrsim(\alpha a)^{-1}\log^2 n$ unless a sharper $L_p$ contraction
  removes the $1/p$ loss in the exponent.

- [x] burned-in version of the depth-two misadjustment bound. Added
  `thm:burn-misadjustment` and `cor:burn-misadjustment-rate`. The finite-start
  burned-in remainder is compared to the stationary augmented-chain
  misadjustment over a window of length $m=n-n_0$, then the accumulated
  startup transfer is added:
  $$
  \|R_{n,n_0,\mathrm{fin}}^{\mathrm{mis,RR}}\|_{L_p}
  \le
  \mathrm{StationaryMis}(m,\alpha,p,q)
  +
  \frac{C p A_{\mathrm{st}}(p,q,\alpha)}
       {\alpha a\sqrt m}
  \exp(-c_{\mathrm{st}}\alpha a n_0/p).
  $$
  Under $\alpha=c n^{-1/2}$, $m\ge n/2$, $p\asymp\log n$, and the same
  startup condition as above, this gives
  $\mathrm{polylog}(n)n^{-1/4}$.

## Remaining work

Перед финальной burn-in theorem stochastic blocks are now closed. Assembly
progress:

- [x] define the finite-start burned-in scalar statistic
  $T_{n,n_0}^{\mathrm{RR}}(u)$ and its normalization. Added the vector statistic
  $\mathcal T_{n,n_0}^{\mathrm{RR}}$, scalar projection
  $T_{n,n_0}^{\mathrm{RR}}(u)$, finite-window normalization by
  $\sigma_{n,n_0}^{\mathrm{bRR}}(u)$, asymptotic normalization by $\sigma(u)$,
  and the burned-in variance lower-bound condition
  $m\alpha a \gtrsim \|u\|^2/\sigma^2(u)$.
- [x] combine deterministic transient, burned-in depth-zero martingale
  approximation, predictable-variance comparison, and burned-in
  misadjustment. Added `thm:burn-M-BE`, `lem:burn-R-bound`, and
  `thm:burn-RR-BE-master`. The finite-start statistic now has the decomposition
  $$
  T_{n,n_0}^{\mathrm{RR}}(u)
  =
  -\frac{u^\top M_{n,n_0}^{\mathrm{bRR}}}{\sqrt m}
  +
  \mathcal R_{n,n_0,\mathrm{fin}}^{\mathrm{bRR}}(u),
  $$
  where the composite remainder is the sum of the deterministic transient,
  the Poisson Abel remainder, and the finite-start depth-two misadjustment.
  The finite-window normalized statistic satisfies a master bound of the form
  $$
  d_K(\Xi_{n,n_0}^{\mathrm{bRR}}(u),N(0,1))
  \le
  C(u)\frac{\log^{3/4} n}{m^{1/4}}
  +
  C(u)\frac{\log n}{\sqrt m}
  +
  \frac{e\|\mathcal R_{n,n_0,\mathrm{fin}}^{\mathrm{bRR}}(u)\|_{L_p}}
       {\sqrt{2\pi}\,\sigma_{n,n_0}^{\mathrm{bRR}}(u)}
  +
  \frac e n .
  $$
- [x] specialize the finite-window master bound: impose the explicit
  logarithmic deterministic-transient and startup assumptions, pass from
  $\sigma_{n,n_0}^{\mathrm{bRR}}(u)$ to $\sigma(u)$, and state the final
  balanced-scale burn-in theorem. Added `lem:burn-normalization-transfer` and
  `thm:burn-final-balanced`. The final statement assumes
  $\alpha=c n^{-1/2}$, $m=n-n_0\ge n/2$, the burned-in variance lower bound,
  and the explicit burn-in conditions
  $$
  n_0\ge \frac{2}{\alpha a}\log n,\qquad
  n_0\ge \frac{p}{c_{\mathrm{st}}\alpha a}\log n,\qquad
  p=\max(2,\lceil\log n\rceil).
  $$
  It gives both
  $$
  d_K(\Xi_{n,n_0}^{\mathrm{bRR}}(u),N(0,1))
  \le C(u,c,\theta_0)\frac{\mathrm{polylog}(n)}{n^{1/4}}
  $$
  and the same bound for the asymptotically normalized statistic
  $\Xi_{n,n_0}^{\mathrm{asy,RR}}(u)=T_{n,n_0}^{\mathrm{RR}}(u)/\sigma(u)$.
  The normalization transfer adds only
  $O((m\alpha a)^{-1})=O(n^{-1/2})$ at the balanced scale.

## Current status

Burn-in transfer theorem is assembled through the final balanced-scale
statement. The stationary-limit lemma for the centered $T^{(1)}$ boundary term
is now stated explicitly in the stationary chapter; the remaining items below
are presentation and transfer-polish tasks.

## Next steps

- [x] Add the stationary-limit lemma in the stationary chapter. This should
  replace the informal "start at time $-m$ and let $m\to\infty$" justification
  in the proof of the centered $T^{(1)}$ bound by a stated lemma with a uniform
  $L_p$ bound.
- [x] Decide how to present the Levin-type startup contraction. Either keep it
  as a clearly named technical assumption/justification, or promote it to a
  self-contained lemma with a precise citation to the augmented-chain
  Wasserstein contraction result. Chosen option 2: the startup section now
  explicitly quotes Levin et al. (2025, Appendix B.2, Eq. (49), Proposition 5,
  Eq. (55), and Corollary 4) and uses Proposition 9 for the `H^{(2)}` bridge.
- [x] Add a final corollary converting the $sqrt(m)$ burned-in statistic to
  the more thesis-facing $sqrt(n)$ statistic. For the logarithmic burn-in
  regime, $sqrt(n/m)=1+O(n_0/n)$, and with
  $n_0\lesssim (\alpha a)^{-1}\log^2 n$ this is lower order at
  $\alpha=c n^{-1/2}$. Added `cor:burn-sqrt-n-transfer`, which defines
  `T_{n,n_0}^{RR,n}(u)` and `Xi_{n,n_0}^{n,RR}(u)` and assumes the matching
  upper burn-in window
  $n_0 \le C_0(\alpha a)^{-1}\log^2 n$.
- [x] Audit the final theorem assumptions for redundancy. In particular check
  whether `alpha, 2 alpha in (0, alpha_infinity]`, `alpha a <= 1/4`, the Levin
  restriction, and the variance lower bound can be grouped into a short
  "for all sufficiently large $n$" clause after the explicit finite-$n$
  statement. Done by introducing
  $\alpha_{\mathrm{adm}}(p,q)=\min\{\alpha_\infty,(2a)^{-1},
  \alpha_*(q,t_{\mathrm{mix}}),\alpha_{\mathrm{st}}(p)\}$ and replacing the
  long list in the final theorem by `m >= n/2`,
  `2 alpha <= alpha_adm(p,q)`, and the scalar variance lower bound. The text now
  says explicitly that the elementary step-size constraints and variance lower
  bound are automatic for large $n$ at $\alpha=c n^{-1/2}$; the only remaining
  non-elementary eventual condition is the Levin admissibility
  $2c n^{-1/2}\le\min\{\alpha_*(q,t_{\mathrm{mix}}),\alpha_{\mathrm{st}}(p)\}$.
- [x] Polish notation and wording in the burn-in chapter. The proof is now
  structurally complete, so the next pass should reduce repeated explanations,
  keep only necessary comments, and make the theorem names consistent with the
  stationary chapter. Done: shortened the chapter opening and several transition
  paragraphs, removed a redundant closure sentence after the misadjustment
  theorem, renamed the smoothing/final sections, and aligned theorem titles with
  the stationary naming pattern (`burned-in PR-averaged RR ...`).

### Decision: Levin-type startup contraction

This refers to `lem:burn-startup-contraction` in `src/burn_in_transfer.typ`.
The burn-in proof needs a pointwise-in-time transfer from finite-start
perturbation remainders to the stationary augmented-chain remainders:

$$
\|R_{k,\mathrm{fin}}^{(w)}-R_{k,\mathrm{aug}}^{(w)}\|_{L_p}
\le
A_{\mathrm{st}}(p,q,w)\exp(-c_{\mathrm{st}}wa k/p).
$$

After summing over $k=n_0,\ldots,n-1$, this gives the startup term

$$
\frac{p A_{\mathrm{st}}(p,q,\alpha)}
     {\alpha a\sqrt m}
\exp(-c_{\mathrm{st}}\alpha a n_0/p).
$$

We chose the self-contained cited-lemma route. The text now first states the
Levin depth-two augmented-chain cost $c_{J,2}^{(w)}$ and cites the componentwise
contraction from Levin et al. (2025, Appendix B.2, Proposition 5, with the cost
in Eq. (49) and constants in Eq. (55)). Corollary 4 supplies the invariant law
for the augmented chain. The proof of `lem:burn-startup-contraction` then
couples the finite-start zero-initialized perturbation coordinates to that
stationary augmented chain, integrates over the stationary initial state using
uniform-in-time moment bounds from Levin Lemmas 4 and 8 and Proposition 8
passed to the invariant law, and handles `H^{(2)}` through the explicit
representation in Appendix D.1, Proposition 9.

The envelope $A_{\mathrm{st}}$ was made deliberately non-sharp
($p^7 t_{\mathrm{mix}}^5$ rather than the raw Proposition-5
$p^{7/2}t_{\mathrm{mix}}^{5/2}$ factor) because the stationary initial cost
and the `H^{(2)}` transfer add fixed polynomial losses. This does not affect
the final balanced-scale conclusion: with $p,q\asymp\log n$ it remains a
polylogarithmic factor.

## Chapter decision

Burn-in лучше вынести в отдельную главу:

- stationary theorem remains readable and does not hide finite-start issues;
- burned-in weights differ structurally from full-window weights;
- the proof has its own normalization choice $m=n-n_0$;
- the final transfer theorem has a separate statement and assumptions on
  $n_0$; with the current $L_p$ startup contraction and $p\asymp\log n$ this
  is $n_0 \gtrsim (\alpha a)^{-1}\log^2 n$.

Created draft chapter:

`src/burn_in_transfer.typ`

Included from `main.typ` as:

`Burn-in Transfer for Deterministic Starts`.

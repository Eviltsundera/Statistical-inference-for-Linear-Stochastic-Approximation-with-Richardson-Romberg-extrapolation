# Removed thesis remarks, 2026-05-17

Эта заметка хранит исследовательские комментарии, которые были убраны или
сильно сжаты из основного Typst-текста, чтобы глава 4 читалась как доказательство,
а не как рабочий лог.

## `src/pr_weights.typ`

### Single-step comparison for RR weights

Для обычных PR-весов
$\|Q_l^{(\alpha)}-\bar A^{-1}\|\lesssim (1-\alpha a)^{k/2}$ и
$\|Q_{l+1}^{(\alpha)}-Q_l^{(\alpha)}\|\lesssim
\alpha(1-\alpha a)^{(k-1)/2}$. Для RR-веса первая сумма остается порядка
$1/(\alpha a)$: около правой границы $\mathcal Q_{n-1}^{RR}=0$, поэтому
$\mathcal Q_{n-1}^{RR}-\bar A^{-1}=-\bar A^{-1}$. RR-выигрыш проявляется в
дискретной производной, но после суммирования геометрического хвоста он не дает
глобального дополнительного множителя $\alpha$.

### Variance comparison

Константа $C_3$ не зависит от $\alpha$ и $n$; зависимость от $a$ идет через
Lyapunov contraction. Доминирующий вклад в
$\|\Sigma_n^{RR}-\Sigma_\infty\|$ приходит из правой границы, поэтому RR не
улучшает порядок variance-comparison относительно single-step PR, но и не
портит ведущую ковариацию.

### Poisson remainder

Оценка $D_{2,n}^{RR}$ детерминирована: используются только sup-нормы
$\mathcal Q_l^{RR}$, их дискретной производной и Poisson solution
$\hat\epsilon$. Эффективный масштаб:
$$
\|D_{2,n}^{RR}\|_{L_p}
  \lesssim \frac{t_{\mathrm{mix}}}{a^2\sqrt n}.
$$
Правая Poisson-boundary исчезает из-за $\mathcal Q_{n-1}^{RR}=0$.

### Bracket concentration and martingale BE

Естественный центр для martingale Berry--Esseen шага:
$s_n^2=n\sigma_{n,RR}^2(u)$, а не $n\sigma^2(u)$. Переход к асимптотической
нормировке $\sigma(u)$ делается отдельно через variance comparison.

В martingale BE константа при ведущем члене растет как
$\sigma(u)^{-1}$; это нормальный вырожденный случай и соответствует условию
$\sigma^2(u)>0$.

### Misadjustment and finite start

The misadjustment theorem is a stationary augmented-chain statement. For a
zero-start recursion, the accumulated startup contribution over a full window
can be of order $1/(\sqrt n\,\alpha a)$, hence at
$\alpha\asymp n^{-1/2}$ it is not negligible. A finite-start theorem should use
burned-in weights $Q_{\ell,n_0}^{(\alpha)}$ and redo the Poisson,
variance-comparison, and misadjustment bookkeeping.

The direct kernel-difference route for the centered RR difference of $S_n$
only gives $O(\sqrt n\,\alpha)=O(1)$ at the working scale. The depth-two route
is used because it separates:

- the bias of $J^{(1)}$, where RR removes the leading $\alpha\Delta$ term;
- the centered part of $J^{(1)}$, controlled by the telescoping identity and
  Levin Corollary 6;
- the $J^{(2)}$ and $H^{(2)}$ terms, whose $\alpha^{3/2}$ scale is already
  small enough.

A possible further improvement would require RR-cancellation at the
$J^{(2)}+H^{(2)}$ level, replacing $\sqrt n\,\alpha^{3/2}$ by
$\sqrt n\,\alpha^2$.

### Stationary theorem vs burn-in

Theorem 3 is an $n_0=0$ stationary augmented-chain theorem for
$S_{n,\mathrm{stat}}^{RR}(u)$, not a deterministic-start theorem for the
actual recursion. Setting $\theta_0=\theta^\star$ removes the deterministic
transient in the finite-start algebra but does not initialize the augmented
variables $(J^{(0)},J^{(1)},J^{(2)},H^{(2)})$ from stationarity.

## `src/last_iterate.typ`

The removed visible remark after the centered shifted-first-order lemma said:
stationarity is mainly used there to avoid boundary notation; a non-stationary
law would add standard exponentially small UGE boundary terms, while the bias
of $T_n^{(1,\alpha)}$ is controlled separately through the stationary expansion
of $\mathbb E J_n^{(1,\alpha)}$ in Levin et al. (2025).

The long kernel-difference discussion from the formerly commented-out Section
3.3 was removed from `src/last_iterate.typ`; the relevant conclusion is the
same as above: the direct kernel-difference route does not improve the
PR-averaged RR misadjustment rate at the working scale.

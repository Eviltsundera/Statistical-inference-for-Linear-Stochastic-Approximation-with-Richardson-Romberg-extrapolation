# Изучение текущей версии диплома

## Вопрос

Изучить текущий диплом и зафиксировать, что в нем сейчас доказывается, как
устроена доказательная цепочка, что показывают эксперименты и какие места
остаются наиболее рискованными.

## Короткий вывод

Текущая версия диплома доказывает неасимптотическую нормальную аппроксимацию
типа Berry--Esseen для скалярных проекций burned-in Polyak--Ruppert averaged
Richardson--Romberg estimator в linear stochastic approximation с марковским
шумом.

Финальная thesis-facing форма результата:

$$
d_K\!\left(
  \frac{
    \sqrt n\,u^\top
    \left(\bar\theta_{n,n_0}^{\mathrm{RR},\alpha}-\theta^\star\right)
  }{
    \sqrt{u^\top\Sigma_\infty u}
  },
  N(0,1)
\right)
\le
C(u,c,\theta_0)\frac{\mathrm{polylog}(n)}{n^{1/4}},
$$

при balanced scale $\alpha=c n^{-1/2}$, non-degenerate
$\sigma^2(u)=u^\top\Sigma_\infty u>0$, $m=n-n_0\ge n/2$ и burn-in window

$$
n_0 \asymp (\alpha a)^{-1}\log^2 n
      = O(n^{1/2}\log^2 n).
$$

Ковариационная цель:

$$
\Sigma_\infty
  =
  \bar A^{-1}\Sigma_\varepsilon^{(M)}\bar A^{-T}.
$$

## Карта текста

- `main.typ` собирает диплом как последовательность: введение, zeroth-order RR,
  last-iterate analysis, PR weight bounds, burn-in transfer, experiments,
  conclusion, appendix.
- `src/introduction/` задает LSA recursion, PR average, RR estimator,
  assumptions, Markovian long-run covariance and main goals.
- `src/zeroth_order_rr/` показывает базовый механизм RR-cancellation для
  deterministic-product leading terms.
- `src/last_iterate/` объясняет, почему single-alpha или depth-one estimates
  недостаточны для финального PR-averaged RR theorem.
- `src/pr_weights/` доказывает stationary augmented-chain Berry--Esseen
  assembly для comparison statistic $S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)$.
- `src/burn_in_transfer/` переносит stationary theorem на deterministic start
  через burn-in и дает финальный $\sqrt n$-statement.
- `src/experiments.typ` эмпирически отделяет две роли: RR in stepsize
  улучшает центр интервала, а OBM/lugsail отвечают за long-run variance
  estimation.
- `src/appendix/external_inputs.typ` выписывает импортированные рабочие формы
  Levin et al. and Samsonov et al. и явно отделяет local extensions.

## Доказательная цепочка

Главная цепочка:

$$
\text{RR weights}
\to
\text{variance comparison}
\to
\text{Poisson martingale approximation}
\to
\text{bracket concentration}
\to
\text{martingale Berry--Esseen}
\to
\text{depth-two misadjustment}
\to
\text{burn-in transfer}.
$$

В `src/pr_weights/` ключевой deterministic kernel:

$$
\mathcal Q_l^{\mathrm{RR}} = 2Q_l^{(\alpha)} - Q_l^{(2\alpha)}.
$$

Он удовлетворяет pointwise comparison

$$
\|\mathcal Q_l^{\mathrm{RR}}-\bar A^{-1}\|
  \lesssim (1-\alpha a)^{(n-l)/2},
$$

и total-variation type bound для successive differences. Эти оценки дают:

$$
\|\Sigma_n^{\mathrm{RR}}-\Sigma_\infty\|
  \lesssim \frac{1}{n\alpha a}.
$$

Poisson equation for Markov noise separates leading weighted sum into
martingale plus Abel boundary remainder:

$$
W^{\mathrm{RR}}
  =
  -\frac{1}{\sqrt n}M_n^{\mathrm{RR}}
  + D_{2,n}^{\mathrm{RR}}.
$$

Martingale Berry--Esseen gives the dominant rate
$\log^{3/4}(n)n^{-1/4}$ because predictable quadratic variation is controlled
only at a concentration scale that enters the martingale BE theorem with
exponent $1/(2p+1)$.

The stationary misadjustment theorem uses depth-two inputs from Levin et al.
and gives, at $\alpha=c n^{-1/2}$ and $p\asymp\log n$,

$$
\|R_n^{\mathrm{mis,RR}}\|_{L_p}
  \le C\,\mathrm{polylog}(n)n^{-1/4}.
$$

## Burn-in Transfer

The burn-in chapter is not cosmetic; it is what turns the stationary
augmented-chain theorem into a deterministic-start theorem. It separately
controls:

- burned-in deterministic weights, including pre-burn-in and post-burn-in
  regimes;
- deterministic transient from $\theta_0-\theta^\star$;
- random initial-product discrepancy;
- Poisson/Abel remainder with burned-in weights;
- predictable variance comparison;
- startup transfer for augmented-chain remainders;
- finite-window normalization and the final $\sqrt m$ to $\sqrt n$ transfer.

The lower burn-in condition comes from exponential contraction with the
Berry--Esseen moment choice $p\asymp\log n$:

$$
n_0 \gtrsim (\alpha a)^{-1}\log^2 n.
$$

The final $\sqrt n$ corollary also needs an upper window
$n_0\lesssim(\alpha a)^{-1}\log^2 n$, so that

$$
\sqrt{n/(n-n_0)}-1 = O(n_0/n)
$$

is lower order at $\alpha=c n^{-1/2}$.

## Experiments

The experiments support the theoretical interpretation:

- single constant-stepsize branches can undercover because the interval center
  is biased;
- Richardson--Romberg in the stepsize reduces this center bias and brings
  scalar coverage close to nominal without simply widening intervals;
- oracle-variance diagnostics suggest that, in the main finite-state
  experiments, the RR center and normal approximation are already adequate at
  large horizons;
- OBM and OBM-LW affect long-run variance estimation, not the point-estimator
  bias;
- lugsail helps when OBM has visible negative Bartlett-window bias, but it can
  be neutral or unstable depending on block size and dependence regime.

The experiments deliberately do not prove a theorem for OBM/lugsail covariance
estimation along RR-averaged constant-stepsize LSA trajectories. That remains
future work.

## Checks Performed

- Read the current top-level structure in `main.typ`.
- Read the active theorem statements in `src/pr_weights/` and
  `src/burn_in_transfer/`.
- Read the current abstract, conclusion, appendix external-input summary and
  experiment section.
- Compiled the thesis with
  `typst compile main.typ /tmp/thesis_study_2026-06-02.pdf`; compilation
  completed without errors.

## Remaining Risks

The main remaining mathematical risk is proof completeness, not an obvious
rate mismatch.

The most load-bearing local extension is the full-state startup transfer for
the depth-two augmented remainder, especially the passage for the
$H^{(2)}$ component. The text now helps by listing this as a local extension
in `src/appendix/external_inputs.typ`, rather than silently attributing it to
Levin et al. Still, for defense-level robustness, this is the first section to
audit in detail.

Specific polish targets:

1. Make the initial law of the Markov chain explicit in the final
   deterministic-start theorem, or state uniformity over that law.
2. In the startup extension, keep the finite-past construction for
   $H^{(2)}$ and the random-time product step as named, auditable lemmas.
3. Add a compact scale audit showing exactly how the
   $p,t_{\mathrm{mix}},a,\alpha$ powers in the $H^{(2)}$ startup bound are
   absorbed into the final $A_{\mathrm{st}}(p,q,\alpha)$-type quantity.
4. If time remains, add a short reader-facing paragraph in the introduction or
   conclusion saying that the proved theorem uses known/asymptotic covariance
   normalization, while OBM/lugsail are empirical variance-estimation tools in
   this thesis.

## Bottom Line

The current thesis is internally coherent as a proof program for scalar
Berry--Esseen inference of burned-in RR-averaged constant-stepsize LSA under
Markovian noise. The main theorem, abstract, conclusion, and experiments point
to the same story: Richardson--Romberg fixes the leading stepsize bias of the
center, while the limiting covariance target remains
$\Sigma_\infty$; practical covariance estimation is investigated
experimentally and left as a separate theoretical problem.

# Ориентация по текущему диплому

## Вопрос

Изучить текущий текст диплома и зафиксировать, что в нем сейчас доказывается,
как устроена цепочка доказательства и где остаются основные риски.

## Короткий вывод

Текущий диплом доказывает неасимптотическую Berry--Esseen аппроксимацию для
скалярных проекций burned-in Polyak--Ruppert averaged
Richardson--Romberg estimator в constant-stepsize linear stochastic
approximation с марковским шумом.

Финальная thesis-facing форма результата:

$$
d_K\!\left(
  \frac{
    \sqrt n\,u^\top
    (\bar\theta_{n,n_0}^{\mathrm{RR},\alpha}-\theta^\star)
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

Здесь

$$
\Sigma_\infty
  =
  \bar A^{-1}\Sigma_\varepsilon^{(M)}\bar A^{-T}
$$

is the Markovian averaged-SA covariance target.

## Карта текста

- `src/introduction.typ` задает LSA recursion, UGE, Hurwitz/Lyapunov
  contraction, bounded noise, Markovian long-run covariance and target
  $\Sigma_\infty$. Введение правильно предупреждает, что stationary theorem
  не является сразу deterministic-start theorem.
- `src/zeroth_order_rr.typ` является preliminary calculation: показывает, как
  RR difference создает дополнительный фактор $\alpha$ в last-iterate
  zero-order kernel.
- `src/last_iterate.typ` объясняет, почему depth-one route недостаточен:
  centered PR-averaged RR misadjustment дает $O(\sqrt n\,\alpha)$, что при
  $\alpha\asymp n^{-1/2}$ не убывает.
- `src/pr_weights.typ` через подпапку `src/pr_weights/` доказывает stationary
  augmented-chain Berry--Esseen theorem for
  $S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)$.
- `src/burn_in_transfer.typ` через подпапку `src/burn_in_transfer/` переносит
  результат на deterministic start после burn-in и дает финальную
  $\sqrt n$-нормировку.
- `src/external_inputs.typ` выписывает рабочие формы внешних inputs из
  Levin et al. 2025 и Samsonov et al. 2025, а также отделяет local extensions
  from direct citations.

## Доказательная цепочка

Главная цепочка сейчас такая:

$$
\text{RR weights}
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

В `pr_weights` ключевые факты:

- full-window RR weight satisfies
  $$
  \mathcal Q_l^{\mathrm{RR}}-\bar A^{-1}
    =
    -\bar A^{-1}(2B_\alpha^{n-l}-B_{2\alpha}^{n-l});
  $$
- variance comparison remains
  $$
  \|\Sigma_n^{\mathrm{RR}}-\Sigma_\infty\|
  \lesssim (n\alpha a)^{-1};
  $$
- RR cancellation is most useful in the discrete derivative of weights:
  $$
  \sum_l
  \|\mathcal Q_{l+1}^{\mathrm{RR}}-\mathcal Q_l^{\mathrm{RR}}\|
  \lesssim a^{-2},
  $$
  which controls the Poisson/Abel remainder;
- martingale Berry--Esseen gives the main
  $\log^{3/4}(n)n^{-1/4}$ term;
- Levin depth-two inputs control the stationary RR misadjustment at
  $\mathrm{polylog}(n)n^{-1/4}$ on the balanced scale.

В `burn_in_transfer` ключевые дополнения:

- burned-in weights $Q_{l;n_0,n}^{\mathrm{RR}}$ have separate pre-burn-in and
  post-burn-in behavior; pre-burn-in weights are not compared to
  $\bar A^{-1}$ but are controlled by their energy;
- deterministic transient and random initial-product discrepancy decay after
  mixing-scale burn-in;
- finite-start depth-two remainders are coupled to stationary augmented-chain
  remainders, producing the startup term
  $$
  \frac{p A_{\mathrm{st}}(p,q,\alpha)}
       {\alpha a\sqrt m}
  \exp(-c_{\mathrm{st}}\alpha a n_0/p);
  $$
- with the Berry--Esseen choice $p\asymp\log n$, this forces
  $n_0\gtrsim(\alpha a)^{-1}\log^2 n$;
- final transfer from $\sqrt m$ to $\sqrt n$ is lower order if
  $n_0\lesssim(\alpha a)^{-1}\log^2 n$.

## Что проверено при чтении

- `typst compile main.typ /tmp/thesis_study_check.pdf` проходит.
- Основной statement согласован с abstract and introduction.
- Стационарный и deterministic-start objects разведены.
- Finite-window normalization
  $\sigma_{n,n_0}^{\mathrm{bRR}}(u)$ and asymptotic normalization
  $\sigma(u)$ are transferred with cost $O((m\alpha a)^{-1})$, which is
  $O(n^{-1/2})$ at $\alpha=c n^{-1/2}$.
- The final $\sqrt n$ corollary correctly requires an upper burn-in window,
  so that $\sqrt{n/m}-1=O(n_0/n)$ remains lower order.

## Remaining risks

The main mathematical risk is not an obvious algebraic error. It is proof
completeness around the local extension
`Full-state startup contraction for the depth-two augmented remainder`.
The $J^{(0)},J^{(1)},J^{(2)}$ coordinates are backed by Levin's augmented-chain
contraction, while $H^{(2)}$ is added through a local finite-past and
random-product argument. The current argument is plausible and rate-consistent,
but it is the most load-bearing part of the deterministic-start theorem.

For a defense or final polish, I would make this section the first target:

1. Spell out exactly how Levin Proposition 9 is passed to the stationary
   finite-past limit for $H^{(2)}$.
2. Keep the random-time product lemma as a named local technical lemma and make
   the empty-product cases explicit.
3. Add a short scale audit explaining why
   $B_H(p,q,w)$ and the convolution term fit into
   $A_{\mathrm{st}}(p,q,w)$.

If this startup extension is accepted as a local lemma, the rest of the proof
chain is internally coherent.


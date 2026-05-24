# Основной результат диплома

## Вопрос

Изучить текущий текст диплома и коротко определить, какой в нем основной
математический результат.

## Короткий ответ

Основной результат диплома -- не просто CLT для обычного PR-усреднения, а
неасимптотическая нормальная аппроксимация типа Berry--Esseen для скалярных
проекций burned-in Polyak--Ruppert averaged Richardson--Romberg estimator в
linear stochastic approximation с марковским шумом.

В thesis-facing форме результат такой. Для двух связанных постоянных шагов
$\alpha$ и $2\alpha$ строится RR-среднее

$$
\bar\theta_{n,n_0}^{\mathrm{RR},\alpha}
  =
  2\bar\theta_{n,n_0}^{(\alpha)}
  -
  \bar\theta_{n,n_0}^{(2\alpha)}.
$$

При стандартных предположениях диплома: uniform geometric ergodicity,
Hurwitz/stability, bounded centered matrix noise, bounded centered vector
noise, non-degenerate scalar variance $\sigma^2(u)>0$, и при balanced scale

$$
\alpha = c n^{-1/2},
$$

после burn-in порядка

$$
n_0 \asymp (\alpha a)^{-1}\log^2 n
     = O(n^{1/2}\log^2 n)
$$

доказывается

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
C(u,c,\theta_0)\frac{\mathrm{polylog}(n)}{n^{1/4}}.
$$

Здесь

$$
\Sigma_\infty
  =
  \bar A^{-1}\Sigma_\epsilon^{(M)}\bar A^{-T}
$$

is the Markovian averaged-SA covariance target.

## Что это означает содержательно

Richardson--Romberg используется для устранения ведущего $O(\alpha)$
stationary bias постоянного шага. При выборе $\alpha=c n^{-1/2}$ остаточные
RR- и startup-члены становятся достаточно малы, чтобы нормальная
аппроксимация работала на масштабе $\sqrt n$.

Ведущий случайный вклад после Poisson/martingale reduction имеет ту же
предельную covariance target $\Sigma_\infty$, что и оптимальное PR-усреднение:
RR меняет bias, но не ухудшает асимптотическую ковариацию.

## Структура доказательства в тексте

1. `src/pr_weights.typ` доказывает stationary augmented-chain
   Berry--Esseen bound для comparison statistic
   $S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)$. Это промежуточная теорема, а не
   конечный deterministic-start result.

2. Там же контролируются RR-веса, variance comparison, Poisson remainder,
   martingale Berry--Esseen step и stationary RR misadjustment.

3. `src/burn_in_transfer.typ` переносит stationary theorem на
   deterministic start через burn-in: отдельно ограничиваются deterministic
   transient, random initial product, burned-in weights, startup transfer и
   finite-window misadjustment.

4. Финальный результат -- Theorem
   `Balanced-scale burned-in PR-averaged RR Berry--Esseen bound` и следующий
   corollary with $\sqrt n$ normalization.

## Важные ограничения формулировки

- Результат скалярный: для фиксированного направления $u$, а не полный
  multivariate Berry--Esseen по выпуклым множествам.
- Это triangular-array statement с $\alpha_n=c n^{-1/2}$, а не fixed-$\alpha$
  CLT centered exactly at $\theta^\star$.
- Теорема использует известную/asymptotic нормировку
  $\sigma(u)=\sqrt{u^\top\Sigma_\infty u}$. Consistency of an empirical
  covariance estimator is not the main proved theorem here.
- Стационарная теорема из `src/pr_weights.typ` сама по себе не является
  практическим deterministic-start theorem; это закрывается отдельной
  burn-in главой.

## Одной фразой для введения или защиты

Диплом доказывает, что Richardson--Romberg PR-усреднение для constant-step LSA
с марковским шумом после mixing-scale burn-in имеет нормальную аппроксимацию
с оптимальной ковариацией $\Sigma_\infty$ и неасимптотической
Berry--Esseen ошибкой $\mathrm{polylog}(n)n^{-1/4}$ для любых фиксированных
скалярных проекций.

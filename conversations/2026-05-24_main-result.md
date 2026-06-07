# Основной результат диплома

## Вопрос

Изучить текущий текст диплома и выписать главный математический результат.

## Короткая формулировка

Главный результат диплома -- неасимптотическая нормальная аппроксимация типа
Berry--Esseen для скалярных проекций burned-in Polyak--Ruppert averaged
Richardson--Romberg estimator в linear stochastic approximation с марковским
шумом.

Если две constant-stepsize траектории с шагами $\alpha$ и $2\alpha$ запускаются
из одной deterministic initial point $\theta_0$ на одной и той же марковской
траектории, то RR-среднее задается как

$$
\bar\theta_{n,n_0}^{\mathrm{RR},\alpha}
  =
  2\bar\theta_{n,n_0}^{(\alpha)}
  -
  \bar\theta_{n,n_0}^{(2\alpha)}.
$$

При предположениях диплома: uniform geometric ergodicity, Hurwitz/Lyapunov
stability, bounded centered matrix noise, bounded centered vector noise,
non-degenerate scalar variance $\sigma^2(u)>0$, admissible small-step
conditions, и при balanced triangular-array scale

$$
\alpha = c n^{-1/2},
$$

после burn-in в mixing-scale window

$$
n_0 \asymp (\alpha a)^{-1}\log^2 n
      = O(n^{1/2}\log^2 n)
$$

доказывается, что для любого фиксированного направления $u$:

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

является марковской long-run covariance target для averaged LSA.

## Где это в дипломе

Финальная формальная версия находится в `src/burn_in_transfer/12_balanced_burn_in_berry_esseen.typ`:

- theorem `Balanced-scale deterministic-start burned-in PR-averaged RR Berry--Esseen bound`;
- corollary `$\sqrt(n)$-normalization for the burned-in RR statistic`.

Промежуточная stationary версия находится в
`src/pr_weights/11_smoothing_assembly.typ`:

- theorem `Stationary augmented-chain Berry--Esseen assembly for the RR comparison statistic`;
- corollary `Stationary balanced-scale augmented-chain Berry--Esseen bound`.

## Содержательный смысл

Richardson--Romberg extrapolation используется для подавления leading
constant-stepsize bias. Обычное constant-step PR average имеет bias, который
может портить confidence intervals на масштабе $\sqrt n$. RR-комбинация
сравнивает две траектории с шагами $\alpha$ и $2\alpha$ и сокращает leading
stepsize term.

При выборе $\alpha=c n^{-1/2}$ остаточные RR-, misadjustment-, Poisson-,
startup- и transient-члены контролируются на уровне
$\mathrm{polylog}(n)n^{-1/4}$. Поэтому итоговая статистика имеет ту же
ковариационную цель, что и оптимальное PR-усреднение:

$$
\Sigma_\infty
  =
  \bar A^{-1}\Sigma_\epsilon^{(M)}\bar A^{-T}.
$$

Иными словами, RR меняет центр статистики, уменьшая bias, но не меняет
асимптотическую covariance target.

## Доказательная цепочка

Доказательство устроено так:

1. Вводится deterministic RR weight kernel и доказываются pointwise,
   energy и total-variation bounds для PR weights.
2. Эти оценки дают comparison между finite-window variance proxy и
   $\Sigma_\infty$.
3. Markovian noise раскладывается через Poisson equation в martingale part и
   Abel/boundary remainders.
4. Для martingale part применяется Berry--Esseen bound с контролем
   predictable quadratic variation.
5. RR misadjustment terms ограничиваются через depth-two estimates.
6. Stationary augmented-chain theorem переносится на deterministic start
   через burn-in: deterministic transient, initial random products, startup
   discrepancy и finite-window normalization.
7. В конце $\sqrt m$-нормировка burned-in window переводится в финальную
   $\sqrt n$-нормировку.

## Что не является главным доказанным результатом

Результат не является multivariate Berry--Esseen theorem для всех выпуклых
множеств; он сформулирован для фиксированных scalar projections $u$.

Результат не является fixed-$\alpha$ CLT centered exactly at $\theta^\star$.
Финальная practical theorem работает в triangular-array режиме
$\alpha_n=c n^{-1/2}$.

Результат не доказывает non-asymptotic consistency of OBM/lugsail covariance
estimation along RR-averaged LSA trajectories. OBM and lugsail в дипломе
используются в экспериментальной части как practical long-run variance
estimators; их теория оставлена для future work.

## Одна фраза для защиты

Диплом доказывает, что burned-in Richardson--Romberg PR-усреднение для
constant-stepsize LSA с марковским шумом имеет нормальную аппроксимацию с
оптимальной ковариационной целью $\Sigma_\infty$ и неасимптотической
Berry--Esseen ошибкой $\mathrm{polylog}(n)n^{-1/4}$ для фиксированных
скалярных проекций.

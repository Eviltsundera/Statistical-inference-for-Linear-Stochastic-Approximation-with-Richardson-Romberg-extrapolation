# Эксперимент для проверки CDF-нормальности RR-статистики

## Вопрос

Поставить эксперимент для проверки главного утверждения диплома: для разных
$n$ сравнить функции распределения нормированной RR-статистики с $N(0,1)$.

## Что именно проверяем

Главный theorem-facing объект:

$$
Z_n^{\mathrm{RR}}(u)
  :=
  \frac{
    \sqrt n\,u^\top
    \left(\bar\theta_{n,n_0}^{\mathrm{RR},\alpha_n}-\theta^\star\right)
  }{
    \sqrt{u^\top\Sigma_\infty u}
  },
\qquad
\alpha_n = c n^{-1/2},
$$

где

$$
\bar\theta_{n,n_0}^{\mathrm{RR},\alpha_n}
  =
  2\bar\theta_{n,n_0}^{(\alpha_n)}
  -
  \bar\theta_{n,n_0}^{(2\alpha_n)}.
$$

Для каждого $n$ запускаем много independent trajectories и строим empirical CDF

$$
\hat F_n(x)
  =
  \frac{1}{M}\sum_{r=1}^M
    \mathbf 1\{Z_{n,r}^{\mathrm{RR}}(u)\le x\}.
$$

Основная метрика:

$$
\hat D_n
  :=
  \sup_x |\hat F_n(x)-\Phi(x)|.
$$

Это прямой экспериментальный аналог Kolmogorov distance из теоремы.

## Почему не coverage-only

Coverage при 95% проверяет только две точки CDF, $x=\pm1.96$. Это полезно, но
слишком грубо. CDF/KS diagnostic показывает всю ошибку распределения:

- mean shift из-за residual bias;
- variance mismatch;
- skewness/tails;
- локальные отклонения в центральной части.

Coverage оставить как secondary metric:

$$
\widehat{\mathrm{cov}}_n
  =
  \frac{1}{M}\sum_{r=1}^M
    \mathbf 1\{|Z_{n,r}^{\mathrm{RR}}(u)|\le 1.96\}.
$$

## Базовая постановка

Лучше начинать не с 100 случайных problems, а с одного фиксированного
finite-state LSA problem. Теорема асимптотическая по $n$ для фиксированной
задачи; random-problem aggregation может замаскировать CDF effect.

Recommended fixed problem:

- finite-state Markovian LSA из текущего кода;
- $d=5$, number of states $=10$;
- generation settings как в thesis experiments:
  `eig_min = 0.25`, `eig_max = 0.60`, `noise_target = 0.35`;
- фиксированная random unit direction $u$;
- exact finite-state oracle variance
  $\sigma^2(u)=u^\top\Sigma_\infty u$;
- deterministic start $\theta_0$, например zero vector или текущий default;
- Markov chain start from a fixed state for deterministic-start test.

Основная сетка:

$$
n\in\{50\,000,\ 100\,000,\ 300\,000,\ 1\,000\,000,\ 3\,000\,000\}.
$$

Step-size:

$$
\alpha_n = c n^{-1/2},
\qquad
(2\alpha_n,\alpha_n)\ \text{for RR}.
$$

Практичный старт: взять $c=20$. Тогда

$$
\alpha_n
\approx
0.089,\ 0.063,\ 0.037,\ 0.020,\ 0.012
$$

на указанной сетке, а старшая ветка $2\alpha_n$ остается в разумном диапазоне.
Если smallest-$n$ branch нестабильна для выбранной задачи, уменьшить до
$c=10$.

Burn-in:

$$
n_0(n)
  =
  \left\lfloor
    \kappa\,(\alpha_n a_{\mathrm{proxy}})^{-1}\log^2 n
  \right\rfloor,
$$

где $a_{\mathrm{proxy}}$ можно взять как stability proxy из generated problem,
например minimum real contraction scale of the mean matrix. Практически нужно
держать $n_0/n\le 0.25$ на smallest horizon; если это нарушается, уменьшить
$\kappa$ или начинать сетку с большего $n$.

Monte Carlo size:

- minimum: $M=5\,000$ trajectories per $n$;
- better: $M=20\,000$ trajectories per $n$ for the fixed-problem CDF plot.

Для $M=20\,000$ Monte Carlo KS noise floor is about

$$
\sqrt{\frac{\log(2/0.05)}{2M}}\approx 0.0096
$$

by the DKW inequality. Поэтому значения $\hat D_n$ ниже примерно $0.01$ уже
почти неразличимы без увеличения $M$.

## Что рисовать

Main figure:

$$
\Delta_n(x)=\hat F_n(x)-\Phi(x),
\qquad x\in[-3,3],
$$

для всех $n$ на одном графике. Добавить horizontal Monte Carlo band
$\pm\varepsilon_M$, где

$$
\varepsilon_M
  =
  \sqrt{\frac{\log(2/\delta)}{2M}},
\qquad \delta=0.05.
$$

Это лучше, чем overlay raw CDFs: все CDFs будут почти совпадать визуально, а
signed CDF error immediately shows where the approximation fails.

Second figure:

$$
\hat D_n
\quad \text{vs}\quad n
$$

in log-log scale, together with:

- Monte Carlo floor $\varepsilon_M$;
- reference slope $n^{-1/4}$, rescaled to the first point;
- optionally $n^{-1/2}$ as a stronger visual benchmark.

Third figure:

Q-Q or P-P plot for the smallest and largest $n$:

$$
\left(\Phi(z_{(i)}),\frac{i-1/2}{M}\right),
$$

where $z_{(i)}$ are sorted values of $Z_{n,r}^{\mathrm{RR}}(u)$.

## Таблица метрик

For every $n$ report:

| metric | definition | why |
|---|---|---|
| `KS_D` | $\sup_x|\hat F_n(x)-\Phi(x)|$ | main CDF diagnostic |
| `KS_D_minus_floor` | $\max(\hat D_n-\varepsilon_M,0)$ | separates signal from MC noise |
| `mean_Z` | sample mean of $Z_n^{RR}$ | residual bias |
| `var_Z` | sample variance of $Z_n^{RR}$ | variance mismatch |
| `skew_Z` | sample skewness | asymmetric remainder |
| `kurt_Z` | excess kurtosis | tail mismatch |
| `cov_95` | $\Pr(|Z_n^{RR}|\le1.96)$ | CI-facing summary |
| `q025/q975_error` | empirical quantile minus normal quantile | tail CI error |

## Baselines

Use the same trajectories and same oracle variance for:

1. `RR balanced`: $\alpha_n=c n^{-1/2}$, pair $(2\alpha_n,\alpha_n)$.
2. `single alpha`: only $\bar\theta^{(\alpha_n)}$.
3. `single 2alpha`: only $\bar\theta^{(2\alpha_n)}$.
4. optional `PR diminishing`: existing PR schedule as practical baseline.

Expected signal:

- `RR balanced`: $\hat D_n$ decreases with $n$ until it hits Monte Carlo floor.
- `single alpha`: because constant-step PR bias is $O(\alpha_n)$, after
  $\sqrt n$ scaling the bias is $O(c)$; its CDF can remain visibly shifted.
- `single 2alpha`: stronger shift than `single alpha`.

This is the cleanest way to show that RR is correcting the center, not merely
changing the variance estimate.

## Two-stage run plan

### Stage 1: clean oracle CDF experiment

One fixed problem, one fixed $u$, exact $\Sigma_\infty$, $M=20\,000$, horizons
as above. No OBM, no lugsail, no estimated variance. This tests the theorem's
normal approximation directly.

Primary acceptance criterion:

$$
\hat D_n^{RR}
\text{ decreases with }n
\text{ and is eventually within the DKW band.}
$$

Secondary acceptance criterion:

$$
|\mathrm{mean}(Z_n^{RR})|\to 0,
\qquad
\mathrm{var}(Z_n^{RR})\to 1,
\qquad
\widehat{\mathrm{cov}}_n\to 0.95.
$$

### Stage 2: robustness across problems

Repeat with, say, 20 fixed generated problems and $M=2\,000$ trajectories per
problem. Report distribution of `KS_D` across problems: median, p10, p90.

This checks that the Stage 1 phenomenon is not a lucky problem instance.

## Important caveat

Do not interpret the fitted slope too literally. The theorem gives an upper
bound

$$
D_n \lesssim \mathrm{polylog}(n)n^{-1/4},
$$

not an exact asymptotic equivalent. In a finite experiment the observed
distance may decay faster, slower, or quickly hit Monte Carlo noise. The right
claim is qualitative and diagnostic:

1. RR standardized CDF approaches $\Phi$.
2. Single-alpha constant-step CDF has a visible bias shift.
3. Oracle variance normalization isolates distributional approximation from
   variance-estimator error.

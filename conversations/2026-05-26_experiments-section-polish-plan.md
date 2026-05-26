# План доработки экспериментальной главы

## Цель

Превратить текущую главу `Numerical Experiments and Covariance-Estimator
Diagnostics` из рабочей заметки в полноценную экспериментальную главу диплома:

- ясно объяснить, что именно проверяют эксперименты;
- отделить уже полученные результаты от планируемых запусков;
- аккуратно ввести OBM и lugsail как практические variance-estimation методы;
- не создавать впечатление, что теория OBM/lugsail для RR-LSA уже доказана в
  дипломе.

## Главная проблема текущей версии

Сейчас глава похожа на research log:

- начинается с технической мотивации OBM/lugsail, а не с целей экспериментов;
- таблицы вставлены без полноценного описания setup/metrics;
- блок `Additional computations needed` выглядит как TODO прямо внутри
  диплома;
- нет четкого narrative: сначала point-estimator bias, потом covariance
  estimation, потом limitations/future work;
- недостаточно явно сказано, какие результаты уже завершены и какие только
  планируются.

## Предлагаемая структура главы

```md
= Numerical Experiments

== Goals and scope

== Experimental setup
=== Random finite-state Markovian LSA problems
=== Estimators compared
=== Confidence intervals and metrics

== Main comparison: point-estimator bias and coverage

== Long-run variance estimation: OBM and lugsail
=== Why long-run variance estimation is needed
=== OBM as a Bartlett-window estimator
=== Lugsail / OBM-RR correction
=== Bias-variance experiment

== Discussion and planned extensions
```

В таком порядке читатель сначала понимает экспериментальный вопрос, затем
видит дизайн, затем результаты, и только после этого читает про OBM/lugsail
как отдельный covariance-estimation layer.

## Исправления в тексте

### 1. Переписать opening paragraph

Текущий opening:

> This chapter records the numerical evidence currently available...

Лучше заменить на thesis-style текст:

> This chapter studies the finite-sample behavior of the Richardson--Romberg
> Polyak--Ruppert estimator in the finite-state Markovian LSA model used in
> prior work. The experiments have two goals. First, they test whether
> step-size Richardson--Romberg extrapolation reduces the point-estimator bias
> enough to restore nominal coverage. Second, they examine how practical
> long-run variance estimators, in particular OBM and lugsail OBM, affect the
> resulting confidence intervals.

### 2. Вынести OBM/lugsail motivation после setup

Сейчас OBM объясняется до описания самого эксперимента. Лучше сначала дать:

- finite-state LSA setup;
- methods compared;
- CI construction;
- metrics.

Потом отдельным subsection объяснить, почему для CI нужен estimator of
$\sigma^2(u)$.

### 3. `Additional computations needed` переименовать

В дипломе не стоит оставлять заголовок, который выглядит как внутренний TODO.
Лучшие варианты:

- `Planned experimental extensions`
- `Limitations of the current experiments`
- `Further diagnostics`

И формулировать не как "надо сделать", а как научные ограничения:

> The current experiments leave several finite-sample effects unresolved.
> We record them as planned extensions rather than as assumptions in the
> theoretical results.

### 4. Четче отделить completed vs planned

Добавить фразу перед результатами:

> All numerical values in this section come from the reports
> `reports/2026-04-23_main_comparison.md` and
> `reports/2026-04-23_lugsail_bias_variance.md`.

А перед future diagnostics:

> The following computations have not yet been included in the numerical
> evidence above.

### 5. Уточнить термин "L2"

В таблице сейчас `L2` выглядит не вполне строго. Лучше написать:

> `L2` denotes the median Euclidean error
> $\|\hat\theta-\theta^\star\|_2$, reported in units of $10^{-3}$.

Если `L2` в коде является empirical error over finite trajectories, не называть
его pure bias без оговорки. Лучше:

> L2 error reflects both finite-sample variance and bias, but in the
> constant-stepsize comparisons the large differences are driven mainly by
> bias.

### 6. Не говорить, что lugsail "debiases standard error"

Точнее:

> lugsail reduces the leading window bias of the long-run variance estimator.

Стандартная ошибка получается после квадратного корня, поэтому лучше не писать
"debiases the standard error" буквально.

### 7. Добавить caveat про PSD / clipping

Для lugsail estimator линейная комбинация может быть отрицательной в скалярном
случае или indefinite в матричном. Надо добавить:

> Because OBM-RR is a signed linear combination of two covariance estimators,
> it need not be positive semidefinite in finite samples. The experiments
> therefore track negative or clamped estimates separately.

Если в текущих completed results это еще не tracked, написать это в planned
diagnostics.

## Что еще добавить содержательно

### A. Таблица setup

Добавить компактную таблицу:

```md
| Quantity | Value |
|---|---|
| dimension | $d=5$ |
| states | $10$ |
| problems | $100$ |
| trajectories per problem | $100$ |
| trajectory length | $T=10^6$ |
| RR pair | $(0.2,0.02)$ in completed comparison |
| PR schedule | $c_0=200,k_0=20000,\gamma=0.65$ |
| confidence level | $95\%$ |
| reported projection | random fixed direction / first coordinate, depending on method |
```

Надо проверить по коду, где exactly first coordinate, а где random direction,
чтобы не смешать.

### B. Таблица methods

Отдельно описать methods:

- constant $\alpha=0.2$;
- constant $\alpha=0.02$;
- RR pair;
- diminishing $0.2/\sqrt{k}$;
- PR + OBM;
- PR + MSB;
- RR + OBM;
- RR + MSB;
- RR + OBM-RR.

Текущая таблица результатов не объясняет, чем они отличаются.

### C. Один абзац про expected behavior

Перед results добавить expected finite-sample picture:

- constant large $\alpha$: fast mixing but large stationary bias;
- constant small $\alpha$: smaller bias but slower effective mixing;
- RR: cancels leading $\alpha$ bias while keeping constant-step benefits;
- PR/diminishing: asymptotically unbiased but can have wider intervals;
- OBM/lugsail: affects width/coverage through variance estimation, not through
  point-estimator bias.

### D. Figure suggestions

Если есть время, добавить в PDF 2--3 графика:

1. Coverage boxplot across problems for main methods.
2. L2 error boxplot across problems for main methods.
3. OBM vs OBM-RR relative bias as a function of block size $b$ at $T=10^5$.

Самый важный график для lugsail:

> relative bias vs block size for OBM and OBM-RR, showing that OBM-RR is near
> zero over a wider block-size range.

### E. Короткая future-theory paragraph

В конце главы:

> The experiments suggest that lugsail corrections can materially improve
> variance estimation in short-horizon regimes, but the present thesis does
> not prove consistency or a Berry--Esseen theorem with a data-driven lugsail
> variance estimator. A natural next step is to prove an OBM/lugsail expansion
> for RR-averaged constant-stepsize LSA, separating window bias, centering
> bias, and the residual SA transient.

## Что еще посчитать

### P0. Theory-aligned RR pair

Текущий сильный comparison использует $(0.2,0.02)$, а теория в дипломе
написана для $(\alpha,2\alpha)$.

Нужно посчитать хотя бы:

- $(0.02,0.04)$;
- $(0.05,0.10)$;
- $(0.10,0.20)$, если stability diagnostics позволяют.

Метрики:

- L2 error;
- coverage;
- CI width;
- divergence / instability count;
- comparison with single $\alpha$ and $2\alpha$.

### P0. Oracle variance intervals

Для finite-state setup можно вычислять analytic $\sigma^2(u)$. Нужно сравнить:

- RR + oracle variance;
- RR + OBM;
- RR + OBM-RR;
- RR + MSB.

Это покажет, что именно портит coverage:

- CLT approximation;
- point-estimator bias;
- variance-estimator bias.

### P1. Coverage over trajectory length

Запустить:

$$
T \in \{2\cdot 10^4, 5\cdot 10^4, 10^5, 3\cdot 10^5, 10^6\}.
$$

Цель: показать, где lugsail помогает, а где становится neutral.

### P1. Block-size sweep

Для каждого $T$:

$$
b = \lfloor T^\eta \rfloor,\qquad \eta \in \{0.3,0.4,0.5,0.6,0.7,0.8\}.
$$

Для OBM и OBM-RR:

- variance-estimator relative bias;
- CI width;
- coverage;
- MSE of variance estimator;
- negative/clamped estimate rate.

### P1. Burn-in sweep

Для deterministic-start story:

$$
n_0 \in \{0,\; c_1(\alpha a)^{-1}\log n,\;
c_2(\alpha a)^{-1}\log^2 n\}.
$$

Цель: показать, насколько теоретический burn-in window консервативен на
практике.

### P2. Stress tests

Варьировать:

- mixing rate of the Markov chain;
- minimum real part / spectral gap of $\bar A$;
- noise amplitude;
- dimension $d$.

Цель: показать, когда RR начинает проигрывать из-за variance inflation or
stability issues.

## Приоритетный порядок правок

1. Перестроить главу по структуре `Goals -> Setup -> Methods -> Results ->
   OBM/lugsail -> Limitations`.
2. Убрать формулировки в стиле working note.
3. Добавить setup table and methods table.
4. Заменить `Additional computations needed` на `Planned experimental
   extensions`.
5. Добавить caveats: OBM/lugsail theory not proved here, OBM-RR may be
   non-PSD/negative, current RR pair differs from theorem pair.
6. После новых запусков заменить planned table на actual results and move
   remaining diagnostics to future work.

## Checklist

### Text restructuring

- [x] Rename the chapter to a simpler thesis-style title, e.g.
  `Numerical Experiments`.
- [x] Replace the opening paragraph with a clear `Goals and scope` subsection.
- [x] Move OBM/lugsail motivation after the experimental setup.
- [x] Add an `Experimental setup` subsection.
- [x] Add a compact setup table.
- [x] Add a methods table explaining each estimator.
- [x] Add a metrics paragraph defining L2 error, CI width, and coverage.
- [x] Clarify whether each reported coverage metric uses the first coordinate
  or a random scalar direction.
- [x] Replace `Additional computations needed` with
  `Planned experimental extensions` or `Limitations`.
- [x] Rewrite planned computations as limitations/future diagnostics, not as
  internal TODO notes.

### OBM and lugsail wording

- [x] State explicitly that OBM estimates long-run variance, not marginal
  variance.
- [x] Explain that OBM is the Bartlett-window / spectral-density-at-zero
  estimator in batch-means form.
- [x] Say that lugsail reduces the leading window bias of the variance
  estimator.
- [x] Avoid saying that lugsail directly debiases the standard error.
- [x] Emphasize that RR in `alpha` and RR/lugsail in `b` target different
  bias sources.
- [x] Add caveat that OBM/lugsail theory for RR-averaged constant-step LSA is
  not proved in the thesis.
- [x] Add caveat that OBM-RR/lugsail covariance estimates may be negative or
  non-PSD in finite samples.

### Current results presentation

- [x] Cite the two completed reports as the source of numerical values:
  `reports/2026-04-23_main_comparison.md` and
  `reports/2026-04-23_lugsail_bias_variance.md`.
- [x] Keep the main comparison table but add explanatory text before it.
- [x] Add a sentence that L2 error reflects both finite-sample variance and
  bias.
- [x] Add a short expected-behavior paragraph before the table.
- [x] Add a separate paragraph interpreting why constant-step single-alpha
  branches undercover.
- [x] Add a separate paragraph interpreting why lugsail is neutral at
  `T=10^6`.
- [x] Add a separate paragraph interpreting why lugsail helps in the
  short-horizon bias-variance experiment.

### Figures to add later

- [ ] Coverage boxplot across problems for main methods.
- [ ] L2 error boxplot across problems for main methods.
- [ ] OBM vs OBM-RR relative bias curve as a function of block size.
- [ ] OBM vs OBM-RR MSE curve as a function of block size.
- [ ] Optional: CI width vs coverage scatter plot for the main methods.

### Additional computations

- [ ] Run theory-aligned RR pair `(0.02, 0.04)`.
- [ ] Run theory-aligned RR pair `(0.05, 0.10)`.
- [ ] Run theory-aligned RR pair `(0.10, 0.20)` if stability diagnostics allow.
- [ ] Compare theory-aligned pairs against the current Huo-style pair
  `(0.02, 0.20)`.
- [ ] Add oracle-variance intervals using analytic `sigma^2(u)`.
- [ ] Compare RR + oracle variance with RR + OBM, RR + OBM-RR, and RR + MSB.
- [ ] Run coverage sweep over
  `T in {2e4, 5e4, 1e5, 3e5, 1e6}`.
- [ ] Run block-size sweep
  `b = floor(T^eta)` for several `eta`.
- [ ] Track negative or clamped lugsail estimates.
- [ ] Run burn-in sweep over `n_0`.
- [ ] Run initialization sweep over `theta_0` and initial law of `Z_0`.
- [ ] Run stress tests for slower mixing chains.
- [ ] Run stress tests for worse conditioning of `Abar`.
- [ ] Run stress tests for larger noise amplitude.
- [ ] Check several random scalar directions per problem.
- [ ] Check full covariance matrix diagnostics and PSD behavior.

### Final polish

- [ ] Replace future-looking planned diagnostics with actual results when the
  new runs are complete.
- [ ] Move any remaining uncomputed diagnostics to a short `Future work`
  paragraph.
- [ ] Make theorem/experiment notation consistent: `n`, `T`, `m`, `b`,
  `n_0`, `alpha`.
- [ ] Rebuild `main.typ`.
- [ ] Review the generated PDF page layout for wide tables.

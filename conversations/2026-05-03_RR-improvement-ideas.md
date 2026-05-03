# Идеи по улучшению RR Berry–Esseen относительно Samsonov 2025

## Контекст

Текущий результат секции 4 (`src/pr_weights.typ`, "Smoothing Assembly", 2026-05-03):

$$
d_K\!\left(\frac{\sqrt n\,u^\top(\bar\theta_n^{\mathrm{RR}}-\theta^*)}
              {\sigma_n^{\mathrm{RR}}(u)},N(0,1)\right)
\le C(u)\,\mathrm{polylog}(n)\,n^{-1/4}
$$

при $\alpha=cn^{-1/2}$, $p=\lceil\log n\rceil$, burn-in $n_0=\lceil 2\log n/(\alpha a)\rceil$.

Это **rate-equivalent** Samsonov 2025 Theorem 1 (diminishing PR, $\alpha_k\asymp k^{-3/4}$),
но в режиме *постоянного* шага. Цена — 2× compute (две сцепленные траектории).

Ниже — список направлений, по которым можно либо (а) усилить уже доказанное,
либо (б) попытаться обойти Samsonov по rate. Отсортированы по убыванию реализуемости.

---

## A. Внутри текущего фреймворка: усилить, не меняя стратегию

### A1. RR-coupling на $J^{(2)}+H^{(2)}$ (open thread, Section 4.9)

**Суть.** В текущем доказательстве `<thm:misadjustment>` пишет
$\|T_n^{(2)}\|_{L_p}+\|T_n^{(H)}\|_{L_p}\lesssim\sqrt n\alpha^{3/2}$,
*суммируя* Levin Prop 8/9 раздельно для шагов $\alpha,2\alpha$ — то есть
RR-разность лечится через triangle inequality с константой $|c_\alpha|+|c_{2\alpha}|=3$.
Никакой межшаговой корреляции $J_k^{(2,\alpha)}\leftrightarrow J_k^{(2,2\alpha)}$
не используется.

**Что доказать.** Чистый RR-аналог Levin Prop 8/9:

$$
\|2 J_k^{(2,\alpha)} - J_k^{(2,2\alpha)}\|_{L_p}
\le D_J^{\mathrm{RR}}\, t_{\mathrm{mix}}^{5/2}\, p^{7/2}\, \alpha^{2}\,\log^{3/2}(1/(\alpha a)),
$$

то есть переход $\alpha^{3/2}\to\alpha^2$ за счёт RR-cancellation.
Аналогично для $H^{(2)}$ (с фактором $d^{1/q}$).

**Ожидаемая отдача.**
$\sqrt n\alpha^{3/2}\to\sqrt n\alpha^2 = n^{-1/2}$ при $\alpha=cn^{-1/2}$ — misadjustment
становится **strictly subleading** к мартингальной BE. Однако ведущий $\log^{3/4}(n)\,n^{-1/4}$
от мартингальной части остаётся. **Rate не улучшится**, но улучшатся константы и
качество `polylog` (логи на $\sqrt\alpha$, $(\alpha n)^{-1/2}$ и $\Phi(p,\alpha)n^{-1/2}$
останутся, но будут легче).

**Блокеры.** Для $J^{(2)}$ нужно сцепить две рекурсии глубины 2 на одной траектории
$\{Z_k\}$ и показать, что разность $2 J^{(2,\alpha)} - J^{(2,2\alpha)}$ имеет
дополнительный фактор $\alpha$ в норме. Это та же проблема, что у наивного
RR-kernel route для $\nabla_\alpha S_n$ в `src/last_iterate.typ` — там она не
закрылась. Для $J^{(2)}$ структура чуть другая ($\alpha\tilde A J^{(1)}$ вместо
$\alpha\tilde A J^{(0)}$), может оказаться легче. Стоит попробовать.

**Реализуемость.** Средняя. Прямое расширение того, что уже есть.

### A2. Уточнить $C_{K,1}(u)$ через explicit Bolthausen constants

**Суть.** В `<thm:M-RR-BE>` константа $C_{K,1}(u)$ оценена сверху через универсальные
$L_B,C_1,C_2$ из Bolthausen 1982 + Fan 2019. Эти константы в их работах
не оптимизировались.

**Что сделать.** Перепроверить Lemma 21 Samsonov 2025 (Bolthausen + Fan)
для частного случая ограниченных приращений и постоянного $\sigma_l^2$, попробовать
получить explicit numeric prefactor вместо абстрактных $L_B,C_1,C_2$.

**Ожидаемая отдача.** Константы в численном эксперименте,
не rate-level.

**Реализуемость.** Высокая, но низкая отдача.

### A3. Расширить на функцию направлений (multivariate Berry–Esseen)

**Суть.** Текущий результат — для скаляра $u^\top(\bar\theta_n^{\mathrm{RR}}-\theta^*)$.
Для уверенностных эллипсоидов нужна $d$-мерная версия.

**Что доказать.** Multivariate $d_K$ или $d_{TV}$ против $N(0,\Sigma_\infty)$,
например через Götze (1991) для мартингалов или Raič (2019) для convex sets.

**Ожидаемая отдача.** Применимость для inference на multidimensional CIs,
но rate в multivariate case обычно $n^{-1/2}/\sqrt d$ или $d^{1/4}n^{-1/4}$ —
не улучшение, а расширение применимости.

**Реализуемость.** Средняя. Стандартная техника.

### A4. Non-stationary start с явным $R_{\mathrm{burn}}$

**Суть.** `<thm:RR-BE>` стартует со стационарного augmented chain. Для
произвольного $(\xi,\theta_0)$ нужен Wasserstein-coupling.

**Что доказать.** $R_{\mathrm{burn}}(n_0,\alpha)\le C\rho^{n_0}(1+\|\theta_0-\theta^*\|)/\sqrt n$
через Levin Prop 1, и формальное вложение в Theorem.

**Ожидаемая отдача.** Чистая теорема для practical use; rate тот же.

**Реализуемость.** Высокая. Шаблон из Samsonov 2025 Section 4.

---

## B. Edgeworth-поправки: возможный rate-выигрыш

### B1. Edgeworth для martingale BE через Bolthausen formal expansion

**Суть.** Bolthausen 1982 даёт one-term Edgeworth для мартингалов под
условиями на третий момент. Если для $u^\top M_n^{\mathrm{RR}}$ выписать
formal Edgeworth-поправку, ведущий $n^{-1/4}$ заменится на одну из:

- $n^{-1/2}$, если $\mathbb E[(\Delta M_l^{\mathrm{RR}})^3]/\sigma_l^3$ суммируется в $O(1/\sqrt n)$;
- что-то промежуточное.

**Что доказать.** $L_p$-контроль третьего conditional moment приращений
$\Delta M_l^{\mathrm{RR}}$, и явное выписывание $E_n(x)\phi(x)$-поправки.

**Ожидаемая отдача.** Если получится, **rate $n^{-1/2}$ — strictly better** Samsonov.
Это будет настоящий novel result.

**Блокеры.**
1. Bolthausen 1982 даёт Edgeworth только при дополнительных moment conditions,
   не во всех случаях $n^{-1/4}\to n^{-1/2}$ возможен.
2. Conditional skewness $\mathbb E[(\Delta M_l)^3|\mathcal F_{l-1}]$ для нашего
   $\Delta M_l^{\mathrm{RR}}=\mathcal Q_l^{\mathrm{RR}}(\hat\varepsilon(Z_l)-\mathsf Q\hat\varepsilon(Z_{l-1}))$
   зависит от трёхмерного Poisson-kernel; нужно проверить, что он суммируется.

**Реализуемость.** Средняя-высокая по технике, но идея амбициозная.
Если получится — главный результат секции.

### B2. Stein–Tikhomirov с RR-cancellation в characteristic function

**Суть.** Альтернатива Bolthausen — Stein method для мартингалов
(Pickett-Berry 2004, Röllin 2018). Для RR-схемы char.func. имеет вид
$\mathbb E[e^{itu^\top(\bar\theta_n^{\mathrm{RR}}-\theta^*)\sqrt n/\sigma}]$
с $u^\top$ выраженным через сумму $\mathcal Q_\ell^{\mathrm{RR}}\varepsilon(Z_\ell)$.

**Что попробовать.** Раскрыть char.func. в порядке $t^3$ (или $t^4$), показать,
что RR-разность кубических кумулянтов исчезает либо имеет дополнительный $\alpha$.
Тогда после inversion $\to n^{-1/2}$ или быстрее.

**Ожидаемая отдача.** Аналогична B1 — rate $n^{-1/2}$.

**Блокеры.** Stein method для time-inhomogeneous мартингалов сложнее.
Известные применения для LSA — единичные (Anastasiou et al., но они для
diminishing step).

**Реализуемость.** Низкая-средняя. Высокий риск, но и высокая отдача.

---

## C. Альтернативные RR-конструкции

### C1. Higher-order Richardson: $\alpha,2\alpha,3\alpha$ или геометрическая сетка

**Суть.** RR с тремя или более шагами через Lagrange-веса даёт cancellation
не только $O(\alpha)$ bias, но и $O(\alpha^2)$. То есть в misadjustment
$\mathbb E_\pi[J^{(1)}]=\alpha\Delta_1+\alpha^2\Delta_2+\dots$, RR-3 уберёт
оба ведущих члена.

**Что доказать.**
- Lagrange-веса для трёх шагов: например $(c_1,c_2,c_3)=(3,-3,1)$ для
  $\alpha,2\alpha,3\alpha$ (Vandermonde inversion);
- $L_p$-bound на тройной кросс-RR разности $J^{(1)}$, $J^{(2)}$, $H^{(2)}$;
- variance comparison: дисперсия может *увеличиться* (Lagrange-веса с
  большой нормой), нужна проверка.

**Ожидаемая отдача.** Misadjustment $\sqrt n\alpha^3$ при working scale
$\alpha=cn^{-1/2}$ даёт $n^{-1}$ — **strictly subleading** даже с учётом
неулучшаемой мартингальной BE. *Rate не изменится*, но constants и
логи станут лучше.

**Цена.** 3× compute, риск увеличения дисперсии (Lagrange-norm).

**Реализуемость.** Средняя. Прямое расширение текущей техники.

### C2. Multilevel Monte Carlo (MLMC)–style RR

**Суть.** Вместо двух траекторий $\alpha,2\alpha$ — каскад
$\alpha,2\alpha,4\alpha,\dots,2^L\alpha$ с весами, дающими телескопическое
сокращение bias. Аналогично MLMC Giles 2008.

**Что попробовать.** Optimal allocation $n_l$ траекторий на уровне $l$ для
минимизации MSE при фиксированном compute $\sum 2^l n_l$.

**Ожидаемая отдача.** Если bias $O(\alpha^L)$ при $L=\log n$, то
$\sqrt n\alpha^L\to n^{-(L-1)/2}$ — exponentially small misadjustment.
Но мартингальная BE остаётся $n^{-1/4}$, поэтому **rate не улучшается**.

**Блокеры.** Compute растёт как $\sum 2^l = O(2^L)=O(n)$ — быстрее, чем 1×.
Может оказаться невыгодно.

**Реализуемость.** Низкая. Скорее как теоретическая параллель MLMC.

---

## D. Радикальные направления (вне RR)

### D1. Bootstrap или studentization для $\sigma$

**Суть.** Вместо replace $\sigma_n^{\mathrm{RR}}\to\sigma$ через variance comparison,
использовать **estimator** $\hat\sigma_n$ (например batch-mean или OBM из
`code/inference.py`). Получить self-normalized BE, который в практических CI
не требует знания $\sigma$.

**Что доказать.** $|\hat\sigma_n-\sigma_n^{\mathrm{RR}}|\le\dots$ в $L_p$,
plus self-normalized BE Bentkus–Götze 1996 / Shao 2010.

**Ожидаемая отдача.** Practical inference без знания $\Sigma_\infty$;
rate того же порядка.

**Реализуемость.** Высокая. Стандартная техника, но требует аккуратного
сцепления с уже доказанной BE.

### D2. Concentration of $\bar\theta_n^{\mathrm{RR}}$ как complement к BE

**Суть.** Для practical CI часто bound concentration более полезен, чем BE.
Доказать $\mathbb P(\|\bar\theta_n^{\mathrm{RR}}-\theta^*\|>t)\le 2\exp(-cnt^2)$
с явной зависимостью от $\alpha,t_{\mathrm{mix}}$.

**Ожидаемая отдача.** Параллельный результат, не улучшение BE как такового,
но complementary для high-probability CIs.

**Реализуемость.** Высокая. Через chaining + Markov concentration уже
имеющейся в Section 2.2.

---

## Ранжирование по приоритету

| Направление | Реализуемость | Отдача | Стоит ли начинать |
|---|---|---|---|
| A4 (non-stationary start) | высокая | clean theorem | да, быстро |
| A1 (RR-coupling на $J^{(2)}+H^{(2)}$) | средняя | strictly subleading misadjustment | да, продолжение Section 4.9 |
| C1 (Richardson-3) | средняя | clean subleading | возможно |
| D1 (studentization) | высокая | practical CI | да, для applications part |
| **B1 (Edgeworth для martingale)** | **средняя** | **rate $n^{-1/2}$, novel** | **главная цель если хватит сил** |
| A2 (constants) | высокая | косметика | в конце |
| A3 (multivariate) | средняя | расширение | если успею |
| B2 (Stein–Tikhomirov) | низкая | rate $n^{-1/2}$ | high-risk thread |
| C2 (MLMC) | низкая | logs only | теоретический интерес |
| D2 (concentration) | высокая | complementary | в конце |

---

## Минимальный roadmap для defence

Если цель — **rate-equivalent с Samsonov + чистая постановка**:
- A4 (non-stationary), A1 (subleading misadjustment), D1 (studentization).
- Это даст полный пакет для constant-step inference, не превосходящий Samsonov,
  но впервые в постоянном шаге.

Если цель — **rate-better Samsonov**:
- B1 (Edgeworth для мартингала). Single direction, but high payoff.
- Если получится — переписать абстракт.

Текущая формулировка thesis совместима с обоими маршрутами: Section 4 закрыта
как "constant-step BE matching the diminishing-step rate of Samsonov 2025"; любые
из этих направлений добавляются как Section 5+.

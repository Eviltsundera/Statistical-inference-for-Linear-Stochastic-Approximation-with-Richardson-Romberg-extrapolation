# Почему в финальной оценке пропадает зависимость от $\alpha$

## Question

Почему в некоторых финальных оценках для RR-весов и RR-misadjustment исчезает
явная зависимость от шага $\alpha$?

## Short Answer

Зависимость от $\alpha$ не исчезает как свойство задачи. Она компенсируется
при суммировании по геометрическому хвосту. У постоянного шага

$$
B_\alpha = I-\alpha\bar A
$$

эффективная длина памяти равна

$$
\frac{1}{\alpha a},
$$

потому что

$$
\|B_\alpha^m\| \lesssim (1-\alpha a)^{m/2}.
$$

Поэтому локальный множитель $\alpha$ или $\alpha^2$, полученный из
Richardson--Romberg telescope identity, может быть полностью съеден суммой по
примерно $1/\alpha$ релевантным лагам.

Иными словами: RR дает локальный gain по $\alpha$, но геометрический слой, на
котором ядро существенно отлично от нуля, становится длиннее как $1/\alpha$.
В финальной summed bound эти два эффекта могут взаимно компенсироваться.

## Example 1: Total Variation of RR PR Weights

В `src/pr_weights.typ` для $k=n-l$ получена точечная оценка

$$
\left\|
  \mathcal Q_{l+1}^{\mathrm{RR}}-\mathcal Q_l^{\mathrm{RR}}
\right\|
\le
C\alpha^2 (k-1)(1-\alpha a)^{(k-2)/2}.
$$

Здесь $\alpha^2$ выглядит как RR-gain: один $\alpha$ уже есть в дискретной
производной PR-веса, второй приходит из разности

$$
B_\alpha^{k-1}-B_{2\alpha}^{k-1}
=
\alpha\bar A
\sum_i B_\alpha^{i-1}B_{2\alpha}^{k-1-i}.
$$

Но total variation требует суммировать по $k$:

$$
\sum_{k\ge 2}
\alpha^2 (k-1)(1-\alpha a)^{(k-2)/2}.
$$

Если $r=\sqrt{1-\alpha a}$, то

$$
\sum_{m\ge 0}(m+1)r^m
=
\frac{1}{(1-r)^2}
\asymp
\frac{1}{(\alpha a)^2}.
$$

Значит

$$
\alpha^2
\sum_{m\ge 0}(m+1)r^m
\asymp
\alpha^2\frac{1}{\alpha^2a^2}
=
\frac{1}{a^2}.
$$

Именно поэтому итоговая оценка имеет вид

$$
\sum_l
\left\|
  \mathcal Q_{l+1}^{\mathrm{RR}}-\mathcal Q_l^{\mathrm{RR}}
\right\|
\le
\frac{C}{a^2},
$$

без явного $\alpha$. Это не означает, что RR дал глобальный фактор $\alpha$ в
Abel-остатке. Это означает, что локальный RR-gain ровно компенсирован длиной
геометрического хвоста.

## Example 2: RR Kernel for the Centered First-Order Term

В `src/last_iterate.typ` RR kernel-difference bound имеет вид, если
$m=n-k$,

$$
\|F_l^{\mathrm{RR}}\|_\infty
\le
C\alpha^2 m(1-\alpha a)^{(m-1)/2}.
$$

При future-centered оценке приходится:

1. суммировать квадраты по $l=1,\dots,m$;
2. затем суммировать по $k$, то есть по $m$.

Это дает variance proxy масштаба

$$
\sum_m
\alpha^4 m^3(1-\alpha a)^{m-1}
\lesssim
\alpha^4\frac{1}{(\alpha a)^4}
=
\frac{1}{a^4}.
$$

После извлечения корня получается ведущий член порядка $1/a^2$, снова без
явного $\alpha$. Поэтому leading bound на $U_M^{\mathrm{RR}}$ получается

$$
\|U_M^{\mathrm{RR}}\|_{L_p}
\le
C p^{3/2}t_{\mathrm{mix}}^{1/2}\|\varepsilon\|_\infty/a^2,
$$

а не с дополнительным убыванием по $\alpha$.

## What This Means for the Final Theorem

Есть три разных явления, которые не нужно смешивать.

Первое: в master-bound зависимость от $\alpha$ сохраняется в отдельных
остатках, например

$$
\frac{1}{\sqrt n\,\alpha a},\qquad
\sqrt n\,\alpha^2,\qquad
\sqrt n\,\alpha^{3/2},\qquad
\sqrt\alpha,\qquad
(\alpha n)^{-1/2}.
$$

Второе: в working corollary мы подставляем

$$
\alpha=c n^{-1/2}.
$$

После этой подстановки явная $\alpha$ превращается в степени $n$ и константу
$c$. Поэтому в финальном rate обычно остается только

$$
\mathrm{polylog}(n)n^{-1/4}.
$$

Третье: некоторые intermediate summed bounds действительно становятся
uniform-in-$\alpha$, например

$$
\sum_l
\|\mathcal Q_{l+1}^{\mathrm{RR}}-\mathcal Q_l^{\mathrm{RR}}\|
\le
C/a^2.
$$

Это не улучшение по $\alpha$, а результат компенсации:

$$
\text{local RR gain in }\alpha
\quad \times \quad
\text{memory length }1/\alpha
\quad \approx \quad
\text{constant}.
$$

## Unresolved Gap

Если нужна строгая subleading оценка misadjustment, например

$$
O(\sqrt n\,\alpha^2)
$$

вместо текущего уровня, то простой deterministic kernel-difference route
недостаточен. Он видит только разность $B_\alpha^m-B_{2\alpha}^m$, но не
использует достаточно сильно стохастическое coupling двух траекторий на общей
цепи $(Z_k)$. Для улучшения нужен другой инструмент: depth-two RR cancellation
на $J^{(2)}+H^{(2)}$, либо более тонкая coupling/decoupling-оценка.

## Can We Remove the Assumption $\alpha=c n^{-1/2}$?

Да, из основного finite-$n$ утверждения это предположение можно убрать.
Правильная структура такая:

1. основной theorem оставляет $\alpha$ свободным и выписывает все
   $\alpha$-зависимые остатки;
2. отдельный working corollary подставляет $\alpha=c n^{-1/2}$, потому что
   это балансирует конкурирующие остатки и дает clean rate
   $\mathrm{polylog}(n)n^{-1/4}$.

В текущей сборке `src/pr_weights.typ` master-bound уже почти в такой форме.
При centered initialization $\theta_0=\theta^*$ его остаточная часть имеет
вид, с точностью до problem constants и polylog factors,

$$
d_K
\lesssim
\frac{\log^{3/4} n}{n^{1/4}}
+
\frac{\log n}{\sqrt n}
+
\frac{1}{\sqrt n}
+
\sqrt n\,\alpha^2
+
\sqrt n\,\alpha^{3/2}\,\mathrm{polylog}\!\left(\frac{1}{\alpha a}\right)
+
\sqrt\alpha\,\mathrm{polylog}(n)
+
\frac{\mathrm{polylog}(n)}{\sqrt{n\alpha}}.
$$

If the normalization is changed from $\sigma_n^{\mathrm{RR}}(u)$ to the
asymptotic $\sigma(u)$, one more variance-comparison term appears:

$$
\frac{1}{n\alpha a}.
$$

Thus the alpha-free theorem should not hide $\alpha$. It should state the
bound with these terms explicitly.

### Conditions for Mere Convergence

Let $\alpha=\alpha_n$. Ignoring logarithmic factors, the
centered-initialization bound converges to zero if

$$
n\alpha_n\to\infty,
\qquad
\sqrt n\,\alpha_n^{3/2}\to0.
$$

Equivalently, for a power law $\alpha_n=n^{-\gamma}$, the admissible window is

$$
\frac13 < \gamma < 1.
$$

The lower side $\gamma<1$ comes from

$$
(n\alpha_n)^{-1/2}\to0,
$$

and the upper side $\gamma>1/3$ comes from the high-order misadjustment term

$$
\sqrt n\,\alpha_n^{3/2}\to0.
$$

With log factors restored, the schematic conditions become

$$
\frac{n\alpha_n}{\mathrm{polylog}(n)}\to\infty,
\qquad
\sqrt n\,\alpha_n^{3/2}\mathrm{polylog}(n,1/\alpha_n)\to0.
$$

One must also keep the finite-step restrictions:

$$
2\alpha_n\le \alpha_\infty,
\qquad
\alpha_n\le \alpha_*(q_n,t_{\mathrm{mix}}),
$$

with $q_n$ the moment parameter used in the Levin depth-two bounds.

### Why $\alpha=n^{-1/2}$ Is Still Special

For $\alpha_n=n^{-\gamma}$, the main $\alpha$-dependent powers are

$$
\sqrt\alpha = n^{-\gamma/2},
\qquad
(n\alpha)^{-1/2}=n^{-(1-\gamma)/2},
\qquad
\sqrt n\,\alpha^{3/2}=n^{1/2-3\gamma/2}.
$$

The first term wants $\gamma$ large, the second wants $\gamma$ small, and the
third wants $\gamma\ge 1/2$ if we want it no larger than $n^{-1/4}$.
Balancing

$$
\sqrt\alpha
\quad\text{and}\quad
(n\alpha)^{-1/2}
$$

gives

$$
\gamma=\frac12.
$$

So $\alpha=c n^{-1/2}$ is not logically required for the theorem, but it is
the unique power scale that preserves the clean leading rate

$$
\mathrm{polylog}(n)n^{-1/4}.
$$

If $\alpha$ is much larger than $n^{-1/2}$, then $\sqrt\alpha$ and
$\sqrt n\,\alpha^{3/2}$ become too large. If $\alpha$ is much smaller than
$n^{-1/2}$, then $(n\alpha)^{-1/2}$ becomes too large.

### Arbitrary Start Versus Centered Initialization

For the full average without burn-in, the deterministic transient is

$$
\frac{\|\theta_0-\theta^*\|}{\sqrt n\,\alpha a}.
$$

At $\alpha=c n^{-1/2}$ this is $O(1)$, so it does not vanish unless
$\theta_0=\theta^*$ or a burn-in transfer is added. Therefore:

- with centered initialization, the optimal working scale
  $\alpha\asymp n^{-1/2}$ is fine;
- with arbitrary initialization and no burn-in, one needs
  $\sqrt n\,\alpha_n\to\infty$ just to kill the deterministic transient;
- with arbitrary initialization and burn-in, the theorem should be rewritten
  using the burned-in weights $Q_{l,n_0}^{(\alpha)}$.

### Recommended Thesis Formulation

The cleanest formulation is:

1. state the master theorem for arbitrary admissible $\alpha$ with all
   $\alpha$-dependent terms visible;
2. add a convergence corollary for any sequence $\alpha_n$ satisfying
   the two-sided window above;
3. keep $\alpha=c n^{-1/2}$ only as the optimal-rate corollary.

This removes the artificial equality assumption while preserving the reason
why the thesis focuses on the $n^{-1/2}$ scale.

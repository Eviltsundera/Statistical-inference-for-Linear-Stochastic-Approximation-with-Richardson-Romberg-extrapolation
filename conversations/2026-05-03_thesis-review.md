# Review of the Current Thesis Draft

Дата: 2026-05-03.

## Вопрос

Проверить текущий текст диплома и отметить математические, структурные и
редакционные места, которые стоит поправить.

Проверены основные файлы: `main.typ`, `src/introduction.typ`,
`src/zeroth_order_rr.typ`, `src/last_iterate.typ`, `src/pr_weights.typ`.
`typst compile main.typ` проходит без ошибок, поэтому замечания ниже относятся
не к синтаксису Typst, а к содержанию и связности текста.

## Главный вывод

Текст уже хорошо отражает текущую исследовательскую нить: RR-веса для
PR-усреднения, сравнение с $\bar A^{-1}$, total variation последовательных
весов и препятствие в первом perturbation term. Но сейчас это скорее набор
сильных proof notes, чем финальная глава диплома. Перед сдачей особенно важно:

1. выровнять обозначения $J^{(0)}$, $H^{(0)}$, $Q_l$ между главами;
2. исправить несколько мест, где утверждение сильнее доказанного;
3. явно отделить доказанные леммы от открытой RR-kernel доработки;
4. добавить main theorem / proof roadmap, иначе введение обещает CLT,
   Berry--Esseen и inference procedure, а основной текст пока доказывает
   только части этого пути.

## Высокий приоритет

### 1. Несогласованность определения $J^{(0)}$ в `zeroth_order_rr.typ`

В `src/zeroth_order_rr.typ:11-23` $J_k^{(0)}$ сначала определяется через
случайный продукт $\Gamma_{l+1:k}$, но дальше в `src/zeroth_order_rr.typ:39-42`
используется как deterministic-product объект

$$
J_n^{(0,\alpha)}
=-\alpha\sum_{j=1}^n (I-\alpha\bar A)^{n-j}\varepsilon(Z_j).
$$

Это два разных объекта. В схеме Samsonov/Levin обычно:

$$
J_k^{(0)}=-\sum_{j=1}^k \alpha_j G_{j+1:k}\varepsilon(Z_j),
$$

где $G$ deterministic, а random-product остаток уходит в $H^{(0)}$ или далее в
misadjustment. Поэтому начало главы лучше переписать так, чтобы $J^{(0)}$
сразу был deterministic-product term. Иначе RR-разность $\tilde J^{(0,\alpha)}$
формально не следует из предыдущего определения.

Связанный симптом: в `src/zeroth_order_rr.typ:22-23` остаток $D_1$ записан с
$\sum J_k^{(0)}$, хотя после выделения leading term $W$ там должен быть
transient plus higher-order/random-product remainder, например $H_k^{(0)}$ или
$R_k$ в выбранной нотации.

### 2. Знак и формулировка Hurwitz-условия

В `src/introduction.typ:12` написано: "$-\bar A$ is Hurwitz (all eigenvalues
have strictly negative real parts)". Это формально можно прочитать правильно
как утверждение о $-\bar A$, но для читателя легко возникает конфликт с
`src/introduction.typ:72-74`, где уже пояснено, что собственные значения
$\bar A$ имеют положительные вещественные части.

Лучше везде писать в одной форме:

$$
-\bar A \text{ is Hurwitz, equivalently }
\operatorname{Re}\lambda(\bar A)>0.
$$

Это особенно важно, потому что Huo et al. используют противоположную
программную конвенцию вида $\theta_{t+1}=\theta_t+\alpha(A\theta_t+b)$.

### 3. Long-run covariance нужно симметризовать или добавить условие

В `src/introduction.typ:88-90` и `src/pr_weights.typ:273-275` long-run
covariance задана как

$$
\Sigma_\varepsilon^{(M)}
=\mathbb E[\varepsilon_0\varepsilon_0^\top]
+2\sum_{\ell\ge 1}\mathbb E[\varepsilon_0\varepsilon_\ell^\top].
$$

Для векторной нереверсивной Markov chain это выражение вообще не обязано быть
симметричным. Стандартная матричная форма:

$$
\Sigma_\varepsilon^{(M)}
=\Gamma_0+\sum_{\ell\ge 1}(\Gamma_\ell+\Gamma_\ell^\top),
\qquad
\Gamma_\ell=\mathbb E_\pi[\varepsilon(Z_0)\varepsilon(Z_\ell)^\top].
$$

Формула с $2\sum \Gamma_\ell$ корректна для скалярных проекций или при
дополнительной симметрии/реверсивности. Так как UGE не означает reversibility,
это место стоит исправить или явно оговорить.

### 4. Перенос bound с $T_n=BJ_n$ на $J_n$ сейчас некорректен

В `src/last_iterate.typ:71-78` сказано, что bound на
$T_n^{(1,\alpha)}=BJ_n^{(1,\alpha)}$ дает такой же bound на
$J_n^{(1,\alpha)}$ up to contraction factor $\|B\|_Q\le 1$.

Это направление неверное: из $\|BJ\|$ нельзя получить $\|J\|$ по contraction
bound. Нужно либо:

1. оставаться с shifted statistic $T_n$ и не утверждать bound на $J_n$;
2. добавить явный uniform inverse bound $\|B^{-1}\|_Q\le C$ для малых
   $\alpha$, и затем писать $J_n=B^{-1}T_n$.

Та же проблема повторяется в `src/last_iterate.typ:316-328`, где bound для
$T_k$ используется для RR-разности $2J_k^{(1,\alpha)}-J_k^{(1,2\alpha)}$.
Там нужны inverse factors $B_\alpha^{-1}$ и $B_{2\alpha}^{-1}$.

### 5. RR-kernel formula в конце `last_iterate.typ` требует перепроверки

В `src/last_iterate.typ:347-360` написано

$$
T_n^{(1,\alpha)}-T_n^{(1,2\alpha)}
=-\sum_k \mathcal H_{k+1}^{RR}\varepsilon(Z_k),
$$

но для RR-combination в misadjustment нужен объект вида

$$
2T_n^{(1,\alpha)}-T_n^{(1,2\alpha)}
$$

или, если возвращаться к $J$, объект

$$
2B_\alpha^{-1}T_n^{(1,\alpha)}
-B_{2\alpha}^{-1}T_n^{(1,2\alpha)}.
$$

Кроме того, по определению $S_n=-\alpha\sum K_\alpha\varepsilon$ и
$T_n=-\alpha S_n$, поэтому $T_n$ несет масштаб $\alpha^2K_\alpha$, а не
масштаб $\alpha K_\alpha$. В текущей формуле kernel содержит множители
$2\alpha$, что выглядит размерностно несовместимым с предыдущим определением
$T_n$.

Так как этот кусок сам помечен как open thread, лучше не оставлять его в форме
"would consequently gain one factor" без отдельной проверенной леммы. Можно
переписать осторожно: "формально ожидается gain, но точная RR-kernel identity
и коэффициенты остаются для отдельной леммы".

### 6. `Sigma_n^RR - Sigma_infinity`: пропущен boundary term

В `src/pr_weights.typ:285-339` определено

$$
\Sigma_n^{RR}
=\frac1n\sum_{l=1}^{n-1}
\mathcal Q_l^{RR}\Sigma(\mathcal Q_l^{RR})^\top.
$$

Если заменить все $\mathcal Q_l^{RR}$ на $\bar A^{-1}$, получится
$\frac{n-1}{n}\Sigma_\infty$, а не $\Sigma_\infty$. Поэтому в разложении
должен появиться дополнительный term $-\Sigma_\infty/n$:

$$
\Sigma_n^{RR}-\Sigma_\infty
=\frac1n\sum_{l=1}^{n-1}
\left[
\mathcal Q_l^{RR}\Sigma(\mathcal Q_l^{RR})^\top-\Sigma_\infty
\right]
-\frac1n\Sigma_\infty.
$$

Он меньше, чем $O((n\alpha a)^{-1})$ при $\alpha a\le 1$, но его нужно либо
добавить в доказательство, либо явно absorbed into the constant.

## Средний приоритет

### 7. Bias order во введении нужно сделать единым

`src/introduction.typ:24-33` одновременно говорит:

- asymptotic bias $=\theta^*+\alpha\Delta+O(\alpha^2)$;
- after two-level RR residual is $O(\alpha^{3/2})$ or higher.

По Levin summary базовая гарантированная форма скорее
$\alpha\Delta+O(\alpha^{3/2})$, а более гладкая power-series версия дает
$O(\alpha^2)$ after two-level RR. Лучше написать:

$$
\mathbb E\theta_\infty^{(\alpha)}
=\theta^*+\alpha\Delta+R(\alpha),
\qquad
\|R(\alpha)\|=O(\alpha^{3/2})
$$

under the Levin assumptions, and then separately:
"under the sharper power-series expansion used by Huo et al.,
$R(\alpha)=O(\alpha^2)$ and multilevel RR cancels successive powers."

Иначе читатель не понимает, какой rate ты реально используешь в BE-плане.

### 8. Ссылки на разделы сейчас хрупкие

Есть несколько ручных ссылок, которые уже не совпадают со структурой:

- `src/introduction.typ:41`: covariance "defined in Section 1.4 below" скорее
  находится в `Key quantities`, то есть не 1.4 после текущих подзаголовков.
- `src/zeroth_order_rr.typ:16`: "cf. Section 4.1 below", хотя глава стоит
  раньше `pr_weights`.
- `src/pr_weights.typ:81`: "controlled in Chapters 2 and 3" звучит так, будто
  полный контроль уже есть, хотя глава 3 сама объясняет потерю cancellation.
- `src/pr_weights.typ:113,123`: ссылки на Sections 4.5/4.2 лучше заменить на
  Typst labels или описательные ссылки.

Рекомендация: завести labels для ключевых секций и ссылаться через них, либо
не использовать номера до финальной сборки структуры.

### 9. `pr_weights.typ` местами звучит кругово

В `src/pr_weights.typ:108-113` сказано: "The CLT identifies the limiting
covariance..." Но CLT как раз еще является целью. Лучше заменить на:
"The target CLT should have covariance..." или "The candidate limiting
covariance is...".

Тогда weight comparison lemma будет выглядеть как подготовительная лемма для
CLT, а не как следствие уже доказанной CLT.

### 10. Нужен точный источник для future-centered bilinear estimate

В `src/last_iterate.typ:256-261` используется weighted
Burkholder/block-coupling estimate for future-centered bilinear Markov sums,
со ссылкой на Samsonov et al. Proposition 9. По текущим summaries Proposition 9
скорее относится к perturbation expansion, а не напрямую к такой inequality.

Если в статье действительно есть ровно такая оценка, стоит указать точную
лемму/номер/форму. Если нет, это надо оформить как отдельную auxiliary lemma
с assumptions, потому что именно этот переход собирает per-$k$ bounds в bound
на $U_M$.

### 11. Осторожнее с "optimal covariance"

В `src/introduction.typ:92-94` утверждение про Hájek--Le Cam optimality сильное.
Если в дипломе нет отдельной LAN/semiparametric argument, лучше смягчить:
"This is the covariance achieved by the averaged linearized recursion and is
the benchmark covariance in the cited LSA inference results." И дать точную
цитату, если хочешь оставить слово optimal.

## Низкий приоритет и редактура

### 12. `main.typ` пока имеет placeholders

В `main.typ:20` стоит `--- ---`, а в `main.typ:24-26` стоит `Your abstract`.
Это нормально для черновика, но перед отправкой бросается в глаза раньше
математики.

### 13. Нет bibliography section

Сейчас ссылки идут в тексте как `(Levin et al., 2025)`, `(Douc et al., 2018)`,
но нет явного списка литературы. Для диплома лучше добавить `.bib`/bibliography
или хотя бы временный References section.

### 14. Некоторые Typst/math notation мелочи

- `src/pr_weights.typ:293`: `$u in RR^d$` лучше заменить на `$u in bb(R)^d$`.
- `src/zeroth_order_rr.typ:108-109`: `$g_i : Z -> bb(R)^d$` лучше заменить на
  `$g_i : sans(Z) -> bb(R)^d$`.
- `src/last_iterate.typ:270-271`: comment `"use sum_m m x^m <= C x^(-2)"`
  лучше заменить на $(1-r)^{-2}$ или $(\alpha a)^{-2}$, потому что сейчас
  $x$ выглядит как сам geometric ratio.
- В proof of `pr_weights.typ` lemma (ii) стоит один раз проверить степени
  $\kappa_Q$: в тексте смешиваются Euclidean and $Q$-operator norms.

## Предлагаемый порядок правок

1. Сначала поправить нотацию `zeroth_order_rr.typ`: deterministic $J^{(0)}$,
   remainder $H^{(0)}/R$, единый $Q_l$.
2. Затем исправить `last_iterate.typ`: убрать неверный перенос через
   contraction, добавить inverse-bound или оставить только shifted statistic,
   переписать RR-kernel paragraph как conjectural/open lemma.
3. Потом поправить covariance definition и boundary term в `pr_weights.typ`.
4. После этого добавить короткий "Main theorem / proof roadmap" перед
   technical lemmas: что уже доказано, что imported from Levin/Samsonov, что
   остается open.
5. В конце пройтись по references, abstract, labels and numbering.

## Unresolved gaps

- Нужно открыть точные PDF-места Samsonov et al. and Levin et al., если текст
  будет оставлять ссылку на Proposition 9 / Proposition 3 / Corollary 6 как
  строгую опору.
- Нужно решить, какой результат диплом реально заявляет: full BE theorem для
  RR-PR average или только proof plan плюс доказанные deterministic RR-weight
  lemmas. Сейчас введение обещает первое, а тело ближе ко второму.

## Update 2026-05-04: Chapter 4, BE Plan, and Improvement Ideas

### Вопрос

Повторно проверить текст диплома, особенно главу 4
`src/pr_weights.typ`, и сверить её с текущим планом
`conversations/2026-05-01_BE-plan-for-RR.md` и списком идей
`conversations/2026-05-03_RR-improvement-ideas.md`.

`typst compile main.typ /tmp/thesis-review-check.pdf` проходит без ошибок.
Замечания ниже относятся к математической строгости и связности результата,
а не к Typst-синтаксису.

### Главный вывод по главе 4

Глава 4 теперь содержит правильный общий маршрут:

$$
W^{\mathrm{RR}}
\to \text{Poisson martingale}
\to \text{quadratic variation concentration}
\to \text{martingale BE}
\to \text{Levin depth-two misadjustment}
\to \text{smoothing assembly}.
$$

Но я бы пока не писал, что "секция 4 закрыта" в строгом смысле. Текущий текст
является хорошим proof skeleton, однако full theorem ещё держится на нескольких
местах, которые нужно поправить до финальной версии.

### 1. Самая серьёзная проблема: смешаны $n_0=0$ и burn-in $n_0>0$

В `src/pr_weights.typ:35-38` явно устанавливается $n_0=0$ and all weights
$Q_l^{(\alpha)}$, $\mathcal Q_l^{\mathrm{RR}}$ are derived for the full average
$n^{-1}\sum_{k=0}^{n-1}\theta_k$.

Позже, в smoothing assembly, `src/pr_weights.typ:1332-1340` вводится burn-in

$$
n_0=\left\lceil \frac{2\log n}{\alpha a}\right\rceil,
$$

and the transient bound in `src/pr_weights.typ:1378-1400` is for the restricted
average over $k=n_0,\ldots,n-1$. Но leading martingale weights, variance proxy,
Poisson remainder and misadjustment are still the $n_0=0$ objects.

For a burned-in PR average

$$
\bar\theta_{n,n_0}^{(\alpha)}
=\frac1{n-n_0}\sum_{k=n_0}^{n-1}\theta_k^{(\alpha)},
$$

the depth-zero weight is not the same:

$$
Q_{\ell,n_0}^{(\alpha)}
=\frac{n}{n-n_0}\,
\alpha\sum_{k=\max(n_0,\ell)}^{n-1}B_\alpha^{k-\ell}
$$

if the statistic is normalized as $-\frac1{\sqrt n}\sum_\ell Q_{\ell,n_0}^{(\alpha)}
\varepsilon(Z_\ell)$. In particular, for $\ell<n_0$ the sum starts at $k=n_0$,
not at $k=\ell$.

**Fix.** Choose one of two clean formulations:

1. stationary theorem with $n_0=0$ everywhere, no deterministic burn-in in the
   main theorem;
2. non-stationary theorem with $n_0>0$, but then rederive the RR-weight bounds,
   Poisson decomposition, variance comparison, and misadjustment for
   $Q_{\ell,n_0}^{(\alpha)}$.

Since the thesis wants practical inference, route 2 is probably the right final
route, but route 1 is the fastest rigorous intermediate theorem.

### 2. Martingale variance proxy uses the wrong index set

In `src/pr_weights.typ:407-420`, the martingale is

$$
M_n^{\mathrm{RR}}=\sum_{\ell=2}^{n-1}\Delta M_\ell^{\mathrm{RR}}.
$$

But `src/pr_weights.typ:268-272` defines

$$
\Sigma_n^{\mathrm{RR}}
=\frac1n\sum_{\ell=1}^{n-1}
\mathcal Q_\ell^{\mathrm{RR}}\Sigma
(\mathcal Q_\ell^{\mathrm{RR}})^\top.
$$

Then `src/pr_weights.typ:760-787` applies the Bolthausen--Fan lemma with
$s_n^2=n\sigma_n^{2,\mathrm{RR}}(u)$. In Samsonov Lemma 21, however,

$$
s_n^2=\sum_i \mathbb E X_i^2
$$

for the actual martingale increments. The actual martingale here starts at
$\ell=2$, so either:

$$
\Sigma_{n,M}^{\mathrm{RR}}
=\frac1n\sum_{\ell=2}^{n-1}
\mathcal Q_\ell^{\mathrm{RR}}\Sigma
(\mathcal Q_\ell^{\mathrm{RR}})^\top
$$

should be used in the martingale theorem, or the martingale should be enlarged
to include an $\ell=1$ increment by introducing $Z_0$.

The current quadratic variation concentration notices the missing term as
$-\pi(h_1)$ in `src/pr_weights.typ:625-630`, but the BE theorem still sets
$s_n^2$ to the larger $n\sigma_n^2$. That is not just a cosmetic boundary:
it changes the object required by Lemma 21. The difference is only $O(1)$ and
will be lower order after normalization, but it must be handled explicitly.

Related: the variance comparison in `src/pr_weights.typ:301-339` still needs
the finite-sum boundary term. Replacing every $\mathcal Q_\ell^{\mathrm{RR}}$
by $\bar A^{-1}$ gives $(n-1)\Sigma_\infty/n$, not $\Sigma_\infty$.
This adds $-\Sigma_\infty/n$ for the $\ell=1,\ldots,n-1$ convention, or
$-2\Sigma_\infty/n$ for the $\ell=2,\ldots,n-1$ martingale convention. It is
lower order but should be in the proof.

### 3. Stationary augmented chain vs. finite recursion from zero

The misadjustment transfer in `src/pr_weights.typ:1068-1157` uses:

- finite recursions with $J_0^{(1,\alpha)}=0$;
- stationary expectations $\mathbb E_\pi[J_\infty^{(1,\alpha)}]$ from Levin
  Proposition 2;
- the phrase "chain is started from stationarity" in
  `src/pr_weights.typ:1069` and `src/pr_weights.typ:1184-1186`.

These are not the same setup. If the augmented chain
$(Z_{t+1},J_t^{(0,\alpha)},J_t^{(1,\alpha)})$ is started from its invariant
law, then $J_0^{(1,\alpha)}$ is generally not zero. If the recursion starts
from $J_0^{(1,\alpha)}=0$ with only $Z_1\sim\pi$, then

$$
\frac1{\sqrt n}\sum_{k=0}^{n-1}
\left(\mathbb E J_k^{(1,\alpha)}
-\mathbb E J_\infty^{(1,\alpha)}\right)
$$

is a missing transient term. It is probably controlled by Levin's Wasserstein
contraction, but it has to appear in Lemma `<lem:T1-bound>` or in a separate
burn-in lemma.

This is the same conceptual issue as point 1: the final theorem currently mixes
stationary proof notation with a non-stationary/burned-in statement.

### 4. Imported Levin bounds need exact hypotheses

The imported statement in `src/pr_weights.typ:1029-1043` says, effectively,
"for every $p\ge2$ and every $q\ge2$". The PDF statement of Levin Proposition 9
has an additional restriction of the form

$$
2\le p\le q/2,
$$

and a step-size upper bound depending on $q$ and $t_{\mathrm{mix}}$.
Therefore the final choice

$$
p=\lceil\log n\rceil,\qquad q=\lceil\log d\rceil
$$

in `src/pr_weights.typ:1205-1206` and `src/pr_weights.typ:1406-1408` is not
valid when $\log d<2\log n$, and is invalid at $d=1$ because $q=0$.

**Fix.** Use something like

$$
q=\max\{2p,\lceil\log(ed)\rceil,2\},
$$

then track the resulting $d^{1/q}$ factor and the $q$-dependent admissible
range for $\alpha$. At $\alpha=c n^{-1/2}$ this should still be fine for large
$n$, but the theorem statement must say so.

Also recheck the exact logarithmic factor in Levin Corollary 6. The current
text uses $\log^{1/p}(1/(\alpha a))$ in `src/pr_weights.typ:1021-1024`; the
PDF extraction should be checked against the original display before this is
made load-bearing.

### 5. Bolthausen--Fan is quoted in a slightly different form

In `src/pr_weights.typ:776-787`, the first term is written as

$$
L_B\frac{(2n+1)\log(2n+1)\kappa^3}{s_n^3}.
$$

Samsonov Lemma 21 states it with a constant $L(\kappa)$ depending on the
increment bound $\kappa$, not necessarily as a universal $L_B\kappa^3$.
The rate is unchanged, but for a thesis proof either quote Lemma 21 exactly
or add a sentence justifying the replacement $L(\kappa)\lesssim L_B\kappa^3$.

### 6. Normalisation perturbation argument has a wrong inequality

In `src/pr_weights.typ:909-919`, the proof of `<cor:M-RR-BE-sigma>` sets
$r=\sigma(u)/\sigma_n^{\mathrm{RR}}(u)$ and claims $r^2\in[1,2]$.
Variance comparison plus the lower bound only gives
$\sigma_n^2\ge\sigma^2/2$; $\sigma_n^2$ can also be larger than $\sigma^2$,
so $r$ need not be at least 1.

The same paragraph also uses a global Lipschitz argument for
$x\mapsto\Phi(rx)$ with constant $1/\sqrt{2\pi}$. A multiplicative perturbation
cannot be bounded by ordinary Lipschitzness uniformly in unbounded $x$ that way.
The later proof in `src/pr_weights.typ:1490-1520` uses the right compact-scaling
argument:

$$
\sup_x|\Phi(x/r)-\Phi(x)|\le C_\Phi |r-1|.
$$

The earlier corollary should be rewritten using the later argument.

### 7. Long-run covariance should be symmetrized or explicitly scalarized

The thesis still defines

$$
\Sigma_\varepsilon^{(M)}
=\mathbb E[\varepsilon_0\varepsilon_0^\top]
+2\sum_{\ell\ge1}\mathbb E[\varepsilon_0\varepsilon_\ell^\top].
$$

For non-reversible vector Markov chains the matrix CLT covariance is normally

$$
\Gamma_0+\sum_{\ell\ge1}(\Gamma_\ell+\Gamma_\ell^\top),
\qquad
\Gamma_\ell=\mathbb E_\pi[\varepsilon(Z_0)\varepsilon(Z_\ell)^\top].
$$

The $2\sum\Gamma_\ell$ notation is harmless for scalar quadratic forms only
if one implicitly takes the symmetric part. Since Chapter 4 uses operator
norms, positive definiteness, and square roots $\sigma(u)$, it is safer to
write the symmetrized matrix in `src/introduction.typ` and `src/pr_weights.typ`.

### 8. Status of the BE plan

The updated plan in `conversations/2026-05-01_BE-plan-for-RR.md` is conceptually
the right route. The important correction already made there is correct:
do **not** make the naive RR-kernel estimate for
$\nabla_\alpha S_n^{(\alpha)}$ load-bearing; use Levin depth-two for the base
BE theorem.

But the status line "готова вся секция 4" is too strong. I would downgrade it to:

> The analytic route is complete, and all required blocks have draft
> statements in `src/pr_weights.typ`; before calling it a theorem, the burn-in
> convention, martingale variance index set, finite-start transfer from Levin,
> and exact imported hypotheses must be fixed.

### 9. Ideas for improvement: revised priority

The list in `conversations/2026-05-03_RR-improvement-ideas.md` is mostly
reasonable, but the priority should change.

1. **A4 non-stationary start is not optional.** It is needed to make the theorem
   match the thesis's burned-in PR average. This should be first.

2. **A1 RR-coupling for $J^{(2)}+H^{(2)}$ is useful but not rate-changing.**
   It can make the misadjustment strictly subleading
   ($\sqrt n\alpha^2=O(n^{-1/2})$), but the martingale BE term remains
   $n^{-1/4}$.

3. **B1 Edgeworth should be downgraded from "medium-high" confidence.**
   The current $n^{-1/4}$ rate is driven not only by third moments but by the
   random predictable quadratic variation term
   $\|\langle M\rangle_n-s_n^2\|_{L_p}=O(\sqrt n)$. An Edgeworth expansion of
   martingale increments alone will not automatically yield $n^{-1/2}$ unless
   this bracket fluctuation is also handled more sharply, or one switches to a
   self-normalized/studentized theorem.

4. **D1 studentization is important for the thesis promise.** The introduction
   promises an inference procedure, but Chapter 4 proves a BE bound with
   $\sigma_n^{\mathrm{RR}}(u)$ or $\sigma(u)$ treated as known. A variance
   estimator / studentized CI section is more valuable for the diploma than
   optimizing constants.

5. **C1 higher-order RR is a clean extension, not a BE breakthrough.** It can
   reduce bias and maybe misadjustment, but it increases the RR coefficient norm
   and leaves the leading martingale BE unchanged.

### Recommended next edits

1. Fix Chapter 4 convention: either stationary $n_0=0$ theorem or burned-in
   $Q_{\ell,n_0}$ theorem.
2. Align $\Sigma_n^{\mathrm{RR}}$ with the actual martingale increments
   $\ell=2,\ldots,n-1$, and add the finite-sum boundary in variance comparison.
3. Add the missing finite-start/augmented-stationarity remainder in the
   misadjustment transfer.
4. Correct the imported Levin hypotheses: $p,q,\alpha$ restrictions and the
   exact logarithmic factor.
5. Then add a short theorem roadmap at the start of Chapter 4 explaining which
   parts are original deterministic RR-weight work and which are imported
   from Levin/Samsonov.

## Update 2026-05-04: Applied high/medium priority fixes

### Что поправлено в тексте

Основной ремонт сделан в `src/pr_weights.typ`.

1. **Конвенция $n_0$ больше не смешивается.** Глава 4 теперь явно доказывает
   full-average результат с $n_0=0$. Burn-in не притворяется "малой
   поправкой" к тем же весам: в начале главы добавлена формула для burned-in
   веса

   $$
   Q_{\ell,n_0}^{(\alpha)}
   =\frac{n}{n-n_0}\alpha
     \sum_{k=\max(n_0,\ell)}^{n-1}B_\alpha^{k-\ell}.
   $$

   В текущем theorem deterministic transient остается явным, а working-rate
   corollary использует centered initialization $\theta_0=\theta^*$.

2. **Variance proxy теперь совпадает с martingale increments.** Поскольку
   Poisson martingale начинается с $\ell=2$, теперь

   $$
   \Sigma_n^{\mathrm{RR}}
   =\frac1n\sum_{\ell=2}^{n-1}
     \mathcal Q_\ell^{\mathrm{RR}}
     \Sigma_\varepsilon^{\mathrm M}
     (\mathcal Q_\ell^{\mathrm{RR}})^\top .
   $$

   Из bracket concentration убран лишний boundary term $-\pi(h_1)$. В variance
   comparison добавлен finite-sum boundary $2\Sigma_\infty/n$.

3. **Long-run covariance symmetrized.** Во введении и главе 4 covariance теперь
   записана как

   $$
   \mathbb E[\varepsilon_0\varepsilon_0^\top]
   +\sum_{\ell\ge1}
     \bigl(\mathbb E[\varepsilon_0\varepsilon_\ell^\top]
     +\mathbb E[\varepsilon_\ell\varepsilon_0^\top]\bigr).
   $$

4. **Levin hypotheses corrected.** Proposition 8/9 inputs now require
   $2\le p\le q/2$ and $\alpha\le\alpha_*(q,t_{\mathrm{mix}})$. Working-scale
   choices are changed to

   $$
   p=\lceil\log n\rceil,\qquad
   q=\max\{2p,\lceil\log(e d)\rceil,2\}.
   $$

5. **Finite-start vs stationary Levin transfer is no longer silent.** The
   depth-two misadjustment section now has an explicit stationary-transfer
   convention: Proposition 2 is stationary, while the zero-start recursion
   requires a contraction/startup transfer term.

6. **Bolthausen--Fan and normalization perturbation fixed.** The
   Bolthausen--Fan quote now uses the bounded-increment constant
   $L_B(\kappa)$, and the normalization change
   $\sigma_n^{\mathrm{RR}}(u)\to\sigma(u)$ uses a compact multiplicative
   scaling argument instead of the invalid global Lipschitz shortcut.

7. **Related consistency fixes outside Chapter 4.** In
   `src/zeroth_order_rr.typ`, $J^{(0,\alpha)}$ is now consistently the
   deterministic-product linearized term, and the random-product difference is
   a remainder $R_k^{(\alpha)}$. In `src/last_iterate.typ`, the transfer from
   $T=BJ$ back to $J$ now explicitly uses an inverse bound
   $\|(I-w\bar A)^{-1}\|\le2$.

### Что осталось открытым

The practical arbitrary-start burn-in theorem is still an extension, not a
proved statement in the current chapter. To get it, one must redo Sections
4.1--4.8 with $Q_{\ell,n_0}^{(\alpha)}$ and track the pre-burn-in noise samples
separately. This is now stated explicitly in the thesis instead of being
implicitly assumed.

## Update 2026-05-04: Second Pass After the Fixes

### Вопрос

Ещё раз изучить текст диплома, особенно часть 4 (`src/pr_weights.typ`), и
сверить текущий текст с планом `conversations/2026-05-01_BE-plan-for-RR.md`
и идеями `conversations/2026-05-03_RR-improvement-ideas.md`.

`typst compile main.typ /tmp/thesis-review-check-2026-05-04.pdf` проходит без
ошибок. Ниже только содержательные замечания.

### Короткий вывод

После последних правок часть 4 стала существенно более согласованной:

$$
W^{\mathrm{RR}}
\to \text{Poisson martingale}
\to \langle M\rangle\text{-concentration}
\to \text{martingale BE}
\to \text{Levin depth-two misadjustment}
\to \text{smoothing}.
$$

Но я бы всё ещё не называл результат полностью доказанным для practical
zero-start / burned-in PR average. Текущая глава доказывает скорее
conditional full-average theorem: полный средний $n_0=0$, deterministic
transient вынесен явно, а misadjustment theorem опирается на
"stationary-transfer convention". Именно этот transfer сейчас главный
load-bearing gap.

### 1. Главный оставшийся gap: stationary-transfer convention слишком сильная

В `src/pr_weights.typ:1089-1096` написано, что для zero-start recursion
разница между finite-start centered sums и stationary counterparts даёт

$$
R_{\mathrm{start}}(n,\alpha)
\le C\rho^n(1+\|\theta_0-\theta^*\|)
$$

или становится negligible после logarithmic burn-in. В такой форме это
неочевидно и, скорее всего, неверно для full average $n_0=0$.

Причина: в bound для $T_n^{(1)}$ фактически появляется

$$
\frac1{\sqrt n}\sum_{k=0}^{n-1}
\left(\mathbb E J_k^{(1,\alpha)}
-\mathbb E_{\Pi_\alpha}J_\infty^{(1,\alpha)}\right).
$$

Если Wasserstein contraction даёт покомпонентно $\rho_\alpha^k$ с
$\rho_\alpha\simeq e^{-c\alpha a}$, то сумма даёт порядок

$$
\frac1{\sqrt n}\sum_{k\ge0}\rho_\alpha^k
\asymp \frac1{\sqrt n\,\alpha a}.
$$

При рабочем масштабе $\alpha=c n^{-1/2}$ это $O(1)$, а не
$o(n^{-1/4})$. Значит, есть только два чистых варианта:

1. стартовать augmented chain сразу из stationary law и честно заявить
   stationary theorem;
2. доказывать practical burned-in theorem, но тогда возвращать
   $Q_{\ell,n_0}^{(\alpha)}$ и переделывать Poisson/variance/misadjustment
   bookkeeping для среднего после burn-in.

Центрированная инициализация $\theta_0=\theta^*$ в working corollary убирает
детерминированный transient, но не автоматически убирает transient
augmented-процессов $J^{(0)},J^{(1)}$.

### 2. В `pr_weights.typ` неверно описан порядок depth-one remainder

В `src/pr_weights.typ:30-31` сказано, что

$$
R_k^{(\alpha)}=J_k^{(1,\alpha)}+H_k^{(1,\alpha)}
$$

собирает члены порядка $\alpha^{3/2}$ или выше. Это не соответствует
дальнейшей же главе: $J^{(1)}$ имеет stationary bias
$\alpha\Delta+O(\alpha^2)$ и centered вклад порядка, который приходится
контролировать отдельно через Levin Corollary 6. Только depth-two хвост
$J^{(2)}+H^{(2)}$ имеет моментный порядок $\alpha^{3/2}$.

Лучше написать: "$R_k$ is the first misadjustment remainder; its leading
$J^{(1)}$ component has order $\alpha$ in stationary bias, and the
Berry--Esseen proof below controls it through the depth-two decomposition."

### 3. `zeroth_order_rr.typ` всё ещё смешивает две decomposition conventions

В `src/zeroth_order_rr.typ:11-22` $J^{(0,\alpha)}$ теперь определён как
deterministic-product term, и это хорошо. Но в `src/zeroth_order_rr.typ:36-44`
следующая RR decomposition снова использует random products
$\Gamma_{1:n}^{(\alpha)}$ and undeclared $J^{(1)},H^{(1)}$:

$$
[2\Gamma_{1:n}^{(\alpha)}-\Gamma_{1:n}^{(2\alpha)}](\theta_0-\theta^*)
+[2J^{(0,\alpha)}-J^{(0,2\alpha)}]+\cdots .
$$

После новой convention там должно быть либо

$$
[2B_\alpha^n-B_{2\alpha}^n](\theta_0-\theta^*)
+[2J^{(0,\alpha)}-J^{(0,2\alpha)}]
+[2R^{(\alpha)}-R^{(2\alpha)}],
$$

либо нужно явно вернуться к Levin random-product expansion and define
$J^{(1)},H^{(1)}$ in that convention. Фраза "The first bracket is the
deterministic transient" сейчас математически неверна, потому что
$\Gamma_{1:n}$ random.

### 4. В `last_iterate.typ` осталась старая неверная фраза про contraction

В `src/last_iterate.typ:75-78` написано, что из bound на
$T_n^{(1,\alpha)}=BJ_n^{(1,\alpha)}$ следует такой же bound на $J_n^{(1,\alpha)}$
"up to the contraction factor $\|B\|_Q\le1$". Направление неверное:
из $\|BJ\|$ contraction bound не даёт $\|J\|$.

Ниже, в `src/last_iterate.typ:319-327`, это уже поправлено через
$\|(I-w\bar A)^{-1}\|\le2$. Нужно заменить и вводную фразу:
"the transfer back to $J_n^{(1,\alpha)}$ requires a local inverse bound for
$B$."

### 5. Введение всё ещё завышает Levin bias expansion

`src/introduction.typ:24-33` одновременно говорит:

$$
\mathbb E\theta_\infty^{(\alpha)}
=\theta^*+\alpha\Delta+O(\alpha^2),
$$

и затем что RR residual is $O(\alpha^{3/2})$ or higher. По Levin et al.:
Proposition 2 даёт $O(\alpha^2)$ remainder for
$\mathbb E J_\infty^{(1,\alpha)}$, но Corollary 1 для полного stationary bias
даёт

$$
\mathbb E\theta_\infty^{(\alpha)}
=\theta^*+\alpha\Delta+O(\alpha^{3/2}).
$$

Поэтому во введении лучше разделить:

- Levin baseline: $\alpha\Delta+O(\alpha^{3/2})$ for full stationary bias;
- sharper/power-series route, if citing Huo under its extra assumptions:
  $\alpha\Delta+O(\alpha^2)$ or higher-order expansion.

Иначе reader видит несогласованность: если до RR остаток $O(\alpha^2)$, то
после two-level RR естественно ожидать $O(\alpha^2)$, а не $O(\alpha^{3/2})$.

### 6. Глава 4: что сейчас выглядит корректно

Следующие ранее критичные места сейчас выглядят исправленными:

- $\Sigma_\varepsilon^{(M)}$ симметризована во введении и в главе 4.
- $\Sigma_n^{\mathrm{RR}}$ теперь суммируется по $\ell=2,\ldots,n-1$, как и
  martingale increments.
- finite-sum boundary $2\Sigma_\infty/n$ добавлен в variance comparison.
- Bolthausen--Fan quote теперь использует $L_B(\kappa)$, что ближе к Samsonov
  Lemma 21.
- нормализация $\sigma_n^{\mathrm{RR}}\to\sigma$ больше не использует
  неверный global Lipschitz argument.
- ограничения Levin Proposition 9 теперь учитывают $2\le p\le q/2$ and
  $\alpha\le\alpha_*(q,t_{\mathrm{mix}})$.

### 7. Статус плана

План `2026-05-01_BE-plan-for-RR.md` концептуально правильный, но его статус
"готова вся секция 4" нужно читать слабее:

> All analytic blocks for a stationary/full-average RR Berry--Esseen theorem
> are drafted. The practical zero-start or burned-in theorem is still open
> until the startup transfer / burned-in weights are proved.

Особенно устарели места, где текущий результат описан с burn-in
$n_0=\lceil 2\log n/(\alpha a)\rceil$. В тексте thesis сейчас выбран другой
промежуточный вариант: full average $n_0=0$ plus centered initialization in
the working corollary.

### 8. Идеи улучшения: обновлённый приоритет

1. **A4 нужно раздробить на две задачи.**  
   A4a: formal stationary/finite-start transfer for augmented chains.  
   A4b: genuine burned-in PR theorem with $Q_{\ell,n_0}^{(\alpha)}$.  
   Для practical inference важна A4b.

2. **D1 studentization / OBM variance estimator важнее, чем Edgeworth.**  
   Сейчас theorem нормирует на known $\sigma_n^{\mathrm{RR}}(u)$ or
   $\sigma(u)$, а thesis promises confidence intervals. Без variance estimator
   inference story остаётся неполной.

3. **A1 RR-coupling for $J^{(2)}+H^{(2)}$ полезен, но не меняет BE rate.**  
   Он делает misadjustment strictly subleading
   ($\sqrt n\alpha^2=O(n^{-1/2})$), но ведущий martingale BE всё равно
   $n^{-1/4}$.

4. **B1 Edgeworth стоит оставить high-risk.**  
   Улучшение до $n^{-1/2}$ невозможно получить только из third moments:
   текущий $n^{-1/4}$ идёт через random bracket fluctuation
   $\|\langle M\rangle_n-s_n^2\|_{L_p}=O(\sqrt n)$. Нужно либо sharpen
   bracket, либо studentized/self-normalized theorem.

5. **Levin equation (26) может быть лучшим practical route.**  
   Для burned-in theorem можно рассмотреть альтернативу weighted-$Q_\ell$
   proof: идти через Levin representation, где leading term is an unweighted
   Markov sum over the post-burn-in window. Это может быть проще, чем
   переделывать все weighted Poisson lemmas for $Q_{\ell,n_0}$.

### Recommended Next Edits

1. Исправить два явных текстовых противоречия:
   `src/pr_weights.typ:30-31` and `src/introduction.typ:24-33`.
2. Починить convention mismatch in `src/zeroth_order_rr.typ:36-44`.
3. Исправить stale contraction sentence in `src/last_iterate.typ:75-78`.
4. Переформулировать `<thm:RR-BE>` as conditional stationary/full-average
   theorem, либо доказать startup lemma with the correct summed
   $\rho_\alpha^k$ dependence.
5. После этого обновить `2026-05-01_BE-plan-for-RR.md` and
   `2026-05-03_RR-improvement-ideas.md`, чтобы они больше не говорили, что
   current theorem already includes burn-in.

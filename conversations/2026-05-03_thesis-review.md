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

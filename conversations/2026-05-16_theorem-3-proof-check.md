# Проверка текущих доказательств и Теоремы 3

## Вопрос

Проверить текущие доказательства после сокращения раздела 3 и понять, верна ли Теорема 3.

## Короткий ответ

Теорема 3 в `src/pr_weights.typ` сейчас является сборочной smoothing-теоремой:

$$
d_K\!\left(
  \frac{\sqrt n\,u^\top(\bar\theta_n^{(\mathrm{RR},\alpha)}-\theta^*)}
       {\sigma_n^{\mathrm{RR}}(u)},
  \mathcal N(0,1)
\right)
$$

оценивается через martingale Berry--Esseen bound и $L_p$-норму остатка
$u^\top\mathcal R_n^{\mathrm{RR}}$.

Как условная сборочная теорема она в целом верна: если уже доказаны

1. martingale Berry--Esseen для $u^\top M_n^{\mathrm{RR}}$;
2. декомпозиция
   $$
   \sqrt n\,u^\top(\bar\theta_n^{(\mathrm{RR},\alpha)}-\theta^*)
   =
   -\frac{u^\top M_n^{\mathrm{RR}}}{\sqrt n}
   + u^\top\mathcal R_n^{\mathrm{RR}};
   $$
3. положительность $\sigma_n^{\mathrm{RR}}(u)$;
4. $L_p$-контроль остатка,

то применение smoothing inequality корректно.

Но как утверждение для обычного full-average алгоритма с zero-start или
произвольной начальной точкой Теорема 3 пока не доказана. Основная причина:
misadjustment-оценка, на которую она опирается, явно сформулирована только
под stationary augmented-chain convention, а finite-start transfer в тексте
сам объявлен недоказанным.

## Главные замечания

### 1. Главный статус: условно верна, но не как finite-start theorem

В `src/pr_weights.typ` перед misadjustment-теоремой написано, что оценки ниже
надо читать под stationary augmented-chain convention, а для zero-start full
average появляется накопленный startup term порядка

$$
\frac{1}{\sqrt n\,\alpha a}.
$$

Это место находится вокруг строк `1087--1104`.

Затем Теорема 3 (`<thm:RR-BE>`, строки `1459--1479`) использует standing
hypotheses Теоремы `<thm:misadjustment>`. Поэтому формально Теорема 3 тоже
наследует stationary augmented-chain convention.

Вывод:

$$
\text{Теорема 3 верна только как stationary/idealized assembly theorem.}
$$

Ее нельзя продавать как theorem для обычного запуска с deterministic
$\theta_0$ без отдельного transfer lemma или burn-in версии весов.

### 2. В короллариях есть конфликт с этой оговоркой

Королларии после Теоремы 3 говорят, что deterministic transient исчезает при
$\theta_0=\theta^*$ (`src/pr_weights.typ`, строки `1492--1495`,
`1535--1539`). Это верно только для явного deterministic transient
$D_{\mathrm{tr}}^{\mathrm{RR}}$.

Но это не устраняет startup error в misadjustment-части. В тексте выше уже
сказано, что даже для zero-start recursion finite-start transfer не включен.
Поэтому рабочий вывод

$$
d_K \lesssim \mathrm{polylog}(n)\,n^{-1/4}
$$

при $\theta_0=\theta^*$ не следует из текущих доказательств, если не добавить
одну из двух вещей:

1. явно оставить stationary augmented-chain convention и в королларии;
2. доказать finite-start/burn-in transfer.

### 3. Условие на шаг для Levin bounds должно быть сильнее

В Lemma `<lem:T2H-bound>` и Теореме `<thm:misadjustment>` применяются Levin
Propositions 8--9 одновременно при шагах $w\in\{\alpha,2\alpha\}$.
Сейчас условие записано как

$$
\alpha \le \alpha_*(q,t_{\mathrm{mix}}).
$$

Но для применения результата при $2\alpha$ нужно

$$
2\alpha \le \alpha_*(q,t_{\mathrm{mix}})
$$

или переопределить рабочую константу как
$\alpha_*^{\mathrm{RR}}=\alpha_*/2$.

То же замечание касается короллариев, где сейчас стоит
$\alpha_n\le\alpha_*(q_n,t_{\mathrm{mix}})$.

### 4. В telescoping identity для stationary версии не хватает левого boundary term

В строках `1106--1125` выведена identity

$$
\sum_{k=0}^{n-1}
\left(J_k^{(1,\alpha)}-\mathbb E_\pi J_\infty^{(1,\alpha)}\right)
=
-\bar A^{-1}\sum_{k=1}^n \bar\psi_\alpha(J_{k-1}^{(0,\alpha)},Z_k)
-\frac1\alpha\bar A^{-1}J_n^{(1,\alpha)}.
$$

Это верно для finite-time recursion с $J_0^{(1,\alpha)}=0$. Но если дальше
используется stationary augmented-chain convention, то обычно
$J_0^{(1,\alpha)}\ne0$, и telescoping дает boundary

$$
\frac1\alpha \bar A^{-1}\bigl(J_0^{(1,\alpha)}-J_n^{(1,\alpha)}\bigr).
$$

Это не ломает rate: дополнительный $J_0$-терм должен оцениваться так же, как
$J_n$, и дает тот же вклад

$$
\Phi(p,\alpha)n^{-1/2}.
$$

Но доказательство в текущем виде имеет пропущенный boundary term.

### 5. В самой smoothing-сборке Теоремы 3 существенной ошибки не видно

Шаги `1348--1391` и `1481--1487` выглядят корректно:

$$
X_n
=
-\frac{u^\top M_n^{\mathrm{RR}}}
       {\sqrt n\,\sigma_n^{\mathrm{RR}}(u)},
\qquad
Y_n
=
\frac{u^\top\mathcal R_n^{\mathrm{RR}}}
       {\sigma_n^{\mathrm{RR}}(u)}.
$$

Smoothing inequality дает

$$
d_K(X_n+Y_n,\mathcal N)
\le
d_K(X_n,\mathcal N)
+ e\|Y_n\|_{L_p}/\sqrt{2\pi}
+ e^{-p}.
$$

При $p=\lceil\log n\rceil$ остаток $e^{-p}\le n^{-1}$, так что написанное
$e/n$ является допустимым грубым вариантом.

Нужно только явно обеспечить $p\ge2$; например писать

$$
p=\max(2,\lceil\log n\rceil)
$$

или добавить $n\ge3$.

### 6. Техническая проблема Typst-ссылок

PDF компилируется, но многие ссылки в тексте отображаются пустыми:
"Theorem ", "Lemma ", "equation ". Это не математическая ошибка, но сейчас
мешает ревью доказательства. Вероятно, кастомные theorem/lemma block labels
и display-math labels не дают печатного номера через обычный `<label>`.

## Что надо сделать, чтобы Теорема 3 стала честной

Минимальный путь:

1. Переименовать Теорему 3 как stationary augmented-chain theorem.
2. В ее statement явно написать, что рассматривается стационарная версия
   augmented recursion, а не arbitrary/zero-start recursion.
3. Везде заменить условие $\alpha\le\alpha_*$ на $2\alpha\le\alpha_*$ там,
   где используются Levin bounds при шагах $\alpha$ и $2\alpha$.
4. Исправить centered telescoping identity: либо оставить finite-start
   версию и не называть ее stationary, либо добавить boundary
   $(1/\alpha)\bar A^{-1}(J_0-J_n)$ и оценить обе границы.
5. В короллариях с $\theta_0=\theta^*$ не утверждать finite-start result,
   пока не доказан transfer/burn-in lemma.

Альтернативный путь:

1. Доказать отдельный finite-start transfer для full average или burned-in
   average.
2. После этого Теорему 3 можно переформулировать как практически применимую
   theorem для deterministic initialization.

## Итог

Теорема 3 не выглядит ошибочной в smoothing-части. Ошибка не в Bobkov--Götze
сборке, а в области применимости: текущий текст доказывает только условную
stationary augmented-chain версию. Для обычного запуска LSA с deterministic
$\theta_0$ доказательства пока недостаточно.

## План исправлений после shared review

Shared chat добавляет к этому разбору несколько более ранних дефектов. Чинить
лучше не с финальной Теоремы 3, а снизу вверх: сначала локальные оценки и
условия применимости, затем misadjustment, затем smoothing assembly.

## Что значит stationary augmented-chain theorem

В обычной LSA-рекурсии состояние не исчерпывается только текущим $Z_t$.
Для доказательств misadjustment мы вводим дополнительные рекурсивные
переменные:

$$
J_t^{(0,\alpha)},\quad
J_t^{(1,\alpha)},\quad
J_t^{(2,\alpha)},\quad
H_t^{(2,\alpha)},\ldots
$$

Они сами обновляются по Markovian recursion вместе с $Z_t$. Поэтому
естественное марковское состояние для доказательства -- это расширенная
цепочка, например

$$
\mathsf X_t^{(\alpha)}
=
\left(
  Z_{t+1},
  J_t^{(0,\alpha)},
  J_t^{(1,\alpha)}
\right)
$$

или более глубокая версия с $J^{(2)}$ и $H^{(2)}$.

`Stationary augmented-chain theorem` означает:

$$
\mathsf X_0^{(\alpha)}
\sim
\Pi_\alpha^{\mathrm{aug}},
$$

где $\Pi_\alpha^{\mathrm{aug}}$ -- invariant distribution этой расширенной
цепочки. То есть мы не запускаем recursion с

$$
J_0^{(0,\alpha)}=J_0^{(1,\alpha)}=0,
$$

а предполагаем, что все вспомогательные переменные уже имеют свое
стационарное распределение. Тогда можно корректно использовать стационарные
центры вроде

$$
\mathbb E_\pi J_\infty^{(1,\alpha)}
$$

и stationary bias expansion из Levin et al.

Это удобная идеализированная теорема: она проверяет Gaussian approximation
после того, как constant-step recursion уже "разогналась".

Важно: условие $\theta_0=\theta^*$ не делает augmented chain stationary.
Оно убирает только deterministic transient

$$
B_\alpha^k(\theta_0-\theta^*).
$$

Но даже при $\theta_0=\theta^*$ вспомогательные переменные finite-start
версии начинаются с нуля:

$$
J_0^{(0,\alpha)}=J_0^{(1,\alpha)}=0,
$$

тогда как в stationary version они случайны и уже имеют свои equilibrium
fluctuations. Поэтому $\theta_0=\theta^*$ не закрывает startup error в
misadjustment part.

## Что значит finite-start theorem

`Finite-start theorem` -- это теорема для реально запускаемого алгоритма:

$$
Z_0\sim \xi,\qquad
\theta_0 \text{ fixed or distributed independently},
$$

и все perturbation variables стартуют из своих recursion initial values,
обычно

$$
J_0^{(0,\alpha)}
=
J_0^{(1,\alpha)}
=
J_0^{(2,\alpha)}
=
H_0^{(2,\alpha)}
=0.
$$

Если брать full PR average,

$$
\bar\theta_n^{(\alpha)}
=
\frac1n\sum_{k=0}^{n-1}\theta_k^{(\alpha)},
$$

то в среднее попадают ранние нестационарные итерации. Для constant step-size
они затухают геометрически примерно как

$$
\rho_\alpha^k
\approx
\exp(-c\alpha a k),
$$

но при суммировании по $k$ дают накопленный вклад

$$
\frac1{\sqrt n}\sum_{k\ge0}\rho_\alpha^k
\asymp
\frac1{\sqrt n\,\alpha a}.
$$

На рабочем масштабе $\alpha\asymp n^{-1/2}$ это порядок $O(1)$, то есть
не малый остаток. Поэтому full-average finite-start theorem не следует
автоматически из stationary theorem.

## Что значит burn-in theorem

`Burn-in theorem` -- это finite-start theorem, но первые $n_0$ итераций
выбрасываются из среднего:

$$
\bar\theta_{n,n_0}^{(\alpha)}
=
\frac{1}{n-n_0}
\sum_{k=n_0}^{n-1}\theta_k^{(\alpha)}.
$$

Идея: выбрать $n_0$ достаточно большим, чтобы расширенная цепочка успела
приблизиться к своему invariant distribution. Типичный порядок:

$$
n_0
\gtrsim
\left(t_{\mathrm{mix}}+\frac1{\alpha a}\right)\log n,
$$

или, если $t_{\mathrm{mix}}$ поглощается в константы,

$$
n_0
\gtrsim
\frac{\log n}{\alpha a}.
$$

Тогда startup contribution становится полиномиально малым, например
$O(n^{-c})$, и его можно добавить к Berry--Esseen remainder.

Но burn-in меняет deterministic PR weights. Для full average вес был

$$
Q_l^{(\alpha)}
=
\alpha\sum_{k=l}^{n-1}B_\alpha^{k-l}.
$$

Для burn-in среднего становится

$$
Q_{l,n_0}^{(\alpha)}
=
\frac{n}{n-n_0}\,
\alpha\sum_{k=\max(n_0,l)}^{n-1}B_\alpha^{k-l}.
$$

Поэтому burn-in theorem требует заново проверить:

1. pointwise bounds on $Q_{l,n_0}^{\mathrm{RR}}$;
2. total variation bounds for Abel summation;
3. variance comparison;
4. Poisson decomposition remainder;
5. finite-start transfer for misadjustment.

## Разница в одну строку

Stationary augmented-chain theorem:

$$
\text{вся расширенная цепочка уже в равновесии с момента }0.
$$

Finite-start theorem:

$$
\text{алгоритм реально стартует из заданного }(Z_0,\theta_0)
\text{ и }J_0=0.
$$

Burn-in theorem:

$$
\text{алгоритм finite-start, но первые }n_0
\text{ шагов не входят в среднее.}
$$

### Этап 0. Зафиксировать выбранный режим теоремы

Сначала надо выбрать, что именно будет доказано в дипломе.

Минимально безопасный вариант для быстрой правки:

$$
\text{Theorem 3 is a stationary augmented-chain theorem.}
$$

Тогда в statement Теоремы 3 и всех rate-corollaries надо явно написать, что
misadjustment bound применяется для стационарной augmented recursion. Все
утверждения про practical finite-start алгоритм остаются как remark/open
extension.

Более сильный, но существенно более длинный вариант:

$$
n_0 \gtrsim
\left(t_{\mathrm{mix}}+\frac{1}{\alpha a}\right)\log n
$$

и доказать burn-in theorem с весами
$Q_{l,n_0}^{(\alpha)}$. Это потребует переделать weight identities, Poisson
decomposition, variance comparison и misadjustment transfer. Для текущей
версии диплома это лучше вынести в отдельную задачу, а не смешивать с
быстрой правкой Теоремы 3.

Рекомендация: сейчас делать минимально безопасный stationary theorem.

### Этап 1. Исправить zeroth-order RR constants в `src/zeroth_order_rr.typ`

В Section 2.4 после оценки

$$
\|H_j^{(n)}\|
\le
\overline C_A(1-\alpha a)^{(n-j-1)/2}\frac{2}{\alpha a}
$$

в bound для $g_j$ потерян множитель $1/a$. Сейчас написано

$$
\|g_j\|_\infty
\le
4\alpha\,\widetilde C_A\|\epsilon\|_\infty
(1-\alpha a)^{(n-j-1)/2}.
$$

Должно быть

$$
\|g_j\|_\infty
\le
\frac{4\alpha}{a}\,\widetilde C_A\|\epsilon\|_\infty
(1-\alpha a)^{(n-j-1)/2}.
$$

Соответственно,

$$
\sum_j\|g_j\|_\infty^2
\lesssim
\frac{\alpha}{a^3}\widetilde C_A^2\|\epsilon\|_\infty^2,
$$

а не $\alpha/a$. После этого надо пересчитать
$\widehat C_A$: он должен нести зависимость порядка
$\sqrt{t_{\mathrm{mix}}/a^3}$, если все остальные обозначения оставить как
сейчас.

Также нужно уточнить Lemma 2: если используемый источник дает концентрацию
только для scalar additive functionals, то формулировать ее для
$u^\top\sum_i g_i(Z_i)$ и затем отдельно получать векторную bound через
координаты/net, либо явно сослаться на векторную версию с нормой.

### Этап 2. Исправить stationary/zero-start смешение в telescoping identity

В `src/pr_weights.typ`, строки вокруг `1087--1125`, сейчас текст говорит о
stationary augmented-chain convention, но telescoping выводится с
$J_0^{(1,\alpha)}=0$.

Для stationary режима identity должна иметь boundary

$$
-\frac1\alpha \bar A^{-1}
\left(J_n^{(1,\alpha)}-J_0^{(1,\alpha)}\right)
$$

вместо только

$$
-\frac1\alpha \bar A^{-1}J_n^{(1,\alpha)}.
$$

Правка:

1. Переписать subsection "Telescoping identity for $J^{(1)}$" сразу в
   stationary notation.
2. В Lemma `<lem:T1-bound>` boundary term оценивать двумя одинаковыми
   членами: $J_n$ и $J_0$.
3. Итоговый порядок не меняется: вклад остается
   $\Phi(p,\alpha)n^{-1/2}$, меняется только константа.

### Этап 3. Усилить все step-size restrictions для RR

Везде, где применяются Levin bounds при $w\in\{\alpha,2\alpha\}$, заменить

$$
\alpha \le \alpha_*(q,t_{\mathrm{mix}})
$$

на

$$
2\alpha \le \alpha_*(q,t_{\mathrm{mix}}).
$$

Это касается:

1. Lemma `<lem:T2H-bound>`;
2. Theorem `<thm:misadjustment>`;
3. Corollary `<cor:misadjustment-rate>`;
4. Theorem `<thm:RR-BE>`;
5. subsequent rate corollaries.

Можно ввести сокращение

$$
\alpha_*^{\mathrm{RR}}(q,t_{\mathrm{mix}})
:=
\frac12\alpha_*(q,t_{\mathrm{mix}})
$$

и дальше писать $\alpha\le\alpha_*^{\mathrm{RR}}$.

### Этап 4. Переформулировать misadjustment theorem как stationary theorem

В Theorem `<thm:misadjustment>` оставить bound, но сделать условия честными:

1. явно сказать, что $R_n^{\mathrm{mis,RR}}$ построен из stationary
   augmented-chain versions of
   $(Z_t,J_t^{(0,w)},J_t^{(1,w)},J_t^{(2,w)},H_t^{(2,w)})$ for
   $w\in\{\alpha,2\alpha\}$;
2. убрать двусмысленность "finite-time notation with zero initial values" из
   самого proof path;
3. remark про finite-start transfer оставить, но перенести после theorem как
   limitation, а не как часть доказательства.

После этого Corollary `<cor:misadjustment-rate>` тоже должен быть stationary.
Нельзя писать, что $\theta_0=\theta^*$ превращает этот result в
finite-start theorem: это убирает только deterministic transient, но не
startup gap в augmented chain.

### Этап 5. Переписать Теорему 3 как conditional/stationary smoothing assembly

В Theorem `<thm:RR-BE>` statement лучше сделать так:

1. "Under the hypotheses of Theorem `<thm:M-RR-BE>` and the stationary
   misadjustment hypotheses of Theorem `<thm:misadjustment>`..."
2. Явно добавить $p=\max(2,\lceil\log n\rceil)$ или условие $n\ge3$.
3. Сказать, что bound applies to the same probability law under which
   Lemma `<lem:R-bound>` is proved.

Сама smoothing-часть proof менять почти не надо.

### Этап 6. Проверить martingale Berry--Esseen Term II

Shared review отдельно пометил нормировку в Term II. Надо сверить точную
форму Bolthausen--Fan inequality из Samsonov et al.:

1. является ли $s_n$ стандартным отклонением или variance proxy;
2. должен ли conditional-variance term использовать
   $|V_n^2-s_n^2|$ или относительную величину;
3. корректен ли переход
   $s_n^{-a_p}=s_n^{-1}s_n^{1/(2p+1)}$ при текущем notation.

Если текущая запись совпадает с источником, оставить rate и добавить
одну поясняющую строку. Если нет, переписать Term II полностью до получения
$\log^{3/4}(n)n^{-1/4}$.

### Этап 7. Почистить финальные corollaries

Для corollaries после Теоремы 3:

1. stationary corollary оставить с balanced scale
   $\alpha=c n^{-1/2}$ и rate
   $\mathrm{polylog}(n)n^{-1/4}$;
2. finite-start wording убрать или заменить на "requires a separate burn-in
   transfer";
3. если упоминается $\theta_0=\theta^*$, пояснить, что это относится только
   к deterministic transient $D_{\mathrm{tr}}^{RR}$.

### Этап 8. Технически исправить Typst-ссылки

PDF сейчас показывает пустые "Theorem ", "Lemma ", "equation ". Это мешает
читать proof chain. После математических правок надо:

1. проверить, как Typst labels работают с кастомными `#theorem[...]`;
2. либо заменить внутренние ссылки на текстовые "the martingale theorem above",
   где номер не критичен;
3. либо переделать theorem environments так, чтобы labels печатали номер.

### Проверка после правок

После каждого крупного этапа:

```bash
typst compile main.typ
pdftotext main.pdf - | rg "Theorem 3|stationary augmented|finite-start|alpha_\\*|sqrt\\(n\\)"
```

Финальный acceptance criterion:

1. `typst compile main.typ` проходит;
2. Теорема 3 больше не читается как finite-start theorem;
3. все ограничения на шаги покрывают $2\alpha$;
4. telescoping identity не смешивает $J_0=0$ со stationary start;
5. Section 2.4 имеет правильную зависимость по $a$;
6. corollaries не утверждают practical zero-start result без burn-in/transfer.

## Что уже исправлено 2026-05-17

По просьбе "сначала stationary augmented-chain theorem, burn-in потом" в
`src/pr_weights.typ` сделаны только statement/proof-scope правки:

1. Theorem `<thm:RR-BE>` теперь применяется к stationary assembled statistic
   $S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)$, а не напрямую к finite-start
   $\sqrt n\,u^\top(\bar\theta_n^{\mathrm{RR}}-\theta^*)$.
2. Composite remainder заменён на
   $\mathcal R_{n,\mathrm{stat}}^{\mathrm{RR}}
   = D_{2,n}^{\mathrm{RR}}+R_n^{\mathrm{mis,RR}}$; deterministic transient
   $D_{\mathrm{tr}}^{\mathrm{RR}}$ явно вынесен за рамки theorem.
3. В telescoping identity для $J^{(1)}$ boundary исправлен с zero-start
   $-\alpha^{-1}\bar A^{-1}J_n^{(1)}$ на stationary/general
   $\alpha^{-1}\bar A^{-1}(J_0^{(1)}-J_n^{(1)})$.
4. Step-size restrictions в misadjustment и final theorem усилены до
   $2\alpha\le \alpha_*(q,t_{\mathrm{mix}})$.
5. В финальных corollaries убрано условие $\theta_0=\theta^*$ как будто оно
   даёт finite-start theorem; теперь они тоже formulated for
   $S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)$.
6. `typst compile main.typ` проходит.

Следующий математический этап всё ещё тот же: отдельный burn-in/finite-start
transfer с весами $Q_{\ell,n_0}^{(\alpha)}$.

# Ревью текущего текста диплома

Файл: `main.pdf`  
Тема: *Statistical Inference for Linear Stochastic Approximation with Richardson–Romberg Extrapolation*  
Формат ревью: проверка текущих материалов, с упором на доказательства, обозначения, стиль и связность.

## 0. Краткий вердикт

Текст выглядит как содержательный и технически серьёзный черновик. Основная линия — собрать Berry–Esseen/CLT-анализ для PR-усреднённого Richardson–Romberg (RR) статистика в Markovian LSA и затем перенести стационарный результат на deterministic-start burned-in statistic — в целом логична. Сильные стороны работы:

- хорошо выделена разница между depth-zero Gaussian term и misadjustment remainder;
- корректно используется Poisson/martingale approximation для Markov-chain noise;
- есть честная оговорка, что zero-start/full-window анализ и stationary augmented-chain анализ не совпадают автоматически;
- финальная структура с burned-in weights выглядит правильным направлением, потому что burn-in действительно меняет веса.

При этом я нашёл несколько мест, которые надо исправить до финальной версии. Самые важные проблемы:

1. Формулировки про “logarithmic burn-in” сейчас неточны: при рабочем масштабе `alpha = c n^{-1/2}` и `p ≍ log n` получающийся burn-in не логарифмический по `n`, а порядка `(alpha a)^{-1} log^2 n`, то есть примерно `n^{1/2} log^2 n`.
2. Startup transfer для depth-two augmented remainder, особенно компонент `H^(2)`, опирается на новый Lemma 24. Это не просто импорт из Levin et al.; это самостоятельный технический результат, и его proof sketch пока недостаточно полон для центральной теоремы.
3. Нужно жёстче развести stationary augmented-chain statement и deterministic-start statistic. В тексте это часто честно проговаривается локально, но введение и названия результатов иногда звучат сильнее, чем доказано.
4. В двух местах используется аргумент “знак не важен для Kolmogorov distance из-за симметрии нормального закона”. В такой форме это не совсем корректно; лучше просто применить martingale Berry–Esseen к мартингальным разностям с противоположным знаком.
5. Серьёзно мешает коллизия обозначений `C_A`: один раз это sup-norm константа из Assumption 2, другой раз chapter-local constant `kappa_Q`.

Ниже — подробные комментарии.

---

## 1. Критические замечания по доказательствам

### 1.1. “Stationary full-window CLT for sqrt(n)(theta_n^(alpha,RR)-theta*)” требует уточнения режима по `alpha`

Во введении заявлено:

```text
A stationary full-window central limit theorem for sqrt(n)(theta_n^(alpha,RR) - theta*)
```

Но для фиксированного `alpha > 0` constant-step LSA имеет стационарное смещение. После RR ведущий `O(alpha)` bias отменяется, но остаётся residual bias `O(alpha^{3/2})` или лучше, в зависимости от используемой декомпозиции. Поэтому CLT вокруг `theta*` при фиксированном `alpha` не является обычной fixed-alpha CLT: центр всё ещё не равен ровно `theta*`.

Чтобы утверждение было математически корректным, надо явно зафиксировать один из вариантов:

- либо формулировать stationary CLT/BE для центрированного объекта вокруг stationary mean/RR stationary center;
- либо формулировать triangular-array statement с `alpha = alpha_n -> 0` и условием, что `sqrt(n) * residual_bias(alpha_n)` уходит в ноль или входит в remainder;
- либо сразу говорить о рабочем balanced scale `alpha = c n^{-1/2}`, где `sqrt(n) alpha^{3/2} = n^{-1/4}` и bias absorbed into Berry–Esseen remainder.

Рекомендованная правка во введении:

```text
We prove a stationary full-window Berry–Esseen bound for the stationary augmented-chain RR assembly. At the thesis-facing balanced scale alpha_n = c n^{-1/2}, the residual RR bias is of the same or smaller order than the non-asymptotic remainder, so the statistic can be normalized around theta*.
```

Иначе читатель может понять, что доказывается fixed-alpha CLT for `sqrt(n)(theta_bar^{RR} - theta*)`, что неверно без дополнительного центрирования.

### 1.2. Stationary augmented-chain theorem и deterministic-start theorem надо развести ещё сильнее

В Chapter 4 вы честно пишете, что stationary result не является deterministic-start full-average result. Особенно важны фразы:

- stationary theorem stated for `S_n,stat^RR(u)`;
- deterministic-start transient and random initial-product discrepancy are not included;
- startup transfer from zero-start recursions to stationary augmented chain is not a terminal `rho^n` term;
- finite-start theorem needs burned-in weights.

Это хорошая и правильная оговорка. Но во введении и в некоторых заголовках результат звучит как theorem directly for `theta_n^(alpha,RR)`. Лучше ввести отдельные обозначения сразу:

- `S_{n,stat}^{RR}(u)` — stationary augmented-chain assembled statistic;
- `T_{n,n0}^{RR}(u)` — deterministic-start burned-in statistic;
- `theta_bar_{n,n0}^{RR}` — actual thesis-facing estimator.

Рекомендую в `Problem statement and goals` заменить пункт 1 на что-то вроде:

```text
A stationary full-window Berry–Esseen/CLT result for the stationary augmented-chain RR assembly associated with the PR-averaged RR statistic, with covariance target Sigma_infty.
```

А пункт 3:

```text
A deterministic-start transfer theorem for the burned-in statistic, under an explicit burn-in condition depending on alpha, p, and a.
```

### 1.3. Lemma 24 — главный технический разрыв в текущей версии

Самое серьёзное место — `Lemma 24. Full-state startup contraction for the depth-two augmented remainder`.

Вы пишете, что Levin Appendix B.2 даёт contraction для координат

```text
(Z_{k+1}, J_k^(0), J_k^(1), J_k^(2)),
```

но не включает `H_k^(2)`. Далее вы вводите full-state contraction для remainder

```text
R_k = J_k^(1) + J_k^(2) + H_k^(2).
```

Это действительно нужно для финального burn-in transfer, но доказательство сейчас выглядит как proof sketch, а не как завершённое доказательство. Причина: `H^(2)` задаётся через случайный продукт и всю историю:

```text
H_k^(2,w) = -w sum_{l=1}^k Gamma_{l+1:k}^{(w)} Atilde(Z_l) J_{l-1}^{(2,w)}.
```

Чтобы сравнить finite-start и stationary versions, надо явно контролировать разность двух таких сумм. Там возникают как минимум три типа членов:

1. разность random products `Gamma_fin - Gamma_aug`;
2. разность `J_fin^(2) - J_aug^(2)`;
3. mismatch до coupling time `T` и вклад после `T`.

Сейчас это проговаривается словами через “random-product stability estimate used inside Levin Proposition 9”, но для основной теоремы этого недостаточно. Нужно либо:

- дать полноценное доказательство Lemma 24 с разложением разности двух `H^(2)` представлений;
- либо явно оформить это как дополнительное предположение, например:

```text
Assumption 4 (depth-two startup contraction). The finite-start and stationary depth-two augmented remainders admit a coupling satisfying ...
```

Если оставить как есть, критичный читатель может сказать, что Theorem 7 и Corollary 13 опираются на не доказанный extension beyond cited Levin results.

Минимум, который стоит добавить в proof of Lemma 24:

- точное определение coupling space для finite and stationary augmented chains;
- decomposition of `H_fin - H_aug` into pre-coupling and post-coupling parts;
- отдельный bound для product difference;
- моментный bound для `Delta H_T` или для начального состояния после coupling time;
- объяснение, почему интегрирование по случайному `T` даёт именно `exp(-c w a k / p)` и амплитуду `A_st(p,q,w)`;
- проверка, что constants не зависят от `n`, `n0`, `m`, `alpha` сверх заявленного.

### 1.4. “Logarithmic burn-in” сейчас вводит в заблуждение

В тексте repeatedly говорится “logarithmic burn-in”. Но ваши условия burn-in имеют вид примерно

```text
n0 >= C/(alpha a) log n,
n0 >= C p/(alpha a) log n.
```

А в balanced scale вы берёте

```text
alpha = c n^{-1/2},
p ≍ log n.
```

Тогда второе условие становится

```text
n0 >= C n^{1/2} log^2 n.
```

Это не логарифмический burn-in по `n`. Это logarithmic-square factor times the mixing scale `(alpha a)^{-1}`. В Corollary 13 вы, кажется, уже пишете “logarithmic-square burn-in”; нужно согласовать все ранние формулировки.

Рекомендация: заменить “logarithmic burn-in” на одну из формулировок:

```text
mixing-scale burn-in of order (alpha a)^{-1} log^2 n at the balanced scale;
```

или

```text
explicit burn-in proportional to the constant-step mixing time (alpha a)^{-1}, with logarithmic factors.
```

В introduction можно написать:

```text
At alpha = c n^{-1/2}, the resulting burn-in is n0 = O(n^{1/2} polylog(n)), not O(log n); throughout we call it a mixing-scale burn-in.
```

### 1.5. Знак в Kolmogorov distance: лучше убрать аргумент через симметрию

В Chapter 4 и Chapter 5 используется идея:

```text
the symmetry of N(0,1) makes the sign irrelevant: d_K(X,N)=d_K(-X,N).
```

В такой форме это не является безопасным утверждением для произвольного `X`. Для continuous distributions часто можно переписать через cdf, но в общем `sup_x |P(-X <= x)-Phi(x)|` связано с левыми/правыми пределами cdf of `X`; при атомах equality может ломаться. Здесь не нужно рисковать.

Правка простая: если ведущий член имеет минус, примените martingale Berry–Esseen directly to the martingale difference sequence with increments multiplied by `-1`. Все условия сохраняются, predictable quadratic variation не меняется.

Заменить фразу на:

```text
We apply the martingale Berry–Esseen theorem to the martingale difference sequence -Delta M_l. The predictable quadratic variation is unchanged, hence the same bound holds for the signed leading term.
```

Это полностью закрывает вопрос.

---

## 2. Важные технические замечания

### 2.1. Коллизия обозначений `C_A`

В Assumption 2 `C_A` — это sup-norm constant:

```text
C_A = max(sup_z ||A(z)||, sup_z ||Atilde(z)||).
```

Позже в Section 2.3 вы вводите chapter-local constant

```text
C_A := kappa_Q,
```

и прямо пишете, что это “distinct from the assumption-2 sup-norm constant”. Это лучше не делать: даже с пояснением формулы становятся нечитаемыми. Например,

```text
tilde C_A := C_A C_A = C_A kappa_Q
```

выглядит как ошибка.

Рекомендованные обозначения:

- `C_A^sup` для sup-norm bound on `A(z), Atilde(z)`;
- `C_Q^eq` или `K_Q` для norm-equivalence constant;
- `C_{A,Q}` для products like `C_A^sup K_Q`.

Это улучшит как стиль, так и проверяемость доказательств.

### 2.2. В identity `X^m - Y^m` нужно явно сказать, почему matrices commute

Вы несколько раз используете

```text
X^m - Y^m = (X-Y) sum_{i=1}^m X^{i-1} Y^{m-i}.
```

Для матриц эта identity в указанной форме требует commutativity of `X` and `Y` или аккуратного порядка множителей. В вашем случае

```text
X = I - alpha A,
Y = I - 2 alpha A,
```

то есть обе матрицы являются polynomial functions of the same matrix `A`; поэтому они коммутируют. Доказательство корректно, но стоит написать:

```text
Since B_alpha and B_{2alpha} are polynomials in A, they commute, and the scalar telescoping identity applies in matrix form.
```

Без этой фразы у читателя может возникнуть ненужное подозрение.

### 2.3. Powers of `a` в Lemma 5 стоит перепроверить

В proof of Lemma 5 для `U_R` получается

```text
||u^T U_R||_Lp <= C p^{1/2} t_mix^{3/2} ||epsilon|| / sqrt(alpha a).
```

После умножения на `alpha` вклад в `S_n - E S_n` должен быть порядка

```text
sqrt(alpha) / sqrt(a)
```

с точностью до других constants. В statement стоит

```text
p^{1/2} t_mix^{3/2} sqrt(alpha) / a.
```

Это более грубо, если вы допускаете поглощение `a^{-1/2}` в `a^{-1}`; но тогда надо явно сказать, что используется, например, `a <= 1` или что constants are allowed to worsen by a fixed power of `a`. Сейчас powers of `a` вроде бы заявлены как explicit, поэтому лучше привести их к одному виду.

Возможная правка:

- либо statement: `... + p^{1/2} t_mix^{3/2} sqrt(alpha/a)`;
- либо в proof добавить: “we loosen `a^{-1/2}` to `a^{-1}` since only powers of the stability constant are tracked coarsely in this chapter.”

### 2.4. В Lemma 14 шаг `Phi + 1 <= C Phi` требует условия

В proof of Lemma 14 есть переход вида

```text
||J_n^(1,w)||_Lp <= C w (Phi(p,w)+1) <= C w Phi(p,w), for p >= 2.
```

Это требует `Phi(p,w) >= c > 0`. Возможно, это верно при ваших constants, но лучше не оставлять неявным. Самый чистый вариант — определить

```text
Phi_+(p,alpha) := 1 + Phi(p,alpha)
```

и использовать его в boundary term. На финальном rate это не влияет.

### 2.5. В Chapter 4 Theorem 3/Corollary 5 нужно явнее указать lower variance condition

Для scalar Berry–Esseen нужно `sigma_n^RR(u)` bounded away from zero. Вы это учитываете через variance lower-bound condition, но в theorem statement стоит сделать это максимально явным:

```text
Assume sigma^2(u) > 0 and n alpha a is large enough so that sigma_n^{2,RR}(u) >= sigma^2(u)/2.
```

Это особенно важно, потому что direction `u` произвольное, а `Sigma_infty` positive definite не всегда явно заявлено в assumptions. Если вы хотите uniform over directions, нужна `lambda_min(Sigma_epsilon^(M)) > 0` или аналог.

### 2.6. Theorem 5: concentration of predictable quadratic variation uses `sqrt(p n)`, not `sqrt(p m)`

В burned-in chapter Lemma 23/Corollary 10 дают bound порядка `sqrt(p n)`, а normalization theorem uses `m = n - n0` with `m >= n/2`. Это нормально, но theorem statement должен явно включать `m >= n/2` как не просто удобное техническое условие, а условие, позволяющее заменить `n/m` constants.

Если в будущем хотите burn-in comparable to `n`, proof надо переписать.

### 2.7. В Theorem 7/Burn-in transfer нужно явно разделить три startup terms

Сейчас финальный theorem объединяет много компонентов. Для читателя будет яснее перед финальным theorem сделать отдельную decomposition table:

| term | source | bound | burn-in condition |
|---|---|---|---|
| deterministic transient | `B_w^k(theta0-theta*)` | exponential | `n0 >= ... log n` |
| random initial-product discrepancy | `Gamma - B` | stability/coupling | `n0 >= ...` |
| augmented-chain startup discrepancy | `R_fin - R_aug` | Lemma 24 | `n0 >= p/(alpha a) log n` |
| martingale BE | Poisson martingale | `n^{-1/4}` | no startup |
| variance comparison | deterministic weights | `(m alpha a)^{-1}` | `m alpha a` large |

Такой table сильно улучшит связность.

---

## 3. Комментарии по главам

### Chapter 1: Introduction, setting, assumptions

Плюсы:

- Контекст constant-stepsize LSA and Markovian bias изложен хорошо.
- Уместно объяснено, зачем RR нужен: PR averaging alone does not remove stationary bias.
- Условия UGE, Hurwitz, boundedness понятны.

Что исправить:

1. Placeholder `Your abstract.` надо заменить реальным abstract.
2. Введение пока обещает чуть более сильные результаты, чем аккуратно доказаны later. Нужно заменить “CLT for theta_n^(alpha,RR)” на “stationary augmented-chain assembly / triangular-array balanced-scale statement”.
3. Пункт “logarithmic burn-in” заменить на “mixing-scale burn-in with logarithmic factors”.
4. `pi(epsilon)=0` следует явно вывести после definitions:

```text
By Assumption 2, pi(Atilde)=0 and pi(btilde)=0, hence pi(epsilon)=0.
```

5. Уточнить, что `Sigma_epsilon^(M)` positive definite или хотя бы что рассматриваются directions `u` with `sigma(u)>0`.

### Chapter 2: Zeroth-order RR last-iterate term

Плюсы:

- Algebraic derivation of zeroth-order last-iterate RR difference is basically correct.
- Bound `O(sqrt(alpha))` for scalar projections is plausible and well motivated.

Что исправить:

1. Добавить commutativity comment for `B_alpha` and `B_{2alpha}`.
2. Убрать collision `C_A`.
3. В Lemma 2 statement лучше явно написать, что functions are scalar. Сейчас дальше используется scalar projection; это окей.
4. `E⟦X|^p]` — typographical glitch. Должно быть `E[|X|^p]`.
5. Эта глава анализирует last-iterate local object, не final PR-averaged RR statistic. Это уже написано, но стоит повторить в short “Role of this chapter” paragraph.

### Chapter 3: Last iterate analysis

Плюсы:

- Хорошо объяснена разница между future-centered bilinear term and ordinary additive functional.
- Разложение `S_n - E S_n = -alpha(U_M + U_R)` логично.
- Вы честно показываете limitation of depth-one route.

Что исправить:

1. Перепроверить powers of `a` в Lemma 5, как указано выше.
2. Imported Lemma 4 стоит назвать не просто “imported”, а дать точную ссылку и условия применимости. Особенно stationarity and boundedness.
3. В Lemma 5 statement `pi(Atilde)=0` следует либо считать следствием definitions, либо не повторять как отдельное assumption. Сейчас выглядит как дополнительное условие.
4. Depth-one limitation section полезен, но стоит сократить/сделать conclusion clearer:

```text
This route is not used in the final theorem because at alpha ~ n^{-1/2} it gives O(1), not o(1), for the centered misadjustment.
```

### Chapter 4: Stationary RR PR-weight and Berry–Esseen assembly

Плюсы:

- Это самая сильная часть текста по структуре.
- Хорошая декомпозиция: deterministic weights, variance comparison, Poisson martingale approximation, quadratic variation concentration, martingale BE, smoothing assembly.
- Важно, что stationary augmented-chain convention explicitly acknowledged.

Что исправить:

1. Не использовать одно и то же обозначение для finite-time zero-start perturbation variables and stationary augmented-chain variables без warning в каждом key theorem. Лучше добавить superscripts `stat` and `fin` хотя бы в Chapter 4–5.
2. Theorem 3/Corollary 5: заменить sign-symmetry argument на direct signed increments.
3. Corollary 4: statement “matching the leading martingale Berry–Esseen rate” лучше заменить на “same polynomial order as the leading martingale Berry–Esseen rate, up to logarithms”, потому что powers of log differ.
4. В Section 4.6 Poisson decomposition: right boundary vanishes because RR terminal weight is zero. Это хороший момент; оставить.
5. В Section 4.10 `S_n,stat^RR` should be introduced much earlier, perhaps at the beginning of Chapter 4.

### Chapter 5: Burn-in transfer

Плюсы:

- Правильно, что burned-in weights restated; нельзя “just set n0>0” in stationary theorem.
- Детерминированный transient and random initial-product transient separated.
- Poisson decomposition for burned-in weights appears structurally correct.

Что исправить:

1. Главная проблема — Lemma 24, см. Section 1.3.
2. “Final corollary converts sqrt(m) to sqrt(n) when burn-in window is logarithmic” — заменить, потому что финальный burn-in at balanced scale не логарифмический.
3. В финальных corollaries explicitly state relation between `m` and `n`, e.g. `n0 <= n/2` and the derived lower bound on `n` ensuring both lower and upper constraints can hold.
4. Если финальная theorem uses both finite-window and asymptotic normalization, лучше в theorem statement дать два separate displays:
   - finite-window normalized bound;
   - asymptotic normalized bound with extra variance-comparison term.
5. В Corollary 13 стоит явно сказать, что `n0 = O(n^{1/2} log^2 n)` is smaller than `n/2` only for sufficiently large `n`; иначе условие может быть impossible for moderate `n`.

---

## 4. Стиль и связность

### 4.1. Нужна карта “что импортировано, что новое”

Сейчас текст много импортирует из Levin et al. and Samsonov et al. Это нормально, но нужно сделать contribution map, например:

```text
Imported inputs:
1. Levin stationary bias expansion for J^(1).
2. Levin centered bilinear Lp bound.
3. Levin depth-two moment bounds for J^(2), H^(2).
4. Samsonov martingale BE/smoothing framework.

New in this thesis:
1. RR deterministic-weight estimates for PR averaging.
2. Stationary RR Poisson-martingale assembly.
3. Burned-in deterministic-weight comparison.
4. Deterministic-start transfer via startup contraction.
```

Это поможет защитить работу от вопроса “что именно сделано автором?”.

### 4.2. Убрать internal source references типа `last_iterate.typ`

В тексте есть ссылка вида “last_iterate.typ”. Для диплома это выглядит как ссылка на исходный файл, не как академический текст. Заменить на “Section 3.1” или “Lemma 5”.

### 4.3. Таблица обозначений очень нужна

В работе много похожих объектов:

- `Q_l^(alpha)`, `mathcal Q_l^RR`, `Q_l; n0,n^RR`, `Q_l^bRR`;
- `W^RR`, `W_{n,n0}^RR`;
- `M^RR`, `M^bRR`;
- `sigma_n^RR`, `sigma_{n,n0}^{bRR}`, `sigma(u)`;
- `S_n,stat^RR`, `T_{n,n0}^RR`, `Xi_{n,n0}^{bRR}`, `Xi_{n,n0}^{asy,RR}`.

Добавьте в начало Chapter 4 или appendix таблицу notation. Это резко повысит читаемость.

### 4.4. Не все constants должны быть named

Некоторые constants named, но потом всё равно absorbed. Лучше придерживаться правила:

- named constants только если они входят в final theorem или burn-in condition;
- остальные — generic `C`.

Сейчас слишком много `C_burn,Q`, `C_burn,V`, `C_burn,3`, `C_D2`, `C_mis`, `C_4`, etc. Это утяжеляет чтение.

### 4.5. Typst/layout артефакты

Нужно пройтись по PDF-рендеру и исправить:

- `𝒵︀` artefacts;
- `𝔼⟦X|^p]` вместо `E[|X|^p]`;
- broken equation numbering like `D_R := ... R_k(98).`;
- разорванные формулы с `√ 𝑛` в знаменателе;
- inconsistent placement of equation numbers;
- repeated hyphenation artefacts like `finite￾start`, `Richard￾son`.

### 4.6. Нужен bibliography в текущем PDF

PDF обрывается после текущих theorem/corollary materials и не содержит полноценного references section. Для диплома это обязательно. Даже если работа не закончена, лучше уже сейчас держать bibliography compiled.

---

## 5. Рекомендованные точечные правки формулировок

### 5.1. Введение: заменить statement goals

Текущее:

```text
A stationary full-window central limit theorem for sqrt(n)(theta_n^(alpha,RR)-theta*), identifying the limiting covariance matrix.
```

Лучше:

```text
A stationary full-window Berry–Esseen/CLT analysis for the stationary augmented-chain RR assembly associated with the PR-averaged RR statistic. At the balanced triangular-array scale alpha_n = c n^{-1/2}, this identifies Sigma_infty as the covariance target and controls the residual RR bias within the non-asymptotic remainder.
```

Текущее:

```text
A deterministic-start transfer theorem with logarithmic burn-in.
```

Лучше:

```text
A deterministic-start transfer theorem with an explicit mixing-scale burn-in. At the balanced scale alpha_n = c n^{-1/2}, the required burn-in is of order n^{1/2} polylog(n).
```

### 5.2. Chapter 4 sign issue

Заменить:

```text
The symmetry -N(0,1) =d N(0,1) makes the sign irrelevant.
```

на:

```text
We apply the martingale Berry–Esseen estimate to the signed martingale increments -Delta M_l. Since the predictable quadratic variation is unchanged, the same bound applies to the leading term with the sign appearing in the Poisson decomposition.
```

### 5.3. Lemma 24 proof opening

Добавить после representation of `H^(2)`:

```text
We emphasize that this is not a direct corollary of Levin Proposition 5, which controls only the J-coordinates. The following proof extends the coupling to H^(2) by comparing the two random-product representations term by term.
```

Затем дать полноценный decomposition.

### 5.4. Constant notation

Заменить:

```text
C_A := kappa_Q
```

на:

```text
K_Q := kappa_Q
```

и дальше:

```text
tilde C_A := C_A^sup K_Q.
```

---

## 6. Проверка логической линии результата

Текущая логика может быть сформулирована так:

1. Constant-step LSA with Markovian noise has stationary bias.
2. RR cancels the leading `O(alpha)` bias.
3. Depth-zero PR-averaged RR term is a deterministic-weighted Markov additive functional.
4. Poisson equation turns it into martingale plus Abel/boundary remainder.
5. The deterministic RR weights converge to `A^{-1}`, so the covariance target is `Sigma_infty`.
6. Martingale Berry–Esseen gives `n^{-1/4}`-type rate for scalar projections.
7. Depth-two Levin transfer controls the RR misadjustment at the same working scale.
8. Stationary augmented-chain result is transferred to deterministic start only after burn-in and with burned-in weights.

This line is coherent. The main danger is not the overall idea; it is the rigor of step 8, especially `H^(2)` startup contraction.

---

## 7. Checklist before continuing

Priority A — mathematical correctness:

- [ ] Rephrase fixed-alpha CLT statements as stationary-centered or triangular-array/balanced-scale statements.
- [ ] Replace “logarithmic burn-in” by “mixing-scale burn-in with logarithmic factors”; compute final scale explicitly.
- [ ] Expand Lemma 24 proof or turn it into an explicit assumption.
- [ ] Remove Kolmogorov sign-symmetry argument; apply martingale BE to signed increments.
- [ ] Add explicit lower-variance assumption for scalar directions.
- [ ] Resolve `C_A` notation collision.

Priority B — clarity:

- [ ] Add contribution map: imported vs new.
- [ ] Add notation table for all RR weights and normalizations.
- [ ] Clearly mark stationary variables with `stat` and finite-start variables with `fin` where possible.
- [ ] Replace `last_iterate.typ` with section/lemma reference.
- [ ] Add real abstract.

Priority C — polish:

- [ ] Fix Typst artefacts and broken brackets.
- [ ] Add bibliography.
- [ ] Standardize theorem names and notation.
- [ ] Reduce named constants where not needed.

---

## 8. Итог

Текущий материал нельзя назвать “ошибочным” в целом: общая математическая стратегия разумная, а большинство imported ingredients используются в правильных местах. Однако есть несколько существенных мест, где формулировки сильнее доказательств или где proof sketch должен стать полноценным доказательством. Если исправить пункты из Priority A, текст станет значительно устойчивее к критическому чтению.

Самая важная правка — Lemma 24. Именно она превращает стационарный результат в deterministic-start theorem. Пока она записана как техническое расширение Levin coupling, но не доказана с достаточной детализацией. Это лучше закрыть до дальнейшего расширения диплома.

# Ревью текущей версии диплома

Файл: `main.pdf`, 53 страницы. Тема: статистическое выведение для constant-stepsize LSA с марковским шумом и Richardson--Romberg extrapolation.

Дата ревью: 2026-05-23.

## 0. Краткий вердикт

Текст в целом выглядит как сильная и содержательная заготовка: структура доказательства правильная по замыслу, основная декомпозиция на RR-веса, пуассоновскую мартингальную аппроксимацию, сравнение вариации, сглаживание и burn-in transfer логически выстроена. Особенно удачно, что в тексте явно отделён стационарный augmented-chain результат от deterministic-start результата с burn-in. Это существенно снижает риск неверной формулировки главной теоремы.

При этом я нашёл несколько мест, которые стоит исправить до защиты или до отправки научруку. Две группы замечаний наиболее важны:

1. **Самосогласованность импортированных результатов.** В нескольких местах финальная теорема опирается на Levin et al. и Samsonov et al. через пороги вида `α_*(q,t_mix)`, `α_st(p)`, startup-contraction constants, stationary depth-two bounds. Сейчас это приемлемо как черновик, но не полностью самодостаточно как диплом: нужно явно выписать точные импортируемые утверждения и все их условия.
2. **Несколько математических/типографских ошибок в формулах.** Самая конкретная: в Lemma 28 интервал для отношения стандартных отклонений должен быть `[1/sqrt(2), sqrt(3/2)]`, а не `[1/sqrt(2), sqrt(3)/2]`. Также в Section 3 есть несогласованность в степени `a` в члене `sqrt(alpha)/a`, который из предыдущей строки выходит как `sqrt(alpha/a)`.

Ниже комментарии разделены на критичность.

---

## 1. Что проверялось

Я проверял текущий материал как математическое ревью, а не как формальную верификацию всех внешних лемм. Основное внимание:

- корректность алгебры RR-весов;
- корректность масштабов по `n`, `m`, `alpha`, `a`, `t_mix`;
- согласованность стационарной и burned-in теорем;
- скрытые условия применимости импортированных результатов;
- стиль, связность и терминологию.

Внешний контекст сверялся с тремя ближайшими работами:

- Samsonov--Sheshukova--Moulines--Naumov, *Statistical inference for Linear Stochastic Approximation with Markovian Noise*, arXiv:2505.19102: Berry--Esseen для PR-averaged LSA с Markovian noise, rate `O(n^{-1/4})` в Kolmogorov distance и bootstrap validity.
- Levin--Naumov--Samsonov, *High-Order Error Bounds for Markovian LSA with Richardson-Romberg Extrapolation*, arXiv:2508.05570: bias decomposition, linear-in-`alpha` bias, RR cancellation, high-order RR moment bounds.
- Huo--Chen--Xie, *Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference*, arXiv:2312.10894: constant stepsize inference and RR bias reduction.

---

## 2. Главные замечания по доказательствам

### P0. Lemma 28: неверно записана верхняя граница для `r_{n,n0}(u)`

**Где:** Section 5.12, Lemma 28, Eq. (334).

**Сейчас:**

```tex
1/\sqrt{2} \le r_{n,n0}(u) \le \sqrt{3}/2.
```

В тексте это выглядит как `sqrt(3)/2`. Но из доказательства используется

```tex
|sigma^2_b - sigma^2| <= sigma^2/2.
```

Следовательно

```tex
sigma_b^2 / sigma^2 \in [1/2, 3/2],
```

и для отношения стандартных отклонений

```tex
r = sigma_b/sigma \in [1/sqrt(2), sqrt(3/2)].
```

`sqrt(3)/2 ≈ 0.866`, тогда как `sqrt(3/2) ≈ 1.225`. Текущая запись делает верхнюю границу меньше 1, что невозможно в случае `sigma_b^2 = 1.5 sigma^2`.

**Как исправить:**

```tex
\frac{1}{\sqrt{2}} \le r_{n,n0}(u) \le \sqrt{\frac{3}{2}}.
```

или

```tex
r_{n,n0}(u) \in [2^{-1/2}, (3/2)^{1/2}].
```

Это типографская ошибка, но математически заметная, потому что используется компактный интервал для Gaussian comparison.

---

### P1. Section 3, Lemma 5: несогласованность степени `a` в члене от `U_R`

**Где:** Lemma 5, Step 1--3, displays around Eq. (73), Eq. (79), and displayed bound Eq. (57).

Из текста:

```tex
||u^T U_R||_{L_p}
  <= C p^{1/2} t_mix^{3/2} ||epsilon||_infty / sqrt(alpha a).
```

Далее, поскольку

```tex
S_n - E S_n = - alpha (U_M + U_R),
```

вклад `U_R` в `S_n - E S_n` должен быть порядка

```tex
alpha / sqrt(alpha a) = sqrt(alpha/a),
```

то есть

```tex
C p^{1/2} t_mix^{3/2} ||epsilon||_infty sqrt(alpha/a).
```

В итоговой формуле написано

```tex
C p^{1/2} t_mix^{3/2} sqrt(alpha) / a.
```

Это сильнее на фактор `1/sqrt(a)` и не следует из предыдущей строки, если только дополнительно не предполагается `a <= 1` или если константа `C` скрывает зависимость от `a`. Но в этом же тексте степени `a` специально отслеживаются явно.

**Как исправить:** один из двух вариантов.

Вариант A, лучше:

```tex
||u^T(S_n - E S_n)||_{L_p}
 <= C ||u|| ||epsilon||_infty
    (p^{3/2} t_mix^{1/2}/a
     + p^{1/2} t_mix^{3/2} sqrt(alpha/a)).
```

Вариант B:

Добавить перед Lemma 5 нормализацию `a <= 1`, либо определить `a_bar = min(a,1)` и вести все оценки через `a_bar`, либо явно сказать, что в этом локальном разделе константы могут поглощать фиксированные степени `a`. Но тогда это конфликтует с фразой, что степени `a` отслеживаются.

---

### P1. Stationary-limit transfer: нужно закрыть вопрос равномерности по `alpha -> 0`

**Где:** Section 4.9, Lemma 13, stationary-limit transfer for `J^(1,alpha)`.

Логика доказательства такая: finite-start bound применяется на отрезке, потом сдвигается назад во времени, затем `m -> infinity` даёт стационарную версию. В тексте прямо отмечено, что dominating bound не является равномерным при `w -> 0`. Но далее результат используется в triangular-array режиме `alpha_n = c n^{-1/2}`, то есть именно при `alpha -> 0`.

Это не обязательно ошибка, но сейчас есть лакуна в изложении. Нужно чётко объяснить, что именно требуется равномерно:

- либо bound Lemma 13 применяется при каждом фиксированном `alpha`, а затем в финальных оценках вся зависимость от `alpha` явно входит через `Phi_+(p,alpha)`;
- либо нужно доказать limit passage с доминированием, допустимым для `alpha` из рассматриваемого диапазона `alpha <= alpha_0`, с константами, которые затем правильно учитываются;
- либо формально импортировать стационарную версию из Levin et al. без finite-past argument.

**Почему это важно:** финальный результат на balanced scale является triangular-array statement. Если стационарный переход доказан только для фиксированного `alpha`, это само по себе нормально, но надо явно связать его с последовательностью `alpha_n`.

**Рекомендация:** добавить после Lemma 13 короткий абзац:

```tex
Although the finite-past domination is not uniform in w as w -> 0, the resulting stationary bound is used only through the explicit quantity Phi_+(p,w). Thus the triangular-array substitution w=alpha_n is legitimate because all constants in Lemma 13 are independent of n and all w-dependence is displayed in Phi_+.
```

Если это утверждение не полностью верно для текущего доказательства, то Lemma 13 нужно усилить.

---

### P1. Импортированные Levin/Samsonov результаты должны быть сформулированы как отдельные self-contained lemmas

**Где:** Section 4.9, Section 5.8--5.10, Theorem 4, Theorem 7.

Сейчас используются фразы вида:

- “under the step-size restrictions of the Levin depth-two and startup-contraction bounds”; 
- `2 alpha <= alpha_*(q,t_mix)`;
- `2 alpha <= alpha_st(p)`;
- “stationary misadjustment constants”; 
- “startup-contraction constants”.

Для черновика это понятно, но для диплома финальная теорема становится не полностью проверяемой. Читатель не знает точных условий: какие моменты нужны, как зависит threshold от `q`, `t_mix`, `d`, `a`, `C_A`, `kappa_Q`, нужна ли стационарность augmented chain, какой exactly cost contraction используется.

**Как исправить:** перед Section 4.9 или в appendix добавить блок “Imported inputs”:

1. **Imported input A: Levin depth-two stationary RR misadjustment bound.** Полная формулировка с точными условиями и результатом.
2. **Imported input B: augmented-chain contraction/startup bound.** Полная формулировка с cost function, moment order and step-size ceiling.
3. **Imported input C: Samsonov martingale Berry--Esseen / smoothing / Markov concentration.** Полная формулировка именно в том виде, в котором используется.

После этого финальные теоремы можно ссылать на “Assume Imported Inputs A--C”. Так работа будет восприниматься как самостоятельная математическая сборка, а не как набор ссылок на невидимые условия.

---

### P1. В Section 4.10 стоит ещё сильнее отделить `S^RR_{n,stat}` от реального RR estimator

**Где:** Section 4.10, stationary assembled statistic.

Это место в целом написано аккуратно: текст явно говорит, что deterministic-start transient и random initial-product discrepancy не входят в stationary result. Но из-за того, что Section 4.1 начинается с finite-start PR decomposition, читатель может решить, что Theorem 3 уже доказывает Berry--Esseen для `sqrt(n)(theta_n^{RR}-theta*)`.

**Рекомендация:** переименовать theorem title и statistic:

```tex
Theorem 3 (Stationary augmented-chain assembly, not a finite-start estimator).
```

и перед statement добавить:

```tex
The random variable S^RR_{n,stat}(u) is an assembled comparison statistic. It is not equal in distribution to the deterministic-start PR-averaged RR statistic unless the corresponding augmented perturbation variables are initialized from their invariant law and the finite-start transients are removed.
```

Такое уточнение важно, потому что главный вклад работы именно в корректном transfer в Section 5.

---

### P1. Burn-in называется “logarithmic” в одном месте, хотя на balanced scale он порядка `sqrt(n) log n` или `sqrt(n) log^2 n`

**Где:** Section 5.1, фраза “A final corollary converts the sqrt(m) statistic to the thesis-facing sqrt(n) statistic when the burn-in window is logarithmic.”

Это неверно по масштабу. Условие Corollary 8:

```tex
n0 >= 2 beta /(alpha a) log n.
```

При `alpha = c n^{-1/2}` это

```tex
n0 >= const * sqrt(n) log n.
```

А для startup terms с `p ~ log n` появляется ещё один логарифм, то есть `n0` порядка `sqrt(n) log^2 n` в финальном Corollary 13. В abstract/introduction формулировка “mixing-scale burn-in with logarithmic factors” корректна. Но “logarithmic burn-in” — нет.

**Как исправить:** заменить на

```tex
... when the burn-in is at the mixing scale, i.e. of order alpha^{-1} times logarithmic factors.
```

или

```tex
... under the balanced-scale burn-in n0 = O(sqrt(n) log^2 n).
```

---

### P1. Theorem 7 и Corollary 13: нужно явно показать, где используется `m >= n/2`

**Где:** Section 5.12--5.13.

В Corollary 10 bound для predictable variation содержит `sqrt(p n)`, а финальная нормировка идёт через `m`. Это нормально, если `m >= n/2`, но лучше явно написать в proof of Theorem 7, что все terms of order `sqrt(n)/m` заменяются на `O(m^{-1/2})` благодаря `m >= n/2`.

Сейчас это частично видно в Corollary 13, но полезно добавить одну строку в proof of Theorem 7:

```tex
Throughout this proof we use m >= n/2 to replace n by m in all polynomial rates, e.g. sqrt(n)/m <= sqrt(2)/sqrt(m).
```

---

### P1. В Lemma 14 нужно уточнить стационарную индексацию пары `(J_{k-1}, Z_k)`

**Где:** Section 4.9, Lemma 14 and Eq. (188).

Функция `psi_alpha` центрирована относительно стационарного закона пары примерно вида `(J_0, Z_1)`. В сумме появляется `(J_{k-1}, Z_k)`. По stationarity это тот же law, но это нужно сказать, потому что иначе центрирование выглядит смещённым на один шаг.

**Предлагаемый текст:**

```tex
Under the stationary augmented-chain convention the pair (J_{k-1}^{(0,alpha)}, Z_k) has the same law as (J_0^{(0,alpha)}, Z_1), hence psi_alpha is centered for every summand.
```

---

### P2. Lemma 23/Corollary 10: концентрация для всех `l=2,...,n-1` включает pre-burn-in indices

**Где:** Section 5.7.

В формуле Eq. (279) суммирование идёт от `l=2` до `n-1`, хотя эффективная выборка имеет длину `m=n-n0`. Это не ошибка, потому что pre-burn-in weights малы и `m >= n/2`. Но в Corollary 10 лучше пояснить, почему bound с `sqrt(p n)` приемлем для финальной нормировки. Сейчас это неочевидно.

---

## 3. Проверка по разделам

### 3.1 Introduction and assumptions

**Сильные стороны:**

- Хорошо сформулирована мотивация: constant step size даёт bias, RR cancels leading `O(alpha)` term, PR averaging alone не убирает stationary bias.
- Правильно сделан акцент, что stationary theorem is an augmented-chain theorem, not a fixed-alpha CLT centered exactly at `theta*`.
- Удачно отделены три цели: stationary assembly, balanced triangular-array specialization, deterministic-start burn-in transfer.

**Что улучшить:**

1. **Contribution map:** фразы вида “Derived in Chapters Section 2 and Section 4” и “Developed in Chapter Section 5” нужно заменить на “Sections 2 and 4” / “Section 5”.
2. **Notation guide:** в notation guide стоит сразу объяснить различие между `Q_l^RR`, `mathcal Q_l^RR`, `Q_l^{bRR}`. Сейчас Section 4 использует calligraphic `mathcal Q`, Section 5 — `Q^{bRR}`, а intro table — `Q^RR`. Это не критично, но тяжело для читателя.
3. **Sign convention:** в вашей работе recursion идёт с минусом и `A` имеет eigenvalues with positive real parts. В Huo et al. часто используется plus-form convention with Hurwitz `Abar`. При цитировании результатов нужно явно напоминать о sign conversion.
4. **Bibliography in introduction:** ссылки на recent 2025 papers должны быть очень точными: arXiv/preprint/journal status, версия, название, authors.

---

### 3.2 Section 2: Zeroth-order RR difference

**Статус:** доказательство в основном корректно.

Проверка ключевой алгебры:

- Формула

```tex
J_tilde^{(0,alpha)} = 2J^{(0,alpha)} - J^{(0,2alpha)}
```

и переход к

```tex
-2 alpha^2 A sum_j H_j epsilon(Z_j)
```

корректны, потому что `B_alpha`, `B_{2alpha}` и `A` являются полиномами от одной матрицы и коммутируют.

- Оценка

```tex
||H_j|| <= K_Q (1-alpha a)^{(n-j-1)/2} * 2/(alpha a)
```

выглядит правильно. Использование геометрического ряда и неравенства

```tex
1 - sqrt((1-2x)/(1-x)) >= x/2
```

при `x=alpha a` корректно при `alpha a <= 1/2`.

- Суммирование квадратов даёт масштаб `alpha/a^3`, поэтому итоговый scalar `L_p` bound `O(sqrt(alpha))` верен.

**Замечания:**

1. Eq. (43) и Eq. (44) имеют typst/LaTeX поломку скобок: `E⟦X|^p]` вместо `E[|X|^p]`. Это нужно исправить.
2. В Lemma 2 вы пишете, что импортированная concentration lemma valid for arbitrary initial law and centered functions. Это важное место; в финальной версии стоит дать точную ссылку на lemma number, statement и constants.
3. Раздел называется “Zeroth-Order Richardson–Romberg Difference”, но объект в Section 2.2 — last-iterate, не PR average. Вы это поясняете, но можно вынести “last-iterate toy/preliminary bound” в title: “Last-iterate zeroth-order RR difference”.

---

### 3.3 Section 3: Last Iterate Analysis

**Статус:** идея правильная, но есть одна формульная несогласованность по `a` и несколько мест требуют уточнения.

Что хорошо:

- Identity `T_n^{(1,alpha)} = B J_n^{(1,alpha)}` корректно выведена.
- Вы честно показываете, что depth-one route не даёт полезный BE remainder at `alpha ~ n^{-1/2}`: centered fluctuation `O(sqrt(n) alpha)` остаётся `O(1)`. Это хорошая часть exposition.
- Decomposition into `U_M` and `U_R` концептуально верная: `U_M` future-centered bilinear term, `U_R` ordinary centered additive functional.

Что нужно поправить:

1. **Степень `a` в Eq. (57)/(79)** — см. P1 выше.
2. **Imported Lemma 4**: нужно точнее оформить. Сейчас это “scalar, constant-stepsize specialization” из Samsonov et al. Appendix D.2. Лучше дать полное statement с условиями на boundedness, stationarity, centering, moment order. Если это не дословная lemma из статьи, а реконструкция proof pattern, так и написать: “The following estimate is extracted from the proof of ...”
3. **В Eq. (71)** вы “factor out the slow rate” как `(1-alpha a)^{(n-k)/2}`. Строго получается `(1-alpha a)^{(n-k)/2}` или `(1-alpha a)^{(n-k)/2}` с возможной константой из `+1/-1`. Это норм, но лучше сказать “up to a universal constant”.
4. **Section 3.2:** при bias estimate `||E D_mis,RR_1|| <= C sqrt(n) alpha^2` стоит явно указать stationarity assumption and whether finite-start burn-in is excluded.

---

### 3.4 Section 4: RR PR weight bounds and stationary Berry--Esseen assembly

**Статус:** основной каркас корректный; это самая сильная часть работы, но импортированные inputs и stationary convention нужно сделать более self-contained.

#### 4.1 PR decomposition

В целом корректно. Хорошо, что `R_{k,init}` отделён от noise-driven remainder. Это важно для burn-in transfer.

Замечание: в Eq. (93) burned-in weight с factor `n/(n-n0)` вводится для normalization `n^{-1/2}`. В Section 5 вы переходите к `sqrt(m)` и весам без этого factor. Нужно явно отметить связь между двумя conventions, иначе читатель может подумать, что веса противоречат друг другу.

#### 4.2--4.5 RR weights and variance comparison

Проверка основных identities:

- `Q_l^(alpha) = A^{-1}(I - B_alpha^{n-l})` корректно.
- `Q_l^RR - A^{-1} = -A^{-1}(2B_alpha^k - B_{2alpha}^k)` корректно.
- Discrete derivative `Q_{l+1}^RR - Q_l^RR = -2 alpha (B_alpha^{k-1} - B_{2alpha}^{k-1})` корректен.
- Bound for derivative with additional alpha gain plausible and important.
- Variance comparison `||Sigma_n^RR - Sigma_infty|| <= C/(n alpha a)` корректен по масштабу.

Замечания:

1. В Lemma 6 constants `C_Q`, `Ctilde_A` выглядят корректно на rendered page. Никакой ошибки в `kappa_Q^{1/2}`/`kappa_Q` я не нашёл.
2. В variance comparison Eq. (125) используется absorption `2||Sigma_infty||/n` into `C/(n alpha a)`. Это верно при `alpha a <= 1`, но лучше явно оставить строку “since alpha a <= 1”. Она уже есть, хорошо.
3. Обозначение `Σ` в variance comparison — это `Σ_epsilon^{(M)}` или `π(V_epsilon)`? Сейчас местами `Σ` вводится локально. Лучше унифицировать: `Σ_epsilon^{(M)}` для Markovian noise covariance and maybe `Σ_Pois = π(V_epsilon)` if needed. Иначе `Σ∞ = A^{-1}Σ(A^{-1})^T` может конфликтовать с `Σ` как arbitrary covariance.

#### 4.6 Poisson martingale approximation

Структура правильная. Boundary/Abel remainder имеет sup-norm order `1/sqrt(n)`, что достаточно.

Рекомендации:

- Написать явно, что `hat epsilon = sum_{j>=0} Q^j epsilon` well-defined by UGE and `π(epsilon)=0`.
- Обратить внимание на index `l=1` boundary and `l=n-1` right boundary. В стационарной части right boundary возникает от `Q_n`? В burned-in right boundary vanishes because `Q_{n-1}^{bRR}=0`; в stationary full-window Eq. (140) left/right terms стоит проверить и описать симметрично.

#### 4.7--4.8 Predictable variation and martingale BE

Каркас корректен. Использование Bolthausen--Fan inequality и control of `V_n^2 - s_n^2` даёт `log^{3/4}(n)n^{-1/4}`. Это согласуется с known results for Markovian LSA PR averaging.

Замечания:

1. `Theorem 1` внутри Section 4 может конфликтовать с global numbering. Лучше использовать section-prefixed numbering: Theorem 4.1, Lemma 4.2 и т.д.
2. В Theorem 1 constants depend on `L_B(kappa(u))`, `C1`, `C2` of Eq. (157). Нужно сделать Eq. (157) отдельной named lemma.
3. Проверить, что `p=ceil(log n)` always `>=2`. В части мест вы пишете `p=max(2,ceil(log n))`, а в Theorem 1 proof — просто `ceil(log n)`. Для `n>=3`, `ceil(log n) >=2`, так что всё нормально; можно оставить.

#### 4.9 Stationary RR misadjustment

Это главная техническая зона риска.

Положительно:

- Вы явно показываете, почему depth-one route недостаточен.
- Вы используете Levin depth-two result, что логично: именно он даёт `alpha^{3/2}` scale and optimal leading covariance.

Что нужно усилить:

1. Сформулировать imported Levin theorem fully.
2. Закрыть stationary-limit transfer uniformity issue.
3. Явно сказать, что `R_mis,RR_n` в stationary assembly is not a zero-start finite recursion, а stationary augmented-chain object.
4. В Lemma 15 triangle bound over `J^(2)` and `H^(2)` даёт correct scale, но скрывает constants and `d^{1/q}` dependence. Если final theorem tracks `q`, нужно указать, где `d^{1/q}` входит.

#### 4.10 Smoothing assembly

Логика корректна:

```tex
S/sigma_n = X_n + Y_n,
```

martingale BE for `X_n`, smoothing for `Y_n`.

Рекомендации:

1. В Theorem 3 statement добавить “assembled stationary comparison statistic”.
2. В Corollary 5 condition Eq. (212) стоит переписать в более читабельной форме:

```tex
sqrt(n) alpha_n^2,
sqrt(n) alpha_n^{3/2},
sqrt(alpha_n),
(alpha_n n)^{-1/2}
```

all vanish at appropriate rate; then say optimized at `alpha_n = c n^{-1/2}`.
3. В Corollary 7 Gaussian comparison constant `C_Phi := sqrt(2)/sqrt(pi e)` is okay, but exact equality `sup_x |x phi(x)| = 1/sqrt(2 pi e)`; multiplying factor with `|r-1|` may not be exactly `sqrt(2)/(sqrt(pi e))` depending on interval and derivative w.r.t. `r`. Since constant is universal, exact constant not important. Можно написать simply `C_Phi < infinity`.

---

### 3.5 Section 5: Burn-in transfer

**Статус:** идея правильная, но нужно поправить масштаб burn-in wording и сделать финальную теорему self-contained.

#### 5.1 Target statistic

Хорошо, что вводятся две нормировки:

- finite-window `sigma_{n,n0}^{bRR}`;
- asymptotic `sigma(u)`.

Ошибка wording: “burn-in window is logarithmic” заменить на “burn-in is of order alpha^{-1} times logarithmic factors”.

#### 5.2--5.4 Burned-in weights and transients

Алгебра burned-in weights выглядит корректной.

Проверка важного boundary jump:

Для single step size

```tex
Q_{n0}^{(w)} - Q_{n0-1}^{(w)} = w(I - B_w^m).
```

Для RR:

```tex
2 alpha(I-B_alpha^m) - 2 alpha(I-B_{2alpha}^m)
= 2 alpha(B_{2alpha}^m - B_alpha^m).
```

Так что Eq. (242) корректна.

Детерминированный transient bound тоже выглядит правильным.

#### 5.4 Random initial-product transient

Использование random-product stability plausible. Но как и выше, это imported input. Нужно дать точную формулировку условий `alpha_st(p)`.

#### 5.5 Burned-in variance comparison

Масштаб

```tex
||Sigma_{n,n0}^{bRR} - Sigma_infty|| <= C/(m alpha a)
```

выглядит правильным. Pre-burn-in weights contribute `O(1/(m alpha a))`; post-burn-in tail also `O(1/(m alpha a))`.

Замечание: finite-index mismatch “two copies of Sigma_infty” absorbed by `1/(m alpha a)` under `alpha a <=1`; лучше явно написать условие в lemma statement.

#### 5.6--5.7 Burned-in Poisson and predictable variance

Корректно, но нужно пояснить почему concentration over `n` terms acceptable after normalization by `m`.

#### 5.8 Startup transfer

Хорошо, что вы явно формулируете discrepancy between zero-start perturbation variables and stationary augmented chain. Это один из важных вкладов текста.

Но:

1. Cost function from Levin et al. should be fully imported.
2. Сумма discrepancy over post-burn-in window likely requires a geometric series with factor `exp(-c alpha a n0/p) / (alpha a)`; это есть. Нужно проверить, что для `p ~ log n` финальный burn-in lower bound includes `log^2 n`. В Corollary 13 это должно быть явно.

#### 5.10 Misadjustment theorem

Theorem 4 right-hand side contains several terms:

- `sqrt(m) alpha^2`,
- `sqrt(m) alpha^{3/2}` with polylogs,
- `sqrt(alpha)`,
- `(alpha m)^{-1/2}`,
- `m^{-1/2}`,
- startup exponential.

At `alpha=c n^{-1/2}`, `m~n`, dominant terms are `n^{-1/4}` up to logs. This is consistent.

Potential issue: Theorem 4 statement says “every m>=2” but terms still depend on `n0` via startup exponential. If `m` is small relative to `n`, final theorem handles `m>=n/2`; but standalone Theorem 4 is okay.

#### 5.11--5.13 Final BE bounds

Theorem 6 is correct as an assembly using finite-window normalization.

Lemma 28 has the concrete typo `sqrt(3)/2` vs `sqrt(3/2)`.

Theorem 7 and Corollary 13 are directionally correct. Add explicit `m>=n/2` conversions.

---

## 4. Style and exposition comments

### 4.1 Numbering and naming

Current theorem/lemma numbering restarts or is not section-prefixed. In a 53-page technical diploma, global numbering is fine, but section-prefixed numbering is much easier to navigate:

```tex
Lemma 2.1, Lemma 2.2, Theorem 4.1, Theorem 5.2, ...
```

This will also make references like “Theorem 1” unambiguous.

### 4.2 “Chapter Section” typo

Replace:

- “Derived in Chapters Section 2 and Section 4”
- “Developed in Chapter Section 5”

with:

- “Derived in Sections 2 and 4”
- “Developed in Section 5”

### 4.3 Обозначения `Q`, `mathcal Q`, `bRR`

Сейчас используются:

- `Q_l^(alpha)`;
- `mathcal Q_l^RR`;
- `Q_l^RR`;
- `Q_l^{bRR}`;
- `Q_{l;n0,n}^RR`.

Лучше выбрать один стиль:

- `Q_l^{RR}` for full-window stationary;
- `Q_{l;n0,n}^{RR}` for burned-in;
- no calligraphic unless needed.

Если calligraphic notation сохраняется, в notation guide нужно прямо сказать:

```tex
In Section 4, mathcal Q_l^RR denotes the full-window RR PR weight; in Section 5, Q_l^{bRR}=Q_{l;n0,n}^{RR} denotes its burned-in analogue.
```

### 4.4 Термин “optimal covariance”

Вы правильно добавляете caveat, что full Hájek--Le Cam optimality не доказывается. Лучше везде писать “averaged-SA optimal covariance target” или “asymptotic PR covariance target” вместо просто “optimal covariance”, чтобы не провоцировать вопрос о minimax lower bound.

### 4.5 English style

Некоторые фразы лучше упростить:

- “The final burned-in statistic has a non-asymptotic normal approximation” → “We obtain a non-asymptotic normal approximation for the burned-in statistic.”
- “The stationary result should be read as...” — хорошая фраза, оставить.
- “thesis-facing” звучит разговорно. В академическом тексте лучше “main” / “final” / “deterministic-start”. Например: “the final deterministic-start consequence”.

### 4.6 References

Список литературы сейчас слишком неполный для диплома. Нужно привести к единому стилю:

- authors;
- year;
- title;
- journal/conference/preprint;
- volume/pages;
- arXiv ID/DOI if preprint.

Особенно важно для recent papers 2024--2025.

---

## 5. Предлагаемые локальные правки

### Patch 1: Lemma 28

```tex
Then
\[
  \frac{1}{\sqrt{2}} \le r_{n,n_0}(u) \le \sqrt{\frac{3}{2}},
\]
```

and later:

```tex
Hence \(r_{n,n_0}(u) \in [1/\sqrt{2},\sqrt{3/2}]\).
```

### Patch 2: Section 3 bound

Replace final display of Lemma 5 by:

```tex
\|u^\top(S_n-\mathbb E S_n)\|_{L_p}
\le C\|u\|\|\epsilon\|_\infty
\left(
  p^{3/2}t_{mix}^{1/2}\frac{1}{a}
  + p^{1/2}t_{mix}^{3/2}\sqrt{\frac{\alpha}{a}}
\right).
```

If you prefer keeping `sqrt(alpha)/a`, add an assumption or convention that `a <= 1`.

### Patch 3: burn-in wording

Replace:

```tex
when the burn-in window is logarithmic
```

with:

```tex
when the burn-in is at the mixing scale, of order \(\alpha^{-1}\) times logarithmic factors.
```

### Patch 4: imported inputs section

Add after assumptions or at the beginning of Section 4:

```tex
Imported input 1 (Markov concentration for time-inhomogeneous centered functions).
Imported input 2 (Bolthausen--Fan martingale Berry--Esseen).
Imported input 3 (Levin depth-two RR misadjustment bound).
Imported input 4 (Levin startup contraction/random-product stability).
```

Each input should have a precise statement and a citation.

### Patch 5: stationary theorem title

```tex
Theorem 3 (Stationary augmented-chain Berry--Esseen assembly for the comparison statistic).
```

### Patch 6: Section 5 theorem proof note

Add:

```tex
Since \(m\ge n/2\), all occurrences of \(n\) in polynomial prefactors can be replaced by \(m\) up to a universal constant; for instance \(\sqrt n/m \le \sqrt 2/\sqrt m\).
```

---

## 6. Итоговый список исправлений по приоритету

### Обязательно исправить

1. `sqrt(3)/2` → `sqrt(3/2)` in Lemma 28.
2. Степень `a` in Lemma 5: `sqrt(alpha/a)` vs `sqrt(alpha)/a`.
3. Уточнить stationary-limit transfer for `alpha -> 0`.
4. Выписать imported Levin/Samsonov assumptions and thresholds.
5. Исправить “logarithmic burn-in” wording.

### Желательно исправить

6. Усилить отделение stationary comparison statistic от actual finite-start estimator.
7. Унифицировать notation for RR weights.
8. Добавить explicit `m>=n/2` conversions in final proof.
9. Исправить broken brackets in Lemma 3.
10. Перевести references в полноценный bibliographic format.

### Можно исправить после основной математики

11. Section-prefixed theorem numbering.
12. Убрать разговорное “thesis-facing”.
13. Сократить некоторые длинные explanatory paragraphs или вынести в remarks.

---

## 7. Общая оценка

После исправления указанных мест работа будет выглядеть математически связной. Самый сильный аспект — аккуратное разделение stationary augmented-chain theorem and deterministic-start burn-in transfer. Главный риск — не алгебра RR-весов, а self-contained статус финальной теоремы: сейчас она зависит от нескольких импортированных bounds, которые нужно оформить в виде чётких предпосылок. Если это сделать, логика доказательства станет намного надёжнее для читателя.

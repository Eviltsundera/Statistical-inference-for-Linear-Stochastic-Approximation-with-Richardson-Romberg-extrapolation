# Ревью доказательств текущей версии диплома

**Файл:** `main.pdf`  
**Тема:** Statistical Inference for Linear Stochastic Approximation with Richardson–Romberg Extrapolation  
**Дата ревью:** 2026-05-20  
**Фокус:** только текущие материалы и доказательства; работа не оценивается как завершённая версия.

---

## 0. Краткий итог

Общая математическая стратегия выглядит состоятельной: работа правильно разделяет last-iterate RR object и PR-averaged RR statistic, использует естественную архитектуру

1. stationary full-window Berry–Esseen / CLT;
2. контроль RR-весов;
3. Poisson / Abel martingale approximation;
4. перенос stationary результата на deterministic-start burned-in statistic.

Однако текущую версию доказательств **нельзя считать полностью закрытой**. Главный пробел находится в **Lemma 22**, в convolution bound для координаты `H^(2)`. Из-за него финальный burned-in theorem и corollary сейчас должны считаться доказанными только условно.

---

## 1. Приоритеты замечаний

| Приоритет | Замечание | Влияние |
|---|---|---|
| P0 | Lemma 22: неверный convolution step для `H^(2)` | Блокирует Theorem 7 / Corollary 12 в текущем виде |
| P1 | Неверная степень `a` в Section 2.4 | Не меняет rate по `alpha`, но портит stated constants |
| P1 | Несогласованность в `a`-зависимости для `U_R` в Lemma 5 | Не меняет rate по `alpha`, но формула некорректна без огрубления |
| P1 | Необоснованный local inverse bound `||(I-wA)^(-1)|| <= 2` | Требуется дополнительное small-step условие |
| P2 | Lemma 13: stationary two-sided transfer написан слишком быстро | Нужна аккуратная фиксация зависимости от `w` |
| P2 | Corollary 5: admissible gamma-window верен, но пояснение неполное | Улучшить читаемость и убедительность |
| P2 | Corollary 12: upper bound на burn-in стоит явно вынести | Важно для финальной thesis-facing statistic |
| P3 | Нотация, опечатки, верстка | Локальные исправления |

---

## 2. P0: Lemma 22, convolution для `H^(2)`

### Где

Lemma 22: **“Full-state startup contraction for the depth-two augmented remainder”**.

### Что написано по смыслу

Для координаты `H^(2,w)` используется представление вида

$$
H^{(2,w)}_k
= -w \sum_{\ell=1}^k \Gamma^{(w)}_{\ell+1:k}\,\widetilde A(Z_\ell)J^{(2,w)}_{\ell-1}.
$$

Дальше применяется product-stability estimate вида

$$
\|\Gamma^{(w)}_{s+1:k}V_s\|_{L^p}
\le C e^{-cwa(k-s)/p}\|V_s\|_{L^{2p}}.
$$

Но в следующем convolution estimate фактически используется ядро без деления на `p`:

$$
e^{-cwa(k-\ell)}.
$$

### Почему это ошибка

Из stated product-stability estimate следует convolution с экспонентой

$$
e^{-cwa(k-\ell)/p},
$$

а не

$$
e^{-cwa(k-\ell)}.
$$

Если аккуратно использовать именно имеющуюся оценку, то получаем

$$
w\sum_{\ell=1}^k
 e^{-cwa(k-\ell)/p}e^{-cwa\ell/p}
= wk e^{-cwak/p}.
$$

После стандартного “lose half the exponent” argument:

$$
wk e^{-cwak/p}
\lesssim \frac{p}{a}e^{-c'wak/p}.
$$

То есть в bound для startup term возникает дополнительный множитель порядка

$$
\frac{p}{a}.
$$

В текущем тексте этот множитель отсутствует.

### Почему это критично

Lemma 22 используется дальше для startup transfer of augmented remainders. Через неё проходят:

- Lemma 24;
- Theorem 7;
- Corollary 12.

Поэтому Theorem 7 и Corollary 12 в текущем виде зависят от неверного шага.

### Как исправить

Есть два возможных пути.

#### Вариант A: доказать более сильную product-stability estimate

Нужно получить оценку без деления на `p` в экспоненте:

$$
\|\Gamma^{(w)}_{s+1:k}V_s\|_{L^p}
\le C e^{-cwa(k-s)}\|V_s\|_{L^{2p}}.
$$

Тогда текущий convolution step можно сохранить.

#### Вариант B: переписать Lemma 22 с дополнительным множителем

Если stronger estimate недоступна, нужно честно включить фактор

$$
\frac{p}{a}
$$

в `A_st(p,q,w)` или аналогичный startup constant, а затем заново пройти цепочку:

```text
Lemma 22 -> Lemma 24 -> Theorem 7 -> Corollary 12
```

Вероятно, итоговая balanced скорость `n^{-1/4}` сохранится up to polylogs, потому что при `p ~ log n` дополнительный `p` может быть поглощён логарифмами, а `a^{-1}` — problem constant. Но это надо явно проверить.

### Рекомендуемая формулировка исправления

В Lemma 22 заменить convolution argument на что-то вроде:

$$
\begin{aligned}
&w\sum_{\ell=1}^k
 e^{-cwa(k-\ell)/p}e^{-cwa\ell/p} \\
&\qquad = wk e^{-cwak/p} \\
&\qquad \le C\frac{p}{a}e^{-c'wak/p},
\end{aligned}
$$

и затем перенести `p/a` в startup constant.

---

## 3. P1: степень `a` в Section 2.4

### Где

Section 2.4, scalar `L^p` bound for the zeroth-order RR last-iterate term.

### Что написано

После суммирования квадратов получается

$$
\sum_j \|g^u_j\|_\infty^2
\le
\frac{16\alpha\|u\|^2\widetilde C_A^2\|\epsilon\|_\infty^2}{a^3}.
$$

Далее используется variance proxy

$$
v_n^2 = 64t_{\rm mix}\sum_j\|g^u_j\|_\infty^2.
$$

В тексте определено

$$
\widehat C_A
:=
32\widetilde C_A\|\epsilon\|_\infty\sqrt{t_{\rm mix}}/a^3.
$$

### Почему это ошибка

Из предыдущих двух формул следует

$$
v_n
\le
32\|u\|\widetilde C_A\|\epsilon\|_\infty
\sqrt{t_{\rm mix}}\frac{\sqrt\alpha}{a^{3/2}}.
$$

Значит естественная константа:

$$
\widehat C_A
=
32\widetilde C_A\|\epsilon\|_\infty\sqrt{t_{\rm mix}}\,a^{-3/2}.
$$

А не `a^{-3}`.

### Как исправить

Либо заменить определение на

$$
\widehat C_A
:=
32\widetilde C_A\|\epsilon\|_\infty\sqrt{t_{\rm mix}}\,a^{-3/2},
$$

либо написать, что константа intentionally enlarged при дополнительном соглашении `a <= 1`:

$$
a^{-3/2}\le a^{-3} \quad \text{if } 0<a\le1.
$$

Сейчас такого соглашения рядом нет, поэтому формально это арифметическая ошибка.

---

## 4. P1: `a`-зависимость для `U_R` в Lemma 5

### Где

Lemma 5, Step 1: bound on `u^T U_R`.

### Что написано

Получается оценка вида

$$
\|u^\top U_R\|_{L^p}
\le
C p^{1/2}t_{\rm mix}^{3/2}\|\epsilon\|_\infty
\frac{1}{\sqrt{\alpha a}}.
$$

Потом используется

$$
S_n - \mathbb E S_n = -\alpha(U_M+U_R).
$$

### Что должно следовать

Вклад `U_R` после умножения на `alpha` равен

$$
C p^{1/2}t_{\rm mix}^{3/2}\|\epsilon\|_\infty
\sqrt{\frac{\alpha}{a}}.
$$

В итоговой формуле записано фактически

$$
C p^{1/2}t_{\rm mix}^{3/2}\|\epsilon\|_\infty
\frac{\sqrt\alpha}{a}.
$$

Это более грубая оценка, но она не следует без дополнительного соглашения вроде `a <= 1`.

### Как исправить

Лучше заменить итоговый second term на

$$
p^{1/2}t_{\rm mix}^{3/2}\|\epsilon\|_\infty\sqrt{\frac{\alpha}{a}}.
$$

Если хочется оставить `sqrt(alpha)/a`, нужно явно написать, что это deliberate enlargement under `a <= 1`.

---

## 5. P1: local inverse bound `||(I-wA)^(-1)|| <= 2`

### Где

Используется при переходе от shifted first-order term

$$
T^{(1,w)}_k = (I-wA)J^{(1,w)}_k
$$

к оценке на `J^(1,w)_k`, а также в boundary / shifted estimates.

### Что написано

Используется

$$
\|(I-wA)^{-1}\|\le 2,
\qquad w\in\{\alpha,2\alpha\}.
$$

### Почему это не следует из текущих assumptions

Lyapunov contraction

$$
\|I-\alpha A\|_Q^2\le 1-\alpha a
$$

не даёт автоматически Euclidean inverse bound

$$
\|(I-wA)^{-1}\|\le2.
$$

Такой bound можно получить, например, из Neumann series, если

$$
w\|A\|\le \frac12.
$$

### Как исправить

Добавить в admissible stepsize ceiling условие, например

$$
2\alpha \le \frac{1}{2\|A\|},
$$

или эквивалентное условие в выбранной норме. Тогда для `w in {alpha, 2alpha}` будет

$$
w\|A\|\le \frac12,
$$

и

$$
\|(I-wA)^{-1}\|
\le \frac{1}{1-w\|A\|}
\le 2.
$$

---

## 6. P2: Lemma 13, stationary two-sided transfer

### Где

Lemma 13: stationary two-sided transfer / construction of stationary solution from finite-past approximations.

### Проблема

В доказательстве используется domination для double series вида

$$
Cw^2s(1-wa)^{s/2}\|\epsilon\|_\infty.
$$

Затем говорится, что

$$
\sum_s s(1-wa)^{s/2}<\infty,
$$

поэтому finite-past approximations являются Cauchy.

### Почему нужно уточнение

Для фиксированного `w > 0` это верно. Но сумма масштабируется как

$$
(wa)^{-2}.
$$

То есть domination не uniform in `w`.

### Как исправить

Добавить фразу:

> The convergence is meant for each fixed admissible `w > 0`; the dependence on `w` of the dominating series is not uniform and is tracked later through the explicit `w`-dependent bounds.

Это снимет возможную претензию к hidden uniform domination.

---

## 7. P2: Corollary 5 и admissible `gamma`-window

### Где

Corollary 5, где рассматривается

$$
\alpha_n = cn^{-\gamma}.
$$

### Текущий вывод

Условия сводятся к

$$
p_n^3(n\alpha_n)^{-1/2}\Lambda_n^{1/p_n}\to0,
$$

и

$$
p_n^{7/2}\sqrt n\,\alpha_n^{3/2}\Lambda_n^{3/2}\to0.
$$

Для `alpha_n = cn^{-gamma}` получается окно

$$
\frac13 < \gamma < 1.
$$

### Статус

Идея правильная modulo logarithms:

$$
(n\alpha_n)^{-1/2}=n^{-(1-\gamma)/2}\to0
\iff \gamma<1,
$$

$$
\sqrt n\,\alpha_n^{3/2}
=n^{1/2-3\gamma/2}\to0
\iff \gamma>1/3.
$$

### Что стоит добавить

Нужно явно сказать, почему остальные terms исчезают:

- `p^{3/2}sqrt(alpha_n) -> 0`, потому что `alpha_n -> 0`;
- `Phi(p_n, alpha_n)n^{-1/2} -> 0` при выбранном logarithmic `p_n`, так как polynomial decay dominates logs.

Это не ошибка результата, но текущий текст может выглядеть так, будто часть terms была забыта.

---

## 8. P2/P0: Theorem 7 зависит от исправления Lemma 22

### Где

Theorem 7: balanced-scale burned-in bound при

$$
\alpha = cn^{-1/2},
\qquad m=n-n_0,
\qquad p\simeq \log n.
$$

### Что выглядит правильно

Логика theorem хорошая:

- deterministic transient гасится burn-in;
- Poisson Abel remainder даёт lower-order term;
- variance comparison даёт contribution порядка `(m alpha a)^(-1)`;
- stationary BE part остаётся главным, порядка `n^{-1/4}` up to logarithms.

### Что не закрыто

Startup transfer for augmented remainders идёт через Lemma 22. Поэтому пока Lemma 22 не исправлена, Theorem 7 нельзя считать доказанным как написано.

### Что проверить после исправления Lemma 22

Если в Lemma 22 появляется дополнительный множитель `p/a`, нужно проверить:

1. как меняется `alpha_st(p)`;
2. как меняется lower burn-in condition;
3. остаётся ли `p ~ log n` достаточным;
4. сохраняется ли итоговая скорость `n^{-1/4}` up to polylogs.

Ожидаемо rate сохранится, но это нужно явно провести в тексте.

---

## 9. P2: Corollary 12 и upper bound на burn-in

### Где

Corollary 12, где осуществляется переход от `sqrt(m)`-normalization к `sqrt(n)`-normalization.

### Замечание

Там появляется дополнительное условие

$$
n_0 \le C_0(\alpha a)^{-1}\log^2 n.
$$

Это условие важно: финальная thesis-facing statistic требует не только достаточно большого burn-in, но и burn-in not too large.

### Как улучшить

Вынести условия на burn-in в theorem/corollary statement как пару:

$$
\frac{c_1}{\alpha a}\log n
\le n_0
\le
\frac{c_2}{\alpha a}\log^2 n.
$$

Если нижняя граница содержит дополнительный factor `p`, то при `p ~ log n` можно записать:

$$
n_0 \gtrsim \frac{\log^2 n}{\alpha a}.
$$

После исправления Lemma 22 эту пару bounds нужно пересчитать.

---

## 10. Нотационные и технические замечания

### 10.1. Переиспользование `C_A`

В assumptions `C_A` обозначает sup-norm constant:

$$
C_A = \max\{\sup_z\|A(z)\|,\sup_z\|\widetilde A(z)\|\}.
$$

В Section 2.3 локально вводится

$$
C_A := \kappa_Q.
$$

Даже если это пояснено, формулы вроде

$$
\widetilde C_A := C_A C_A
$$

становятся плохо читаемыми.

**Рекомендация:** заменить локальное обозначение на

$$
C_{\rm Lyap}:=\kappa_Q.
$$

Тогда

$$
\widetilde C_A = C_A C_{\rm Lyap}.
$$

### 10.2. Опечатка в Lemma 3

Сейчас:

```text
E⟦X|^p]
```

Должно быть:

$$
\mathbb E|X|^p.
$$

### 10.3. Сломанная верстка в Eq. (84)

Сейчас встречается фрагмент вида

```text
Phi(84) p, alpha)
```

Нужно заменить на нормальное

$$
\Phi(p,\alpha).
$$

### 10.4. Разделить last-iterate RR и PR-averaged RR в формулировках

Ты уже правильно пишешь, что Chapter 2 last-iterate RR object не совпадает с PR-averaged RR estimator. Это стоит сохранить и, возможно, дополнительно подчеркнуть в начале Chapter 2, чтобы читатель не переносил bounds из last-iterate analysis напрямую на PR statistic.

---

## 11. Что выглядит корректно и сильное

### 11.1. Общая architecture доказательства

Хорошо устроена цепочка:

```text
RR deterministic weights
-> Poisson / Abel decomposition
-> martingale Berry–Esseen
-> stationary full-window theorem
-> burned-in deterministic-start transfer
```

Это естественная и убедительная архитектура для задачи.

### 11.2. Отделение last-iterate RR от PR-averaged RR

Это важная методологическая аккуратность. Last-iterate RR object имеет другую весовую структуру, и ты не смешиваешь его напрямую с финальным PR-averaged RR statistic.

### 11.3. RR cancellation логически встроен правильно

Стационарный bias вида

$$
\theta^\ast + \alpha\Delta + \text{higher order terms}
$$

и RR-комбинация

$$
2\theta^{(\alpha)} - \theta^{(2\alpha)}
$$

используются в правильном направлении: leading linear bias должен отменяться.

### 11.4. Правильно замечена проблема zero-start full average

В тексте правильно отмечено, что stationary augmented-chain theorem нельзя напрямую применять к zero-start full average, потому что accumulated startup contribution имеет порядок

$$
\frac{1}{\sqrt n\,\alpha a}.
$$

Это зрелое замечание: действительно нужен burn-in или отдельная transfer lemma.

### 11.5. Target covariance выбран разумно

Цель

$$
\Sigma_\infty = A^{-1}\Sigma_\epsilon^{(M)}A^{-\top}
$$

корректна как averaged-SA covariance target. Хорошо, что в тексте осторожно сказано, что полный Hájek–Le Cam optimality statement не доказывается.

---

## 12. Минимальный план исправления

### Шаг 1. Починить Lemma 22

Выбрать один из двух путей:

- доказать product stability без `/p` в экспоненте;
- или внести дополнительный factor `p/a` в startup constant.

### Шаг 2. Обновить downstream constants

После правки Lemma 22 пройти:

```text
Lemma 22
-> Lemma 24
-> Theorem 7
-> Corollary 12
```

и явно проверить balanced rate.

### Шаг 3. Исправить степени `a`

Исправить:

- `widehat C_A` в Section 2.4;
- contribution of `U_R` в Lemma 5;
- все места, где используется огрубление `a^{-1/2} <= a^{-1}` или похожее, снабдить assumption `a <= 1` либо не огрублять.

### Шаг 4. Добавить small-step ceiling для inverse bound

В общий admissible stepsize condition добавить что-то вроде

$$
2\alpha\|A\|\le \frac12.
$$

### Шаг 5. Уточнить stationary-past construction

В Lemma 13 явно написать, что convergence is for fixed `w`, а domination не uniform in `w`.

### Шаг 6. Привести финальные assumptions в theorem statements

Особенно для burned-in theorem:

- lower burn-in;
- upper burn-in для перехода `sqrt(m)` -> `sqrt(n)`;
- small-step ceiling;
- dependence on `p ~ log n`.

---

## 13. Suggested patch text snippets

### 13.1. Patch для Lemma 22

```text
Using the stated product-stability bound with the exponent divided by p, the convolution gives

w sum_{ell=1}^k exp{-cwa(k-ell)/p} exp{-cwa ell/p}
= wk exp{-cwak/p}.

Consequently, after losing a constant factor in the exponent,

wk exp{-cwak/p} <= C (p/a) exp{-c'wak/p}.

Thus the startup coefficient A_st must be enlarged by a factor p/a unless one invokes a stronger product-stability estimate with exponent independent of p.
```

### 13.2. Patch для Section 2.4

```text
Plugging the square-sum estimate into the variance proxy gives

v_n <= 32 ||u|| Ctilde_A ||epsilon||_infty sqrt(t_mix) sqrt(alpha) / a^{3/2}.

Thus we set

Chat_A := 32 Ctilde_A ||epsilon||_infty sqrt(t_mix) a^{-3/2}.
```

Если хочется оставить `a^{-3}`:

```text
Assuming without loss of generality that 0 < a <= 1, we may enlarge a^{-3/2} to a^{-3}.
```

### 13.3. Patch для local inverse bound

```text
We further restrict the admissible stepsize so that 2 alpha ||A|| <= 1/2. Then, for w in {alpha, 2 alpha}, the Neumann series gives

||(I-wA)^{-1}|| <= (1-w||A||)^{-1} <= 2.
```

---

## 14. Финальный статус доказательств после ревью

На текущий момент я бы классифицировал proof status так:

| Блок | Статус |
|---|---|
| Общая постановка и motivation | Хорошо |
| RR cancellation intuition | Хорошо |
| Deterministic RR weight bounds | В целом хорошо |
| Zeroth-order scalar bound | Rate верен, constants по `a` требуют правки |
| Lemma 5 / first-order centered bound | Rate по `alpha` ок, `a`-зависимость требует правки |
| Stationary full-window theorem | Логика хорошая, есть технические уточнения |
| Lemma 22 startup transfer | Требует исправления; главный blocker |
| Theorem 7 burned-in theorem | Условно, зависит от Lemma 22 |
| Corollary 12 thesis-facing statistic | Условно, зависит от Lemma 22 и burn-in upper bound |

---

## 15. Литературный контекст для ориентации

Эти замечания не требуют менять основную идею работы. Она согласуется с современным контекстом:

- Samsonov, Sheshukova, Moulines, Naumov (2025), *Statistical inference for Linear Stochastic Approximation with Markovian Noise*, arXiv:2505.19102. В этой работе получены non-asymptotic Berry–Esseen bounds для PR-averaged LSA under Markovian noise со скоростью порядка `O(n^{-1/4})`.
- Levin, Naumov, Samsonov (2025), *High-Order Error Bounds for Markovian LSA with Richardson–Romberg Extrapolation*, arXiv:2508.05570. В этой работе анализируется Markovian LSA with RR extrapolation, leading linear bias in `alpha`, и high-order bounds для RR iterates.

Текущий диплом, судя по структуре, пытается совместить эти две линии: stationary Berry–Esseen для PR-averaged LSA и RR cancellation / high-order remainder analysis, затем добавить deterministic-start burned-in transfer. Это разумная цель, но startup transfer надо технически закрыть аккуратнее.

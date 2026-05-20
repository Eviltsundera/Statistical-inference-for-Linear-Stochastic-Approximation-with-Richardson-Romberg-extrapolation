# Ревью текущих доказательств в дипломе

**Работа:** *Statistical Inference for Linear Stochastic Approximation with Richardson–Romberg Extrapolation*  
**Файл:** `main.pdf`  
**Фокус ревью:** ошибки и пробелы в текущих доказательствах, без оценки незавершённых частей как финального текста.

---

## Общий вердикт

В текущих материалах есть несколько ошибок и пробелов в доказательствах. При этом общая архитектура работы выглядит жизнеспособной: RR-веса, Poisson–martingale decomposition, variance comparison и smoothing assembly в целом согласованы.

Главные проблемы сейчас не в основной идее, а в нескольких технических местах:

1. константы и степени параметра `a` в Section 2.4;
2. векторная концентрация в Lemma 2;
3. перенос stationary augmented-chain bounds на finite-start / burn-in setting;
4. формулировки некоторых теорем, где не хватает self-contained assumptions;
5. startup transfer для depth-two remainder в Section 5.8.

**Fatal conceptual error в основной RR/Poisson/Berry–Esseen схеме я не увидел.** Но в текущем виде доказательства нельзя считать полностью строгими из-за перечисленных ниже технических дыр.

Самая явная математическая ошибка: **потерянный множитель `1/a` в Section 2.4**.  
Самый важный недоказанный технический мост: **startup transfer для depth-two remainder, особенно для компоненты `H^{(2,w)}` в Section 5.8**.

---

## 1. Явная алгебраическая ошибка в Section 2.4: потерян множитель `1/a`

Это место стоит исправлять первым.

В Section 2.3 получено

```tex
\|H_j^{(n)}\| \le C_A (1-\alpha a)^{(n-j-1)/2}\frac{2}{\alpha a}.
```

Дальше в Section 2.4 подставляется

```tex
g_j(z)=-2\alpha^2 A H_j^{(n)}\epsilon(z).
```

Тогда должно получаться

```tex
\|g_j\|_\infty
\le
2\alpha^2 \|A\|
\cdot
C_A (1-\alpha a)^{(n-j-1)/2}
\frac{2}{\alpha a}
\|\epsilon\|_\infty
=
\frac{4\alpha}{a}\|A\|C_A\|\epsilon\|_\infty
(1-\alpha a)^{(n-j-1)/2}.
```

В тексте же в Eq. (37) написано

```tex
\|g_j\|_\infty
\le
4\alpha \widetilde C_A\|\epsilon\|_\infty
(1-\alpha a)^{(n-j-1)/2},
```

то есть множитель `1/a` исчезает.

После исправления, если не прятать `a^{-1}` внутрь константы, должно быть примерно

```tex
\sum_{j=1}^n \|g_j\|_\infty^2
\lesssim
\frac{\alpha \widetilde C_A^2\|\epsilon\|_\infty^2}{a^3},
```

а значит

```tex
u_n^2
\lesssim
\frac{\alpha\,t_{\rm mix}\,\widetilde C_A^2\|\epsilon\|_\infty^2}{a^3}.
```

Итоговый порядок `O(sqrt(alpha))` остаётся верным, но константа по `a` сейчас неправильная.

Дополнительная внутренняя несогласованность: даже если читать Eq. (38) буквально, из него получается `u_n^2 \lesssim alpha/a`, а значит `\widehat C_A` должен иметь зависимость `1/sqrt(a)`, а не `1/a`, если специально не используется дополнительное грубое неравенство.

### Рекомендация

- Переименовать локальный `C_A = \kappa_Q` в `C_H` или `C_Q^{loc}`, чтобы не конфликтовать с sup-norm constant из Assumption 2.
- Переписать Section 2.4 с явным учётом всех степеней `a`.
- Решить, какие параметры считаются “problem constants”, и придерживаться этого соглашения во всех главах.

---

## 2. Крайний индекс `j = n` в оценке `H_j^{(n)}`

В сумме для `H_j^{(n)}` при `j=n` внутренняя сумма пуста, то есть

```tex
H_n^{(n)}=0.
```

Но формулы с фактором

```tex
(n-j-1)/2
```

дают показатель `-1/2`, а геометрическая сумма в доказательстве фактически рассчитана для `j <= n-1`.

Это небольшая, но формальная ошибка.

### Исправление

Написать:

```tex
1 \le j \le n-1
```

и отдельно добавить:

```tex
H_n^{(n)}=0.
```

На итоговую оценку это не влияет.

---

## 3. Lemma 2: векторная концентрация требует отдельного обоснования

В Lemma 2 заявлена векторная sub-Gaussian tail bound для

```tex
\left\|\sum_i g_i(Z_i)\right\|
```

с prefactor `2`, без зависимости от размерности `d`, при

```tex
g_i: Z \to \mathbb R^d.
```

Это сильное утверждение. Если ссылка на Durmus et al. даёт scalar Markov concentration для вещественных функций, то текущая формулировка не следует автоматически.

Есть два безопасных варианта.

### Вариант A: сделать Lemma 2 скалярной

Сформулировать Lemma 2 только для проекций:

```tex
\sum_i u^\top g_i(Z_i), \qquad \|u\|=1.
```

Это хорошо совместимо с Berry–Esseen-частью, потому что основной результат всё равно формулируется для фиксированной проекции

```tex
u^\top(\cdot).
```

### Вариант B: доказать настоящую vector-valued bound

Если нужна именно векторная формулировка, нужно явно доказать или процитировать Hilbert-valued / vector-valued Markov concentration. Тогда нужно объяснить, почему нет `d`-фактора, либо добавить соответствующую зависимость:

```tex
d^{1/q}, \quad \sqrt d, \quad \text{net argument}, \quad \text{или Pinelis-type bound}.
```

### Где это влияет

Проблема влияет не только на Section 2.4, но и на Section 3.1. Там сначала делается проекция на фиксированный `u`, а затем результат используется как bound на векторную `L_p`-норму выражения с `H_{k+1}^{(w)}\epsilon(Z_k)`.

Нужно либо полностью перевести этот участок в scalar-projection режим, либо добавить отдельный векторный аргумент.

---

## 4. Section 3.1: зависимость от `a` в `U_R`

Из Eq. (70) получается

```tex
\|U_R\|_{L_p}
\lesssim
p^{1/2}t_{\rm mix}^{3/2}\|\epsilon\|_\infty
\frac{1}{\sqrt{\alpha a}}.
```

После умножения на `alpha` в сборке

```tex
S_n - \mathbb E S_n = -\alpha(U_M + U_R)
```

это даёт

```tex
\alpha\|U_R\|_{L_p}
\lesssim
p^{1/2}t_{\rm mix}^{3/2}\|\epsilon\|_\infty
\frac{\sqrt\alpha}{\sqrt a}.
```

В Eq. (78) написано с `sqrt(alpha)/a`.

Это не проблема по порядку `alpha`, но снова проблема со степенями `a`. Если используется грубая переоценка

```tex
1/\sqrt a \le C/a,
```

нужно явно предположить допустимый диапазон `a` или сказать, что все зависимости от `a` поглощены в problem constants.

Сейчас зависимости от `a` показываются явно, поэтому лучше держать их точно.

---

## 5. Section 3.2: Eq. (84) нужно переписать как centered bound

В Section 3.2 правильно замечено, что bias сам по себе не является препятствием: RR cancellation убирает linear `alpha Delta`, а stationary bias становится `O(alpha^2)`.

Но после этого в Eq. (84) написано

```tex
\|D_1^{mis,RR}\|_{L_p}=O(\sqrt n\,\alpha).
```

Строго из предыдущего вывода следует скорее

```tex
\|D_1^{mis,RR}-\mathbb E D_1^{mis,RR}\|_{L_p}
\lesssim
\sqrt n\,\alpha\,\Phi(p,\alpha),
```

и отдельно

```tex
\|\mathbb E D_1^{mis,RR}\|
\lesssim
\sqrt n\,\alpha^2.
```

То есть Eq. (84) лучше переписать как centered statement plus bias term.

Вывод о том, что при `alpha \asymp n^{-1/2}` прямой depth-one bound даёт `O(1)` и поэтому непригоден для Berry–Esseen remainder, выглядит верным. Это не ошибка в стратегии, потому что позже используется Levin depth-two route. Но локальная формулировка Eq. (84) сейчас неточная.

---

## 6. RR weight bounds и Poisson/Abel part выглядят в основном корректно

Я специально проверил блок Section 4.2–4.6, потому что там легко получить sign/index error.

Базовые identities

```tex
\mathcal Q_l^{RR}-A^{-1}
=
-A^{-1}(2B_\alpha^k-B_{2\alpha}^k),
```

```tex
\mathcal Q_{l+1}^{RR}-\mathcal Q_l^{RR}
=
-2\alpha(B_\alpha^{k-1}-B_{2\alpha}^{k-1})
```

согласованы с определениями `Q_l^{(alpha)}`.

Оценка discrete derivative с дополнительным `alpha`-gain тоже выглядит правильной.

Poisson/Abel decomposition также выглядит знаково согласованной: boundary term справа исчезает за счёт

```tex
\mathcal Q_{n-1}^{RR}=0,
```

и remainder bound через total variation weights используется корректно.

Variance comparison

```tex
\Sigma_n^{RR}\to\Sigma_\infty
```

через

```tex
\sum_l\|\mathcal Q_l^{RR}-A^{-1}\|^2
```

выглядит как грубая, но допустимая оценка. Суммарные deterministic bounds в Eq. (108)–(109) тоже выглядят согласованными.

---

## 7. Stationary augmented-chain convention: нужно жёстче развести theorem statements

В Section 4.9 правильно замечено, что stationary augmented-chain theorem не является finite-start theorem.

Для zero-start full average перенос к stationary centered sums даёт accumulated startup contribution порядка

```tex
\frac{1}{\sqrt n\,\alpha a},
```

а не просто terminal `rho^n`.

Это очень важное замечание, и оно правильное.

### Риск текущей версии

Читатель может не сразу понять, какие результаты относятся к stationary augmented-chain statistic, а какие — к настоящему finite-start / burn-in statistic.

Section 5 действительно строит burn-in version, но в формулировках Section 4 лучше прямо добавить в заголовки theorem/corollary слова:

```text
stationary augmented-chain only
```

Иначе легко случайно процитировать stationary bound как finite-start bound.

---

## 8. Lemma 21 / startup transfer для `H^{(2,w)}` — самый слабый технический мост в Section 5

Section 5.8 — ключевой мост от stationary augmented-chain к finite-start. Идея правильная: после burn-in нужно контролировать discrepancy между finite-start perturbation variables и stationary augmented version.

Но Lemma 21 в текущем виде выглядит скорее как сильное импортированное утверждение, чем как полностью доказанная лемма.

### В чём проблема

Levin contraction даёт componentwise estimates для

```tex
J^{(0)}, \quad J^{(1)}, \quad J^{(2)}.
```

Но remainder `R_k^{(w)}` включает ещё

```tex
H_k^{(2,w)}.
```

Дальше используется representation

```tex
H_k^{(2,w)}
=
-w\sum_{l=1}^k
\Gamma_{l+1:k}^{(w)}\widetilde A(Z_l)J_{l-1}^{(2,w)}
```

и deterministic-product estimate из Levin Proposition 9; затем это поглощается в `A_{st}`.

Это правдоподобно, но как доказательство пока слишком сжато.

### Что нужно сделать

Нужно выбрать один из трёх путей:

1. **Расширить augmented state**, включив `H^{(2,w)}`, и доказать Wasserstein contraction для полного state.
2. **Сформулировать startup contraction для `H^{(2,w)}` как отдельную imported lemma** из Levin, если такая лемма там реально есть.
3. **Дать полноценную оценку разности двух `H^{(2,w)}`**, включая разность random products `\Gamma_{l+1:k}^{(w)}` в coupled trajectories.

Сейчас это главный proof gap в финальном burn-in theorem.

---

## 9. Theorem 7: assumptions нужно собрать в clean self-contained формулировку

Theorem 7 формулируется через:

- UGE;
- `pi(epsilon)=0`;
- bounded `epsilon`;
- Lyapunov contraction;
- Levin depth-two / startup-contraction bounds;
- `sigma^2(u)>0`.

Но для читателя этого недостаточно как самостоятельной теоремы. Нужно явно сказать, что действуют Assumptions 1–3 из начала работы, плюс boundedness of `A`, `\widetilde A`, плюс small-step admissibility conditions для всех imported Levin bounds.

Текущая формулировка

```text
the Levin depth-two and startup-contraction bounds
```

выглядит как assumption, а не как следствие предыдущих assumptions.

### Возможная формулировка

```text
Assume Assumptions 1–3. Moreover, assume the step size satisfies the admissibility conditions required by Levin et al. depth-two bounds, summarized by alpha_adm(p,q). Then ...
```

И отдельно указать, что Lemma 21 либо доказана, либо принимается как imported proposition.

---

## 10. Predictable quadratic variation: по сути нормально, но есть опасная нотация

В burned-in quadratic variation block задаются scalar functions `h_l`, `g_l` и используется Markov concentration для scalar centered time-inhomogeneous functions. Это выглядит корректнее, чем векторная Lemma 2, потому что здесь действительно scalar quantity.

Но в stationary chapter и вокруг Poisson martingale decomposition нужно аккуратно развести обозначения:

- `epsilon(z)` — noise vector;
- объект вида

```tex
P\widehat\epsilon\widehat\epsilon^\top(z)
-
P\widehat\epsilon(z)P\widehat\epsilon(z)^\top
```

— matrix conditional covariance.

Лучше не обозначать их похожими символами. Иначе в variance comparison / quadratic variation легко спутать vector noise и matrix conditional covariance.

---

## 11. Что выглядит уже хорошо и не требует радикальной переделки

Основная линия Section 4 выглядит здоровой:

- deterministic RR weights получены правильно;
- discrete derivative действительно получает extra `alpha`-factor;
- Poisson equation используется с deterministic coefficients, поэтому Abel summation работает;
- martingale Berry–Esseen step структурно согласован;
- stationary-to-asymptotic normalization через Gaussian comparison выглядит корректно;
- balanced scale `alpha = c n^{-1/2}` действительно даёт `n^{-1/4}`-type remainder для нескольких competing terms.

Особенно хорошо, что работа явно отделяет “bad” depth-one misadjustment estimate `O(sqrt(n) alpha)` от later depth-two Levin route. Это правильное решение, а не ошибка.

---

## Приоритетный список правок

### Высокий приоритет

1. Исправить Section 2.4: восстановить потерянный `1/a`, пересчитать Eq. (37)–(40), убрать конфликт обозначений `C_A` vs `C_A`.
2. Решить, является ли Lemma 2 scalar или vector-valued. Если scalar — переписать все применения через projections или net argument.
3. Усилить Lemma 21: отдельно доказать startup contraction для `H^{(2,w)}` или явно вынести её в assumption / imported proposition.
4. Сделать Theorem 7 self-contained по assumptions.

### Средний приоритет

5. В Section 3.1 поправить переход “project onto `u`” ⇒ vector `L_p`-bound.
6. В Section 3.2 переписать Eq. (84) как centered estimate plus bias.
7. В Section 4 явно маркировать результаты как “stationary augmented-chain only”, где это применимо.

### Низкий, но важный для строгости приоритет

8. Добавить edge-case `j=n` в Section 2.3.
9. Почистить нотацию matrix conditional covariance vs noise vector `epsilon`.
10. Унифицировать соглашение о том, какие зависимости входят в constants, особенно по `a`, `t_mix`, `C_A`, `kappa_Q`.

---

## Suggested patch notes для текста

Ниже — короткие текстовые правки, которые можно почти напрямую вставлять в диплом.

### Patch для Section 2.3

```tex
For 1 \le j \le n-1, the estimate above gives ...
For j=n, the inner sum defining H_n^{(n)} is empty, hence H_n^{(n)}=0, and the same upper bound holds trivially after increasing the constant if needed.
```

### Patch для Section 2.4

```tex
Using the bound on H_j^{(n)}, we obtain
\[
\|g_j\|_\infty
\le
\frac{4\alpha}{a}\widetilde C_A\|\epsilon\|_\infty
(1-\alpha a)^{(n-j-1)/2}.
\]
Consequently,
\[
\sum_{j=1}^n \|g_j\|_\infty^2
\le
\frac{C\alpha}{a^3}\widetilde C_A^2\|\epsilon\|_\infty^2,
\]
where C is an absolute constant.
```

Точные constants нужно пересчитать в зависимости от того, как ты хочешь нормировать `\widetilde C_A`.

### Patch для Lemma 2, scalar version

```tex
Lemma 2'. Assume UGE. Let g_i:Z\to\mathbb R be measurable, bounded, and centered under \pi. Then ...
```

Если дальше нужен векторный результат, после scalar lemma можно добавить:

```tex
All applications below are to fixed one-dimensional projections. Therefore the scalar form is sufficient for the Berry--Esseen bounds for u^\top(\cdot).
```

### Patch для Eq. (84)

```tex
Therefore,
\[
\|D_{1}^{\rm mis,RR}-\mathbb E D_{1}^{\rm mis,RR}\|_{L_p}
\le
C\sqrt n\,\alpha\,\Phi(p,\alpha),
\]
and the stationary RR bias satisfies
\[
\|\mathbb E D_{1}^{\rm mis,RR}\|\lesssim \sqrt n\,\alpha^2.
\]
Thus the depth-one control is not sufficient at the balanced scale \alpha\asymp n^{-1/2}.
```

### Patch для Section 4 theorem statements

```tex
The following result is stated for the stationary augmented-chain statistic associated with the full-window weights. It should not be interpreted as a finite-start theorem for the original zero-start PR average.
```

### Patch для Lemma 21

```tex
The proof requires a startup contraction estimate for the full depth-two augmented state, including H^{(2,w)}. This estimate is either established below or assumed as an imported proposition from Levin et al. In particular, the bound is not a direct consequence of componentwise contraction for J^{(0)}, J^{(1)}, J^{(2)} alone.
```

---

## Итог

Текущая версия уже содержит сильную и логичную схему доказательства. Но для строгой дипломной версии нужно закрыть несколько технических мест:

- исправить явные константы в Section 2.4;
- не использовать без доказательства vector-valued concentration;
- чётко отделить stationary augmented-chain result от finite-start / burn-in theorem;
- усилить startup transfer для depth-two remainder;
- сделать финальную теорему self-contained.

После этих правок основная линия доказательства должна выглядеть значительно надёжнее.

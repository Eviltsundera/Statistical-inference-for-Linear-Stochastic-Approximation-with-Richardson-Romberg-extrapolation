# Ревью текущей версии диплома: замечания к доказательствам

**Файл:** `main.pdf`  
**Тема:** Statistical Inference for Linear Stochastic Approximation with Richardson–Romberg Extrapolation  
**Статус ревью:** текущие материалы, работа не закончена  
**Дата:** 2026-05-20

## Краткий итог

Общая архитектура доказательства выглядит перспективно: идея разложить RR PR-average на leading Markov-noise term, Poisson/martingale approximation и остатки естественна для Berry–Esseen анализа. Однако текущую версию нельзя считать полностью строгой. Есть несколько существенных логических разрывов:

1. смешиваются last iterate и Polyak–Ruppert average;
2. часть оценок заявлена для векторной нормы, но фактически доказана только для фиксированной скалярной проекции;
3. conditional concentration для future-centered сумм используется слишком быстро;
4. finite-start/burn-in transfer держится на Lemma 21, доказательство которой пока эскизное;
5. есть локальные ошибки в степенях параметра `a` и в обозначениях констант.

Ниже замечания разделены по приоритету.

---

## 1. Критические замечания

### 1.1. Смешиваются last iterate и PR average

**Место:** начало работы, Chapter 2, Chapter 4.  
**Приоритет:** критический.

В начале `theta_n^(alpha)` определяется как Polyak–Ruppert average, а RR-оценка записана как

```tex
\theta_n^{(\alpha, RR)} = 2\theta_n^{(\alpha)} - \theta_n^{(2\alpha)}.
```

Но в Chapter 2 анализируется last-iterate объект `J_n^(0,alpha)` и разложение для `theta_k - theta^*`. Далее в Chapter 4 используется PR-weight representation. Из-за этого есть риск, что last-iterate estimate применяется как estimate для averaged statistic без корректного суммирования.

**Почему это проблема:** last iterate и PR average имеют разные kernels, разные transient terms и разные нормировки. Bound для

```tex
J_n^{(0,\alpha)}
```

не является автоматически bound для

```tex
\frac{1}{n-n_0}\sum_{k=n_0}^{n-1}J_k^{(0,\alpha)}.
```

**Что исправить:** развести обозначения:

```tex
\theta_k^{(\alpha)} \quad \text{last iterate},
```

```tex
\bar\theta_{n,n_0}^{(\alpha)}
= \frac{1}{n-n_0}\sum_{k=n_0}^{n-1}\theta_k^{(\alpha)}.
```

Тогда RR average должен быть записан как

```tex
\bar\theta_{n,n_0}^{RR}
= 2\bar\theta_{n,n_0}^{(\alpha)}
- \bar\theta_{n,n_0}^{(2\alpha)}.
```

Объект из Chapter 2 лучше назвать, например,

```tex
\widetilde J_{n,\mathrm{last}}^{(0,\alpha)}
```

или прямо: “last-iterate zeroth-order RR difference”.

---

### 1.2. Lemma 4 заявлена как векторная, но доказана только для скалярной проекции

**Место:** Lemma 4, Eq. (54)–(55), Step 2, Eq. (71)–(78), затем Eq. (85)–(87).  
**Приоритет:** критический.

Lemma 4 утверждает bound вида

```tex
\|S_n - \mathbb E S_n\|_{L_p} \le \cdots,
```

то есть bound для евклидовой нормы векторного объекта. Но в доказательстве Step 2 начинается словами “Project onto a deterministic unit vector `u`”, и дальше доказывается estimate для скалярной проекции.

**Почему это проблема:** из оценки

```tex
\|u^\top X\|_{L_p} \le C
```

для фиксированного `u` не следует dimension-free оценка

```tex
\|X\|_{L_p} \le C.
```

Чтобы получить векторную оценку, нужен дополнительный аргумент. Например, в фиксированной размерности можно использовать координатный базис:

```tex
\|X\|_{L_p(\ell_2)}
\le
\left(\sum_{i=1}^d \|e_i^\top X\|_{L_p}^2\right)^{1/2}.
```

Но это даст явный dimension factor порядка `sqrt(d)` или другой зависимый от размерности множитель.

**Что исправить:** есть два варианта.

**Вариант A:** переформулировать Lemma 4 как скалярную:

```tex
\|u^\top(S_n-\mathbb E S_n)\|_{L_p}
\le C\|u\|\,\|\varepsilon\|_\infty(\cdots).
```

Это особенно естественно, если Berry–Esseen theorem дальше доказывается только для фиксированной projection `u`.

**Вариант B:** оставить векторную формулировку, но добавить отдельный шаг vectorization и явно указать зависимость от `d`.

---

### 1.3. Conditional concentration для future-centered term `U_M` пока не обоснована

**Место:** Lemma 4, Step 2, Eq. (71)–(76).  
**Приоритет:** критический.

В Step 2 рассматривается future-weighted объект

```tex
H_{k+1}^{(w)}u = \sum_{l=1}^{n-k} g_{k,l}(Z_{k+l}).
```

Дальше говорится, что `g_{k,l}` центрированы под `pi`, и применяется Markov concentration lemma к future chain conditionally on `F_k`.

**Почему это проблема:** условно на `F_k` будущая цепь стартует из `Z_k`, а не из stationary distribution `pi`. Поэтому `pi(g_{k,l}) = 0` не означает, что сумма центрирована относительно условного закона будущего path. В частности,

```tex
\mathbb E[g_{k,l}(Z_{k+l})\mid Z_k]
= (Q^l g_{k,l})(Z_k),
```

и это обычно не равно нулю.

**Что исправить:** нужно заменить рассуждение на одно из двух.

**Вариант A:** работать с условно центрированными функциями:

```tex
g_{k,l}(Z_{k+l}) - (Q^l g_{k,l})(Z_k).
```

Тогда отдельно контролировать deterministic/Markovian correction.

**Вариант B:** импортировать готовую proposition для future-centered bilinear Markov sums, например аналог Proposition 9 из Samsonov et al./Durmus et al., но с точной формулировкой:

- какие функции допускаются;
- какая норма используется;
- какой порядок по `p`, `t_mix`, `alpha`, `a` получается;
- требуется ли stationarity;
- что именно является centered object.

Сейчас Eq. (74)–(76) выглядит как главный логический разрыв в доказательстве Lemma 4.

---

### 1.4. Lemma 2 о Markov concentration для произвольного initial distribution нуждается в проверке

**Место:** Lemma 2, Eq. (33)–(35).  
**Приоритет:** высокий/критический, зависит от используемой версии результата.

Lemma 2 заявлена для любой initial distribution `xi`, если `pi(g_i)=0`. Но если цепь стартует не из `pi`, то сумма

```tex
\sum_i g_i(Z_i)
```

вообще говоря не центрирована относительно initial distribution `xi`.

**Почему это проблема:** tail bound вокруг нуля для non-stationary start без bias/correction term обычно не следует только из `pi(g_i)=0`. Может быть нужна либо stationarity, либо дополнительный term, либо специальная версия concentration inequality.

**Что исправить:**

1. Проверить точную формулировку Lemma 9 из Durmus et al. Если она действительно даёт такую форму для произвольного старта, нужно процитировать её дословно и объяснить, почему нет initial-bias term.
2. Если такой версии нет, то Lemma 2 нужно ограничить случаем `Z_1 ~ pi`.
3. Для finite-start theorem добавить correction term, зависящий от mixing от initial distribution.

---

### 1.5. Lemma 21 — главный незакрытый кусок finite-start theorem

**Место:** Chapter 5, Lemma 21, Theorem 7.  
**Приоритет:** критический.

Theorem 7 для deterministic-start/burn-in версии опирается на startup transfer. В тексте признаётся, что Levin contraction покрывает augmented chain с координатами примерно

```tex
(Z, J^{(0)}, J^{(1)}, J^{(2)}),
```

но не включает `H^{(2)}`. Затем вводится Lemma 21 как full-state startup contraction для remainder

```tex
R_k^{(w)} = J_k^{(1,w)} + J_k^{(2,w)} + H_k^{(2,w)}.
```

**Почему это проблема:** для `H^{(2)}` нужно сравнивать две копии

```tex
H_k^{(2,w)}
= -w\sum_{\ell=1}^k
\Gamma_{\ell+1:k}^{(w)}
\widetilde A(Z_\ell)
J_{\ell-1}^{(2,w)}.
```

При сравнении finite-start copy и stationary copy возникают одновременно:

- разности random products `Gamma - Gamma_tilde`;
- разности `A_tilde(Z_l) - A_tilde(Z_l_tilde)`;
- разности `J^{(2)} - J_tilde^{(2)}`;
- terms до coupling time и после coupling time.

One-trajectory moment bound для `H^{(2)}` не даёт автоматически contraction между двумя копиями.

**Что исправить:** Lemma 21 нужно вынести в отдельное полноценное technical proposition. Минимальный skeleton доказательства должен включать:

1. explicit coupling construction для `Z` и `Z_tilde`;
2. recursion для разности `Delta J^{(0)}`, `Delta J^{(1)}`, `Delta J^{(2)}`;
3. отдельный bound для `Delta H^{(2)}`;
4. summation over pre-coupling and post-coupling intervals;
5. final dependence on `n_0`, `alpha`, `a`, `t_mix`, `p`;
6. проверку, что после burn-in этот transfer subleading относительно целевого `n^{-1/4}` Berry–Esseen rate.

Пока Theorem 7 лучше формулировать условно:

```tex
Assume the full-state startup contraction of Lemma 21 holds. Then ...
```

---

## 2. Локальные алгебраические и нормировочные ошибки

### 2.1. Неверная степень `a` в константе `C_hat_A`

**Место:** Eq. (38)–(40).  
**Приоритет:** высокий.

Из Eq. (38):

```tex
\sum_j \|g_j^u\|_\infty^2
\le
\frac{16\alpha \|u\|^2 \widetilde C_A^2\|\varepsilon\|_\infty^2}{a^3}.
```

Так как

```tex
v_n^2 = 64t_{mix}\sum_j\|g_j^u\|_\infty^2,
```

получается

```tex
v_n^2
\le
\frac{1024\alpha\|u\|^2\widetilde C_A^2\|\varepsilon\|_\infty^2t_{mix}}{a^3}.
```

Значит корректная константа должна быть

```tex
\widehat C_A
= \frac{32\widetilde C_A\|\varepsilon\|_\infty\sqrt{t_{mix}}}{a^{3/2}},
```

а не с `a^{-3}`.

**Когда текущая запись могла бы быть допустима:** если явно предполагается `a <= 1` и делается намеренное ухудшение. Но это нужно написать. Без такого предположения bound с `a^{-3}` может быть меньше нужного при `a > 1` и не следовать из предыдущей строки.

---

### 2.2. В Eq. (78) у второго слагаемого, вероятно, неверная степень `a`

**Место:** Lemma 4, Step 1 и assembly, Eq. (70), Eq. (78).  
**Приоритет:** высокий.

Из Step 1 получается

```tex
\|U_R\|_{L_p}
\lesssim
p^{1/2}t_{mix}^{3/2}\|\varepsilon\|_\infty
\frac{1}{\sqrt{\alpha a}}.
```

После умножения на `alpha`:

```tex
\alpha\|U_R\|_{L_p}
\lesssim
p^{1/2}t_{mix}^{3/2}\|\varepsilon\|_\infty
\sqrt{\frac{\alpha}{a}}.
```

В тексте Eq. (78) выглядит как

```tex
p^{1/2}t_{mix}^{3/2}\frac{\sqrt\alpha}{a}.
```

Это снова допустимо только как ухудшение при `a <= 1`, но такое предположение нужно явно указать. Иначе степень `a` некорректна.

---

### 2.3. Опасный reuse символа `C_A`

**Место:** Assumption 2, Eq. (30), Eq. (36).  
**Приоритет:** средний/высокий.

В Assumption 2 `C_A` — это sup-norm bound на `A(z)` и `A_tilde(z)`. В Chapter 2 заново задаётся chapter-local constant

```tex
C_A := \kappa_Q,
```

после чего появляется

```tex
\widetilde C_A := C_A C_A = C_A\kappa_Q.
```

**Почему это проблема:** это не просто стилистика. Такое переиспользование может скрыть неверную зависимость констант от `C_A`, `kappa_Q`, `a`.

**Что исправить:** переименовать локальную константу, например:

```tex
C_H := \kappa_Q,
```

```tex
C_{sup} := \max(\sup_z\|A(z)\|, \sup_z\|\widetilde A(z)\|),
```

```tex
\widetilde C_A := C_{sup}C_H.
```

---

### 2.4. В Lemma 19 / Eq. (259) есть typographical/norm-order ошибка

**Место:** Lemma 19, Eq. (259), затем Lemma 23.  
**Приоритет:** средний/высокий.

В burned-in Poisson approximation bound должен быть порядок

```tex
\|D^{bRR}_{2,n,n_0}\|_\infty
\lesssim
\frac{t_{mix}\|\varepsilon\|_\infty}{\sqrt m}(\cdots).
```

В тексте вокруг Eq. (259) выглядит как `sqrt(||epsilon||_infty)/m` или как минимум неверно сверстанная дробь. Далее Lemma 23 использует именно порядок `C/sqrt(m)`, поэтому это, вероятно, typographical error.

**Что исправить:** привести Eq. (259) и все downstream constants к одному порядку. Проверить размерности: noise amplitude должен входить линейно как `||epsilon||_infty`, а не как `sqrt(||epsilon||_infty)`.

---

### 2.5. Формулировка Hurwitz condition в начале сбивает с толку

**Место:** Introduction и Assumption 2.  
**Приоритет:** средний.

В одном месте написано примерно:

```text
Assuming that -A is a Hurwitz matrix (all eigenvalues have strictly negative real parts)
```

Строго корректная формулировка:

```tex
-A \text{ is Hurwitz}
\quad \Longleftrightarrow \quad
\operatorname{Re}\lambda(A) > 0.
```

То есть eigenvalues of `-A` have negative real parts, equivalently eigenvalues of `A` have positive real parts. В Assumption 2 это уже написано корректно, но в Introduction лучше убрать двусмысленность.

---

## 3. Замечания к Chapter 3 и misadjustment route

### 3.1. Bound Eq. (85)–(87) слишком грубый и не даёт нужный Berry–Esseen remainder

**Место:** Section 3.2, Eq. (79)–(87).  
**Приоритет:** содержательный, но автор уже фактически это замечает.

В тексте выводится

```tex
\|D_{1,c}^{mis,RR}\|_{L_p}
\le C\sqrt n\,\alpha\Phi(p,\alpha).
```

При `alpha ~ n^{-1/2}` это даёт порядок `O(1)` для centered fluctuation, то есть не subleading относительно Gaussian leading term.

**Комментарий:** это не ошибка, если Section 3 задуман как демонстрация, что depth-one route недостаточен. Но нужно явно сказать, что Section 3 не используется как финальное доказательство Berry–Esseen theorem, а служит motivation для depth-two/Levin transfer.

---

### 3.2. Stationary bias cancellation для `J^(1)` требует аккуратной формулировки

**Место:** Eq. (81)–(82).  
**Приоритет:** средний.

Написано, что

```tex
\mathbb E_\pi[J_\infty^{(1,\alpha)}] = \alpha\Delta + O(\alpha^2),
```

поэтому RR-combination имеет bias `O(alpha^2)`. Это выглядит правильно, но нужно уточнить:

1. под какой stationary law берётся expectation: stationary law augmented chain for step `alpha` или stationary distribution of `Z` alone;
2. одинаков ли `Delta` для step sizes `alpha` и `2alpha`;
3. почему remainder достаточно uniform для применения к PR-scaled statistic.

---

## 4. Замечания к PR-weight / Poisson part

### 4.1. RR weight identities выглядят корректно

**Место:** Eq. (100)–(103).  
**Приоритет:** положительное замечание.

Следующие identities выглядят правильными:

```tex
Q_l^{(\alpha)} = A^{-1}(I-B_\alpha^{n-l}),
```

```tex
Q_{l+1}^{(\alpha)} - Q_l^{(\alpha)}
= -\alpha B_\alpha^{n-l-1},
```

```tex
\mathcal Q_l^{RR} - A^{-1}
= -A^{-1}(2B_\alpha^k - B_{2\alpha}^k),
```

```tex
\mathcal Q_{l+1}^{RR} - \mathcal Q_l^{RR}
= -2\alpha(B_\alpha^{k-1}-B_{2\alpha}^{k-1}).
```

Матрицы коммутируют, потому что это полиномы от `A`. Здесь существенной ошибки не видно.

---

### 4.2. Burned-in weight normalization нужно держать отдельно от full-window case

**Место:** Eq. (90), Chapter 5.  
**Приоритет:** средний.

Для burned-in average вводится вес

```tex
Q_{l,n_0}^{(\alpha)}
= \frac{n}{n-n_0}\alpha\sum_{k=\max(n_0,l)}^{n-1}B_\alpha^{k-l}.
```

Это правильное направление, но нужно строго отделить:

- stationary full-window theorem (`n_0 = 0`);
- finite-start theorem with burn-in (`n_0 > 0`);
- deterministic transient;
- dropped pre-burn-in samples;
- normalization by `sqrt(n)` vs `sqrt(m)`, where `m=n-n_0`.

В финальной теореме лучше зафиксировать одну нормировку и не менять её между леммами.

---

### 4.3. Poisson martingale approximation structure выглядит стандартно, но boundary terms нужно проверить

**Место:** Poisson equation / Abel summation часть.  
**Приоритет:** средний.

Схема

```tex
\widehat\varepsilon - Q\widehat\varepsilon = \varepsilon
```

и переход к martingale increments плюс Abel boundary remainder выглядит стандартно и правдоподобно. Но нужно явно проверить:

1. sup-norm bound на `epsilon_hat` через `t_mix ||epsilon||_infty`;
2. boundary terms при `l=1`, `l=n-1`;
3. изменение weights при burned-in window;
4. variance comparison между finite `Q_l^RR` и asymptotic `A^{-1}`;
5. nondegeneracy `sigma^2(u)>0` для Berry–Esseen normalization.

---

## 5. Finite-start / burn-in theorem

### 5.1. Stationary result и deterministic-start result правильно разведены, но transfer пока не закрыт

**Место:** Chapter 4–5, Theorem 7.  
**Приоритет:** критический для финальной теоремы.

Положительный момент: в тексте явно сказано, что stationary `n_0=0` result не является deterministic-start result, и нужен отдельный burn-in transfer. Это правильно.

Главная проблема остаётся Lemma 21. Пока её доказательство не доведено, финальный deterministic-start Berry–Esseen theorem должен быть либо:

1. условным;
2. перенесён в раздел “conjectured/remaining technical lemma”;
3. доказан полностью отдельным блоком.

---

### 5.2. Burn-in scale выглядит согласованным

**Место:** Theorem 7 assumptions.  
**Приоритет:** положительное замечание.

Если

```tex
\alpha = c n^{-1/2},
```

то burn-in порядка

```tex
n_0 \asymp (\alpha a)^{-1}\log^2 n
```

даёт

```tex
n_0 = O(\sqrt n\log^2 n) = o(n).
```

То есть условие `m=n-n_0 >= n/2` выполнимо для больших `n`. Это внутренне согласовано с целевым rate `polylog(n)n^{-1/4}`.

---

## 6. Замечания к формулировке новизны и литературному контексту

### 6.1. Формулировка “open problem” должна быть аккуратной

**Место:** Introduction, Problem statement and goals.  
**Приоритет:** высокий для позиционирования.

В текущей версии утверждается, что distributional approximation / CLT / Berry–Esseen для averaged RR iterates остаётся open problem. Это можно оставить, но формулировку нужно сделать точной.

Внешний контекст:

- Samsonov, Sheshukova, Moulines, Naumov получают non-asymptotic Berry–Esseen bounds для PR-averaged LSA under Markovian noise с rate `O(n^{-1/4})` и bootstrap validity results.
- Huo, Chen, Xie рассматривают statistical inference for constant-stepsize Markovian LSA и используют RR для bias reduction.
- Levin, Naumov, Samsonov получают high-order moment bounds для Markovian LSA with RR, включая leading term aligned with asymptotically optimal covariance.

Поэтому безопаснее писать не просто “CLT for constant-step LSA is open”, а:

```text
A non-asymptotic Berry–Esseen analysis for the PR-averaged Richardson–Romberg extrapolated statistic under Markovian noise, including deterministic-start / burn-in transfer, is not covered by the existing results in the exact form needed here.
```

Или:

```text
The contribution is an assembly of Berry–Esseen / CLT inference for the PR-averaged RR statistic, with deterministic-start and burn-in transfer, rather than a CLT for constant-stepsize LSA per se.
```

---

### 6.2. Нужно явно вынести hypotheses финальной теоремы

**Место:** final theorem statements.  
**Приоритет:** средний.

Даже если часть условий следует из определений, в theorem statement стоит явно указать:

```tex
\pi(\widetilde A)=0,
```

```tex
\pi(\varepsilon)=0,
```

```tex
\sigma^2(u)>0.
```

Также нужно указать:

- stationarity или non-stationary initial law;
- dependence constants;
- range of `alpha` and `2alpha`;
- relation между `n`, `n_0`, `m=n-n_0`;
- whether the result is scalar-projected or vector-valued.

---

## 7. Что выглядит корректно / удачно

### 7.1. RR deterministic weight identities

Как отмечено выше, identities Eq. (100)–(103) выглядят корректно. Они хорошо объясняют, где RR coupling реально даёт улучшение, а где triangle inequality его не видит.

### 7.2. Разделение stationary и finite-start результатов

Идея сначала доказать stationary full-window theorem, а затем делать burn-in transfer — правильная. Это лучше, чем пытаться сразу смешать все errors в одном proof.

### 7.3. Balanced scale `alpha ~ n^{-1/2}`

Выбор

```tex
\alpha \asymp n^{-1/2}
```

согласован с tradeoff между RR residual bias и stochastic approximation remainder. Burn-in scale также выглядит совместимым с `m ~ n`.

### 7.4. Poisson equation / martingale approximation

Использование Poisson equation для Markov noise и Abel summation по deterministic RR weights — естественная и стандартная структура для Berry–Esseen proof. Основная задача здесь не концептуальная, а техническая: аккуратно закрыть boundary terms, variance comparison и burned-in weights.

---

## 8. Рекомендованный порядок исправлений

### Шаг 1. Почистить нотацию

Развести:

```tex
\theta_k^{(\alpha)} \quad \text{last iterate},
```

```tex
\bar\theta_{n,n_0}^{(\alpha)} \quad \text{PR average},
```

```tex
\bar\theta_{n,n_0}^{RR} \quad \text{RR PR average}.
```

После этого пройтись по всем theorem/lemma statements и проверить, к какому объекту относится каждое утверждение.

### Шаг 2. Исправить Lemma 4

Лучший вариант для текущей цели — сделать Lemma 4 скалярной:

```tex
\|u^\top(S_n-\mathbb E S_n)\|_{L_p}\le\cdots.
```

Если нужен vector bound, добавить explicit dimension factor.

### Шаг 3. Переписать proof of `U_M`

Не применять обычную `pi`-centered concentration условно на `F_k`. Использовать условно центрированный объект или импортировать точную proposition для future-centered bilinear sums.

### Шаг 4. Доказать или условно вынести Lemma 21

Это главный bottleneck финального deterministic-start theorem. Без неё Theorem 7 лучше не подавать как полностью доказанный.

### Шаг 5. Исправить степени `a` и константы

Особенно:

- Eq. (39): `a^{-3/2}` вместо `a^{-3}`, если нет явного `a <= 1`;
- Eq. (78): `sqrt(alpha/a)` вместо `sqrt(alpha)/a`, если нет явного ухудшения;
- унифицировать `C_A`, `C_sup`, `kappa_Q`.

### Шаг 6. Переписать novelty paragraph

Сделать акцент на точном вкладе:

```text
Berry–Esseen / CLT assembly for the PR-averaged Richardson–Romberg statistic under Markovian noise, including deterministic-start / burn-in transfer.
```

Не позиционировать результат как первый CLT для constant-stepsize Markovian LSA вообще.

---

## 9. Краткий checklist перед следующей версией

- [ ] Везде различаются last iterate и PR average.
- [ ] RR estimator записан через averaged iterates, а не через ambiguous `theta_n`.
- [ ] Lemma 4 либо scalar, либо vector with dimension factor.
- [ ] Conditional future-chain concentration переписана строго.
- [ ] Lemma 2 проверена на arbitrary initial distribution или ограничена stationarity.
- [ ] Lemma 21 доказана полноценно или theorem сделан conditional.
- [ ] Eq. (39) исправлена по степени `a`.
- [ ] Eq. (78) исправлена по степени `a`.
- [ ] `C_A` больше не используется в двух разных смыслах.
- [ ] Eq. (259) исправлена по order и размерности.
- [ ] Hurwitz condition сформулирована единообразно.
- [ ] В theorem statements явно есть `sigma^2(u)>0`.
- [ ] В theorem statements явно указано, scalar-projected result или vector result.
- [ ] Literature positioning уточнено относительно Samsonov et al., Huo et al., Levin et al.

---

## 10. Внешние источники для позиционирования

Эти источники важны именно для формулировки новизны, а не для проверки всех внутренних доказательств:

1. Samsonov, Sheshukova, Moulines, Naumov, *Statistical inference for Linear Stochastic Approximation with Markovian Noise*, arXiv:2505.19102.  
   URL: https://arxiv.org/abs/2505.19102

2. Levin, Naumov, Samsonov, *High-Order Error Bounds for Markovian LSA with Richardson-Romberg Extrapolation*, arXiv:2508.05570.  
   URL: https://arxiv.org/abs/2508.05570

3. Huo, Chen, Xie, *Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference*, arXiv:2312.10894 / AAAI 2024.  
   URL: https://arxiv.org/abs/2312.10894


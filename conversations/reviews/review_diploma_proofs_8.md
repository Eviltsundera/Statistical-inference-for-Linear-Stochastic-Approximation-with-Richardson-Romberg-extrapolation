# Новые замечания по текущей версии диплома

Дата ревью: 2026-05-25  
Файл: `main.pdf`, текущая исправленная версия

## Общий вывод

Текущая версия стала существенно сильнее предыдущей. Главная проблема прошлой версии — скрытые внешние зависимости и неотделённый deterministic-start transfer — в основном закрыта. В тексте теперь явно есть:

- отдельный appendix с external inputs и local extensions;
- разделение stationary augmented-chain theorem и deterministic-start burned-in theorem;
- явные admissibility thresholds `alpha_stat`, `alpha_burn`, `alpha_st`;
- честная формулировка burn-in scale: при `alpha = c n^{-1/2}` burn-in имеет порядок `n^{1/2} log^2 n`, а не просто логарифмический порядок;
- явное указание, что full-state startup contraction и construction with `H^{(2)}` являются локальными леммами, а не дословными внешними теоремами.

Тем не менее, перед следующей отправкой я бы исправил несколько точечных проблем. Две из них proof-critical: формула Markov concentration и согласованность `A_st` с Corollary 11. Остальные — citation hygiene и аккуратность формулировок.

---

## Сводная таблица замечаний

| Приоритет | Место | Тип | Суть |
|---|---|---|---|
| Высокий | Lemma 2 / Lemma 27 / Eq. 36, Eq. 389 | Доказательство | В Markov concentration потерян квадратный корень по `sum c_i^2`; текущая формула размерностно неверна и не даёт последующие bounds. |
| Высокий | Lemma 23 / Corollary 11 / Eq. 318, Eq. 340 | Доказательство | `A_st(p,q,w)` содержит первый член без `sqrt(w/a)`, но Corollary 11 использует `A_st = polylog(n) alpha^{1/2}`. Это противоречие. |
| Средний | Lemma 30 / Eq. 394 | Citation / formula hygiene | `log^{1/p}` выглядит несогласованным с Levin-type working form; скорее должен быть `log^{1/2}` или нужно не называть это direct citation. |
| Средний | Section 3.2 / Eq. 78 | Citation hygiene | Ссылка на “Step (S8) of the Samsonov scheme” всё ещё осталась, хотя такой named step не определён локально. |
| Низкий | References | Стиль | Библиография всё ещё выглядит как рабочая, а не финальная: стоит унифицировать статусы, venues и arXiv references. |

---

## 1. Markov concentration: в формуле потерян квадратный корень

### Где

- Lemma 2, Eq. 36:

```tex
\left\|\sum_{i=1}^n g_i(Z_i)\right\|_{L_p(\xi)}
\le C_{MC,0}\sqrt p\, t_{mix}\sum_{i=1}^n c_i^2.
```

- Appendix Lemma 27, Eq. 389:

```tex
\left\|\sum_{i=1}^N g_i(Z_i)\right\|_{L_p(\xi)}
\le C_{MC}\sqrt p\, t_{mix}\sum_{i=1}^N c_i^2.
```

- Далее эта же проблема проявляется в proof of Lemma 9, особенно Eq. 154 / Eq. 300, где текст использует форму с `sum c_i^2`, но следующий bound фактически соответствует root-form.

### Проблема

В таком виде правая часть квадратична по коэффициентам `c_i`, а должна быть линейна. Если заменить все `g_i` на `lambda g_i`, левая часть масштабируется как `lambda`, а текущая правая часть — как `lambda^2`. Поэтому current display не может быть верной концентрационной оценкой.

Кроме того, дальнейшие оценки уже используют правильную root-form. Например, в Eq. 39 получено

```tex
\sum_j \|g_j^u\|_\infty^2 = O(\alpha),
```

а в Eq. 41 из этого делается вывод порядка `O(sqrt(alpha))`. Такой вывод возможен только из оценки вида

```tex
\left(\sum_j \|g_j^u\|_\infty^2\right)^{1/2},
```

а не из самой суммы `sum_j ||g_j^u||^2`.

### Исправление

Заменить Lemma 2 и Lemma 27 на согласованную форму. Я бы использовал такую:

```tex
\left\|\sum_{i=1}^N g_i(Z_i)\right\|_{L_p(\xi)}
\le
C_{MC}\sqrt{p\,t_{mix}}
\left(\sum_{i=1}^N c_i^2\right)^{1/2}.
```

Если в твоём источнике удобнее оставить более грубую зависимость от mixing time, можно написать:

```tex
\left\|\sum_{i=1}^N g_i(Z_i)\right\|_{L_p(\xi)}
\le
C_{MC}\sqrt p\,t_{mix}
\left(\sum_{i=1}^N c_i^2\right)^{1/2}.
```

Но тогда нужно последовательно пересчитать powers of `t_mix`. Текущие дальнейшие оценки, например predictable-variation bound с `t_mix^{5/2}`, выглядят согласованнее с первой формой `sqrt{p t_mix}`.

### Что нужно поправить дальше по тексту

После исправления основной леммы нужно поправить фразы вида:

```tex
The only dependence on the coefficient sequence is through
\sqrt p\, t_{mix}\sum_i c_i^2.
```

на

```tex
The dependence on the coefficient sequence is through
\left(\sum_i c_i^2\right)^{1/2}.
```

Также в proof of Lemma 9 и Lemma 21 нужно заменить промежуточные displays типа

```tex
C_{MC}\sqrt p\,t_{mix}\sum_i c_i^2
```

на root-form. Например, в Lemma 9 должно быть примерно:

```tex
\left\|\sum_{l=2}^{n-1} g_l(Z_{l-1})\right\|_{L_p(\xi)}
\le
C_{MC}\sqrt{p\,t_{mix}}
\left(\sum_{l=2}^{n-1} c_l^2\right)^{1/2}
\le
C\,C_{\mathcal Q}^2\|u\|^2\|\epsilon\|_\infty^2
 t_{mix}^{5/2}\sqrt{pn}.
```

И аналогично в Lemma 21.

### Severity

High. Сейчас это выглядит как typographical error, но формально ломает цепочку Lemma 2 → Eq. 41 → Lemma 9 / Lemma 21 → martingale Berry–Esseen assembly.

---

## 2. `A_st(p,q,w)` не согласован с Corollary 11

### Где

Lemma 23, Eq. 318:

```tex
A_{st}(p,q,w)
:= C_{st}(1+d^{1/q})p^7
+ \frac{p^8}{a}t_{mix}^5\sqrt{w/a}\log^3(1/(wa)).
```

Corollary 11 далее утверждает, что при `alpha = c n^{-1/2}`:

```tex
A_{st}(p,q,alpha) = polylog(n) alpha^{1/2},
```

и поэтому startup discrepancy имеет порядок

```tex
polylog(n) n^{-1/4-\beta}.
```

### Проблема

Из текущей формулы Eq. 318 это не следует, потому что первый член

```tex
C_{st}(1+d^{1/q})p^7
```

не убывает по `w`. Тогда

```tex
A_{st}(p,q,alpha)/(alpha sqrt m)
```

на balanced scale `alpha = c n^{-1/2}`, `m ~ n`, будет порядка `polylog(n)`, а не `polylog(n)n^{-1/4}`. В этом случае Corollary 11 даёт startup term порядка `polylog(n)n^{-\beta}`, а не `polylog(n)n^{-1/4-\beta}`.

При этом в proof of Lemma 23 сам `J`-scale выводится как

```tex
A_J(p,q,w)
= C(1+d^{1/q})p^7 t_{mix}^5 \sqrt{w/a}\log^3(1/(wa)),
```

то есть с множителем `sqrt(w/a)`. Похоже, в итоговом display для `A_st` этот множитель случайно потерян в первом слагаемом.

### Исправление

Лучший вариант — исправить Eq. 318 так, чтобы весь `A_st` имел `sqrt(w/a)`:

```tex
A_{st}(p,q,w)
:=
C_{st}(1+d^{1/q})t_{mix}^5\sqrt{w/a}\log^3(1/(wa))
\left(p^7 + \frac{p^8}{a}\right).
```

Можно оставить две строки, если хочешь сохранить происхождение terms:

```tex
A_{st}(p,q,w)
:=
C_{st}(1+d^{1/q})p^7 t_{mix}^5\sqrt{w/a}\log^3(1/(wa))
+
C_{st}(1+d^{1/q})\frac{p^8}{a}t_{mix}^5\sqrt{w/a}\log^3(1/(wa)).
```

После этого Corollary 11 становится согласованным:

```tex
A_{st}(p,q,\alpha)=polylog(n)\alpha^{1/2}
```

при `p,q` logarithmic and `d^{1/q}=O(1)`.

### Альтернативное исправление

Если ты по какой-то причине хочешь оставить Eq. 318 как есть, тогда нужно ослабить Corollary 11:

```tex
\|\mathcal U^{start,RR}_{n,n_0}\|_{L_p}
\le polylog(n)n^{-\beta}
```

instead of

```tex
polylog(n)n^{-1/4-\beta}.
```

Но это хуже стилистически и, судя по proof, не нужно: правильнее добавить потерянный `sqrt(w/a)` в Eq. 318.

### Severity

High. Это proof-critical для precise startup rate и для утверждения в Theorem 7, что startup transfer является lower order при выбранном burn-in.

---

## 3. Lemma 30: `log^{1/p}` выглядит подозрительно для direct citation

### Где

Appendix Lemma 30, Eq. 394:

```tex
\left\|\sum_{t=0}^{r-1}\psi_w(J_t^{(0,w)},Z_{t+1})\right\|_{L_p}
\le
c_{W,1}p^{3/2}\sqrt{wr}
+
c_{W,2}p^3 w^{-1/2}\log^{1/p}(1/(wa)).
```

### Проблема

Для Levin-type centered bilinear bounds ожидаемый логарифмический множитель выглядит как

```tex
\log^{1/2}(1/(wa))
```

или, в некоторых statements, как другой фиксированный power of log. Степень `1/p` выглядит нетипично и слишком сильной, особенно если Lemma 30 называется direct citation.

В текущем тексте это не ломает polynomial rate, потому что всё равно затем поглощается в `polylog(n)`. Но как citation hygiene это опасно: если direct citation в источнике имеет `log^{1/2}`, а в дипломе стоит `log^{1/p}`, то statement выглядит как неявное усиление внешнего результата.

### Исправление

Если источник действительно даёт `log^{1/2}`, заменить Eq. 394 на:

```tex
\left\|\sum_{t=0}^{r-1}\psi_w(J_t^{(0,w)},Z_{t+1})\right\|_{L_p}
\le
c_{W,1}p^{3/2}\sqrt{wr}
+
c_{W,2}p^3 w^{-1/2}\log^{1/2}(1/(wa)).
```

И затем заменить `log^{1/p}` на `log^{1/2}` во всех местах применения:

- Eq. 203;
- Eq. 208–209;
- Eq. 214;
- Corollary 4 proof;
- Theorem 4 burned-in analogue, if формула перепереносится туда.

Финальный rate останется тем же:

```tex
polylog(n)n^{-1/4}.
```

Если у тебя есть собственное доказательство формы `log^{1/p}`, тогда лучше не маркировать Lemma 30 как direct citation. Можно написать:

```tex
Lemma 30. (Local working form derived from Levin et al. Corollary 6.)
```

и добавить короткое объяснение, откуда появляется именно `1/p`.

### Severity

Medium. Не ломает финальный rate, но важно для корректности внешних ссылок.

---

## 4. Осталась ссылка на “Step (S8) of the Samsonov scheme”

### Где

Section 3.2, перед Eq. 78:

```tex
The PR-averaged Richardson–Romberg expansion produces, after Step (S8) of the Samsonov scheme applied separately at step sizes alpha and 2alpha, ...
```

### Проблема

Раньше в summary по цитированию уже отмечалось, что в локальном Samsonov PDF нет literal named “Step (S8)”. Поэтому такая ссылка выглядит неаккуратно: читатель не сможет быстро найти соответствующий named step.

Это не proof-critical, потому что section 3.2 является exploratory и прямо говорит, что depth-one route не используется в final assembly. Но лучше не оставлять несуществующий label.

### Исправление

Заменить на одну из формулировок:

```tex
The PR-averaged Richardson–Romberg expansion produces, after applying the first perturbation-expansion step underlying Samsonov et al. (2025, Proposition 9) separately at step sizes alpha and 2alpha, ...
```

или:

```tex
The PR-averaged Richardson–Romberg expansion produces, after the first deterministic-product perturbation step described in Section 4.2 and used in Samsonov et al. (2025, Proposition 9), ...
```

Если хочется сохранить “Step (S8)”, нужно локально определить этот step в тексте, например в начале Section 3.2:

```tex
For reference, by Step (S8) we mean the replacement of the random-product first misadjustment by the depth-one perturbation term J^{(1)} plus the residual H^{(1)}.
```

Но проще убрать label.

### Severity

Medium / low. Это citation hygiene, не центральная ошибка доказательства.

---

## 5. Библиографию лучше привести к финальному виду

### Где

References section.

### Проблема

Сейчас bibliography всё ещё выглядит как working bibliography. Например:

```text
Fan, X. (2019). Berry–Esseen bounds for martingales and applications. arXiv preprint.
```

и

```text
Huo, D., Chen, Y., and Xie, Q. (2023). Effectiveness of constant stepsize in Markovian LSA and statistical inference. Proceedings of the AAAI Conference on Artificial Intelligence; arXiv:2312.10894.
```

Стоит перепроверить точный статус публикаций, titles, venues, years и arXiv identifiers. Сейчас это не мешает доказательствам, но создаёт ощущение, что список литературы не финализирован.

### Исправление

Для каждого arXiv / published paper выбрать один стиль:

```text
Author, A. (Year). Title. Journal/Conference, volume(issue), pages. arXiv:xxxx.xxxxx.
```

или, если paper только preprint:

```text
Author, A. (Year). Title. arXiv preprint arXiv:xxxx.xxxxx.
```

Особенно стоит проверить:

- Fan (2019): точный journal title и citation;
- Huo–Chen–Xie: exact title, year, conference / arXiv status;
- Samsonov et al. (2025): arXiv identifier and title capitalization;
- Levin–Naumov–Samsonov (2025): arXiv identifier and title capitalization;
- Bobkov–Götze: spelling of Goetze/Götze should be consistent.

### Severity

Low. Это финальная полировка.

---

## Рекомендуемый порядок исправлений

1. Исправить Markov concentration Lemma 2 / Lemma 27 на root-form и синхронизировать все места применения.
2. Исправить `A_st(p,q,w)` так, чтобы первый член тоже имел `sqrt(w/a)`, либо ослабить Corollary 11.
3. Проверить Lemma 30 по Levin source: заменить `log^{1/p}` на `log^{1/2}`, если это direct citation.
4. Убрать или локально определить “Step (S8)”.
5. Финализировать bibliography.

После первых двух исправлений основная proof assembly будет выглядеть существенно устойчивее. После третьего замечания external-input appendix будет намного аккуратнее и безопаснее для чтения научруком или рецензентом.

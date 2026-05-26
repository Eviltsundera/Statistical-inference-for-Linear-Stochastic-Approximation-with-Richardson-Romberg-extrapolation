# Ревью текущей версии диплома `main.pdf`

Дата ревью: 2026-05-25  
Файл: `main.pdf`, 60 страниц  
Тема: non-asymptotic inference for constant-stepsize Markovian LSA with Polyak–Ruppert averaging and Richardson–Romberg extrapolation.

## 0. Краткий вердикт

Текст в целом выглядит как серьезная и уже достаточно связная заготовка математической работы. Главная линия — stationary augmented-chain Berry–Esseen assembly, затем balanced triangular-array specialization и transfer к deterministic starts through burn-in — понятна и хорошо мотивирована.

Однако в текущей версии есть несколько проблем, которые я бы считал обязательными к исправлению до защиты или отправки научному руководителю как «почти финальной» версии:

1. **Есть одна явная индексная ошибка/несогласованность в определении augmented-chain kernel в Section 1.6.** В текущем виде kernel соответствует цепи `(θ_k, Z_k)`, а не заявленной цепи `(θ_k, Z_{k+1})`.
2. **Есть существенная формульная ошибка в Lemma 23, Eq. (318): первый член `A_st(p,q,w)` теряет фактор порядка `sqrt(w/a) log^3(1/(wa))` и `t_mix^5`.** Без этой поправки Corollary 11 и последующие burn-in rates в заявленном виде не следуют.
3. **В burn-in части есть формальный пробел вокруг Lemma 17 / product stability.** Лемма используется как ключевой input для random initial-product transient и startup coupling, но не оформлена как прямой импорт и не доказана локально.
4. **Стационарный theorem действительно относится к comparison/assembled augmented-chain statistic, а не к исходному finite-start RR estimator.** В самой работе это в основном честно проговорено, но Abstract/Introduction стоит сделать еще более осторожными.

После исправления этих пунктов основная proof pipeline выглядит правдоподобно: RR deterministic weights → Poisson/martingale reduction → predictable-variance comparison → martingale Berry–Esseen → depth-two misadjustment → burn-in transfer. Но я не стал бы говорить, что доказательство полностью проверено, пока не будут формально закрыты пункты 1–3.

---

## 1. Объем и метод проверки

Я проверял текущие материалы как рецензент, а не как формальный proof assistant. Проверка включала:

- чтение структуры работы и theorem dependency chain;
- проверку основных алгебраических разложений и индексов;
- проверку, где используются imported inputs и local extensions;
- визуальную проверку PDF-верстки;
- сопоставление заявленных результатов с релевантными работами Samsonov–Sheshukova–Moulines–Naumov (2025) и Levin–Naumov–Samsonov (2025), на которые опирается диплом.

Я не перепроверял заново все внешние леммы Levin et al. и Samsonov et al. с нуля. Для них я проверял, что в дипломе они используются в заявленном диапазоне параметров и что локальные расширения действительно формально оформлены.

---

## 2. Самые важные замечания по доказательствам

### 2.1. Критично: off-by-one inconsistency в augmented-chain kernel, Section 1.6, Eq. (14)

**Место:** Section 1.6, definition of joint process and kernel `P_α`, Eq. (14).  
**Статус:** математическая/индексная ошибка, требующая исправления.

В тексте сказано, что поскольку `θ_k^{(α)}` alone is generally not Markovian, рассматривается joint process

```text
(θ_k^{(α)}, Z_{k+1})
```

с kernel

```text
P_α f(θ,z) = ∫ Q(z,dz') f(F_{z'}(θ), z'),
F_z(θ) = (I - α A(z))θ + α b(z).
```

Но этот kernel соответствует state convention `(θ_k, Z_k)`, потому что если текущее состояние содержит `Z_k=z`, то следующий шум `Z_{k+1}=z'`, и обновление `θ_{k+1}=F_{z'}(θ_k)`.

Если же текущее состояние действительно `(θ_k, Z_{k+1})`, то уже известный шум для следующего обновления равен `z`, и переход должен выглядеть как

```text
P_α f(θ,z) = ∫ Q(z,dz') f(F_z(θ), z').
```

**Почему это важно:** весь язык stationary augmented chain, invariant law and startup coupling потом опирается на эту convention. Позже в работе часто используются состояния вида `(Z_{k+1}, J_k^{(0)}, J_k^{(1)}, ...)`, поэтому лучше привести Section 1.6 к этой convention.

**Рекомендация:** выбрать один из двух вариантов и держать его всюду:

- либо писать joint process `(θ_k, Z_k)` и оставить kernel as written;
- либо оставить joint process `(θ_k, Z_{k+1})`, но заменить kernel на `f(F_z(θ), z')`.

Я рекомендую второй вариант, потому что он лучше согласуется с later notation `Y_k=(Z_{k+1},J_k,...)`.

---

### 2.2. Критично: пропущен `sqrt(w/a)`-factor в Lemma 23, Eq. (318)

**Место:** Section 5.8, Lemma 23, Eq. (318).  
**Статус:** формульная ошибка с последствиями для Corollary 11, Corollary 12 и Theorem 7.

В текущей версии:

```text
A_st(p,q,w) := C_st(1+d^{1/q})p^7
              + (p^8/a) t_mix^5 sqrt(w/a) log^3(1/(wa)).
```

Но непосредственно ниже proof of Lemma 23 использует pure J-coordinate scale

```text
A_J(p,q,w) := C(1+d^{1/q})p^7 t_mix^5 sqrt(w/a) log^3(1/(wa)).
```

Текст также говорит, что `p^7`-summand in `A_st` is exactly this pure J-coordinate scale. Значит, первый summand в Eq. (318) должен тоже содержать `t_mix^5 sqrt(w/a) log^3(1/(wa))`.

**Почему это критично:** Corollary 11 утверждает, что at balanced scale `α = c n^{-1/2}`,

```text
A_st(p,q,α) = polylog(n) α^{1/2}.
```

Это верно только если первый член `A_st` тоже имеет `sqrt(α)`-factor. С текущей Eq. (318) первый член имеет порядок `polylog(n)` без `α^{1/2}`, поэтому startup bound после burn-in становится слабее. Тогда Corollary 12 в заявленном виде уже не следует для произвольного `β>0`; нужно было бы требовать как минимум дополнительный запас типа `β ≥ 1/4`, что явно не планируется.

**Рекомендованная правка:** заменить Eq. (318) на что-то вида

```text
A_st(p,q,w) := C_st (1+d^{1/q}) t_mix^5 sqrt(w/a) log^3(1/(wa))
               · (p^7 + p^8/a),
```

или определить через already introduced scale:

```text
A_st(p,q,w) := C [ A_J(p,q,w) + (p/a) A_J(2p,q,w) ],
```

после чего в Corollary 11 proof станет корректным.

---

### 2.3. Высокий приоритет: Lemma 17 используется, но не оформлена как доказанная или импортированная

**Место:** Section 4.1, Section 5.3, Section 5.7, Section 5.8.  
**Статус:** формальный пробел в burn-in transfer.

В Section 4.1 вводится

```text
α_st(p) := min(α_prod(2p), α_rand-prod(2p), α_full-start(2p)).
```

Там сказано, что first ceiling belongs to the deterministic product-stability working form Lemma 17, а Lemma 22 and Lemma 23 are local extensions. Однако в Appendix 6.2 как local extensions перечислены только Lemma 10, Lemma 22, Lemma 23. Lemma 17 не включена в direct imported working forms и, насколько видно по текущей версии, не имеет полноценного внешнего citation/proof block.

**Где используется:**

- Lemma 18 / random initial-product transient;
- Lemma 22 / conditional product stability at random coupling time;
- Lemma 23 / full-state startup contraction.

**Рекомендация:** добавить отдельный clearly marked block:

```text
Lemma 17 (Direct citation / local proof: deterministic product stability).
...
```

и указать:

- точную ссылку на источник, если это импорт;
- exact sign convention conversion;
- exact range of `w`, `p`, `t_mix`;
- whether the statement is unconditional or conditional.

Если это не прямой импорт, лучше перенести доказательство в Appendix 6.2 рядом с Lemma 22/23.

---

### 2.4. Высокий приоритет: Lemma 22 требует conditional version of product stability

**Место:** Section 5.7, Lemma 22.  
**Статус:** неполная формализация.

В proof Lemma 22 фактически используется условная версия product stability:

```text
E[ ||Γ_{l+1:k}^{(w)} U_l||^p | F_l ]^{1/p}
≤ C B exp(-c w a (k-l)/p) exp(-c0 w a l/p).
```

Текст объясняет это фразой “the deterministic-vector estimate proved before the final Hölder step in Lemma 17”. Но поскольку Lemma 17 в текущей версии не доказана и не процитирована в полном виде, reader cannot verify the conditional step.

**Рекомендация:** либо усилить statement Lemma 17 и включить conditional estimate как часть леммы, либо в Lemma 22 дать самостоятельный proof of this conditional estimate.

---

### 2.5. Средний/высокий приоритет: stationary construction for both RR levels should be explicit

**Место:** Section 4.10, Lemma 10 and subsequent misadjustment bounds.  
**Статус:** не ошибка, но стоит явно оформить.

Lemma 10 строит stationary full depth-two augmented state для фиксированного step size `w`. Однако RR misadjustment использует разности

```text
2J_k^{(j,α)} - J_k^{(j,2α)},
2H_k^{(2,α)} - H_k^{(2,2α)}.
```

Эти объекты должны быть построены jointly on the same two-sided stationary base chain, because RR levels use the same Markov trajectory.

Скорее всего, это следует автоматически: построить finite-past limits для `w=α` и `w=2α` на одной two-sided chain and pass to joint `L_p` limits. Но сейчас это стоит проговорить.

**Рекомендация:** после Lemma 10 добавить короткий corollary/paragraph:

> Applying Lemma 10 simultaneously to `w=α` and `w=2α` on the same two-sided stationary chain gives a joint stationary RR augmented state. All RR differences below are read under this joint construction.

---

### 2.6. Средний приоритет: stationary theorem vs actual estimator

**Место:** Abstract, Section 1.3, Section 4.11, Theorem 3.  
**Статус:** формально в тексте почти правильно, но нужно усилить в начале.

В Section 4.11 хорошо сказано, что theorem is for

```text
S_n,stat^RR(u) := -u^T M_n^RR / sqrt(n) + u^T R_n,stat^RR
```

and not for deterministic-start RR average. Также сказано, что finite-start transient and random initial-product discrepancy are not part of stationary result.

Но Abstract/Introduction могут читаться так, будто stationary theorem непосредственно относится к PR-averaged RR estimator. Лучше добавить в Abstract или Section 1.3 фразу:

> The stationary result is a theorem for the assembled augmented-chain comparison statistic. Statements for the deterministic-start RR estimator are obtained only after the burn-in transfer of Section 5.

Это особенно важно для внешнего читателя.

---

## 3. Замечания по разделам

### 3.1. Section 1: setup and assumptions

**Сильные стороны:** setup хорошо мотивирован: constant-stepsize bias, RR cancellation, distinction between stationary and deterministic-start cases. Contribution map полезен и должен остаться.

**Замечания:**

1. Исправить augmented-chain kernel in Eq. (14), см. Section 2.1 этого ревью.
2. Стандартизировать notation for process indices: в Section 1 recursion uses `Z_k`, а Section 4.1 говорит “throughout this thesis we use `θ_{k+1}=θ_k-w(...Z_{k+1})`”. Это эквивалентно, но вкупе с augmented-state convention приводит к off-by-one confusion.
3. `π(Ã)=0` and `π(ε)=0` are consequences of definitions and Assumptions 2–3. In theorem statements later можно писать “by construction” rather than repeat them as if independent assumptions.
4. Bias formula `θ* + αΔ + O(α^{3/2})` is consistent with the Levin decomposition, but since the thesis also says the first misadjustment component has `O(α^2)` remainder, it would be useful to state precisely whether the `O(α^{3/2})` is from the full iterate due to `J^(2)+H^(2)` moment scale, or merely a safe bound.

### 3.2. Section 2: zeroth-order last-iterate RR difference

This section is algebraically useful and the derivation of the extra `αA` factor in the difference of powers is clear. The scalar `L_p` bound is plausible and pedagogically helpful.

**Main issue:** it is not integrated into the later theorem pipeline. The later proof uses PR-averaged RR weights, Poisson decomposition, and depth-two misadjustment; the last-iterate `J̃_{n,last}^{(0,α)}` estimate is not a formal input to Theorem 3/7.

**Recommendation:** add a short opening/closing paragraph:

> This chapter is a local warm-up estimate and is not used directly in the main Berry–Esseen assembly.

or move it to an appendix if final length matters.

### 3.3. Section 3: last-iterate analysis and depth-one limitation

The section is valuable because it explains why depth-one misadjustment is insufficient: at `α≈n^{-1/2}`, the centered fluctuation remains `O(1)`, so it cannot feed into an `n^{-1/4}` BE remainder.

**Recommendations:**

1. Make explicit that the depth-one bound is intentionally crude and not used in final theorems.
2. If keeping this in the main text, add a one-sentence bridge to Section 4:

   > This motivates importing the depth-two Levin transfer rather than attempting to close the Berry–Esseen argument at depth one.

### 3.4. Section 4: stationary RR PR weight bounds and assembly

This is the core and, modulo the comments above, the logical structure is strong.

**Good points:**

- The distinction between full-window `𝒬_l^RR` and burned-in `Q_{l;n0,n}^RR` is clear.
- The finite-window variance comparison and asymptotic covariance target are well motivated.
- The stationary assembled statistic is correctly separated from deterministic-start estimator.
- Theorem 2 / Corollaries 4–7 give a readable rate hierarchy.

**Issues to fix/check:**

1. Make simultaneous stationary construction for RR levels explicit, as discussed in 2.5.
2. The statements “Use imported inputs summarized in Section 6” are too compressed for final version. Consider defining formal assumption blocks, e.g.

   ```text
   Assumption C: stationary depth-two external inputs.
   Assumption D: burn-in startup product stability inputs.
   ```

   Then the theorem statements can say “under Assumption C” rather than carrying a vague reference.
3. In Theorem 3, the martingale Berry–Esseen bound is applied to signed scalar martingale increments. This is fine, but write explicitly that bounded increment constant and predictable variance are invariant under `u -> -u`.
4. In Corollary 5, the admissible window `1/3 < γ < 1` follows modulo logs. This is correct from the displayed conditions, but maybe state that the small-step ceiling `2α_n ≤ α_stat(q_n)` also requires large enough `n`, because `q_n≈log n` enters `α_stat(q_n)`.

### 3.5. Section 5: burn-in transfer

This section is ambitious and mostly coherent, but it is the part where the most formal repairs are needed.

**Main issues:**

1. Lemma 17 must be either proved or explicitly imported.
2. Lemma 22 needs the conditional product-stability estimate stated/proved.
3. Lemma 23 Eq. (318) must be corrected.
4. Corollary 11 and Corollary 12 need to be rechecked after correcting Eq. (318).
5. Theorem 5 uses martingale BE with a martingale indexed on ambient `n` but scaled by effective window `m`. The argument is okay because `m ≥ n/2`, but the proof should explicitly show inequalities of the form

   ```text
   (2n+1)log(2n+1) / s^3 ≤ C log(n)/sqrt(m)
   ```

   after using variance lower bounds.
6. In Lemma 19 / burned-in variance comparison, spell out the exact number of post-burn-in indices. Some sums run over `2,...,n-1` and some over effective window `m`; current bounds are safe but the text would benefit from “at most two boundary terms” language.

### 3.6. Appendix / external inputs

The appendix is a good idea: it reduces ambiguity about which results are imported and which are local.

**But:** Section 6.2 says the local extensions are Lemma 10, 22, 23. Since Lemma 17 is essential, add it either to Section 6.1 or 6.2.

Also ensure all imported lemma numbers match the actual source versions. For example, “Levin et al. (2025, Lemma 11)” / “Proposition 5” / “Corollary 4” should match the arXiv version used in bibliography, because arXiv/conference versions may renumber.

---

## 4. Style and exposition

### 4.1. Overall style

The prose is generally clear and technically mature. The work reads like a serious research draft rather than a standard undergraduate-style thesis. That is a strength, but it creates an exposition burden: the reader needs more help navigating dependencies.

### 4.2. High-value style improvements

1. **Add a one-page “proof dependency graph” before Section 4 or at the end of Introduction.**  
   You already have a contribution map. Make it more formal:

   ```text
   Lemma 5–9 → Theorem 1 → Theorem 3 → Corollaries 6–7
   Lemma 17–23 → Theorem 6 → Theorem 7 → Corollary 13
   ```

2. **Standardize British/American spelling.**  
   Both “normalisation” and “normalization” appear. Pick one. Since most surrounding prose is US-style, I recommend “normalization”.

3. **Standardize “Hájek–Le Cam”.**  
   I saw both “Hajek–Le Cam” and “Hájek–Le Cam”. Use the accented form consistently or omit accents consistently.

4. **Reduce meta-commentary in theorem paragraphs.**  
   Phrases like “The theorem just proved…” are useful in a draft but can be tightened in final version:

   ```text
   The preceding result is a stationary augmented-chain statement only; the finite-start analogue requires...
   ```

5. **Make “stationary comparison statistic” a named object early.**  
   Since it is not the original estimator, name it early and use this name everywhere.

6. **Title page placeholder.**  
   The title page has “— —” under the title. Add author, supervisor, institution, program, year, or remove placeholder.

### 4.3. References and citation consistency

1. Huo et al. are cited as 2023 in some prose and 2024 in references. Decide whether you cite arXiv preprint, AAAI proceedings, or both.
2. Levin et al. is cited as arXiv 2025 plus conference version 2026. That is fine, but in theorems/imports refer to the exact version whose numbering you use.
3. Samsonov et al. 2025 appears as NeurIPS 38 / arXiv. Make sure the bibliography has final venue info only if accepted and stable.
4. If some results are from arXiv versions rather than proceedings, state “extended arXiv version” in the citation because lemma/proposition numbering often differs.

---

## 5. Coherence and narrative

The narrative is good but currently split into three “modes”:

1. pedagogical local last-iterate estimates;
2. stationary augmented-chain theorem;
3. deterministic-start burn-in transfer.

This is mathematically sensible, but the transition from mode 1 to mode 2 needs more signposting. A reader might overestimate the role of Sections 2–3 in the final theorem. I recommend explicitly labeling Sections 2–3 as preliminary/motivational unless their lemmas are later used.

The strongest part of the exposition is the repeated warning that stationary `n0=0` theorem is not the deterministic-start theorem. Keep that warning. It protects the work from a common criticism.

---

## 6. Suggested action plan

### Immediate fixes before sharing as a polished draft

1. Fix Section 1.6 kernel/indexing.
2. Fix Lemma 23 Eq. (318) and then re-check Corollary 11, Corollary 12, Theorem 7.
3. Add/prove/cite Lemma 17 explicitly.
4. Strengthen Lemma 22 statement with conditional product stability.
5. Add joint stationary construction for `α` and `2α` RR augmented states.
6. Add a short statement in Abstract/Introduction clarifying that the stationary theorem is for an assembled comparison statistic.

### Secondary fixes

1. Add dependency graph.
2. Standardize spelling and citation years.
3. Move or label Sections 2–3 as preliminary if they are not used downstream.
4. Add final conclusion section after Corollary 13.
5. Clean title page placeholder.

---

## 7. Final assessment

I would not describe the current proof as “wrong overall”. The intended proof strategy is coherent and aligns with the modern literature on Markovian LSA, PR averaging, Berry–Esseen bounds, and RR bias reduction. The core idea — combine RR deterministic weights with Markov-chain Poisson/martingale reduction and depth-two Levin misadjustment, then transfer to deterministic starts through burn-in — is sound.

But the current version still has several formal defects. Two of them are serious enough that a careful examiner could stop there:

- the Section 1.6 augmented-chain indexing mismatch;
- the missing `sqrt(w/a)` factor in Lemma 23 Eq. (318).

Once these are fixed and Lemma 17 is properly documented, the thesis will read much more robustly.

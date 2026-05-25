# Ревью текущей версии диплома

**Файлы:** `main.pdf`, `2026-05-24_external-unexpanded-statements.md`  
**Тема:** statistical inference for constant-stepsize Markovian LSA with Richardson--Romberg extrapolation  
**Фокус ревью:** корректность доказательств, полнота внешних ссылок, стиль, связность и структура текста.

## 0. Краткий вердикт

Текущая версия выглядит как сильный и содержательно цельный черновик: основная архитектура доказательства понятна, разделение на stationary augmented-chain theorem и deterministic-start burn-in transfer хорошо мотивировано, RR-weight algebra и Poisson/martingale part в целом выглядят математически осмысленно.

Однако как законченный proof document работа пока не самодостаточна. Главный риск не в одной явной алгебраической ошибке, а в том, что несколько proof-critical мест опираются на внешние/локально-расширенные результаты, которые в тексте либо не сформулированы полностью, либо сформулированы как “imported input” без достаточного списка условий. Особенно это касается:

1. `alpha_*(q,t_mix)` и `alpha_st(p)`;
2. finite-past construction for stationary `H^{(2)}`;
3. conditional product stability at a random coupling time;
4. full-state startup contraction including `H^{(2)}`;
5. полного списка assumptions в финальной deterministic-start theorem.

Если эти блоки явно оформить как леммы/assumptions или доказать локально, основная линия доказательства станет гораздо надежнее.

---

## 1. Самые важные proof-critical замечания

### 1.1. Hidden external inputs: `Input C` и `Input D` пока слишком черные ящики

**Где:** Section 4.1, `Imported Inputs and Admissibility Thresholds`; далее используются в burn-in transfer и финальной теореме.

В тексте хорошо выписаны `Input A` и `Input B`: Markov concentration и martingale Berry--Esseen даны с формулами. Но `Input C` и `Input D` остаются существенно менее прозрачными:

- `Input C` говорит, что существует ceiling `alpha_*(q,t_mix)`, при котором работают stationary bias, centered bilinear и depth-two moment estimates Levin et al.;
- `Input D` говорит, что существует `alpha_st(p)` для product-stability и full-state startup contractions.

Проблема: именно эти два input-а несут основную доказательную нагрузку в stationary misadjustment и deterministic-start transfer. Сейчас читатель не видит:

- какие именно Levin statements входят в `alpha_*`;
- для каких moment orders они применяются;
- как threshold применяется одновременно к `w=alpha` и `w=2alpha`;
- какие constants разрешено считать fixed problem constants;
- что именно входит в startup threshold `alpha_st(p)`.

**Что исправить:** добавить отдельную subsection, например:

```text
4.1.1. Admissibility thresholds and imported local extensions
```

и там явно написать:

```math
2\alpha \le \alpha_*(q,t_{\mathrm{mix}}),\qquad
2\alpha \le \alpha_{\mathrm{st}}(p),\qquad
2\alpha \le \alpha_{\mathrm{inv}},\qquad
\alpha a \le 1/4,
```

с расшифровкой, что `alpha_*` является минимумом ceilings, нужных для Levin Proposition 2, Corollary 6, Propositions 8--9 и invariant-chain results, а `alpha_st(p)` является минимумом startup/product-stability ceilings, включая moment order `2p` там, где нужен Hölder.

---

### 1.2. В Chapter 2 нужно явно требовать `2 alpha <= alpha_infty`

**Где:** Chapter 2, особенно Lemma 1 / bounds for `H_j^{(n)}`, equations (29)--(32).

В оценке `H_j^{(n)}` используются Lyapunov contractions одновременно для

```math
B_\alpha = I-\alpha A,\qquad B_{2\alpha}=I-2\alpha A.
```

Но Lemma 1 дает contraction только для step size в `[0, alpha_infty]`. Значит, для `B_{2alpha}` нужно явно предположить

```math
2\alpha \le \alpha_\infty.
```

В последующих главах это условие часто уже аккуратно ставится, например в Chapter 3 и Chapter 4. Но в Chapter 2 оно должно быть сказано до Lemma 2 / Proposition on zeroth-order RR term.

**Статус:** это не ломает доказательство, если подразумевалось `2alpha <= alpha_infty`, но в текущей формулировке условие неполное.

**Правка:** перед Section 2.3 или в Lemma 1 application написать:

```text
In all RR estimates of this chapter we assume `2 alpha <= alpha_infty`, so that the Lyapunov contraction applies at both levels `alpha` and `2 alpha`.
```

---

### 1.3. Несогласованное отслеживание степеней `a`

**Где:** Lemma 5, Step 1/Step 3, equations (73)--(79), а также некоторые более поздние estimates.

В Step 1 получается, что

```math
\|U_R\|_{L_p}
\lesssim p^{1/2} t_{\mathrm{mix}}^{3/2}\|\epsilon\|_\infty (\alpha a)^{-1/2}.
```

После умножения на `alpha` это дает

```math
\alpha\|U_R\|_{L_p}
\lesssim p^{1/2} t_{\mathrm{mix}}^{3/2}\|\epsilon\|_\infty \sqrt{\alpha/a}.
```

В тексте итоговая форма записана как примерно

```math
p^{1/2}t_{\mathrm{mix}}^{3/2}\sqrt{\alpha}/a.
```

Это более грубо, если предполагается `a <= 1`, либо если `a` поглощается в constants. Но рядом текст говорит, что powers of `a` сохраняются явно. Тогда нужно выбрать одну конвенцию:

- либо честно отслеживать точные степени `a`;
- либо заранее объявить, что все fixed problem constants, включая powers of `a^{-1}`, могут поглощаться в `C`.

Сейчас смешиваются оба стиля.

**Правка:** добавить глобальную convention:

```text
Unless explicitly displayed, constants may depend polynomially on `a^{-1}`, `kappa_Q`, `C_A`, `||A^{-1}||`, `||epsilon||_infty`, and fixed dimension parameters, but never on `n, alpha, p, q`.
```

Если же хочешь сохранить powers of `a`, перепроверь все места, где `(alpha a)^{-1/2}` превращается в `sqrt(alpha)/a`.

---

### 1.4. Stationary augmented-chain construction for `H^{(2)}` нужно сделать локальной леммой

**Где:** burn-in transfer; места, связанные с finite-past stationary construction and full augmented state including `H^{(2)}`.

Судя по summary по цитированию, Levin Corollary 4 дает invariant law для depth-two chain в координатах

```math
(Z_{t+1},J_t^{(0)},J_t^{(1)},J_t^{(2)}),
```

но не включает `H_t^{(2)}`. В дипломе же stationary augmented-chain theorem фактически использует стационарную копию с `H^{(2)}`.

Это одно из самых важных мест. Текущая идея через finite-past Cauchy construction правильная, но ее нужно явно оформить:

**Нужная лемма:** на двухсторонней стационарной копии Markov chain стартовать recursions с нуля в момент `-m`, вычислить в момент `0`:

```math
(J_{0,m}^{(0,w)},J_{0,m}^{(1,w)},J_{0,m}^{(2,w)},H_{0,m}^{(2,w)}).
```

Затем показать, что это Cauchy в `L_p`, например в форме

```math
\|H_{0,m}^{(2,w)}-H_{0,m'}^{(2,w)}\|_{L_p}
\le
C p^{7/2}t_{\mathrm{mix}}^{5/2}
 w^{3/2}\log^{3/2}(1/(wa))
 e^{-cwa\min(m,m')/p}.
```

После этого можно определить stationary full augmented state as the `L_p` limit.

**Почему это критично:** без этой леммы стационарный объект, в котором живет `H^{(2)}`, не полностью построен. Тогда stationary misadjustment theorem опирается на не до конца определенный augmented state.

---

### 1.5. Conditional product stability at random coupling time — это локальное расширение, не прямая цитата

**Где:** Lemma around “Conditional product stability at a coupling time”, особенно proof, где говорится, что Levin product-stability proof is conditional on the past and can be applied on each event `T=s`.

Это утверждение правдоподобно, но его нельзя оставлять как очевидное следствие внешней Proposition без формулировки. Нужно явно написать:

- что `T` — exact coupling time;
- какие хвосты у `T`;
- что `V_T` is adapted / measurable w.r.t. the past at time `T`;
- как обрабатываются events `{T=s}`;
- что делать при empty product `T=k`;
- почему constants не зависят от `s`.

**Рекомендация:** либо доказать полностью локально, либо оформить как explicit technical input. Но не писать это как прямую ссылку на Levin Proposition 9, потому что в citation summary правильно указано: такого отдельного external statement в Levin нет.

---

### 1.6. Full-state startup contraction including `H^{(2)}` — главное место для усиления

**Где:** Lemma `Full-state startup contraction for the depth-two augmented remainder` and its proof.

Текущая схема:

1. Levin Proposition 5 controls `J^{(0)},J^{(1)},J^{(2)}` in Wasserstein/cost;
2. Levin Proposition 8 controls `J^{(2)}` moments;
3. Levin Proposition 9 controls one-trajectory `H^{(2)}`;
4. product stability controls random products;
5. combine to control finite-start vs stationary augmented remainder.

Схема правильная, но это не direct Levin corollary. Нужно явно показать local extension:

```math
R_{k,\mathrm{fin}}^{(w)}
=J_{k,\mathrm{fin}}^{(1,w)}+J_{k,\mathrm{fin}}^{(2,w)}+H_{k,\mathrm{fin}}^{(2,w)},
```

and a stationary augmented copy satisfy

```math
\|R_{k,\mathrm{fin}}^{(w)}-R_{k,\mathrm{aug}}^{(w)}\|_{L_p}
\le A_{\mathrm{st}}(p,q,w)e^{-c_{\mathrm{st}}wak/p}.
```

**Что добавить в proof:**

- exact decomposition of the `H^{(2)}` difference;
- term-by-term product stability bounds;
- where `L_{2p}` is used;
- how invariant initial cost is controlled;
- exact allowed dependence of `A_st(p,q,w)`;
- whether the estimate is uniform over the initial distribution of `Z_0`/`Z_1`.

Без этого финальная deterministic-start theorem выглядит условной.

---

### 1.7. Финальная теорема должна быть self-contained по assumptions

**Где:** final theorem in Chapter 5.

Финальная theorem ссылается на imported Levin and Samsonov inputs and defines an admissibility threshold. Это приемлемо для черновика, но для защиты/препринта theorem statement должен быть более автономным.

В statement нужно явно включить:

```math
2\alpha\le \alpha_*(q,t_{\mathrm{mix}}),\qquad
2\alpha\le \alpha_{\mathrm{st}}(p),\qquad
2\alpha\le \alpha_{\mathrm{inv}},\qquad
\alpha a\le 1/4,
```

а также:

- `p <= q/2`;
- `sigma^2(u)>0`;
- lower variance bound for `sigma_n^{2,bRR}(u)` if normalization uses finite-window variance;
- burn-in lower bound, e.g. `n_0 >= C alpha^{-1} p log(...)`;
- relation between `m=n-n_0` and `n`, e.g. `m >= n/2`;
- whether constants are uniform over initial laws of the base Markov chain.

**Текущий риск:** результат выглядит сильнее, чем реально доказано, потому что часть admissibility условий спрятана в prose.

---

## 2. Замечания по отдельным главам

### 2.1. Introduction

Сильная сторона: введение хорошо объясняет, почему stationary theorem не является fixed-`alpha` CLT centered exactly at `theta^*`, а deterministic-start result получается at balanced triangular-array scale `alpha_n = c n^{-1/2}`. Это важное уточнение и оно защищает от неправильной интерпретации результата.

Что поправить:

1. Claims about geometric forgetting, Huo power-series expansion, Levin residual RR bias and Samsonov Berry--Esseen/bootstrap should either remain clearly background-level claims, or be moved to a short “Literature theorem statements” appendix.
2. В формуле bias expansion стоит явно сказать, что statement требует small-step/admissibility assumptions.
3. “matching the minimax optimal rate” лучше переформулировать осторожно: в тексте уже есть фраза, что full Hájek--Le Cam argument is not part of thesis. Тогда лучше писать “matching the usual averaged-SA leading covariance scale” или “matching the leading covariance scale known from averaged LSA”.

---

### 2.2. Chapter 2: Zeroth-order RR difference

Главная алгебра с

```math
B_\alpha^m-B_{2\alpha}^m
=(B_\alpha-B_{2\alpha})\sum_i B_\alpha^{i-1}B_{2\alpha}^{m-i}
```

корректна, потому что оба matrices являются polynomials in the same matrix `A`. Хорошо, что это явно сказано.

Замечания:

1. Добавить `2alpha <= alpha_infty`.
2. Lemma 3 has a typographical/formula rendering issue: `E⟦X|^p]` should be `E[|X|^p]`.
3. В Lemma 2/3 можно добавить one-line derivation of the tail-to-moment inequality или citation; сейчас это стандартно, но для self-contained proof лучше дать короткое доказательство через layer-cake.
4. Если claim “uniformly in n” важен, стоит подчеркнуть, что all constants do not depend on `n`.

---

### 2.3. Chapter 3: Last iterate analysis

Плюсы:

- хорошо объяснена разница между `J^{(1)}` and shifted `T^{(1)}`;
- корректно отмечено, что depth-one route не дает полезного Berry--Esseen remainder at `alpha ~ n^{-1/2}`.

Замечания:

1. Lemma 4 is imported from Samsonov-style block/Berbee coupling. It is written in working form, which is good. Но надо аккуратно сказать, что это not literally Proposition 9 but a scalar extracted form / local specialization.
2. В Step 1 используется “weighted Markov concentration/Rosenthal bound”. Если это Input A, нужно проверить фактор: для `L_p` bound из sub-Gaussian concentration обычно получается `sqrt{p t_mix sum c_i^2}`, а не `sqrt p * t_mix * sqrt{sum c_i^2}`. В PDF visually Input A has square root over `p t_mix sum c_i^2`, но text extraction может искажать. Нужно убедиться, что во всех местах используется одна и та же версия.
3. Степени `a` в Lemma 5 лучше согласовать, как отмечено выше.
4. “Step (S8) of the Samsonov scheme” лучше не цитировать, если в Samsonov PDF нет literal Step (S8). Лучше написать локально: “the perturbation step replacing `H^{(0)}` by `J^{(1)}+H^{(1)}`”.

---

### 2.4. Chapter 4: RR PR Weight Bounds and stationary BE assembly

Это один из наиболее сильных разделов. Особенно удачны:

- closed-form identity for `Q_l^{(alpha)}`;
- exact RR identities for `mathcal Q_l^{RR}`;
- separation between variance comparison and Abel/Poisson variation;
- clear explanation that RR cancellation appears in the discrete derivative but not in the asymptotic-weight error.

Замечания:

1. `Input C` and `Input D` need expansion, as discussed.
2. In Lemma 7, the finite-sum boundary term is handled carefully. Good. But in final theorem, remind reader that martingale variance proxy is built from indices `l=2,...,n-1`, not all `l`.
3. In Poisson section, covariance identity `pi(V_epsilon)=Sigma_epsilon^{(M)}` should be stated as a lemma. This is standard, but it is a load-bearing identity for the covariance target.
4. If `Sigma_epsilon^{(M)}` is used in operator norm, state absolute convergence from boundedness + UGE.
5. Non-degeneracy `sigma^2(u)>0` should be attached to every Kolmogorov-normalized theorem.

---

### 2.5. Chapter 5: Burn-in transfer

Плюсы:

- правильно отделены deterministic transient, random initial-product transient, startup discrepancy;
- хорошая идея использовать mixing-scale burn-in with logarithmic factors;
- final balanced-scale result is the right target.

Главные проблемы:

1. Lemma 24/25 need either full proof or explicit technical-input status.
2. Burn-in conditions should be stated once in a clean “admissible burn-in regime” definition.
3. The theorem should explicitly say if constants are uniform over initial law. Сейчас это не полностью ясно.
4. Нужно перепроверить, где используется `m=n-n0` and whether assumptions such as `m >= n/2` are always active before replacing `m` by `n` in rates.
5. In the final balanced-scale statement, write exact dependency of logarithmic factors. If using `polylog(n)`, define it as a placeholder and state what it hides.

---

## 3. Проверка summary по цитированию

Summary `2026-05-24_external-unexpanded-statements.md` в целом корректно выделяет проблемные места. Я бы использовал его не просто как чеклист, а как основу для отдельного appendix.

### 3.1. Proof-critical items from summary

Следующие пункты из summary действительно нужно закрыть до финальной версии:

1. Levin admissibility threshold `alpha_*(q,t_mix)`;
2. startup threshold `alpha_st(p)`;
3. random-product stability behind Levin Appendix D.1 / Proposition 9;
4. Levin Appendix B.2 Proposition 5 and constants from Eq. (55);
5. invariant law for depth-two augmented chain;
6. Levin Proposition 9 for `H^{(2)}`;
7. finite-past Cauchy construction for stationary `H^{(2)}`;
8. conditional product stability at random coupling time;
9. full-state startup contraction extension;
10. Levin Proposition 8 for initial cost;
11. generic step-size restrictions in burned-in theorem;
12. final theorem imported inputs and thresholds.

Особенно критичны пункты 7--9: они являются не просто “missing citations”, а local mathematical extensions that are not direct consequences in the cited papers unless proved.

### 3.2. Lower-priority citation items

Пункты 13--23 из summary в основном background/expository. Их тоже желательно поправить, но они не блокируют основную proof architecture так же сильно.

Recommended handling:

- Intro/background claims can stay in prose if marked as background;
- but claims used inside proofs should be statements with assumptions and exact formulas.

### 3.3. Как превратить summary в appendix

Предлагаю добавить appendix:

```text
Appendix A. External inputs and local extensions
A.1 Markov concentration and martingale Berry--Esseen
A.2 Levin stationary depth-two inputs and admissibility ceilings
A.3 Product stability and startup transfer inputs
A.4 Poisson covariance identity and Markov-chain CLT
A.5 Background-only literature statements
```

В основной части тогда можно ссылаться не на статьи напрямую, а на локальные statements Appendix A. Это резко повысит читаемость и надежность.

---

## 4. Стиль, связность, оформление

### 4.1. Сильные стороны стиля

- Текст хорошо объясняет мотивацию: зачем RR, почему stationary theorem separate from deterministic start, почему depth-one route insufficient.
- Хорошо выделены roles of proof blocks.
- Много локальных comments в proof помогают читать сложную алгебру.

### 4.2. Что улучшить

1. Исправить повторения типа `Section Section 2`, `Chapter Section 3`.
2. Убрать переносы внутри слов в таблицах/подписях: `discrep-ancy`, `Berry--Es-seen`, etc.
3. Привести theorem/lemma labels к единому стилю.
4. Ввести global notation convention for:
   - whether chain starts at `Z_0` or `Z_1`;
   - filtration `F_l`;
   - initial law `xi`;
   - stationary versions `Z_infty`, `J_infty`, etc.
5. Не перегружать `theta_n^{RR}`: clearly distinguish last-iterate RR, PR-averaged full-window RR, and burned-in RR statistic.
6. References сейчас выглядят как черновой список. Нужен единый BibTeX/APA/AMS-style format.

---

## 5. Suggested fix order

### Priority 0: theorem-scope fixes

1. Add explicit `2alpha <= alpha_infty` wherever both `alpha` and `2alpha` are used.
2. Define `alpha_stat(q)` and `alpha_burn(p,q)` once and use consistently.
3. Make final theorem assumptions self-contained.

### Priority 1: proof-critical external/local inputs

1. Add local statement/proof for finite-past stationary construction including `H^{(2)}`.
2. Add local statement/proof for conditional product stability at random coupling time.
3. Expand full-state startup contraction proof or mark it as a technical input.
4. Add exact Levin Proposition 5/8/9 working forms in the appendix.

### Priority 2: citation and background cleanup

1. Convert citation summary into appendix.
2. State Poisson covariance identity as a lemma.
3. Add concise statements for Huo expansion, Levin residual RR bias, Samsonov BE/bootstrap only if they are used beyond background.

### Priority 3: style/polish

1. Fix duplicated words and hyphenation.
2. Standardize notation.
3. Clean references.
4. Add a “Theorem map” at the end of Introduction.

---

## 6. Checklist before continuing

Priority A — mathematical correctness:

- [x] Add explicit `2alpha <= alpha_infty` in Chapter 2 and every theorem where both `alpha` and `2alpha` are used.
- [x] Define admissibility thresholds once, preferably `alpha_stat(q)` for stationary results and `alpha_burn(p,q)` for deterministic-start burn-in results.
- [x] State the constants convention explicitly: fixed problem constants may absorb fixed powers of `a^{-1}`, but constants must not hide dependence on `n`, `alpha`, `p`, or `q`.
- [x] Make the final stationary and burned-in theorem assumptions self-contained: include `p <= q/2`, `sigma^2(u)>0`, variance lower bounds, small-step restrictions, and `m=n-n0 >= n/2` where used.
- [x] Add or import a precise finite-past construction for the stationary full augmented state including `H^{(2)}`.
- [x] Prove conditional product stability at a random coupling time, including the tail of `T`, measurability/adaptedness, empty products, and uniform constants.
- [x] Expand the full-state startup contraction proof for `J^{(1)}+J^{(2)}+H^{(2)}`, or explicitly turn it into a technical input with a complete statement.
- [x] Check the scale of `A_st(p,q,w)` in the startup contraction, especially the powers of `p`, `t_mix`, `w`, and `a`.
- [x] Add the Poisson covariance identity `pi(V_epsilon)=Sigma_epsilon^(M)` and absolute convergence of `Sigma_epsilon^(M)` from boundedness plus UGE.
- [x] Recheck the Markov concentration input everywhere: use one consistent form `C sqrt(p t_mix sum c_i^2)`.

Priority B — imported inputs and structure:

- [x] Add an appendix `External inputs and local extensions`.
- [x] Move precise working forms of Markov concentration and Bolthausen--Fan martingale Berry--Esseen into that appendix.
- [x] Add precise working forms of Levin Proposition 2, Corollary 6, Propositions 8--9, Appendix B.2 Proposition 5, and the invariant depth-two law.
- [x] Replace the black-box wording of `Input C` by an explicit statement that `alpha_*(q,t_mix)` is the minimum of the required Levin stationary depth-two ceilings.
- [x] Replace the black-box wording of `Input D` by an explicit statement that the random-coupling product stability and full-state startup contraction are local extensions, not direct citations.
- [x] Make the main proof body cite the appendix/local lemmas instead of referring informally to “Levin inputs”.
- [x] Define an “admissible burn-in regime” once and reuse it in finite-window and balanced-scale theorems.
- [x] Separate direct citations from locally proved extensions in the appendix.

Priority C — clarity and notation:

- [x] Clearly distinguish the stationary augmented-chain assembly from the deterministic-start burned-in estimator in theorem titles and notation.
- [x] Standardize notation for `Z_0` versus `Z_1`, filtrations, initial law `xi`, and stationary copies.
- [x] State whether startup bounds and final theorems are uniform over the initial law of the base Markov chain.
- [x] Define `polylog(n)` if it remains in the balanced corollary, or replace it by explicit logarithmic powers.
- [x] Check every replacement of `m` by `n`; it should only occur after `m >= n/2` is active.
- [x] Add a theorem map in the Introduction: stationary assembly, burn-in transfer, balanced triangular-array corollary.
- [x] Soften “matching minimax optimal rate” unless a full Hájek--Le Cam lower-bound argument is included.

Priority D — polish and mechanical checks:

- [x] Fix the typo `E⟦X|^p]`.
- [x] Fix duplicated phrases like `Section Section` and `Chapter Section`.
- [x] Remove awkward hyphenation such as `discrep-ancy` and `Berry--Es-seen`.
- [x] Standardize theorem/lemma labels and theorem names.
- [x] Clean the bibliography/reference style.
- [x] Run `typst compile main.typ` after each proof-critical block.
- [x] Run a final search for unresolved markers: `rg "TODO|ПРОВЕРИТЬ|Section Section|Chapter Section|polylog|black box|input" src`.

Minimal acceptable route if time is short:

- [x] Fix admissibility, variance, and theorem-scope assumptions.
- [x] State `H^{(2)}` finite-past construction, random-time product stability, and full-state startup contraction as explicit technical inputs.
- [x] Make the final deterministic-start theorem conditional on those technical inputs.
- [x] Mark in the appendix which statements are direct Levin/Samsonov citations and which are local extensions.

---

## 7. Bottom line

I would not say that the current draft contains an obvious fatal algebraic error in the core RR-weight / Poisson-martingale assembly. The main proof idea is coherent.

But I would not yet present the final deterministic-start theorem as fully proved unless the local extensions around stationary `H^{(2)}` and startup contraction are either:

1. proved in the thesis, or
2. explicitly elevated to assumptions/imported technical inputs with exact statements.

The citation summary is accurate and useful. Its proof-critical part should be integrated into the thesis, preferably as an appendix of external inputs and local technical extensions.

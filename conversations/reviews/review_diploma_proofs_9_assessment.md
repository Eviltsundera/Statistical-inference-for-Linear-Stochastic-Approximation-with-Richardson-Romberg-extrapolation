# Assessment of `review_diploma_proofs_9.md`

Дата: 2026-05-26
Проверено против текущих исходников после коммита `ddaf036`.

## Короткий вывод

В review 9 есть три реально важных новых пункта:

1. Section 1.6 действительно содержит off-by-one inconsistency в kernel для
   заявленного state convention `(theta_k, Z_{k+1})`.
2. Product-stability input `lem:burn-product-stability` был превращен в
   explicit technical input с детерминированной и conditional формами.
3. Conditional product-stability estimate, используемая в Lemma 22, теперь
   явно включена в statement Lemma 17 и цитируется в proof Lemma 22.

Пункт про `A_st(p,q,w)` в review 9 уже устарел: в текущих исходниках
`A_st` содержит общий множитель `t_mix^5 sqrt(w/a) log^3(1/(wa))`.

Средние и стилистические замечания также закрыты: joint stationary
construction для RR levels, clarification in Abstract, preliminary-label для
Sections 2--3, spelling `normalization`, citation-year consistency, and
title-page placeholder.

## 1. Section 1.6 augmented-chain kernel

Статус: исправлено в источниках.

Review 9 correctly noted that the old text in `src/introduction.typ`
considered the joint
process

$$
(\theta_k^{(\alpha)}, Z_{k+1})
$$

but wrote the kernel as

$$
\overline P_\alpha f(\theta,z)
= \int Q(z,dz') f(F_{z'}(\theta),z').
$$

Эта формула соответствует convention `(theta_k, Z_k)`: сначала переход
`z -> z'`, затем update через `z'`. Для convention `(theta_k, Z_{k+1})`,
который лучше согласуется с later notation

$$
(Z_{k+1},J_k^{(0,w)},J_k^{(1,w)},\ldots),
$$

следующий update должен использовать уже известный coordinate `z`:

$$
\overline P_\alpha f(\theta,z)
= \int Q(z,dz') f(F_z(\theta),z').
$$

Сделанная правка: оставлен state convention `(theta_k, Z_{k+1})`, kernel
заменен на форму с `F_z(theta)`, и добавлена поясняющая фраза that the current
second coordinate is the observation used to update `theta`.

## 2. `A_st(p,q,w)`

Статус: уже исправлено / замечание устарело.

В текущем файле
`src/burn_in_transfer/08_startup_transfer_augmented_remainders.typ` стоит

$$
A_{\mathrm{st}}(p,q,w)
= C_{\mathrm{st}}(1+d^{1/q})
\left(p^7+\frac{p^8}{a}\right)
t_{\mathrm{mix}}^5\sqrt{w/a}\log^3(1/(wa)).
$$

Поэтому Corollary 11 statement
`A_st(p,q,alpha)=polylog(n) alpha^(1/2)` at balanced scale согласован с
displayed definition. Здесь ничего менять не нужно.

## 3. Lemma 17 / product stability

Статус: исправлено в источниках.

В текущем тексте product stability записана в
`src/burn_in_transfer/04_deterministic_transient.typ` как
`lem:burn-product-stability`:

$$
\|\Gamma_{s+1:k}^{(w)}V_s\|_{L_p}
\le C_{\mathrm{prod}}
e^{-c_{\mathrm{prod}}wa(k-s)/p}\|V_s\|_{L_{2p}}.
$$

Теперь она оформлена как “Technical input: deterministic and conditional
product stability” и содержит полный deterministic/conditional statement. При
этом она остается load-bearing:

- random initial-product transient;
- finite-past construction for `H^(2)`;
- random-time product stability in Lemma 22;
- full-state startup contraction in Lemma 23.

Сделанная правка: statement расширен до explicit technical input; appendix
теперь отдельно перечисляет этот product-stability input и указывает, что
conditional display is the only form used at random coupling times.

## 4. Conditional product stability in Lemma 22

Статус: исправлено в источниках.

Lemma 22 доказывает random-time product stability через conditional estimate:

$$
\mathbb E[
  \|\Gamma_{s+1:k}^{(w)}W_s\|^p \mid \mathcal G_s
]^{1/p}
\le C e^{-cwa(k-s)/p}\|W_s\|.
$$

Proof Lemma 22 теперь ссылается на explicit conditional display
`eq:burn-product-stability-conditional` из Lemma 17, а не на невыписанный
intermediate step.

## 5. Joint stationary construction for RR levels

Статус: исправлено в источниках.

`lem:finite-past-full-augmented-state` строит full augmented stationary state
для фиксированного step size `w`. Ниже есть convention paragraph saying that
stationary versions with `w in {alpha, 2 alpha}` are used. Но не сказано
явно, что они constructed simultaneously on the same two-sided base chain.

После Lemma 10 добавлен paragraph “Joint RR stationary construction”, где
finite-past construction применяется одновременно at `w=alpha` and `w=2alpha`
on the same two-sided stationary chain.

## 6. Stationary theorem versus actual estimator

Статус: исправлено в Abstract.

Introduction already contains a good clarification: stationary result is an
augmented-chain theorem, and deterministic-start estimator requires burn-in
transfer. But Abstract still opens with “main object is the PR-averaged RR
estimator” and only then says “stationary augmented-chain assembly”.

В Abstract добавлена фраза, что stationary theorem is for an assembled
augmented-chain comparison statistic, while deterministic-start statements are
obtained only after burn-in transfer.

## 7. Secondary/style items

- The introduction prose now uses the same Huo et al. (2024) year as the
  bibliography.
- The spelling is standardized to `normalization`.
- The text uses accented `Hájek--Le Cam`.
- The title-page placeholder was removed.
- Sections 2 and 3 now include explicit preliminary/motivational labels.
- In `src/pr_weights/10_misadjustment_depth_two.typ`, the draft phrase about
  the preceding theorem was tightened.
- Theorem 5 burn-in BE proof now includes the explicit `m >= n/2` conversion
  for the first Bolthausen--Fan term.
- Lemma 19 already says the bracket concentration sum runs over ambient
  indices and is converted using `m >= n/2`; adding “at most boundary terms”
  language is optional.

## Recommended Fix Order

### Priority A — proof-critical

- [x] Fix Section 1.6 kernel/indexing convention.
- [x] Promote `lem:burn-product-stability` to an explicit technical input or
      prove/import it fully.
- [x] Add the conditional product-stability estimate to Lemma 17 and make
      Lemma 22 cite that display.
- [x] Add joint stationary RR augmented-state construction for
      `w=alpha,2alpha` on the same two-sided chain.

### Priority B — statement clarity

- [x] Add the Abstract sentence distinguishing stationary comparison statistic
      from deterministic-start estimator.
- [x] Label Sections 2--3 as preliminary/motivational and not direct inputs to
      the final BE assembly.
- [x] Make the burn-in martingale BE `n/m` conversion explicit.

### Priority C — polish

- [x] Fix `Huo 2023/2024` citation-year mismatch.
- [x] Standardize `normalization` and `Hájek--Le Cam`.
- [x] Remove the title-page placeholder.
- [x] Tighten draft theorem-transition phrasing.

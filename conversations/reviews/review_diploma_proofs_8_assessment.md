# Assessment of `review_diploma_proofs_8.md`

Дата: 2026-05-25
Проверено против текущих исходников после коммита `6d358af`.

## Короткий вывод

Из пяти замечаний в `review_diploma_proofs_8.md` два высокоприоритетных
замечания уже не соответствуют текущим исходникам: Markov concentration и
масштаб `A_st(p,q,w)` сейчас записаны в правильной форме. Оставались два
содержательных hygiene-пункта, которые были исправлены после этого assessment:

1. заменить `log^(1/p)` в импортированном Levin Corollary 6 на
   `log^(1/2)`;
2. убрать или локально определить ссылку на `Step (S8)`.

Библиография также приведена к более финальному виду отдельной правкой.

## 1. Markov concentration

Статус: уже исправлено / замечание устарело.

В текущем appendix:

- `src/external_inputs.typ:18-20`

стоит форма

$$
\left\|\sum_{i=1}^N g_i(Z_i)\right\|_{L_p(\xi)}
\le C_{\mathrm{MC}}\sqrt{p\,t_{\mathrm{mix}}\sum_{i=1}^N c_i^2}.
$$

То есть зависимость по коэффициентам уже равна
`(sum c_i^2)^(1/2)`, а не `sum c_i^2`. Та же форма используется в
`src/zeroth_order_rr.typ` и `src/last_iterate.typ`.

Дополнительно проверен локальный PDF Levin et al. (2025), Lemma 11: tail
parameter там имеет вид

$$
u_n = 8\left(\sum_i c_i^2\right)^{1/2}\sqrt{t_{\mathrm{mix}}}.
$$

Поэтому текущая working form с `sqrt(p t_mix sum c_i^2)` согласована с
источником.

Действие: ничего не менять, кроме возможной пересборки `main.pdf`, если review
был сделан по старому PDF.

## 2. `A_st(p,q,w)`

Статус: уже исправлено / замечание устарело.

В текущем тексте:

- `src/burn_in_transfer/08_startup_transfer_augmented_remainders.typ:192-196`

уже стоит

$$
A_{\mathrm{st}}(p,q,w)
= C_{\mathrm{st}}(1+d^{1/q})
\left(p^7+\frac{p^8}{a}\right)
t_{\mathrm{mix}}^5\sqrt{w/a}\log^3(1/(wa)).
$$

Значит первый `p^7`-term уже содержит `sqrt(w/a)`, и Corollary
`A_st(p,q,alpha)=polylog(n) alpha^(1/2)` при balanced scale согласован с
доказательством.

Действие: ничего не менять.

## 3. Levin Corollary 6: `log^(1/p)`

Статус: подтверждено и исправлено.

В текущем appendix:

- `src/external_inputs.typ:79`

стоит

$$
\log^{1/p}(1/(wa)).
$$

Но в локальном PDF Levin et al. (2025), Corollary 6, Eq. (67), displayed factor
is

$$
\sqrt{\log(1/(\alpha a))}.
$$

Значит statement нельзя оставлять как direct citation с `log^(1/p)`.
Правильная правка: заменить imported working form и все downstream displays на
`log^(1/2)`. Это не меняет финальные rates, потому что эти terms поглощаются в
`polylog(n)`.

Места, найденные поиском:

- `src/external_inputs.typ:79`;
- `src/burn_in_transfer/09_depth_two_misadjustment_bound.typ:47`;
- `src/pr_weights/10_misadjustment_depth_two.typ:328`;
- `src/pr_weights/10_misadjustment_depth_two.typ:377`;
- `src/pr_weights/10_misadjustment_depth_two.typ:384`;
- `src/pr_weights/10_misadjustment_depth_two.typ:448`;
- `src/pr_weights/11_smoothing_assembly.typ:112`;
- `src/pr_weights/11_smoothing_assembly.typ:260`.

## 4. `Step (S8) of the Samsonov scheme`

Статус: подтверждено и исправлено.

В текущем тексте:

- `src/last_iterate.typ:333-334`

осталась фраза:

```text
after Step (S8) of the Samsonov scheme
```

Такой локально определенный named step в дипломе отсутствует. Лучше заменить на
описательную ссылку, например:

```text
after applying the first deterministic-product perturbation step underlying
Samsonov et al. (2025, Proposition 9) separately at step sizes alpha and
2 alpha
```

или сослаться на локальную imported lemma `lem:future-centered-bilinear-input`.

## 5. Bibliography

Статус: низкий приоритет, исправлено в пределах локально проверяемых entries.

`main.typ` приведен к более единому стилю для entries, затронутых review:
Fan записан как JMAA 2019 article, Huo как AAAI 2024 article, Levin как AAAI
2026 article with arXiv extended version, Samsonov as NeurIPS 2025 with arXiv.

## Рекомендуемый следующий план

- [x] Replace Levin Corollary 6 logarithmic factor `log^(1/p)` by `log^(1/2)`
      in the appendix and all applications.
- [x] Remove the undefined `Step (S8)` wording from `src/last_iterate.typ`.
- [x] Rebuild `main.pdf` and confirm that no stale displays remain.
- [x] Separately polish bibliography metadata before final submission.

# Berry–Esseen Plan for the PR-averaged Richardson–Romberg Iterate

## Уточнение по пункту 3 (2026-05-03)

Пункт **Lemma "RR-misadjustment"** относится прежде всего к **новой секции 4**
про Berry--Esseen/CLT для RR-усреднения. Это не попытка просто улучшить
`src/last_iterate.typ` как самостоятельный результат.

Роль `src/last_iterate.typ` такая:

- он даёт прототип: shifted kernel, условное центрирование,
  разложение на $U_M+U_R$;
- он показывает, почему single-$\alpha$ оценка слишком груба для RR;
- он также фиксирует важную негативную проверку: наивное поднятие этой схемы
  на $\nabla_\alpha S_n^{(\alpha)}$ ловит детерминированную RR-разность
  ядер, но теряет выигрыш на суммировании по $l$, поэтому заявленная оценка
  $C\alpha^2(\dots)$ пока **не закрыта этим способом**.

Значит, для первой версии секции 4 правильнее не делать пункт 3 load-bearing
леммой. Для BE-rate $n^{-1/4}$ misadjustment следует контролировать через
depth-two разложение Levin et al.:

$$
H^{(0,\alpha)}=J^{(1,\alpha)}+J^{(2,\alpha)}+H^{(2,\alpha)},
$$

используя centered bound для $\sum \tilde A J^{(0)}$, high-order bounds для
$J^{(2)}$, $H^{(2)}$ и RR-bias cancellation для
$\mathbb E[J^{(1,\alpha)}]$. А оценка

$$
\|U^{\mathrm{RR}}_M\|_{L_p}+\|U^{\mathrm{RR}}_R\|_{L_p}
\le C\alpha^2(\dots)
$$

остаётся отдельным усилением/open technical thread: если её удастся доказать,
она сделает RR-misadjustment строго меньше мартингального BE-вклада, но она не
нужна для базового результата секции 4.

## Проверка пункта 1 (2026-05-02)

Формулировка Lemma "RR-Q-bounds" из плана ниже в заявленном виде неверна
равномерно по малому $\alpha$:

$$
\|\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}\|
  \le C\alpha(1-\alpha a)^{(n-\ell)/2},
\qquad
\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|
  \le C\alpha^2(1-\alpha a)^{(n-\ell)/2}.
$$

Контрпример уже находится на последнем PR-весе. При $\ell=n-1$:

$$
Q_{n-1}^{(\alpha)}=\alpha I,\qquad
Q_{n-1}^{(2\alpha)}=2\alpha I,\qquad
\mathcal Q_{n-1}^{\mathrm{RR}}=0,
$$

поэтому

$$
\mathcal Q_{n-1}^{\mathrm{RR}}-\bar A^{-1}=-\bar A^{-1},
$$

что не может быть $O(\alpha)$ с константой, не зависящей от $\alpha$.
Правильная версия, записанная в секции
`src/pr_weights.typ` "Richardson--Romberg PR Weight Bounds", такая:

$$
\|\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}\|
  \le C(1-\alpha a)^{(n-\ell)/2},
$$

и, при $k=n-\ell$,

$$
\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|
  \le C\alpha^2(k-1)(1-\alpha a)^{(k-2)/2}.
$$

Фактор $k-1$ нельзя убрать при той же экспоненте: для $k\asymp1/\alpha$
разность степеней $B_\alpha^{k-1}-B_{2\alpha}^{k-1}$ имеет порядок
$\alpha k$, так что вся дискретная разность имеет порядок $\alpha$,
а не $\alpha^2$.

Следствия для плана:

- variance comparison даёт естественную оценку
  $|\sigma_n^{2,\mathrm{RR}}(u)-\sigma^2(u)|\lesssim 1/(n\alpha)$,
  а не $C/n$ с $\alpha$-независимой константой;
- при рабочем масштабе $\alpha=c\,n^{-1/2}$ это всё равно $O(n^{-1/2})$,
  то есть ниже мартингального BE-rate $n^{-1/4}$;
- Abel-остаток в пуассоновском разложении контролируется через
  $\sum_\ell\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|=O(1)$,
  поэтому его вклад порядка $O(n^{-1/2})$.

## Проверка пункта 2 (2026-05-02)

Пункт 2 по существу верен: сравнение дисперсий следует из уже исправленной
леммы о RR-весах и не требует нового RR-сокращения. Однако в формулировке
плана и в `src/pr_weights.typ` была лишняя степень $a$ в знаменателе.

Пусть

$$
\Delta_\ell=\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1},\qquad
\Sigma=\Sigma_\varepsilon^{(\mathrm M)}.
$$

Тогда

$$
\mathcal Q_\ell^{\mathrm{RR}}\Sigma
(\mathcal Q_\ell^{\mathrm{RR}})^\top-\Sigma_\infty
=
\Delta_\ell\Sigma\bar A^{-\top}
+\bar A^{-1}\Sigma\Delta_\ell^\top
+\Delta_\ell\Sigma\Delta_\ell^\top.
$$

Из пункта 1:

$$
\sum_{\ell=1}^{n-1}\|\Delta_\ell\|
\le \frac{6C_Q}{\alpha a},
\qquad
\sum_{\ell=1}^{n-1}\|\Delta_\ell\|^2
\le \frac{9C_Q^2}{\alpha a}.
$$

Поэтому правильная оценка с константой

$$
C_3=
12C_Q\|\bar A^{-1}\|\,\|\Sigma\|
+9C_Q^2\|\Sigma\|
$$

имеет вид

$$
\|\Sigma_n^{\mathrm{RR}}-\Sigma_\infty\|
\le \frac{C_3}{n\alpha a},
\qquad
|\sigma_n^{2,\mathrm{RR}}(u)-\sigma^2(u)|
\le \frac{C_3\|u\|^2}{n\alpha a}.
$$

Версия с $a^2$ в знаменателе не следует из доказательства с этой константой
и вообще может быть сильнее правильной оценки, если $a>1$. Если хочется
писать $1/(n\alpha a^2)$, то нужно либо явно нормировать $a\le1$, либо
перенести дополнительный множитель $a$ в константу.

При рабочем масштабе $\alpha=c\,n^{-1/2}$ вывод плана не меняется:
variance-comparison вклад равен $O(n^{-1/2})$ и остаётся ниже
мартингального Berry--Esseen порядка $n^{-1/4}$.

## Цель

Доказать неасимптотическую границу типа Берри–Эссеена

$$
\mathrm{d}_K\!\left(
  \frac{\sqrt{n}\,u^\top(\bar\theta_n^{(\mathrm{RR},\alpha)}-\theta^*)}
       {\sigma_n^{\mathrm{RR}}(u)},\;
  \mathcal N(0,1)
\right)
  \le
  \mathrm B_n^{\mathrm{RR}},
$$

где

$$
\bar\theta_n^{(\mathrm{RR},\alpha)}
  = 2\bar\theta_n^{(\alpha)} - \bar\theta_n^{(2\alpha)},
\qquad
\bar\theta_n^{(\alpha)}
  = \frac1{n-n_0}\sum_{k=n_0}^{n-1}\theta_k^{(\alpha)}.
$$

Постоянный шаг $\alpha,2\alpha\in(0,\alpha_\infty)$, обе траектории сцеплены
по одной и той же реализации $\{Z_k\}$.

## Обновлённый план достижения цели (2026-05-03)

Рабочая цель секции 4: доказать Berry--Esseen bound для RR-PR среднего без
использования пока не закрытой оценки
$\|U_M^{\mathrm{RR}}\|_{L_p}+\|U_R^{\mathrm{RR}}\|_{L_p}\le C\alpha^2(\dots)$.
Эта RR-kernel оценка остаётся полезным усилением, но не должна быть
load-bearing в первой версии доказательства.

### 0. Формулировка теоремы

Сначала формулируем результат в стационарной версии цепи $Z_k\sim\pi$.
Для произвольного старта добавляем отдельный burn-in/UGE remainder.
Фиксируем направление $u$ и предполагаем невырожденность

$$
\sigma^2(u)=u^\top\Sigma_\infty u>0.
$$

Рабочий диапазон шагов:

$$
0<\alpha\le \alpha_\infty/2,\qquad n\alpha\gtrsim \log n,
\qquad p=\lceil \log n\rceil.
$$

Итоговая короллария будет при $\alpha=c n^{-1/2}$:

$$
\mathrm d_K\!\left(
  \frac{\sqrt n\,u^\top(\bar\theta_n^{(\mathrm{RR},\alpha)}-\theta^*)}
       {\sigma_n^{\mathrm{RR}}(u)},N(0,1)
\right)
\le C(u)\,\mathrm{polylog}(n)\,n^{-1/4}
+ \text{startup/burn-in terms}.
$$

### 1. Декомпозиция RR-среднего

Разложить

$$
\sqrt n\,u^\top(\bar\theta_n^{(\mathrm{RR},\alpha)}-\theta^*)
=W_n^{\mathrm{RR}}+R_n^{\mathrm{tr,RR}}+R_n^{\mathrm{mis,RR}},
$$

где

$$
W_n^{\mathrm{RR}}
=-\frac1{\sqrt n}\sum_{\ell=1}^{n-1}
u^\top\mathcal Q_\ell^{\mathrm{RR}}\varepsilon(Z_\ell),
\qquad
\mathcal Q_\ell^{\mathrm{RR}}=2Q_\ell^{(\alpha)}-Q_\ell^{(2\alpha)}.
$$

`src/pr_weights.typ` уже даёт нужные deterministic bounds:

$$
\|\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}\|
\le C(1-\alpha a)^{(n-\ell)/2},
$$

$$
\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|
\le C\alpha^2(n-\ell-1)(1-\alpha a)^{(n-\ell-2)/2}.
$$

### 2. Ведущий мартингал

К $W_n^{\mathrm{RR}}$ применить тот же Poisson route, что у Samsonov:

$$
W_n^{\mathrm{RR}}
=\frac1{\sqrt n}M_n^{\mathrm{RR}}+D_{2,n}^{\mathrm{RR}}.
$$

Нужно выписать три RR-леммы:

1. Poisson boundary/Abel remainder:

$$
\|D_{2,n}^{\mathrm{RR}}\|_{L_p}
\le C\frac{\mathrm{poly}(p,t_{\mathrm{mix}})}{\sqrt n}.
$$

2. Концентрация predictable quadratic variation:

$$
\left\|
  \langle M^{\mathrm{RR}}\rangle_n
  -n(\sigma_n^{\mathrm{RR}}(u))^2
\right\|_{L_p}
\le C\,\mathrm{poly}(p,t_{\mathrm{mix}})\,n^{1/2}.
$$

3. Мартингальная Berry--Esseen:

$$
\mathrm d_K\!\left(
  \frac{M_n^{\mathrm{RR}}}{\sqrt n\,\sigma_n^{\mathrm{RR}}(u)},N(0,1)
\right)
\le C\,\mathrm{polylog}(n)\,n^{-1/4}.
$$

Здесь RR входит только через boundedness и total variation of weights; новых
misadjustment-идей не нужно.

### 3. Сравнение дисперсий

Использовать уже доказанную в `src/pr_weights.typ` оценку

$$
\|\Sigma_n^{\mathrm{RR}}-\Sigma_\infty\|
\le \frac{C_3}{n\alpha a}.
$$

Она нужна в двух местах:

- чтобы показать, что $\sigma_n^{\mathrm{RR}}(u)$ отделена от нуля при
  достаточно большом $n\alpha$;
- чтобы получить вариант нормировки на $\sigma(u)$ вместо
  $\sigma_n^{\mathrm{RR}}(u)$.

При $\alpha=c n^{-1/2}$ этот вклад равен $O(n^{-1/2})$, то есть ниже
мартингального $n^{-1/4}$.

### 4. Misadjustment через Levin depth-two

Это главный пересмотр плана. Не опираемся на наивную RR-kernel оценку для
$\nabla_\alpha S_n^{(\alpha)}$. Вместо этого переводим Levin et al. (2025)
в наши обозначения:

$$
H^{(0,\alpha)}
=J^{(1,\alpha)}+J^{(2,\alpha)}+H^{(2,\alpha)}.
$$

Нужная lemma-transfer:

$$
\|R_n^{\mathrm{mis,RR}}\|_{L_p}
\le C\,\mathrm{poly}(p,t_{\mathrm{mix}})
\left(\sqrt n\,\alpha^{3/2}+\sqrt n\,\alpha^2\right)
+R_{\mathrm{burn}}(n_0,\alpha).
$$

Смысл слагаемых:

- $\sqrt n\,\alpha^2$ приходит из RR-bias cancellation для
  $\mathbb E[J^{(1,\alpha)}]$;
- $\sqrt n\,\alpha^{3/2}$ приходит из Levin bounds на $J^{(2,\alpha)}$ и
  $H^{(2,\alpha)}$;
- centered $\sum\tilde A J^{(0)}$ закрывается Levin Proposition 3 /
  Corollary 6 после сверки с equation (26).

При $\alpha=c n^{-1/2}$:

$$
\sqrt n\,\alpha^{3/2}=n^{-1/4},
\qquad
\sqrt n\,\alpha^2=n^{-1/2}.
$$

То есть misadjustment не обязан быть строго ниже мартингальной BE-части; для
первой теоремы достаточно, что он не хуже $n^{-1/4}$.

### 5. Сборка через smoothing

Пишем

$$
\frac{\sqrt n\,u^\top(\bar\theta_n^{(\mathrm{RR},\alpha)}-\theta^*)}
     {\sigma_n^{\mathrm{RR}}(u)}
=
\frac{M_n^{\mathrm{RR}}}{\sqrt n\,\sigma_n^{\mathrm{RR}}(u)}
+\mathcal R_n^{\mathrm{RR}}.
$$

По Samsonov smoothing inequality:

$$
\mathrm d_K(X+\mathcal R,N)
\le \mathrm d_K(X,N)
  + C\eta
  + \mathbb P(|\mathcal R|>\eta).
$$

Берём $p=\lceil\log n\rceil$ и контролируем вероятность через
$L_p$-bounds на $D_2^{\mathrm{RR}}$, transient и Levin-misadjustment.
Получаем рабочую форму

$$
\mathrm B_n^{\mathrm{RR}}
\le
C\frac{\mathrm{polylog}(n)}{n^{1/4}}
+\frac{C}{n\alpha}
+C\sqrt n\,\alpha^2
+C\sqrt n\,\alpha^{3/2}\mathrm{polylog}(n)
+R_{\mathrm{burn}}(n_0,\alpha).
$$

При $\alpha=c n^{-1/2}$:

$$
\mathrm B_n^{\mathrm{RR}}
\le C\,\mathrm{polylog}(n)\,n^{-1/4}
+R_{\mathrm{burn}}(n_0,n^{-1/2}).
$$

### 6. Что писать в секции 4

1. Theorem: RR Berry--Esseen with normalization $\sigma_n^{\mathrm{RR}}(u)$.
2. Lemma: RR PR-weight bounds and variance comparison (`src/pr_weights.typ`).
3. Lemma: Poisson martingale approximation for $W_n^{\mathrm{RR}}$.
4. Lemma: predictable quadratic variation concentration.
5. Lemma: Levin-to-our-notation misadjustment transfer.
6. Proof: smoothing assembly and choice $\alpha=c n^{-1/2}$.
7. Corollary: replacement of $\sigma_n^{\mathrm{RR}}(u)$ by $\sigma(u)$.

### 7. Optional refinement

Если позже удастся доказать

$$
\|U_M^{\mathrm{RR}}\|_{L_p}+\|U_R^{\mathrm{RR}}\|_{L_p}
\le C\alpha^2(\dots),
$$

то Levin-term $\sqrt n\,\alpha^{3/2}$ можно будет заменить более тонким
$\sqrt n\,\alpha^2$ для соответствующего centered first-order remainder.
Это даст strictly subleading RR-misadjustment, но не требуется для достижения
цели выше.

## Статус готовности плана (2026-05-03)

Коротко: готова вся **детерминированная RR-weight часть**, диагностика
misadjustment, **Poisson martingale approximation** для $W^{\mathrm{RR}}$
(boundary/Abel remainder), **predictable quadratic variation concentration**
для $\langle M^{\mathrm{RR}}\rangle_n$, и теперь **martingale Berry--Esseen
step** через Bolthausen--Fan для $u^\top M_n^{\mathrm{RR}}/(\sqrt n\,\sigma_n^{\mathrm{RR}}(u))$.
Ещё не готовы Levin-misadjustment transfer и smoothing-сборка.

Уточнение по нумерации:

- если считать по **старому списку задач**, то пункты 1--2 готовы, а пункт 3
  (*RR-misadjustment через $\nabla_\alpha S_n^{(\alpha)}$*) **не готов** и
  перенесён в optional refinement;
- если считать по **обновлённому плану**, то пункт 1 готов почти полностью
  (с caveat про burn-in), пункт 2 готов целиком (Poisson approximation +
  quadratic variation concentration + Bolthausen/Fan), пункт 3 готов.
  Остаются пункты 4--6 (Levin transfer, smoothing, $\sigma_n\to\sigma$
  corollary для полной BE).

### Готово в тексте thesis

1. **RR PR decomposition and weights.**  
   В `src/pr_weights.typ` уже выведено

   $$
   W^{\mathrm{RR}}
   =-\frac1{\sqrt n}\sum_{\ell=1}^{n-1}
     \mathcal Q_\ell^{\mathrm{RR}}\varepsilon(Z_\ell),
   \qquad
   \mathcal Q_\ell^{\mathrm{RR}}=2Q_\ell^{(\alpha)}-Q_\ell^{(2\alpha)}.
   $$

   Caveat: там для простоты стоит $n_0=0$; для финальной теоремы нужно явно
   вернуть burn-in или сказать, где он прячется.

2. **Closed-form identities for RR weights.**  
   В `src/pr_weights.typ` уже есть точные формулы

   $$
   \mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}
   =-\bar A^{-1}(2B_\alpha^{n-\ell}-B_{2\alpha}^{n-\ell}),
   $$

   $$
   \mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}
   =-2\alpha(B_\alpha^{n-\ell-1}-B_{2\alpha}^{n-\ell-1}).
   $$

3. **Pointwise and summed RR-weight bounds.**  
   В `src/pr_weights.typ` доказаны нужные оценки

   $$
   \|\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}\|
   \le C(1-\alpha a)^{(n-\ell)/2},
   $$

   $$
   \|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|
   \le C\alpha^2(n-\ell-1)(1-\alpha a)^{(n-\ell-2)/2},
   $$

   и суммарные следствия

   $$
   \sum_\ell\|\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}\|^2=O((\alpha a)^{-1}),
   \qquad
   \sum_\ell\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|=O(a^{-2}).
   $$

4. **Variance comparison.**  
   В `src/pr_weights.typ` уже доказано

   $$
   \|\Sigma_n^{\mathrm{RR}}-\Sigma_\infty\|
   \le \frac{C_3}{n\alpha a},
   \qquad
   |\sigma_n^{2,\mathrm{RR}}(u)-\sigma^2(u)|
   \le \frac{C_3\|u\|^2}{n\alpha a}.
   $$

   Это закрывает пункт 3 обновлённого плана.

5. **Zeroth-order RR gain.**  
   В `src/zeroth_order_rr.typ` уже есть proof thread для
   $\tilde J_n^{(0,\alpha)}=2J_n^{(0,\alpha)}-J_n^{(0,2\alpha)}$ и
   $L^2$-оценка порядка $O(\sqrt\alpha)$. Это полезный локальный RR sanity
   check, но не заменяет martingale BE.

6. **Single-$\alpha$ shifted last-iterate bound and RR-kernel diagnostic.**  
   В `src/last_iterate.typ` уже есть:

   - centered bound для $S_n^{(\alpha)}-\mathbb E S_n^{(\alpha)}$;
   - применение к грубой RR-misadjustment оценке;
   - честная проверка RR-kernel route для $\nabla_\alpha S_n^{(\alpha)}$;
   - вывод, что наивный RR-kernel route не закрывает
     $O(\sqrt n\,\alpha^2)$.

7. **Poisson martingale approximation для $W^{\mathrm{RR}}$.**  
   В `src/pr_weights.typ` (секция "Poisson Martingale Approximation",
   2026-05-03) доказано

   $$
   W^{\mathrm{RR}}=-\frac1{\sqrt n}M_n^{\mathrm{RR}}+D_{2,n}^{\mathrm{RR}},
   $$

   с явными мартингальными приращениями

   $$
   \Delta M_l^{\mathrm{RR}}=\mathcal Q_l^{\mathrm{RR}}\bigl(\hat\varepsilon(Z_l)-\mathsf Q\hat\varepsilon(Z_{l-1})\bigr),\qquad 2\le l\le n-1,
   $$

   и Poisson boundary/Abel остатком

   $$
   D_{2,n}^{\mathrm{RR}}=-\frac1{\sqrt n}\Bigl[\mathcal Q_1^{\mathrm{RR}}\hat\varepsilon(Z_1)+\sum_{l=1}^{n-2}(\mathcal Q_{l+1}^{\mathrm{RR}}-\mathcal Q_l^{\mathrm{RR}})\mathsf Q\hat\varepsilon(Z_l)\Bigr].
   $$

   Правый boundary $\mathcal Q_{n-1}^{\mathrm{RR}}\mathsf Q\hat\varepsilon(Z_{n-1})$
   отсутствует, потому что $\mathcal Q_{n-1}^{\mathrm{RR}}=0$ —
   RR-сокращение убирает один из двух Poisson-граничных членов outright.
   Получена *детерминированная* sup-norm оценка

   $$
   \|D_{2,n}^{\mathrm{RR}}\|_\infty\le \frac{3 t_{\mathrm{mix}}\|\varepsilon\|_\infty}{\sqrt n}\Bigl(C_{\mathcal Q}+\frac{C_2}{a^2}\Bigr),
   $$

   откуда $\|D_{2,n}^{\mathrm{RR}}\|_{L_p}\le C\,t_{\mathrm{mix}}/(a^2\sqrt n)$
   для всех $p\ge 1$ — лучше, чем заявленное в плане
   $\mathrm{poly}(p,t_{\mathrm{mix}})/\sqrt n$ ($p$-зависимость отсутствует,
   потому что используются только pointwise bounds).

8. **Predictable quadratic variation concentration.**
   В `src/pr_weights.typ` (секция "Predictable Quadratic Variation
   Concentration", 2026-05-03) доказан RR-аналог Samsonov Lemmas 22--23.
   Сначала прямым вычислением условного второго момента приращений
   $\Delta M_l^{\mathrm{RR}}$ получено
   $$
   \langle M^{\mathrm{RR}}\rangle_n
   = \sum_{l=2}^{n-1}\mathcal Q_l^{\mathrm{RR}}\,\bar\varepsilon(Z_{l-1})\,(\mathcal Q_l^{\mathrm{RR}})^\top,
   \qquad
   \bar\varepsilon(z) := \mathsf Q(\hat\varepsilon\hat\varepsilon^\top)(z)
   - (\mathsf Q\hat\varepsilon)(z)(\mathsf Q\hat\varepsilon)(z)^\top,
   $$
   с $\pi(\bar\varepsilon)=\Sigma_\varepsilon^{(\mathrm M)}$. Затем оценка
   $\|\bar\varepsilon\|_\infty\le 18\,t_{\mathrm{mix}}^2\|\varepsilon\|_\infty^2$
   и Markov-концентрация (Levin et al. 2025, Lemma 11), переписанные через
   $h_l(z)=u^\top\mathcal Q_l^{\mathrm{RR}}\bar\varepsilon(z)(\mathcal Q_l^{\mathrm{RR}})^\top u$,
   дают
   $$
   \bigl\|u^\top\langle M^{\mathrm{RR}}\rangle_n u
        - n\,\sigma_n^{2,\mathrm{RR}}(u)\bigr\|_{L_p}
   \le C_4\,C_{\mathcal Q}^2\|u\|^2\|\varepsilon\|_\infty^2\,t_{\mathrm{mix}}^{5/2}\sqrt{p\,n},
   \qquad C_4 \le 850.
   $$
   Corollary с заменой $\sigma_n^{2,\mathrm{RR}}\to\sigma^2$ через variance
   comparison добавляет $C_3\|u\|^2/(\alpha a)$. При $\alpha=cn^{-1/2}$ и
   $p=\lceil\log n\rceil$ это даёт нужный вход в Bolthausen/Fan для
   Berry--Esseen rate $n^{-1/4}\mathrm{polylog}(n)$.

### Готово только как notes / перенос из литературы

1. **Levin extraction.**  
   В нижней части этой заметки уже выписано, какие результаты Levin et al.
   нужны: stationary bias, equation (26), centered $\sum\tilde A J^{(0)}$,
   bounds на $J^{(2)}$ и $H^{(2)}$. Но это ещё не оформлено как lemma-transfer
   в thesis.

2. **Обновлённая стратегия BE.**  
   Раздел “Обновлённый план достижения цели” уже фиксирует правильный маршрут,
   но это пока план, не доказательство.

### Не готово / нужно написать

1. **Формальная Theorem statement для RR Berry--Esseen.**  
   В thesis пока нет секции с полным theorem statement, диапазоном
   $\alpha,n,p$, burn-in и условием $\sigma^2(u)>0$. Мартингальная часть
   зафиксирована (Theorem в `src/pr_weights.typ`, секция "Martingale
   Berry--Esseen Step"); нужна композитная теорема для всего
   $\sqrt n\,u^\top(\bar\theta_n^{\mathrm{RR}}-\theta^*)$.

2. ~~**Martingale Berry--Esseen step.**~~ Готово (`src/pr_weights.typ`,
   секция "Martingale Berry--Esseen Step", 2026-05-03). Применяется
   Lemma 21 Samsonov 2025 (Bolthausen 1982 + Fan 2019) с
   $\varkappa = 6\,t_{\mathrm{mix}}\,C_{\mathcal Q}\|\varepsilon\|_\infty\|u\|$,
   $s_n^2 = n\,\sigma_n^{2,\mathrm{RR}}(u)$, $p=\lceil\log n\rceil$.
   Result:
   $$
   d_K\!\left(\frac{u^\top M_n^{\mathrm{RR}}}{\sqrt n\,\sigma_n^{\mathrm{RR}}(u)},N(0,1)\right)
   \le \frac{C_{K,1}(u)\log^{3/4}n}{n^{1/4}}+\frac{C_{K,2}(u)\log n}{\sqrt n}
   $$
   при $n\alpha a\ge 2C_3\|u\|^2/\sigma^2(u)$ (что выполняется при
   $\alpha=cn^{-1/2}$ для всех достаточно больших $n$).
   Постоянная $C_{K,1}(u)\asymp C_{\mathcal Q}\|u\|^2 t_{\mathrm{mix}}^{5/4}\|\varepsilon\|_\infty/\sigma(u)$.
   Включён corollary с заменой нормировки $\sigma_n^{\mathrm{RR}}\to\sigma$
   (Lipschitz cdf normalization argument), который добавляет $O(n^{-1/2})$
   при working scale.

4. **Levin-to-our-notation misadjustment transfer.**  
   Самый важный оставшийся non-martingale кусок: надо аккуратно переписать
   Levin depth-two decomposition в наши обозначения и получить

   $$
   \|R_n^{\mathrm{mis,RR}}\|_{L_p}
   \le C\mathrm{poly}(p,t_{\mathrm{mix}})
      (\sqrt n\,\alpha^{3/2}+\sqrt n\,\alpha^2)
      +R_{\mathrm{burn}}.
   $$

5. **Smoothing assembly.**  
   Нужно собрать martingale BE + $L_p$-остатки через smoothing inequality и
   выбрать $p=\lceil\log n\rceil$, $\alpha=c n^{-1/2}$.

6. **Corollary for replacing $\sigma_n^{\mathrm{RR}}(u)$ by $\sigma(u)$.**  
   После theorem со $\sigma_n^{\mathrm{RR}}(u)$ это коротко следует из variance
   comparison, но пока не написано.

### Минимальный следующий шаг

Martingale Berry--Esseen готов (см. `src/pr_weights.typ`, секция "Martingale
Berry--Esseen Step", 2026-05-03). Следующий кирпич:

> Levin-misadjustment transfer: переписать depth-two разложение
> $H^{(0,\alpha)}=J^{(1,\alpha)}+J^{(2,\alpha)}+H^{(2,\alpha)}$ в наших
> обозначениях и доказать
> $\|R_n^{\mathrm{mis,RR}}\|_{L_p}\le C\,\mathrm{poly}(p,t_{\mathrm{mix}})(\sqrt n\,\alpha^{3/2}+\sqrt n\,\alpha^2)+R_{\mathrm{burn}}$.

После него — финальная smoothing-сборка через Samsonov Proposition 12.

### Следующий шаг для misadjustment

Да: если сейчас закрывать именно non-leading remainder, то нужно делать
**Misadjustment через Levin depth-two**, а не возвращаться к наивному
RR-kernel proof для $\nabla_\alpha S_n^{(\alpha)}$.

Цель этого блока:

$$
\|R_n^{\mathrm{mis,RR}}\|_{L_p}
\le C\,\mathrm{poly}(p,t_{\mathrm{mix}})
  \left(\sqrt n\,\alpha^{3/2}+\sqrt n\,\alpha^2\right)
  +R_{\mathrm{burn}}(n_0,\alpha).
$$

Нужно аккуратно перенести из Levin et al. в наши обозначения три вещи:

1. depth-two expansion
   $H^{(0,\alpha)}=J^{(1,\alpha)}+J^{(2,\alpha)}+H^{(2,\alpha)}$;
2. RR-cancellation of the stationary bias
   $\mathbb E[J^{(1,\alpha)}]=\alpha\Delta+O(\alpha^2)$;
3. high-order moment bounds on $J^{(2,\alpha)}$ and $H^{(2,\alpha)}$,
   plus the centered bound for $\sum\tilde A J^{(0)}$.

At $\alpha=c n^{-1/2}$ this gives

$$
\sqrt n\,\alpha^{3/2}=n^{-1/4},
\qquad
\sqrt n\,\alpha^2=n^{-1/2},
$$

so the misadjustment matches, but does not exceed, the martingale
Berry--Esseen scale. That is enough for the first theorem.

## Краткий конспект схемы Samsonov 2025 (Theorem 1)

Цепочка декомпозиций для одного шага $\alpha$ диминишингового вида:

**(S1) Ошибка → транзиент + флуктуация.**
$$
\theta_k-\theta^*
  =\Gamma_{1:k}^{(\alpha)}(\theta_0-\theta^*)
  -\sum_{j=1}^k\alpha_j\Gamma_{j+1:k}^{(\alpha)}\varepsilon(Z_j).
$$

**(S2) Случайные → детерминированные произведения.**
$$
u^\top\tilde\theta_k^{(\mathrm{fl})}
  = u^\top J_k^{(0)} + u^\top H_k^{(0)},
\quad
J_k^{(0)}=-\sum_j\alpha_j G_{j+1:k}\varepsilon(Z_j),
\quad
G_{m:k}=\prod_{i=m}^k(I-\alpha_i\bar A).
$$

**(S3) PR-усреднение даёт линейный функционал.**
$$
\sqrt n(\bar\theta_n-\theta^*)=W+D_1,\qquad
W=-\frac1{\sqrt n}\sum_{\ell=1}^{n-1}Q_\ell\varepsilon(Z_\ell),\qquad
Q_\ell=\alpha_\ell\sum_{k=\ell}^{n-1}G_{\ell+1:k}.
$$

**(S4) Уравнение Пуассона: $W=n^{-1/2}M+D_2$.**
С $\hat\varepsilon=\sum_{k\ge0}\mathsf Q^k\varepsilon$ имеем
$$
\Delta M_\ell=Q_\ell(\hat\varepsilon(Z_\ell)-\mathsf Q\hat\varepsilon(Z_{\ell-1})),
\qquad
D_2=-\tfrac1{\sqrt n}Q_1\hat\varepsilon(Z_1)+\tfrac1{\sqrt n}Q_{n-1}\mathsf Q\hat\varepsilon(Z_{n-1})
   +\tfrac1{\sqrt n}\sum_\ell(Q_\ell-Q_{\ell+1})\mathsf Q\hat\varepsilon(Z_\ell).
$$

**(S5) Концентрация квадратичной характеристики.**
$\langle M\rangle_n-n\Sigma_n$ оценивается через Пуассон-уравнение для функции
$\bar\varepsilon(z)=\mathsf Q(\hat\varepsilon\hat\varepsilon^\top)(z)-\mathsf Q\hat\varepsilon(z)\mathsf Q\hat\varepsilon(z)^\top$,
$\pi(\bar\varepsilon)=\Sigma_\varepsilon$.

**(S6) Мартингальная BE (Bolthausen + Fan).**

**(S7) $L_p$-контроль остатков $D_1+D_2$.**

**(S8) Перестановка $H_k^{(0)}=J_k^{(1)}+H_k^{(1)}$** + Berbee coupling +
Burkholder для билинейной части.

## Что меняется при переходе к RR

В работе сцеплены **две** последовательности с шагами $\alpha$ и $2\alpha$ на
одной реализации $\{Z_k\}$. Все объекты получают индекс шага. Введём
оператор $\nabla_\alpha X(\alpha):=2X(\alpha)-X(2\alpha)$. Тогда

$$
\bar\theta_n^{(\mathrm{RR},\alpha)}-\theta^*
  =\nabla_\alpha\bar\theta_n^{(\alpha)}-\theta^*
  =\frac1{n-n_0}\sum_{k=n_0}^{n-1}
    \nabla_\alpha\bigl(\theta_k^{(\alpha)}-\theta^*\bigr).
$$

Применяем (S1)–(S2) к каждому шагу и суммируем с весами $(2,-1)$:

$$
\sqrt n(\bar\theta_n^{(\mathrm{RR},\alpha)}-\theta^*)
  = W^{\mathrm{RR}}
  + D_1^{\mathrm{tr,RR}}
  + \frac1{\sqrt n}\sum_{k=0}^{n-1}\nabla_\alpha H_k^{(0,\alpha)},
$$

где

$$
W^{\mathrm{RR}}
  =-\frac1{\sqrt n}\sum_{\ell=1}^{n-1}
    \mathcal Q_\ell^{\mathrm{RR}}\,\varepsilon(Z_\ell),
\qquad
\mathcal Q_\ell^{\mathrm{RR}}
  := 2Q_\ell^{(\alpha)}-Q_\ell^{(2\alpha)}.
$$

В постоянном шаге

$$
Q_\ell^{(\alpha)}
  =\alpha\sum_{k=\ell}^{n-1}(I-\alpha\bar A)^{n-k-1}
  =\bar A^{-1}\bigl[I-(I-\alpha\bar A)^{n-\ell}\bigr],
$$
$$
Q_\ell^{(\alpha)}-\bar A^{-1}=-\bar A^{-1}(I-\alpha\bar A)^{n-\ell}.
$$

Поэтому

$$
\mathcal Q_\ell^{\mathrm{RR}}
  = \bar A^{-1}-\bar A^{-1}\bigl[
      2(I-\alpha\bar A)^{n-\ell}-(I-2\alpha\bar A)^{n-\ell}
    \bigr].
$$

Скобка — это RR-разность матричных степеней, но она не мала равномерно по
$\alpha$: при $n-\ell=1$ она равна $I$. Поэтому корректная поточечная
оценка сравнения с асимптотическим весом имеет вид

$$
\|\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}\|
  \le C\bigl(1-\alpha a\bigr)^{(n-\ell)/2}.
$$

RR-сокращение проявляется в дискретной разности:

$$
\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|
  \le C\alpha^2(n-\ell-1)
      \bigl(1-\alpha a\bigr)^{(n-\ell-2)/2}.
$$

**Главное наблюдение:** ведущий мартингальный член $W^{\mathrm{RR}}$
имеет такие же асимптотические веса $\bar A^{-1}$, как и в обычной PR-схеме;
RR не меняет ведущую дисперсию.

## Адаптация шагов S3–S6: ведущий мартингал

**Асимптотическая дисперсия.** Та же, что у PR:

$$
\sigma^2(u)=u^\top\bar A^{-1}\Sigma_\varepsilon^{(\mathrm M)}\bar A^{-\top}u,
\qquad
\Sigma_n^{\mathrm{RR}}
  =\frac1n\sum_\ell
    \mathcal Q_\ell^{\mathrm{RR}}\,
    \Sigma_\varepsilon^{(\mathrm M)}\,
    {\mathcal Q_\ell^{\mathrm{RR}}}^\top.
$$

**Поправка к Lemma 9 Samsonov.** Нужна аналогичная оценка
$|\sigma_n^2(u)-\sigma^2(u)|\le C_\infty^{\mathrm{RR}} f(\alpha)$
для констант-step случая. Разложение

$$
\Sigma_n^{\mathrm{RR}}-\Sigma_\infty
  = R_1^{\mathrm{RR}}+R_2^{\mathrm{RR}},
$$
где $R_1^{\mathrm{RR}}$ — линейная часть по $\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}$,
а $R_2^{\mathrm{RR}}$ — квадратичная. Через корректную оценку
$\|\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}\|\le C(1-\alpha a)^{(n-\ell)/2}$
получаем

$$
\|R_1^{\mathrm{RR}}\|
  \le C\,\|\Sigma_\varepsilon\|\,\frac1n\sum_\ell(1-\alpha a)^{(n-\ell)/2}
  \le C\frac1{n\alpha},
\qquad
\|R_2^{\mathrm{RR}}\|\le C\frac1{n\alpha}.
$$

То есть $|\sigma_n^2(u)-\sigma^2(u)|\le C/(n\alpha)$. При
$\alpha=c\,n^{-1/2}$ это $O(n^{-1/2})$, что всё ещё ниже целевого
Berry--Esseen порядка $n^{-1/4}$.

**Шаги S4–S6 проходят дословно** с заменой $Q_\ell\to\mathcal Q_\ell^{\mathrm{RR}}$.
Достаточные ингредиенты:

- *RR-аналог Lemma 3 Samsonov*: $\|\mathcal Q_\ell^{\mathrm{RR}}\|\le\mathcal L_Q^{\mathrm{RR}}$
  (вытекает напрямую из $\|Q_\ell^{(\alpha)}\|\le\|\bar A^{-1}\|+C$).
- *RR-аналог Lemma 6 Samsonov*: $\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|
  \le C\alpha^2(n-\ell-1)(1-\alpha a)^{(n-\ell-2)/2}$, и после суммирования
  $\sum_\ell\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|=O(1)$.
- *RR-аналог Lemma 22–23 Samsonov*: концентрация $\langle M^{\mathrm{RR}}\rangle_n-n\sigma_n^{2,\mathrm{RR}}(u)$
  получается тем же приёмом (Пуассон + Paulin).

В итоге для ведущего члена

$$
\mathrm{d}_K\!\left(
  \frac{u^\top M^{\mathrm{RR}}}{\sqrt n\,\sigma_n^{\mathrm{RR}}(u)},
  \mathcal N(0,1)
\right)
  \le \frac{C\log^{3/4}n}{n^{1/4}}+\frac{C\log n}{n^{1/2}}.
$$

## Адаптация шагов S7–S8: остатки $D^{\mathrm{RR}}$

В старой версии плана именно здесь предполагалась главная новая техническая
работа. В обновлённом плане load-bearing контроль misadjustment переносится
на Levin depth-two transfer; прямой RR-kernel анализ ниже читается как
диагностика и optional refinement. Остаток разбивается на

$$
D^{\mathrm{RR}}
  = \underbrace{D_{1}^{\mathrm{tr,RR}}}_{\text{инициализация}}
  + \underbrace{D_{1}^{\mathrm{mis,RR}}}_{\text{misadjustment }H^{(0)}\text{-членов}}
  + \underbrace{D_{2}^{\mathrm{RR}}}_{\text{Poisson boundary + Abel}}.
$$

### Транзиент $D_1^{\mathrm{tr,RR}}$

$$
D_1^{\mathrm{tr,RR}}
  =\frac1{\sqrt n}\sum_{k=0}^{n-1}
    \nabla_\alpha\Gamma_{1:k}^{(\alpha)}(\theta_0-\theta^*).
$$

Random matrix products стабильны при обоих шагах (Samsonov Prop 10), поэтому

$$
\|D_1^{\mathrm{tr,RR}}\|_{L_p}
  \le \frac{C\,\|\theta_0-\theta^*\|}{\sqrt n}
      (e^{-\alpha a n/24}+e^{-2\alpha a n/24})
  = O(e^{-c\alpha n}/\sqrt n).
$$

### Poisson-граница $D_2^{\mathrm{RR}}$

Дословно как в S4 с $\mathcal Q_\ell^{\mathrm{RR}}$. Из *RR-аналог Lemma 6 Samsonov*:

$$
\|D_2^{\mathrm{RR}}\|_{L_p}
  \le \frac{C t_{\mathrm{mix}}\|\varepsilon\|_\infty}{\sqrt n}.
$$

(Abel-член того же порядка, что и граничные члены: используется
$n^{-1/2}\sum_\ell\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|=O(n^{-1/2})$.)

### Misadjustment $D_1^{\mathrm{mis,RR}}$ — старый RR-kernel route

Этот подраздел описывает первоначальную попытку сделать centered
misadjustment strictly subleading через прямую RR-разность shifted kernels.
После проверки в `src/last_iterate.typ` этот маршрут не является текущим
доказательством секции 4: он полезен как диагностика и optional refinement,
но базовый BE-план выше использует Levin depth-two bounds.

Это PR-усреднение от RR-разности первой перестановки:

$$
D_1^{\mathrm{mis,RR}}
  =\frac1{\sqrt n}\sum_{k=1}^{n-1}u^\top
    \bigl[2H_k^{(0,\alpha)}-H_k^{(0,2\alpha)}\bigr].
$$

Разложение глубины $L=1$ (как в S8):

$$
H_k^{(0,\alpha)}=J_k^{(1,\alpha)}+H_k^{(1,\alpha)},
\qquad
\nabla_\alpha H_k^{(0,\alpha)}
  =\nabla_\alpha J_k^{(1,\alpha)}+\nabla_\alpha H_k^{(1,\alpha)}.
$$

Дальше рассматриваем по отдельности **смещение** и **флуктуацию**.

#### Смещение (bias)

Для каждого $\alpha$ Levin et al. (2025, Proposition 2):

$$
\mathbb E_\pi[J_k^{(1,\alpha)}]\xrightarrow{k\to\infty}\alpha\Delta+R(\alpha),
\quad\|R(\alpha)\|\le C\alpha^2.
$$

Тогда

$$
\mathbb E_\pi[\nabla_\alpha J_k^{(1,\alpha)}]
  =2\alpha\Delta-2\alpha\Delta+\nabla_\alpha R(\alpha)
  =O(\alpha^2).
$$

И значит

$$
\frac{\sqrt n}{n-n_0}
\sum_{k=n_0}^{n-1}\mathbb E_\pi[\nabla_\alpha J_k^{(1,\alpha)}]
  = O(\sqrt n\,\alpha^2).
$$

Это исчезает при $\alpha=o(n^{-1/4})$. Для оптимального шага $\alpha\sim n^{-1/2}$
из Levin (Corollary 3) bias-вклад $\sqrt n\alpha^2=n^{-1/2}\to0$ — годится.

(Транзиент к $\Pi_\alpha$ здесь нужно учесть отдельно: $|\mathbb E_\pi J_k^{(1,\alpha)}-\mathbb E_\pi J_\infty^{(1,\alpha)}|\le C\rho_\alpha^k$, что даёт
поправку $O((\alpha n)^{-1})$ через геометрическую сумму.)

#### Флуктуация

Нужна $L_p$-граница на

$$
U_k^{\mathrm{RR}}
  := \nabla_\alpha\bigl(J_k^{(1,\alpha)}-\mathbb E[J_k^{(1,\alpha)}]\bigr)
$$

и далее на $\frac1{\sqrt n}\sum_k U_k^{\mathrm{RR}}$.

Здесь становится важным результат, доказанный в `src/last_iterate.typ`:
для shifted last-iteration статистики

$$
S_n^{(\alpha)}=\sum_{t=0}^{n-1}B_\alpha^{n-t}\tilde A(Z_{t+1})J_t^{(0,\alpha)},
\quad B_\alpha=I-\alpha\bar A,
$$

имеет место

$$
\|S_n^{(\alpha)}-\mathbb ES_n^{(\alpha)}\|_{L_p}
  \le C\|\varepsilon\|_\infty
    \Bigl(p^{3/2}t_{\mathrm{mix}}^{1/2}\frac1a
          +p^{1/2}t_{\mathrm{mix}}^{3/2}\sqrt{\alpha/a}\Bigr).
$$

Поскольку $J_k^{(1,\alpha)}=-\alpha S_k^{(\alpha)}$ (с точностью до сдвига
индекса), это даёт

$$
\|J_k^{(1,\alpha)}-\mathbb EJ_k^{(1,\alpha)}\|_{L_p}
  \le C\alpha\|\varepsilon\|_\infty
    \Bigl(p^{3/2}\frac{t_{\mathrm{mix}}^{1/2}}{a}
          +p^{1/2}t_{\mathrm{mix}}^{3/2}\sqrt{\alpha/a}\Bigr).
$$

Покомпонентно $U_k^{\mathrm{RR}}$ получается из нескольких таких членов
с шагами $\alpha,2\alpha$. Грубая оценка:

$$
\|U_k^{\mathrm{RR}}\|_{L_p}\le C\alpha\cdot\text{(rhs)}.
$$

PR-сумма даёт

$$
\Bigl\|\frac1{\sqrt n}\sum_{k=1}^{n-1}U_k^{\mathrm{RR}}\Bigr\|_{L_p}
  \le \sqrt n\cdot \text{(грубо)}\;\alpha\cdot O(1/a)
  = O(\sqrt n\,\alpha).
$$

Для $\alpha=n^{-1/2}$ это $O(1)$, что **слишком плохо**. Чтобы получить
$o(1)$, нужно использовать **межшаговую корреляцию** между $S_k^{(\alpha)}$
и $S_k^{(2\alpha)}$ — RR-разность двух близких процессов имеет дополнительный
порядок $\alpha$.

Эта именно структура декомпозиции, которая ещё не разработана в `src/last_iterate.typ`:
она даёт оценку для ОДНОГО $\alpha$, но не для разности $\nabla_\alpha S_k$.

**Это была центральная идея старого маршрута.** После проверки
`src/last_iterate.typ` она остаётся отдельным open thread, а не обязательной
леммой для первого Berry--Esseen результата.

#### Шаг 8 для RR-разности: что делать

Идея: применить уже разработанную в `src/last_iterate.typ` схему

1. shift-формализация: $S_n^{(\alpha)}\to T_n^{(1,\alpha)}=-\alpha S_n^{(\alpha)}$;
2. условное центрирование $H^{(w)}\varepsilon\to v_k^{(w,\varepsilon)}+\text{martingale}$;
3. $U_M+U_R$-разложение;

— ко всей **разности** $\nabla_\alpha T_n^{(1,\alpha)}$. Тогда

$$
\nabla_\alpha S_n^{(\alpha)}
 = \sum_{t=0}^{n-1}
   \bigl[2 B_\alpha^{n-t}\tilde A(Z_{t+1})J_t^{(0,\alpha)}
         -B_{2\alpha}^{n-t}\tilde A(Z_{t+1})J_t^{(0,2\alpha)}\bigr].
$$

Подставляя $J_t^{(0,\alpha)}=-\alpha\sum_{k\le t}B_\alpha^{t-k}\varepsilon(Z_k)$,
получаем двойную сумму

$$
\nabla_\alpha S_n^{(\alpha)}
  =-\sum_{k=1}^{n-1}\mathcal H_{k+1}^{\mathrm{RR}}\varepsilon(Z_k),
$$

с **RR-ядром**

$$
\mathcal H_{k+1}^{\mathrm{RR}}
  =\sum_{l=1}^{n-k}\Bigl[
    2\alpha B_\alpha^{n-k-l+1}\tilde A(Z_{k+l})B_\alpha^{l-1}
    -2\alpha B_{2\alpha}^{n-k-l+1}\tilde A(Z_{k+l})B_{2\alpha}^{l-1}
  \Bigr].
$$

Ключевая оценка:

$$
\Bigl\|2\alpha B_\alpha^{n-k-l+1}M B_\alpha^{l-1}
       -2\alpha B_{2\alpha}^{n-k-l+1}M B_{2\alpha}^{l-1}\Bigr\|
  \le C\alpha\cdot\alpha(n-k)\,(1-\alpha a)^{(n-k-2)/2}\,\|M\|.
$$

Это даёт **дополнительный порядок $\alpha$** в норме $\mathcal H^{\mathrm{RR}}$ по сравнению с $H^{(w)}$:

$$
\|v_k^{(w,\mathrm{RR})}\|_\infty
  \le C\alpha\, t_{\mathrm{mix}}\,\alpha(n-k)\,
       (1-\alpha a)^{(n-k)/2}.
$$

Перенося это через схему `src/last_iterate.typ` (с заменой $H^{(w)}\to\mathcal H^{\mathrm{RR}}$),
получаем

$$
\|U_M^{\mathrm{RR}}\|_{L_p}+\|U_R^{\mathrm{RR}}\|_{L_p}
  \le C\alpha\cdot
    \Bigl(\frac{p^{3/2}t_{\mathrm{mix}}^{1/2}}{a}
          +p^{1/2}t_{\mathrm{mix}}^{3/2}\sqrt{\alpha/a}\Bigr).
$$

PR-сумма для misadjustment:

$$
\|D_1^{\mathrm{mis,RR}}\|_{L_p}
  \le \sqrt n\cdot \alpha\cdot \alpha\cdot O(1)
  = O(\sqrt n\,\alpha^2).
$$

При $\alpha=n^{-1/2}$ это $n^{-1/2}$ — то, что нужно для BE-rate $n^{-1/4}$.

## Устаревшая optimistic структура $\mathrm B_n^{\mathrm{RR}}$

Этот блок оставлен как исторический черновик старого маршрута. Канонический
план после проверки `src/last_iterate.typ` находится выше, в разделе
“Обновлённый план достижения цели (2026-05-03)”. Главная поправка: вклад
$D_1^{\mathrm{mis,RR}}=O(\sqrt n\,\alpha^2)$ пока не доказан наивным
RR-kernel route и не должен быть load-bearing для первой BE-теоремы.

Собирая все вклады:

$$
\mathrm B_n^{\mathrm{RR}}
  = \underbrace{\frac{C\log^{3/4}n}{n^{1/4}}}_{\text{мартингальная BE}}
  + \underbrace{\frac{C\log n}{n^{1/2}}}_{\text{Lindeberg}}
  + \underbrace{\frac{C\|\theta_0-\theta^*\|e^{-c\alpha n}}{\sqrt n}}_{D_1^{\mathrm{tr,RR}}}
  + \underbrace{\frac{C}{\sqrt n}}_{D_2^{\mathrm{RR}}\text{ boundary}}
  + \underbrace{C\sqrt n\,\alpha^2}_{D_1^{\mathrm{mis,RR}}}.
$$

При $\alpha=c\,n^{-1/2}$:

$$
\mathrm B_n^{\mathrm{RR}}
  \lesssim_{\log n}\frac1{n^{1/4}}.
$$

Это **тот же порядок $n^{-1/4}$**, что у Samsonov для диминишинга, но
получен в постоянно-шаговой схеме.

## Старый список задач (заменён обновлённым планом выше)

Ниже старый список задач. Его нужно читать как журнал проверки: пункты 1--2
остаются актуальными, пункт 3 перенесён в optional refinement, а базовый путь
к цели теперь идёт через Levin depth-two misadjustment transfer.

1. **Lemma "RR-Q-bounds":** пункт проверен и исходная формулировка исправлена.
   Верные оценки:
   $\|\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}\|\le C(1-\alpha a)^{(n-\ell)/2}$
   и $\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|
   \le C\alpha^2(n-\ell-1)(1-\alpha a)^{(n-\ell-2)/2}$.
   Нужные суммарные следствия:
   $\sum_\ell\|\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}\|^2=O(1/\alpha)$,
   $\sum_\ell\|\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}\|=O(1)$.

2. **Lemma "Variance comparison":** доказана в `src/pr_weights.typ`
   (секция "Variance Comparison").
   Утверждение:
   $\|\Sigma_n^{\mathrm{RR}}-\Sigma_\infty\|\le C_3/(n\alpha a)$
   с $\alpha,n$-независимой константой
   $C_3=12 C_Q\|\bar A^{-1}\|\|\Sigma_\varepsilon^{(\mathrm M)}\|+9C_Q^2\|\Sigma_\varepsilon^{(\mathrm M)}\|$,
   откуда $|\sigma_n^{2,\mathrm{RR}}(u)-\sigma^2(u)|\le C_3\|u\|^2/(n\alpha a)$.
   Доказательство: разложение $\mathcal Q_\ell^{\mathrm{RR}}\Sigma(\mathcal Q_\ell^{\mathrm{RR}})^\top-\Sigma_\infty=R_{1,\ell}+R_{2,\ell}$
   на линейную и квадратичную части от $\Delta_\ell=\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}$;
   суммирование линейной части использует $\sum_\ell\|\Delta_\ell\|\le 6C_Q/(\alpha a)$
   (из (i) предыдущей леммы и геометрической суммы),
   квадратичной — $\sum_\ell\|\Delta_\ell\|^2\le 9C_Q^2/(\alpha a)$ из Corollary 1.
   При $\alpha=c\,n^{-1/2}$ это даёт $O(n^{-1/2})$, ниже целевого BE-rate $n^{-1/4}$.

3. **Lemma "RR-misadjustment" (усиление/open thread):** обобщить
   `src/last_iterate.typ` на разность
   $\nabla_\alpha S_n^{(\alpha)}$, доказав
   $\|U^{\mathrm{RR}}_M\|_{L_p}+\|U^{\mathrm{RR}}_R\|_{L_p}\le C\alpha^2(\dots)$.
   Это было задумано как главная техническая часть для получения strictly
   subleading RR-misadjustment. После проверки в `src/last_iterate.typ`:
   наивный RR-kernel route не закрывает эту оценку, поэтому для базового
   BE-rate $n^{-1/4}$ пункт лучше заменить Levin depth-two контролем, а эту
   лемму оставить как отдельное усиление.

4. **Lemma "RR-bias of $J^{(1)}$":** перенести Levin Prop 2 в форму
   $\|\mathbb E_\pi[\nabla_\alpha J_k^{(1,\alpha)}]\|\le C\alpha^2$.
   Уже есть в Levin.

5. **RR-версия Theorem 1 Samsonov:** собрать всё через smoothing inequality
   (Prop 12 Samsonov) с $p=\log n$ и оптимальным $\alpha\asymp n^{-1/2}$.

6. **Corollary "asymptotic variance":** $\sigma\to\sigma_n$, использует
   п. 2.

## Замечания и оговорки

- **Стационарность.** Все оценки в плане формулируются в стационарной
  версии цепи; для произвольного $\xi$ добавляются стандартные UGE-граничные
  поправки порядка $\rho^n$.
- **Бутстрап (Theorem 2 Samsonov).** Этот план накрывает только Berry–Esseen.
  Расширение до OBM-bootstrap для RR-итерации требует отдельной работы:
  оценщик $\hat\sigma_\theta^2$ нужно скорректировать с учётом lugsail/RR
  для дисперсии (см. `summaries/RR_for_OBM_variance_estimator.md` и
  `inference.py:obm_rr_ci`), и это уже частично исследовано численно в
  `run_lugsail_bias_variance.py`.
- **Связь с существующими файлами.** Lemma 3 из `src/last_iterate.typ` уже
  даёт нужную оценку для $\|S_n^{(\alpha)}-\mathbb ES_n^{(\alpha)}\|_{L_p}$
  для одного шага. Шаг 3 плана — это **ректификация** того же доказательства
  с заменой ядра на RR-разность; основные ингредиенты (UGE, Markov-условное
  центрирование, weighted Burkholder) переносятся без изменений.

## Извлечение из Levin et al. (2025) для текущего BE-плана

Источник: `papers/Levin и др. - 2025 - High-Order Error Bounds for Markovian LSA with Richardson-Romberg Extrapolation.pdf`.
Ниже перечислены только те элементы статьи, которые реально полезны для
доказательства Berry--Esseen для PR-усреднённого RR-итерата.

### 1. Нормализация и важное предупреждение про burn-in

В статье Levin et al. в доказательстве Theorem 2 берут $n_0=n/2$ и оценивают
$\bar A(\bar\theta_n^{(\alpha,\mathrm{RR})}-\theta^*)$. Поэтому ведущий член
имеет вид

$$
-\frac{2}{n}\sum_{t=n/2}^{n-1}\varepsilon(Z_{t+1}),
$$

и при масштабировании через $\sqrt n$ его ковариация несёт множитель $2$ по
сравнению с полным средним длины $n$.

Для нашей BE-формулировки лучше ввести $m=n-n_0$ и либо масштабировать
$\sqrt m$, либо явно учитывать множитель $n/m$. Если $n_0=o(n)$, то ведущая
ковариация остаётся $\Sigma_\infty$; если $n_0=n/2$ и масштабирование
$\sqrt n$, то ведущая ковариация равна $2\Sigma_\infty$.

### 2. Базовая perturbation-декомпозиция

Levin et al. используют произведение случайных матриц

$$
\Gamma_{m:n}^{(\alpha)}
  = \prod_{i=m}^n (I-\alpha A(Z_i)).
$$

Ошибка одного constant-step итерата раскладывается как

$$
\theta_n^{(\alpha)}-\theta^*
  = \tilde\theta_n^{(\mathrm{tr})}
    + \tilde\theta_n^{(\mathrm{fl})},
$$

где

$$
\tilde\theta_n^{(\mathrm{tr})}
  = \Gamma_{1:n}^{(\alpha)}(\theta_0-\theta^*),
\qquad
\tilde\theta_n^{(\mathrm{fl})}
  = -\alpha\sum_{j=1}^n\Gamma_{j+1:n}^{(\alpha)}\varepsilon(Z_j).
$$

Для детерминированной линеаризации они вводят

$$
J_n^{(0,\alpha)}
  = (I-\alpha\bar A)J_{n-1}^{(0,\alpha)}-\alpha\varepsilon(Z_n),
$$

$$
H_n^{(0,\alpha)}
  = (I-\alpha A(Z_n))H_{n-1}^{(0,\alpha)}
    -\alpha\tilde A(Z_n)J_{n-1}^{(0,\alpha)}.
$$

Дальше для $\ell\ge1$

$$
J_n^{(\ell,\alpha)}
  = (I-\alpha\bar A)J_{n-1}^{(\ell,\alpha)}
    -\alpha\tilde A(Z_n)J_{n-1}^{(\ell-1,\alpha)},
$$

$$
H_n^{(\ell,\alpha)}
  = (I-\alpha A(Z_n))H_{n-1}^{(\ell,\alpha)}
    -\alpha\tilde A(Z_n)J_{n-1}^{(\ell,\alpha)}.
$$

Имеем рекурсивное тождество $H_n^{(\ell,\alpha)}
=J_n^{(\ell+1,\alpha)}+H_n^{(\ell+1,\alpha)}$ и, при глубине $L=2$,

$$
\theta_n^{(\alpha)}-\theta^*
  =\tilde\theta_n^{(\mathrm{tr})}
   +J_n^{(0,\alpha)}+J_n^{(1,\alpha)}
   +J_n^{(2,\alpha)}+H_n^{(2,\alpha)}.
$$

Это ровно оправдывает нашу схему
$H^{(0)}=J^{(1)}+H^{(1)}$ и показывает, что для high-order контроля лучше
раскрывать ещё один уровень до $J^{(2)}+H^{(2)}$.

### 3. Стационарные цепи для bias-членов

Levin et al. рассматривают совместные цепи

$$
(\theta_t^{(\alpha)},Z_{t+1}),\qquad
Y_t=(Z_{t+1},J_t^{(0,\alpha)},J_t^{(1,\alpha)}),
\qquad
V_t=(J_t^{(0,\alpha)},Z_{t+1}).
$$

Под A1, A2 и UGE 1, при достаточно малом $\alpha$, эти цепи имеют единственные
стационарные распределения $\Pi_\alpha$, $\Pi_{J^{(1)},\alpha}$,
$\Pi_{J,\alpha}$. Доказательство идёт через Wasserstein-сжатие с метриками
типа

$$
c_J((J,z),(\tilde J,\tilde z))
  = \|J-\tilde J\|
    +(\|J\|+\|\tilde J\|+\sqrt{\alpha a}\|\varepsilon\|_\infty)
      \mathbf 1_{\{z\ne\tilde z\}}.
$$

Практическое значение: можно легально говорить о стационарных средних
$\mathbb E_{\Pi_{J,\alpha}}[\tilde A(Z_{t+1})J_t^{(0,\alpha)}]$ и переносить
их на finite-time оценки через геометрические boundary terms.

### 4. Главная bias-формула: Proposition 2

Под стационарным распределением для
$(Z_{t+1},J_t^{(0,\alpha)},J_t^{(1,\alpha)})$ выполняется
$\mathbb E[J_\infty^{(0,\alpha)}]=0$ и

$$
\mathbb E[J_\infty^{(1,\alpha)}]
  = \alpha\Delta+R(\alpha),
$$

где

$$
\Delta
  = \bar A^{-1}\sum_{k=1}^\infty
      \mathbb E\bigl[(Q^k\tilde A)(Z_\infty)\varepsilon(Z_\infty)\bigr],
$$

а остаток удовлетворяет

$$
\|R(\alpha)\|
  \le
  12\|\bar A^{-1}\| C_A^2 t_{\mathrm{mix}}^2
  \alpha^2\|\varepsilon\|_\infty.
$$

Самая полезная для нас форма получается прямо из стационарного уравнения для
$J^{(1)}$:

$$
\bar A\,\mathbb E[J_\infty^{(1,\alpha)}]
  =
  -\mathbb E[\tilde A(Z_{\infty+1})J_\infty^{(0,\alpha)}].
$$

Следовательно,

$$
\mathbb E[\tilde A(Z_{\infty+1})J_\infty^{(0,\alpha)}]
  =
  -\bar A(\alpha\Delta+R(\alpha)).
$$

Для RR-комбинации стационарных средних член $\alpha\Delta$ сокращается:

$$
\mathbb E[\tilde A J^{(0,2\alpha)}]
  -2\mathbb E[\tilde A J^{(0,\alpha)}]
  =
  \bar A\{2R(\alpha)-R(2\alpha)\}
  =O(\alpha^2).
$$

Именно эту формулу нужно использовать в bias-части PR-суммы. Формулировка
через $\mathbb E[J^{(1,\alpha)}]$ тоже верна, но в RR-разложении Levin
появляется статистика $\tilde A(Z_{t+1})J_t^{(0,\alpha)}$, поэтому знак и
множитель $\bar A$ лучше отслеживать через последнее равенство.

### 5. Второй bias-коэффициент: Proposition 6 и Remark 1

В appendix B.4 доказано

$$
\mathbb E[J_\infty^{(2,\alpha)}]
  =
  \alpha^2\Delta_2+R_2(\alpha),
$$

где

$$
\Delta_2
  =
  -\sum_{k=1}^\infty\sum_{i=0}^\infty
    \mathbb E[
      \tilde A(Z_{\infty+k+i+1})
      \tilde A(Z_{\infty+i+1})
      \varepsilon(Z_\infty)
    ],
$$

и

$$
\|R_2(\alpha)\|
  \le D_b\,t_{\mathrm{mix}}^4\alpha^{5/2}\|\varepsilon\|_\infty.
$$

Это не обязательно нужно для первого BE-результата, но полезно как источник
для будущего улучшения остаточного bias с $O(\alpha^{3/2})$ до $O(\alpha^2)$.

### 6. RR-разложение усреднённого итерата: equation (26)

Пусть $m=n-n_0$ и

$$
e(\theta,z)
  = \varepsilon(z)+\tilde A(z)(\theta-\theta^*).
$$

Из рекурсии следует точное тождество

$$
\bar A(\bar\theta_n^{(\alpha,\mathrm{RR})}-\theta^*)
  =
  \frac{1}{2\alpha m}
  \Bigl[
    4(\theta_{n_0}^{(\alpha)}-\theta_n^{(\alpha)})
    -(\theta_{n_0}^{(2\alpha)}-\theta_n^{(2\alpha)})
  \Bigr]
  +
  \frac1m\sum_{t=n_0}^{n-1}
  \{e(\theta_t^{(2\alpha)},Z_{t+1})
    -2e(\theta_t^{(\alpha)},Z_{t+1})\}.
$$

После раскрытия $\theta_t-\theta^*$ по perturbation-декомпозиции:

$$
\sum_{t=n_0}^{n-1} e(\theta_t^{(\alpha)},Z_{t+1})
  =
  E_n^{(\mathrm{tr},\alpha)}
  +\sum_{t=n_0}^{n-1}\varepsilon(Z_{t+1})
  +\sum_{\ell=0}^2\sum_{t=n_0}^{n-1}
      \tilde A(Z_{t+1})J_t^{(\ell,\alpha)}
  +\sum_{t=n_0}^{n-1}
      \tilde A(Z_{t+1})H_t^{(2,\alpha)}.
$$

Поэтому ведущий RR-член в этом представлении равен

$$
-\frac1m\sum_{t=n_0}^{n-1}\varepsilon(Z_{t+1})
$$

после умножения на $\bar A^{-1}$. Это даёт альтернативный путь к BE:
применить пуассоновскую мартингальную аппроксимацию напрямую к unweighted
Markov-сумме $\sum\varepsilon(Z_t)$, а не к весам
$\mathcal Q_\ell^{\mathrm{RR}}$. Оба пути должны вести к той же
асимптотической ковариации, но unweighted-путь ближе к Levin.

### 7. Центрированная статистика $\tilde A J^{(0)}$: Proposition 3 / Corollary 6

Определим

$$
\psi_\alpha(J,z)=\tilde A(z)J,\qquad
\bar\psi_\alpha(J,z)
  =\psi_\alpha(J,z)
   -\mathbb E_{\Pi_{J,\alpha}}[\psi_\alpha(J_0,Z_1)].
$$

Для любого начального распределения и достаточно малого $\alpha$ Levin et al.
получают bound вида

$$
\left\|
  \sum_{t=0}^{r-1}
    \bar\psi_\alpha(J_t^{(0,\alpha)},Z_{t+1})
\right\|_{L_p}
\le
c_{W,1}^{(2)}p^{3/2}(\alpha r)^{1/2}
+
c_{W,2}^{(2)}p^3\alpha^{-1/2}
  \log^{1/p}\!\frac1{\alpha a}.
$$

Грубее можно писать второй логарифм как $\log^{1/2}(1/(\alpha a))$ для
$p\ge2$. После деления на $m$ и масштабирования $\sqrt n$ при $m\asymp n$:

$$
\sqrt n\,m^{-1}
\left\|\sum_{t=n_0}^{n-1}\bar\psi_\alpha\right\|_{L_p}
\lesssim
p^{3/2}\sqrt\alpha
+
p^3(\alpha n)^{-1/2}\log^{1/p}\!\frac1{\alpha a}.
$$

При $\alpha\asymp n^{-1/2}$ это даёт вклад порядка $n^{-1/4}$ с логами.
Вывод: для BE-rate $n^{-1/4}$ можно использовать Levin напрямую; чтобы
сделать misadjustment строго меньшим остатком, нужен наш более тонкий
RR-kernel анализ $\nabla_\alpha S_n^{(\alpha)}$ из `src/last_iterate.typ`.

### 8. High-order moment bounds для $J^{(2)}$ и $H^{(2)}$

Proposition 8:

$$
\|J_n^{(2,\alpha)}\|_{L_p}
  \le
  D_J\,t_{\mathrm{mix}}^{5/2}
  p^{7/2}\alpha^{3/2}
  \log^{3/2}\!\frac1{\alpha a}.
$$

Proposition 9:

$$
\|H_n^{(2,\alpha)}\|_{L_p}
  \le
  D_H\,d^{1/q}t_{\mathrm{mix}}^{5/2}
  p^{7/2}\alpha^{3/2}
  \log^{3/2}\!\frac1{\alpha a}.
$$

В proof of Theorem 2 эти оценки используются так:

$$
\sum_{t=n_0}^{n-1}\tilde A(Z_{t+1})J_t^{(1,\alpha)}
$$

выражается через endpoint-разность $J^{(2,\alpha)}$ и сумму
$\sum\tilde A J^{(2,\alpha)}$ с помощью рекурсии для $J^{(2)}$; аналогично
$\sum\tilde A H^{(2,\alpha)}$ контролируется через Proposition 9.

После усреднения и масштабирования $\sqrt n$ эти члены дают вклад порядка

$$
\sqrt n\,\alpha^{3/2}p^{7/2}\log^{3/2}\!\frac1{\alpha a},
$$

то есть $n^{-1/4}$ при $\alpha\asymp n^{-1/2}$.

### 9. Theorem 2 как sanity check для всех остатков

Основной high-probability/moment результат Levin et al.:

$$
\|\bar A(\bar\theta_n^{(\alpha,\mathrm{RR})}-\theta^*)\|_{L_p}
\le
2C_{\mathrm{Rm},1}
\{\mathrm{Tr}\,\Sigma_\varepsilon^{(M)}\}^{1/2}
p^{1/2}n^{-1/2}
+R_{n,p,\alpha}^{(\mathrm{fl})}
+R_{n,p,\alpha}^{(\mathrm{tr})}
\|\theta_0-\theta^*\|e^{-\alpha an/24}.
$$

В упрощённой форме

$$
R_{n,p,\alpha}^{(\mathrm{fl})}
\lesssim
pn^{-3/4}
+
\bigl(p^{3/2}(\alpha n)^{-1/2}+\alpha^{1/2}\bigr)
p^{3/2}n^{-1/2}
+
p^{7/2}\alpha^{3/2}
\log^{3/2}\!\frac1{\alpha a},
$$

$$
R_{n,p,\alpha}^{(\mathrm{tr})}\lesssim(\alpha n)^{-1}.
$$

Более точная appendix-форма (equation (87)) заменяет последний член на

$$
\bigl(D_3^{(\mathrm{RR})}\alpha+D_4^{(\mathrm{RR})}n^{-1}\bigr)
p^{7/2}\alpha^{1/2}\log^{3/2}\!\frac1{\alpha a}.
$$

При $\alpha=\alpha_{p,\infty}^{(b)}n^{-1/2}$ unscaled ошибка имеет ведущий
порядок $n^{-1/2}$ и остатки порядка $n^{-3/4}$; после умножения на
$\sqrt n$ это ровно даёт residual scale $n^{-1/4}$. Это не Berry--Esseen,
но это сильная проверка, что выбранная $\alpha\asymp n^{-1/2}$ совместима с
остатками нашего BE-доказательства.

### 10. Технические инструменты из appendix, которые стоит цитировать

**Lyapunov contraction (Proposition 10).** Если $-\bar A$ Hurwitz, то существует
$Q\succ0$ такое, что $\bar A^\top Q+Q\bar A=I$, и для малых $\alpha$

$$
\|I-\alpha\bar A\|_Q^2\le1-a\alpha,\qquad \alpha a\le\frac12.
$$

Это источник всех оценок матричных степеней $B_\alpha^k$.

**Berbee decoupling (Lemma 6).** Для каждого $m$ можно построить
$Z_k^*$, независимую от future $\sigma$-поля $\sigma(Z_\ell:\ell\ge k+m)$,
с тем же законом, что $Z_k$, и

$$
\mathbb P(Z_k^*\ne Z_k)\le\Delta(Q^m).
$$

Это ровно тот инструмент, который нужен для future-dependent ядер
$H_{k+1}^{(w)}\varepsilon(Z_k)$.

**Weighted Markov concentration (Lemma 11).** Если $g_i:Z\to\mathbb R^d$,
$\pi(g_i)=0$, $\|g_i\|_\infty=c_i$, то

$$
\mathbb P_\xi\!\left(
  \left\|\sum_{i=1}^n g_i(Z_i)\right\|\ge t
\right)
\le
2\exp\!\left\{
  -\frac{t^2}{2u_n^2}
\right\},
\qquad
u_n=8\sqrt{t_{\mathrm{mix}}}
\left(\sum_i c_i^2\right)^{1/2}.
$$

Это уже используется в `src/zeroth_order_rr.typ` для RR-разности
$J^{(0)}$.

**Rosenthal for Markov chains (Theorem 3).** Для bounded centered
$f:Z\to\mathbb R^d$:

$$
\left\|
  \sum_{i=0}^{n-1}\{f(Z_i)-\pi(f)\}
\right\|_{L_p}
\le
C_{\mathrm{Rm},1}p^{1/2}n^{1/2}\sigma_\pi(f)
+
C_{\mathrm{Ros},1}n^{1/4}t_{\mathrm{mix}}p^{3/4}\log^2(2p)
+
C_{\mathrm{Ros},2}t_{\mathrm{mix}}p\log^2(2p).
$$

Для BE нам всё равно нужен пуассоновский martingale route, но эта оценка
полезна для $L_p$-контроля остатков и sanity checks.

### 11. Что Levin закрывает, а что остаётся нашим

Levin et al. закрывают:

1. существование стационарных распределений для augmented chains;
2. точный $O(\alpha)$ bias coefficient $\Delta$ и RR-cancellation до
   $O(\alpha^2)$ для стационарного среднего $\tilde A J^{(0)}$;
3. $L_p$-контроль centered $\sum\tilde A J^{(0)}$;
4. $L_p$-контроль $J^{(2)}$ и $H^{(2)}$;
5. полный moment/high-probability sanity check для RR-среднего.

Levin et al. не закрывают:

1. Kolmogorov/Berry--Esseen distance для
   $\sqrt n\,u^\top(\bar\theta_n^{(\alpha,\mathrm{RR})}-\theta^*)$;
2. концентрацию predictable quadratic variation для martingale
   approximation в нужной one-dimensional нормировке;
3. сравнение $\sigma_n^{\mathrm{RR}}(u)$ с $\sigma(u)$ в нашей
   weighted-$Q_\ell$ формулировке;
4. улучшенную RR-разность centered misadjustment до
   $O(\sqrt n\,\alpha^2)$ вместо Levin-достаточного $O(n^{-1/4})$ при
   $\alpha=n^{-1/2}$.

Итог: для первого BE-результата можно опираться на Levin для всех
non-leading остатков и получить тот же порядок $n^{-1/4}$, что и
мартингальная BE-часть. Если цель — сделать RR-остатки строго ниже
мартингального BE-rate, тогда центральной новой леммой остаётся
RR-kernel bound для $\nabla_\alpha S_n^{(\alpha)}$.

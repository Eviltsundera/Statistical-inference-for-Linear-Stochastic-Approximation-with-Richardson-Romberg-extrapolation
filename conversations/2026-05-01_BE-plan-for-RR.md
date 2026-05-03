# Berry–Esseen Plan for the PR-averaged Richardson–Romberg Iterate

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

Здесь происходит **главная техническая работа**. Остаток разбивается на

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

### Misadjustment $D_1^{\mathrm{mis,RR}}$ — критическая часть

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

**Это центральная задача нового анализа.**

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

## Итоговая структура $\mathrm B_n^{\mathrm{RR}}$

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

## План работы (по убыванию приоритета)

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

3. **Lemma "RR-misadjustment":** обобщить `src/last_iterate.typ` на разность
   $\nabla_\alpha S_n^{(\alpha)}$, доказав
   $\|U^{\mathrm{RR}}_M\|_{L_p}+\|U^{\mathrm{RR}}_R\|_{L_p}\le C\alpha^2(\dots)$.
   Это **главная техническая часть**: схема та же (shifted kernel + условное
   центрирование), но **базовая оценка ядра $\mathcal H^{\mathrm{RR}}$
   несёт дополнительный $\alpha$**.

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

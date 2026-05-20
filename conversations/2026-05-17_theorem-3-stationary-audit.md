# Проверка Theorem 3 как stationary augmented-chain theorem

## Задача

Проверить, что Theorem 3 в `src/pr_weights.typ` действительно является
утверждением про stationary augmented-chain режим, а не неявным
finite-start/burn-in результатом. Отдельно отслеживаем, какие места надо
закрыть перед будущим burn-in transfer theorem.

Рабочая идентификация: Theorem 3 -- это stationary $n_0=0$ PR-averaged RR
Berry--Esseen bound для
$$
\frac{S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)}
     {\sigma_n^{\mathrm{RR}}(u)},
$$
блока `<thm:RR-BE>` в `src/pr_weights.typ`. Misadjustment theorem выше
является ингредиентом, а не финальной Theorem 3.

## План проверки

- [x] Найти точные границы Theorem 3 и связанных ингредиентов.
- [x] Проверить, что объект теоремы называется
  $S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)$, а не
  $\sqrt n(\bar\theta_n^{\mathrm{RR}}-\theta^*)$ для arbitrary start.
- [x] Проверить, что deterministic transient
  $D_{\mathrm{tr}}^{\mathrm{RR}}$ явно исключен.
- [x] Проверить, что misadjustment input используется только под stationary
  augmented-chain convention.
- [x] Проверить, что Poisson remainder $D_{2,n}^{\mathrm{RR}}$ и martingale
  term $M_n^{\mathrm{RR}}$ совместимы с той же stationary assembly.
- [x] Проверить proof step: smoothing applies to $X_n+Y_n$ with
  $Y_n = u^\top \mathcal R_{n,\mathrm{stat}}^{\mathrm{RR}} /
  \sigma_n^{\mathrm{RR}}(u)$.
- [x] Убрать двусмысленное "standing hypotheses" из формулировки Theorem 3.
- [x] Переименовать/уточнить формулировки: не "main theorem", а stationary
  $n_0=0$ PR-averaged RR bound; явно сказать, что burn-in будет отдельным
  transfer theorem.
- [x] Починить ссылки системно: theorem-like labels сделать referenceable,
  equation references перевести с `<eq:...>` на `@eq:...`.
- [ ] Позже: оформить отдельный finite-start/burn-in transfer theorem.

## Текущий вывод

После правки формулировки Theorem 3 можно честно читать как stationary
augmented-chain theorem. Ключевой объект:
$$
S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)
  = -\frac{u^\top M_n^{\mathrm{RR}}}{\sqrt n}
    + u^\top \mathcal R_{n,\mathrm{stat}}^{\mathrm{RR}},
$$
где
$$
\mathcal R_{n,\mathrm{stat}}^{\mathrm{RR}}
  = D_{2,n}^{\mathrm{RR}} + R_n^{\mathrm{mis,RR}}.
$$

Это не утверждение про фактическую finite-start RR average. В тексте это уже
сказано явно: deterministic transient
$$
D_{\mathrm{tr}}^{\mathrm{RR}}
  = 2D_{\mathrm{tr}}^{(\alpha)} - D_{\mathrm{tr}}^{(2\alpha)}
$$
не входит в stationary $n_0=0$ result; startup discrepancy между zero-start
perturbation variables и stationary augmented chain также вынесен в будущий
burn-in transfer.

## Что было исправлено в тексте

В `src/pr_weights.typ:1289` формулировка Theorem 3 больше не начинается с
расплывчатого "Under the standing hypotheses...". Теперь в самом утверждении
явно стоят:

- *UGE 1*;
- $\pi(\bar\epsilon)=0$;
- $\|\bar\epsilon\|_\infty < \infty$;
- $\sigma^2(u)>0$;
- $\alpha,2\alpha \in (0,\alpha_\infty]$;
- stationary augmented-chain convention;
- ограничения на $p$, $q$, variance lower bound и
  $2\alpha \le \alpha_*(q,t_{\mathrm{mix}})$.

Это делает область действия теоремы явной и не позволяет прочитать ее как
burned-in или arbitrary-start утверждение.

## Пункт 3: naming and scope

В `src/pr_weights.typ` формулировки теперь разведены так:

- misadjustment theorem назван
  "Stationary PR-averaged RR misadjustment bound";
- финальная Berry--Esseen формулировка названа
  "Stationary $n_0=0$ PR-averaged RR Berry--Esseen bound";
- начало главы больше не говорит "final Berry--Esseen theorem", а говорит про
  stationary $n_0=0$ bound;
- заключение больше не говорит "main theorem above"; вместо этого явно сказано,
  что это утверждение для $S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)$, а не для
  deterministic-start RR average itself;
- burn-in вынесен в отдельный future transfer theorem with
  $Q_{\ell,n_0}^{(\alpha)}$ plus corresponding Poisson,
  variance-comparison, and misadjustment bounds.

## Пункт 4: references and labels

Системная правка сделана так:

- в `src/defs.typ` theorem-like окружения (`theorem`, `lemma`, `corollary`,
  `remark`) теперь реализованы как captionless `figure(kind: ...)`, поэтому
  labels вида `<lem:...>`, `<thm:...>`, `<cor:...>` можно нормально
  референсить через `@lem:...`, `@thm:...`, `@cor:...`;
- в `main.typ` включена нумерация display equations с supplement `Eq.`, чтобы
  equation labels можно было референсить через `@eq:...`;
- в `src/pr_weights.typ` текстовые псевдоссылки вида `<eq:...>` заменены на
  настоящие Typst references `@eq:...`; сами labels после displayed equations
  оставлены в виде `<eq:...>`.

Правило на будущее: `<...>` используется только для постановки label после
элемента, а ссылка в тексте должна быть `@...`.

## Локальная проверка доказательства

Smoothing step корректен в stationary statement:
$$
\frac{S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)}
     {\sigma_n^{\mathrm{RR}}(u)}
  = X_n + Y_n,
$$
где
$$
X_n =
-\frac{u^\top M_n^{\mathrm{RR}}}
       {\sqrt n\,\sigma_n^{\mathrm{RR}}(u)},
\qquad
Y_n =
\frac{u^\top \mathcal R_{n,\mathrm{stat}}^{\mathrm{RR}}}
     {\sigma_n^{\mathrm{RR}}(u)}.
$$

Martingale Berry--Esseen применяется к $-X_n$; знак не влияет из-за симметрии
standard normal. Для $Y_n$ используется Markov inequality и $L_p$-bound из
предыдущей lemma. Условие variance lower bound гарантирует, что
$\sigma_n^{\mathrm{RR}}(u)$ не вырождается.

## Остаточные долги перед burn-in

- Нужна отдельная transfer lemma от finite-start recursion к stationary
  augmented-chain variables. Нельзя заменять ее одним terminal
  $\rho^n$-членом, потому что PR average суммирует startup error по
  $k=0,\ldots,n-1$.
- Для burned-in theorem нужны веса
  $Q_{\ell,n_0}^{(\alpha)}$, а не full-window веса
  $Q_\ell^{(\alpha)}$.
- Нужно отдельно перенести Poisson remainder, variance comparison и
  misadjustment bounds на burned-in weights.
- В proof of $T^{(1)}$ остается техническое место: переход к stationary
  version через "start at time $-m$ and let $m\to\infty$" надо при строгой
  версии оформить как lemma с uniform $L_p$ bound и convergence аргументом.

## Статус

Theorem 3 сейчас годится как stationary $n_0=0$ augmented-chain theorem.
Главный внутренний долг перед строгой версией -- оформить stationary-limit
lemma в proof of $T^{(1)}$. Burn-in остается отдельным transfer theorem.

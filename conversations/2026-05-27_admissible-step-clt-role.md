# Role of the stationary triangular-array admissible-step CLT

## Question

Нужно ли дальше использовать королларий
`Stationary triangular-array admissible-step CLT`
из `src/pr_weights/11_smoothing_assembly.typ`?

## Short answer

Для текущей финальной линии доказательства он, похоже, не нужен как
используемая ссылка. Label

```typst
<cor:RR-BE-admissible-alpha>
```

не цитируется дальше по репозиторию. Финальный маршрут идет через:

1. stationary assembled Berry--Esseen theorem `@thm:RR-BE`;
2. balanced stationary corollaries `@cor:RR-BE-working` and `@cor:RR-BE-sigma`;
3. deterministic-start burn-in theorem `@thm:burn-final-balanced`;
4. final $\sqrt n$ transfer `@cor:burn-sqrt-n-transfer`.

То есть для theorem chain при $\alpha_n = c n^{-1/2}$ достаточно
balanced-scale corollaries, а не общего admissible-step CLT.

## Why it may still be useful

Королларий полезен как exposition/theorem-map statement. Он говорит, что
stationary augmented-chain result не только работает в balanced scale
$\alpha_n=c n^{-1/2}$, но и дает CLT для более широкого triangular-array окна.
При power scale $\alpha_n=c n^{-\gamma}$ условия дают, up to logs,

$$
\frac{1}{3}<\gamma<1.
$$

Это помогает объяснить, что:

- fixed-$\alpha$ statement centered exactly at $\theta^*$ не является тем, что
  доказывается;
- нужен triangular-array reading с $\alpha_n\to0$;
- balanced scale $\gamma=1/2$ находится внутри admissible window.

## Recommendation

Если цель главы — короткая proof chain к финальному theorem, этот королларий
можно удалить или перенести в remark после `@cor:RR-BE-working`.

Если цель — показать более общую stationary CLT window, его стоит оставить,
но он должен быть явно представлен как самостоятельный consequence of
`@thm:RR-BE`, а не как промежуточный шаг, который нужен дальше.

## Small consistency note

В условии сейчас стоит

$$
p_n^3 (n\alpha_n)^{-1/2}\Lambda_n^{1/p_n}\to0.
$$

Но из bound `@lem:R-bound` соответствующий term выглядит как

$$
p^3(\alpha n)^{-1/2}\log^{1/2}(1/(\alpha a)).
$$

Поэтому если королларий остается, показатель у $\Lambda_n$ стоит проверить:
ожидаемо должен быть $\Lambda_n^{1/2}$, если нет отдельного сглаживающего
аргумента, который меняет степень.

## What about the other two stationary corollaries?

После admissible-step CLT идут:

1. `@cor:RR-BE-working`: stationary balanced-scale augmented-chain
   Berry--Esseen bound;
2. `@cor:RR-BE-sigma`: stationary asymptotic-normalization version.

Они тоже не являются строгими prerequisites для deterministic-start final
theorem: burn-in chapter заново доказывает finite-window burned-in assembly and
normalization transfer through `@thm:burn-RR-BE-master` and
`@lem:burn-normalization-transfer`.

Но, в отличие от general admissible-step CLT, эти два следствия лучше оставить.
Они фиксируют две thesis-facing интерпретации stationary chapter:

- `@cor:RR-BE-working` сворачивает master bound `@thm:RR-BE` в читаемый rate
  at the intended scale $\alpha=c n^{-1/2}$:

  $$
  d_K\left(
    \frac{S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)}
         {\sigma_n^{\mathrm{RR}}(u)},
    N(0,1)
  \right)
  \le
  C(u)\,\mathrm{polylog}(n)\,n^{-1/4}.
  $$

- `@cor:RR-BE-sigma` replaces finite-window normalization
  $\sigma_n^{\mathrm{RR}}(u)$ by the asymptotic normalization $\sigma(u)$.
  This is the stationary analogue of the normalization step later used for
  the burned-in final statistic.

So the recommended hierarchy is:

- keep `@thm:RR-BE` as the technical master bound;
- keep `@cor:RR-BE-working` as the readable balanced-scale stationary result;
- keep `@cor:RR-BE-sigma` if the text wants to identify the final covariance
  target $\Sigma_\infty$ already in the stationary chapter;
- demote or remove `@cor:RR-BE-admissible-alpha` unless the wider window
  $1/3<\gamma<1$ is important for the story.

# Проверка доказательств текущего диплома

## Вопрос

Проверить, что в доказательствах текущего текста диплома нет ошибок.

## Краткий вывод

Я не нашел явной алгебраической ошибки или несогласованности в главной
цепочке:

$$
\text{RR weights}
\to
\text{Poisson martingale approximation}
\to
\text{bracket concentration}
\to
\text{martingale Berry--Esseen}
\to
\text{misadjustment}
\to
\text{burn-in transfer}.
$$

Текущая финальная теорема в `src/burn_in_transfer.typ` согласована с
заявленным результатом:

$$
d_K\left(
  \Xi_{n,n_0}^{\mathrm{asy,RR}}(u), N(0,1)
\right)
\le
C(u,c,\theta_0)\frac{\mathrm{polylog}(n)}{n^{1/4}}
$$

при $\alpha=c n^{-1/2}$, $m=n-n_0\ge n/2$ и
$n_0\gtrsim(\alpha a)^{-1}\log^2 n$.

Но я бы не писал "доказательства полностью без ошибок" без оговорки:
есть один load-bearing участок, который все еще выглядит как сжатое
расширение импортированного результата Levin et al., а не как полностью
самодостаточно доказанная лемма.

## Проверено

- `typst compile main.typ` проходит.
- Прежняя ошибка в normalization transfer исправлена:
  в `src/burn_in_transfer.typ` стоит
  $r_{n,n_0}(u)\in[1/\sqrt 2,\sqrt{3/2}]$, а не ошибочное
  $\sqrt 3/2$.
- Потерянный stochastic initial-product discrepancy теперь включен:
  `cal(I)_{n,n_0}^{init,RR}` входит в composite remainder.
- Знак в smoothing/martingale step обработан правильно: используется
  signed martingale increments, а не сомнительная симметрия
  Kolmogorov distance.
- Stationary theorem в `src/pr_weights.typ` явно отделена от
  deterministic-start theorem: объектом является
  $S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)$.
- Burned-in weights $Q_{\ell;n_0,n}^{\mathrm{RR}}$ не подменяются
  full-window weights: pre-burn-in и post-burn-in индексы разведены.
- Переход от finite-window normalization
  $\sigma_{n,n_0}^{\mathrm{bRR}}(u)$ к
  $\sigma(u)=\sqrt{u^\top\Sigma_\infty u}$ имеет правильный порядок
  $O((m\alpha a)^{-1})=O(n^{-1/2})$ на balanced scale.

## Findings

### P1. Full-state startup contraction is still the main proof gap

Где: `src/burn_in_transfer.typ`, lemma
`Full-state startup contraction for the depth-two augmented remainder`,
roughly lines 774--898.

Эта лемма нужна для deterministic-start transfer. Для координат
$J^{(0)},J^{(1)},J^{(2)}$ текст ссылается на Levin Appendix B.2. Для
$H^{(2)}$ текст строит расширение через finite-past construction и
random-product stability:

$$
H_k^{(2,w)}
  =
  -w\sum_{\ell=1}^k
  \Gamma_{\ell+1:k}^{(w)}
  \widetilde A(Z_\ell)J_{\ell-1}^{(2,w)}.
$$

Идея правильная, и я не вижу явного неверного множителя в итоговом rate.
Но это самое нагруженное место доказательства: здесь одновременно используются
coupling time, conditional product stability at random time, finite-past
stationary construction for $H^{(2)}$, one-trajectory Levin Proposition 9,
и convolution estimate. В текущем виде часть переходов остается
proof-sketch style.

Что стоит добавить, чтобы закрыть замечание:

1. Явно сформулировать, что Levin Proposition 9 применяется к finite-past
   stationary copy uniformly in the past truncation, затем предел сохраняет
   тот же $L_p$ bound.
2. В lemma `Conditional product stability at a coupling time` отдельно
   разобрать крайний случай $T=k$ или $l=k$, где product пустой.
3. В строке с
   $B_H(p,q,w)\le A_{\mathrm{st}}(p,q,w)$ явно показать поглощение степеней
   $p,t_{\mathrm{mix}},a,w$ и сказать, какие степени $a$ считаются
   tracked, а какие absorbed into constants.

Без этого финальная burn-in theorem опирается на правдоподобную, но
не полностью self-contained extension of Levin.

### P2. Финальная теорема должна явно сказать, что начальный закон Markov chain произвольный

Где: `src/burn_in_transfer.typ`, final theorem lines 1332--1396.

Локальные леммы bracket concentration и startup transfer сформулированы
uniformly over initial law $\xi$ of the base chain. Финальная теорема говорит
о deterministic start через $\theta_0$, но не проговаривает явно начальный
закон $Z_1$ или $Z_0$.

Это не ломает proof, потому что оценки выше uniform in $\xi$. Но для читателя
лучше добавить в theorem statement фразу:

> for any initial distribution of the Markov chain, uniformly over that
> distribution.

Или, если intended setting is deterministic $Z_0=z$, написать это.

### P2. Imported thresholds are now named, but still not fully auditable

Где: `src/pr_weights.typ`, `Imported Inputs and Admissibility Thresholds`;
`src/burn_in_transfer.typ`, final theorem.

Текст уже сильно лучше: есть $\alpha_*(q,t_{\mathrm{mix}})$ и
$\alpha_{\mathrm{st}}(p)$, а финальная theorem собирает их в
$\alpha_{\mathrm{adm}}(p,q)$. Но если диплом должен быть проверяем без
открытия Levin/Samsonov, нужно еще точнее выписать:

- какие именно условия входят в $\alpha_*(q,t_{\mathrm{mix}})$;
- какие constants входят в $\alpha_{\mathrm{st}}(p)$;
- какие moment/cost assumptions нужны для product stability and full-state
  startup contraction.

Это dependency/auditability issue, не найденная ошибка в ставках по
$n,\alpha,p$.

## Места, где ошибок не нашел

### RR weights

Full-window identities:

$$
\mathcal Q_\ell^{\mathrm{RR}}-\bar A^{-1}
  =
  -\bar A^{-1}(2B_\alpha^k-B_{2\alpha}^k),
$$

$$
\mathcal Q_{\ell+1}^{\mathrm{RR}}-\mathcal Q_\ell^{\mathrm{RR}}
  =
  -2\alpha(B_\alpha^{k-1}-B_{2\alpha}^{k-1})
$$

согласованы. Burned-in pre/post formulas тоже согласованы, включая boundary
$\ell=n_0-1$.

### Poisson decomposition

И в stationary, и в burned-in case right boundary действительно исчезает:

$$
Q_{n-1}^{\mathrm{RR}}
  =
  2\alpha I - 2\alpha I
  =
  0.
$$

Индекс martingale increments starts at $\ell=2$, а variance proxy использует
тот же index set.

### Variance comparison

Оценки

$$
\|\Sigma_n^{\mathrm{RR}}-\Sigma_\infty\|
  \lesssim \frac{1}{n\alpha a},
\qquad
\|\Sigma_{n,n_0}^{\mathrm{bRR}}-\Sigma_\infty\|
  \lesssim \frac{1}{m\alpha a}
$$

согласованы с energy bounds на RR weights. Pre-burn-in weights в burned-in
case учтены отдельно через $\sum_{\ell<n_0}\|Q_\ell^{\mathrm{bRR}}\|^2$.

### Martingale Berry--Esseen rate

Использование Bolthausen--Fan inequality дает

$$
\frac{\log^{3/4}n}{n^{1/4}}
$$

или burned-in version

$$
\frac{\log^{3/4}n}{m^{1/4}},
$$

потому что bracket concentration has order $\sqrt{pn}$ and final theorem
assumes $m\ge n/2$.

### Final normalization

Переход от $\sqrt m$ к $\sqrt n$ в финальном corollary корректен при upper
burn-in window:

$$
n_0 \le C_0(\alpha a)^{-1}\log^2 n.
$$

Тогда

$$
\sqrt{n/m}-1
  \lesssim \frac{n_0}{n}
  =
  O(n^{-1/2}\log^2 n),
$$

что ниже основного $n^{-1/4}\mathrm{polylog}(n)$ rate.

## Итог

Явных ошибок уровня "формула неверна, theorem не следует" в текущей версии не
нашел. Самый важный оставшийся риск -- не algebraic, а proof-completeness:
лемма full-state startup contraction for $H^{(2)}$ должна быть либо
расширена до более детального самостоятельного доказательства, либо явно
помечена как импортируемый technical input.

Если эту лемму принять как input, главная stationary + burn-in proof chain
выглядит согласованной.

# Are the Last Iterate Analysis estimates used later?

## Question

Используются ли оценки из `Last Iterate Analysis` в дальнейшей работе?

## Short answer

Да, но только одна часть главы используется как настоящий downstream input.
Оценка

$$
\|u^\top (T_n^{(1,\alpha)}-\mathbb E T_n^{(1,\alpha)})\|_{L_p}
  \lesssim
  \alpha \|u\|\|\epsilon\|_\infty
  \left(
    p^{3/2} t_{\mathrm{mix}}^{1/2} a^{-1}
    + p^{1/2}t_{\mathrm{mix}}^{3/2}\sqrt{\alpha/a}
  \right)
$$

из леммы `lem:last-shifted-first-order` используется в Chapter 4 для контроля
первого stationary misadjustment boundary term. Остальная depth-one
Richardson--Romberg дискуссия в `Last Iterate Analysis` является
диагностической: она показывает, почему прямой depth-one маршрут слишком груб,
и поэтому не входит в финальную Berry--Esseen assembly как самостоятельный
input.

## Dependency chain

1. В `src/last_iterate/00_centered_shifted_first_order.typ` глава прямо
   говорит, что она partly preliminary, а retained downstream input is the
   centered shifted first-order bound. Это лемма
   `lem:last-shifted-first-order`.

2. В `src/pr_weights/10_misadjustment_depth_two.typ` эта лемма используется
   для stationary-limit transfer:

   $$
   T_t^{(1,w)} = B_w J_t^{(1,w)}
   $$

   и после inverse-bound для $B_w^{-1}$ дает контроль

   $$
   \|J_t^{(1,w)}-\mathbb E_\pi J_t^{(1,w)}\|_{L_p}
     \lesssim w\,\Phi_+(p,w).
   $$

   Это записано как `lem:stationary-limit-J1`.

3. Затем `lem:stationary-limit-J1` используется в доказательстве
   `lem:T1-bound`, именно для boundary term в centered telescoping identity for
   $J^{(1)}$:

   $$
   \frac{1}{w}
   \|J_0^{(1,w)}-J_n^{(1,w)}\|_{L_p}
   \lesssim \Phi_+(p,w).
   $$

4. `lem:T1-bound` вместе с depth-two bounds for $J^{(2)}$ and $H^{(2)}$
   образует theorem `thm:misadjustment`.

5. `thm:misadjustment` входит в stationary smoothing assembly
   `thm:RR-BE` through the composite remainder

   $$
   \mathcal R_{n,\mathrm{stat}}^{\mathrm{RR}}
     = D_{2,n}^{\mathrm{RR}} + R_n^{\mathrm{mis,RR}}.
   $$

6. Burn-in chapter then uses the stationary augmented-chain misadjustment bound
   `thm:misadjustment` as one input in `thm:burn-misadjustment`.

So the live path is:

$$
\texttt{lem:last-shifted-first-order}
\to
\texttt{lem:stationary-limit-J1}
\to
\texttt{lem:T1-bound}
\to
\texttt{thm:misadjustment}
\to
\texttt{thm:RR-BE}
\to
\texttt{thm:burn-misadjustment}.
$$

## What is not used

The subsection `A Depth-One RR Misadjustment Bound and Its Limitation` is not
used as a final proof input. Its role is negative/diagnostic:

$$
\|D_{1,c}^{\mathrm{mis,RR}}\|_{L_p}
  \lesssim \sqrt n\,\alpha\,\Phi(p,\alpha),
$$

which is $O(1)$ at the balanced scale $\alpha \asymp n^{-1/2}$. This is too
large for the target $n^{-1/4}$ Berry--Esseen remainder, so the final proof
switches to the Levin depth-two decomposition

$$
R_k^{(\alpha)}
  = J_k^{(1,\alpha)} + J_k^{(2,\alpha)} + H_k^{(2,\alpha)}.
$$

Likewise, the zeroth-order last-iterate RR bound in Chapter 2 is mainly
pedagogical and explains the cancellation mechanism in a simpler terminal
iterate setting. The final Berry--Esseen theorem uses the PR-weight algebra and
martingale/Poisson/misadjustment decomposition of Chapter 4 instead.

## Unresolved or clarity gap

Mathematically the dependency is present, but exposition-wise it is easy to
miss because the useful part of `Last Iterate Analysis` is only the shifted
first-order centered estimate, while the RR depth-one conclusion is deliberately
not used. If polishing the thesis, it would help to add one explicit sentence
near `lem:stationary-limit-J1`:

> The only input imported from the last-iterate chapter is
> `lem:last-shifted-first-order`; the depth-one RR averaged bound from Section 3.2
> is used only to motivate the depth-two route.


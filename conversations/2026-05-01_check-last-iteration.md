# Corrected Last-Iteration Decomposition for $J_n^{(1,\alpha)}$

## Task

Проверить и аккуратно переписать фрагмент из `src/new_lemma.typ`, где
анализируется последний взвешенный член, связанный с $J_n^{(1,\alpha)}$.

## What Was Fixed

Первоначальная идея была правильной: нужно переписать первый perturbation term
как будущезависимую взвешенную сумму по траектории Марковской цепи, затем
отделить обычный centered additive functional от genuinely future-centered
bilinear part.

Но исходная выкладка содержала несколько критических ошибок:

- в явной формуле для $J_n^{(1,\alpha)}$ был потерян множитель
  $J_{j-1}^{(0,\alpha)}$;
- знак double-sum был неверным;
- ядро $H_{k+1}^{(w)}$ имело неаккуратно объясненный one-step shift;
- условное центрирование было проведено неполно;
- residual term был ошибочно заменен на выражение через $Qv_k(Z_{k-1})$ без
  добавления martingale difference;
- сумма Dobrushin coefficients должна оцениваться через UGE:

$$
\Delta(Q^l)
  \le
  \left(\frac14\right)^{\lfloor l/t_{\mathrm{mix}}\rfloor},
\qquad
\sum_{l\ge1}\Delta(Q^l)
  \le
  C t_{\mathrm{mix}}.
$$

Фрагмент в `src/new_lemma.typ` теперь переписан в виде lemma + proof + remark.

## Deterministic-Product Expansion

Пусть

$$
B = I-\alpha\bar A.
$$

The deterministic-product term satisfies

$$
J_n^{(0,\alpha)}
  =
  -\alpha\sum_{k=1}^n B^{n-k}\varepsilon(Z_k).
$$

For the first perturbation term,

$$
\begin{aligned}
J_n^{(1,\alpha)}
  &=
  -\alpha\sum_{j=1}^n
  B^{n-j}\tilde A(Z_j)J_{j-1}^{(0,\alpha)}
\\
  &=
  \alpha^2
  \sum_{1\le k<j\le n}
  B^{n-j}\tilde A(Z_j)B^{j-1-k}\varepsilon(Z_k)
\\
  &=
  \alpha^2
  \sum_{k=1}^{n-1}\sum_{r=1}^{n-k}
  B^{n-k-r}\tilde A(Z_{k+r})B^{r-1}\varepsilon(Z_k).
\end{aligned}
$$

The sign is positive in the double sum because
$J_{j-1}^{(0,\alpha)}$ already contains a minus sign.

## Shifted Kernel and Weighted Matrix

The section now studies the shifted statistic

$$
S_n
  =
  \sum_{t=0}^{n-1}
  B^{n-t}\tilde A(Z_{t+1})J_t^{(0,\alpha)}.
$$

Since $J_0^{(0,\alpha)}=0$,

$$
S_n
  =
  -\alpha\sum_{k=1}^{n-1}
  H_{k+1}^{(w)}\varepsilon(Z_k),
$$

where

$$
H_{k+1}^{(w)}
  =
  \sum_{l=1}^{n-k}
  B^{n-k-l+1}\tilde A(Z_{k+l})B^{l-1}.
$$

The extra factor $B$ in the left exponent is intentional: $S_n$ is the shifted
last-iteration statistic, while $J_n^{(1,\alpha)}$ itself would have
$B^{n-k-l}$.

The corresponding shifted first-order contribution is

$$
T_n^{(1,\alpha)}
  =
  -\alpha S_n.
$$

## Additive/Future Decomposition

Define

$$
\mu_k^{(w)}
  =
  \mathbb E_\pi[
    H_{k+1}^{(w)}\varepsilon(Z_k)
  ].
$$

Then

$$
S_n-\mathbb E S_n
  =
  -\alpha\sum_{k=1}^{n-1}
  \left[
    H_{k+1}^{(w)}\varepsilon(Z_k)
    -
    \mu_k^{(w)}
  \right].
$$

Let $\mathcal F_k=\sigma(Z_1,\ldots,Z_k)$. By the Markov property,

$$
\mathbb E[
  H_{k+1}^{(w)}\varepsilon(Z_k)
  \mid
  \mathcal F_k
]
  =
  v_k^{(w,\varepsilon)}(Z_k),
$$

where

$$
v_k^{(w)}(z)
  =
  \sum_{l=1}^{n-k}
  B^{n-k-l+1}(Q^l\tilde A)(z)B^{l-1},
\qquad
v_k^{(w,\varepsilon)}(z)
  =
  v_k^{(w)}(z)\varepsilon(z).
$$

Under stationarity, $\mu_k^{(w)}=\pi(v_k^{(w,\varepsilon)})$. Hence

$$
S_n-\mathbb E S_n
  =
  -\alpha(U_M+U_R),
$$

with

$$
U_M
  =
  \sum_{k=1}^{n-1}
  \left[
    H_{k+1}^{(w)}\varepsilon(Z_k)
    -
    v_k^{(w,\varepsilon)}(Z_k)
  \right],
$$

and

$$
U_R
  =
  \sum_{k=1}^{n-1}
  \left[
    v_k^{(w,\varepsilon)}(Z_k)
    -
    \pi(v_k^{(w,\varepsilon)})
  \right].
$$

This is the clean decomposition:

- $U_R$ is an ordinary centered additive functional of the Markov chain;
- $U_M$ is a future-centered bilinear term and is not a forward martingale sum.

## $L_p$ Bounds and Consequence

The corrected proof yields

$$
\|U_R\|_{L_p}
  \le
  C p^{1/2}t_{\mathrm{mix}}^{3/2}
  \|\varepsilon\|_\infty
  \frac{1}{\sqrt{\alpha a}},
$$

and, using the weighted Burkholder/block-coupling estimate from Samsonov et al.
(2025, Proposition 9),

$$
\|U_M\|_{L_p}
  \le
  C p^{3/2}t_{\mathrm{mix}}^{1/2}
  \|\varepsilon\|_\infty
  \frac{1}{\alpha a}.
$$

Therefore

$$
\|S_n-\mathbb E S_n\|_{L_p}
  \le
  C\|\varepsilon\|_\infty
  \left(
    p^{3/2}t_{\mathrm{mix}}^{1/2}\frac1a
    +
    p^{1/2}t_{\mathrm{mix}}^{3/2}
    \sqrt{\frac{\alpha}{a}}
  \right).
$$

For

$$
T_n^{(1,\alpha)}
  =
  -\alpha S_n,
$$

we get

$$
\|T_n^{(1,\alpha)}-\mathbb E T_n^{(1,\alpha)}\|_{L_p}
  \le
  C\alpha\|\varepsilon\|_\infty
  \left(
    p^{3/2}t_{\mathrm{mix}}^{1/2}\frac1a
    +
    p^{1/2}t_{\mathrm{mix}}^{3/2}
    \sqrt{\frac{\alpha}{a}}
  \right).
$$

Thus the centered shifted first-order contribution is $O(\alpha)$, with the
displayed dependence on $p$, $t_{\mathrm{mix}}$, and $a$.

## Stationarity and Bias Caveats

The local lemma is written in the stationary-chain version to keep the notation
clean. For a non-stationary initial distribution one should add the standard
UGE boundary terms. The bias of the first perturbation term is also handled
separately, through the stationary bias expansion of Levin et al. (2025).

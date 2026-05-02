#import "defs.typ": *

== Setup and Closed-Form Identities

The Polyak--Ruppert version of the LSA expansion replaces the trajectory $theta_k - theta^*$ by its time average and rewrites the leading martingale contribution as a single sum
$
W = -frac(1, sqrt(n)) sum_(l = 1)^(n - 1) Q_l^((alpha)) thin epsilon.alt(Z_l),
quad
Q_l^((alpha))
  = alpha sum_(k = l)^(n - 1) (I - alpha overline(A))^(n - k - 1),
$
so that $Q_l^((alpha))$ collects the deterministic-product weight with which the noise $epsilon.alt(Z_l)$ enters the PR average. (For constant step this is the specialization of the time-dependent kernel
introduced in Samsonov et al. (2025).) Coupling two trajectories with steps $alpha$ and $2 alpha$
through the same noise realization $\{Z_k\}$ and forming the Richardson--Romberg combination,
$
cal(Q)_l^("RR") := 2 Q_l^((alpha)) - Q_l^((2 alpha)),
$
we obtain the leading martingale weight for $sqrt(n)(bar(theta)_n^(("RR", alpha)) - theta^*)$. The Berry--Esseen analysis of the RR-averaged iterate hinges on two contraction estimates for $cal(Q)_l^("RR")$: comparison with the asymptotic weight $overline(A)^(-1)$, and a bound on the total variation of the successive differences $cal(Q)_(l+1)^("RR") - cal(Q)_l^("RR")$. The latter controls the Abel-summation term in the Poisson-equation remainder.

Throughout this chapter write
$
B_alpha := I - alpha overline(A),
quad
B_(2 alpha) := I - 2 alpha overline(A),
quad
k := n - l.
$
We assume $alpha, 2 alpha in (0, alpha_infinity]$, so that the Lyapunov contraction
$|| B_alpha^m ||_Q^2 <= (1 - alpha a)^m$ and $|| B_(2 alpha)^m ||_Q^2 <= (1 - 2 alpha a)^m$ hold (and we use freely $1 - 2 alpha a <= 1 - alpha a$).

The geometric-series identity gives the closed form
$
Q_l^((alpha))
  = alpha sum_(j = 0)^(n - l - 1) B_alpha^j
  = alpha thin (alpha overline(A))^(-1) thin (I - B_alpha^(n - l))
  = overline(A)^(-1) (I - B_alpha^(n - l)).
$
Two immediate consequences are
$
Q_l^((alpha)) - overline(A)^(-1) = - overline(A)^(-1) B_alpha^(n - l),
quad
Q_(l + 1)^((alpha)) - Q_l^((alpha)) = - alpha B_alpha^(n - l - 1).
$
The first identity, applied to $alpha$ and $2 alpha$, yields the basic RR identity
$
cal(Q)_l^("RR") - overline(A)^(-1)
  = - overline(A)^(-1) thin (2 B_alpha^k - B_(2 alpha)^k),
$
and the second yields the discrete-difference identity
$
cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")
  = - 2 alpha thin (B_alpha^(k - 1) - B_(2 alpha)^(k - 1)).
$

== Pointwise Bounds for the RR Weights

#lemma[
  Let $alpha, 2 alpha in (0, alpha_infinity]$, set $C_Q := kappa_Q^(1 slash 2) || overline(A)^(-1) ||$ and $tilde(C)_A := kappa_Q || overline(A) ||$, and write $k = n - l$.

  *(i)* For every $1 <= l <= n - 1$,
  $
  || cal(Q)_l^("RR") - overline(A)^(-1) || <= 3 C_Q (1 - alpha a)^(k slash 2).
  $

  *(ii)* For every $1 <= l <= n - 2$,
  $
  || cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR") ||
    <= 2 tilde(C)_A thin alpha^2 thin (k - 1) thin (1 - alpha a)^((k - 2) slash 2).
  $
]

_Proof of (i)._ Take norms in the basic RR identity above. Using the equivalence $|| dot || <= kappa_Q^(1 slash 2) || dot ||_Q$ and the Lyapunov contraction at both step sizes,
$
|| 2 B_alpha^k - B_(2 alpha)^k ||_Q
  <= 2 || B_alpha^k ||_Q + || B_(2 alpha)^k ||_Q
  <= 2 (1 - alpha a)^(k slash 2) + (1 - 2 alpha a)^(k slash 2)
  <= 3 (1 - alpha a)^(k slash 2).
$
Submultiplicativity in $|| dot ||_Q$ followed by norm equivalence then gives
$
|| cal(Q)_l^("RR") - overline(A)^(-1) ||
  <= kappa_Q^(1 slash 2) || overline(A)^(-1) || dot || 2 B_alpha^k - B_(2 alpha)^k ||_Q
  <= 3 C_Q (1 - alpha a)^(k slash 2).
$

_Proof of (ii)._ Apply the elementary identity
$X^m - Y^m = (X - Y) sum_(i = 1)^m X^(i - 1) Y^(m - i)$
with $X = B_alpha$, $Y = B_(2 alpha)$, $X - Y = alpha overline(A)$, and $m = k - 1$:
$
B_alpha^(k - 1) - B_(2 alpha)^(k - 1)
  = alpha overline(A) sum_(i = 1)^(k - 1)
    B_alpha^(i - 1) thin B_(2 alpha)^(k - 1 - i).
$
Each summand obeys, by submultiplicativity in $|| dot ||_Q$ and Lyapunov contraction,
$
|| B_alpha^(i - 1) thin B_(2 alpha)^(k - 1 - i) ||
  &<= kappa_Q^(1 slash 2) thin (1 - alpha a)^((i - 1) slash 2) (1 - 2 alpha a)^((k - 1 - i) slash 2) \
  &<= kappa_Q^(1 slash 2) thin (1 - alpha a)^((k - 2) slash 2),
$
where in the last step we used $1 - 2 alpha a <= 1 - alpha a$. Summing over $i in {1, dots, k - 1}$ and absorbing constants,
$
|| B_alpha^(k - 1) - B_(2 alpha)^(k - 1) ||
  <= alpha kappa_Q || overline(A) || (k - 1) (1 - alpha a)^((k - 2) slash 2).
$
Inserting this bound into the discrete-difference identity finishes the proof. $square$

== Summed Bounds and Comparison with the Single-Step Case

#corollary[
  Under the assumptions of the previous lemma, uniformly in $n >= 2$,
  $
  sum_(l = 1)^(n - 1) || cal(Q)_l^("RR") - overline(A)^(-1) ||^2
    <= frac(C_1, alpha a),
  quad
  sum_(l = 1)^(n - 2) || cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR") ||
    <= frac(C_2, a^2),
  $
  with $C_1 = 9 C_Q^2$ and $C_2 = 32 tilde(C)_A$.
]

_Proof._ For the first sum apply (i) and the geometric series $sum_(k >= 1) (1 - alpha a)^k <= 1 / (alpha a)$. For the second, apply (ii) and
$
sum_(k >= 2) (k - 1) thin (1 - alpha a)^((k - 2) slash 2)
  = sum_(m >= 0) (m + 1) (1 - alpha a)^(m slash 2)
  <= frac(1, (1 - sqrt(1 - alpha a))^2)
  <= frac(4, (alpha a)^2),
$
where the last step uses $1 - sqrt(1 - alpha a) >= alpha a / 2$ for $alpha a <= 1 slash 2$. Multiplying by $2 tilde(C)_A alpha^2$ gives the claim, up to the stated universal constant. $square$

#remark[
  Comparison with the single-step case. The same identities for $Q_l^((alpha))$ give
  $|| Q_l^((alpha)) - overline(A)^(-1) || <= C_Q (1 - alpha a)^(k slash 2)$ and
  $|| Q_(l + 1)^((alpha)) - Q_l^((alpha)) || <= alpha kappa_Q^(1 slash 2) (1 - alpha a)^((k - 1) slash 2)$.
  Part (i) of the lemma above is therefore the same order as the single-step bound: the leading
  $- overline(A)^(-1)$ term of $cal(Q)_(n - 1)^("RR")$ is exactly $- overline(A)^(-1)$ (since
  $cal(Q)_(n - 1)^("RR") = 0$ identically), so no factor of $alpha$ is gained on the boundary.
  The Richardson--Romberg cancellation is visible in the successive difference of
  (ii): for fixed lag $k$ the discrete derivative gains an extra factor of $alpha$.
  This local gain comes with the factor $k - 1$, so after summing over the geometric
  tail it should not be read as a global extra factor of $alpha$ in the Abel term.
  What is needed for the Poisson-equation remainder is the uniform total-variation
  bound in the corollary. On the variance side the corollary gives
  $sum || cal(Q)_l^("RR") - overline(A)^(-1) ||^2 = O(1 / (alpha a))$, which yields
  the finite-horizon comparison error $O(1 / (n alpha a))$.
]

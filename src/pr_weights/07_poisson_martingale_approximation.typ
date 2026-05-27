#import "../defs.typ": *

== Poisson Martingale Approximation

We convert $W^("RR")$ into a martingale plus an Abel remainder via the
Poisson equation.
// The variance comparison of the previous section identifies the limiting
// covariance of $W^("RR")$, but it does not by itself produce a martingale. The
// deterministic kernel bounds of Sections 4.3--4.4 enter once and for all in
// the boundary/Abel control of the remainder.

*Poisson kernel.* Let $sans(Q)$ be the one-step transition kernel:
$
(sans(Q) f)(z) = bb(E) lr([f(Z_(k + 1)) | Z_k = z]).
$
Under UGE 1,
$|| sans(Q)^k epsilon.alt ||_infinity <= 2 || epsilon.alt ||_infinity (1 slash 4)^(floor(k slash t_"mix"))$
(valid for centered $epsilon.alt$, $pi(epsilon.alt) = 0$) makes the Poisson
series
$
hat(epsilon.alt) := sum_(k = 0)^infinity sans(Q)^k epsilon.alt
$
absolutely convergent in sup-norm:
$
|| hat(epsilon.alt) ||_infinity
  <= 2 || epsilon.alt ||_infinity sum_(k = 0)^infinity (1 slash 4)^(floor(k slash t_"mix"))
  <= 3 thin t_"mix" thin || epsilon.alt ||_infinity,
quad
|| sans(Q) hat(epsilon.alt) ||_infinity
  <= || hat(epsilon.alt) ||_infinity,
$
and $hat(epsilon.alt)$ solves
// The second inequality uses that $sans(Q)$ is a Markov kernel and contracts
// the sup-norm.
$
hat(epsilon.alt) - sans(Q) hat(epsilon.alt) = epsilon.alt.
$ <eq:poisson-eq>

*Conditional centering.* Let $cal(F)_l := sigma(Z_0, dots, Z_l)$. For $l >= 2$,
$
epsilon.alt(Z_l)
  = m_l + t_l,
quad l >= 2,
$
where
$
m_l := hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)),
quad
t_l := sans(Q) hat(epsilon.alt)(Z_(l - 1)) - sans(Q) hat(epsilon.alt)(Z_l),
quad l >= 2,
$
and $m_l$ is centered conditionally on $cal(F)_(l - 1)$.
// The Markov property gives
// $bb(E)[hat(epsilon.alt)(Z_l) | cal(F)_(l - 1)]
//   = sans(Q) hat(epsilon.alt)(Z_(l - 1))$.
// The $l = 1$ term is kept as the left Abel boundary rather than included in
// $M_n^("RR")$. Abel summation then leaves the discrete derivative
// $cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")$, bounded in Section 4.4.

#lemma[
  *(Stationary Poisson martingale decomposition.)*
  Assume *UGE 1* and $pi(epsilon.alt) = 0$. Set
  $
  Delta M_l^("RR")
    := cal(Q)_l^("RR") thin (hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1))),
  quad 2 <= l <= n - 1,
  $
  and let $M_n^("RR") := sum_(l = 2)^(n - 1) Delta M_l^("RR")$. Then
  ${Delta M_l^("RR")}_(l = 2)^(n - 1)$ is a sequence of $cal(F)_l$-martingale
  differences, and
  $
  W^("RR")
    = -frac(1, sqrt(n)) M_n^("RR") + D_(2, n)^("RR"),
  $ <eq:poisson-decomp>
  with the *Abel remainder*
  $
  D_(2, n)^("RR")
    := -frac(1, sqrt(n)) lr([
        cal(Q)_1^("RR") thin hat(epsilon.alt)(Z_1)
        + sum_(l = 1)^(n - 2)
            (cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")) thin sans(Q) hat(epsilon.alt)(Z_l)
      ]).
  $
  Moreover, with
  // The right boundary
  // $cal(Q)_(n - 1)^("RR") thin sans(Q) hat(epsilon.alt)(Z_(n - 1))$
  // vanishes identically because $cal(Q)_(n - 1)^("RR") = 0$.
  $C_(cal(Q)) := || overline(A)^(-1) || + 3 C_Q$ a uniform bound on
  $|| cal(Q)_l^("RR") ||$, and $C_2$ the constant from the Corollary of
  Section 4.4,
  $
  || D_(2, n)^("RR") ||_infinity
    <= frac(3 thin t_"mix" thin || epsilon.alt ||_infinity, sqrt(n))
       lr((C_(cal(Q)) + frac(C_2, a^2))).
  $ <eq:D2-bound>
  Consequently, for every $p >= 1$,
  $
  || D_(2, n)^("RR") ||_(L_p)
    <= frac(C thin t_"mix" thin || epsilon.alt ||_infinity, a^2 thin sqrt(n)),
  $
  with a constant $C$ depending only on $|| overline(A)^(-1) ||$, $kappa_Q$,
  and $|| overline(A) ||$.
] <lem:poisson-martingale-decomp>

_Proof._ The increments are martingale differences by the Markov property.
// Here $cal(Q)_l^("RR")$ is deterministic and
// $bb(E)[hat(epsilon.alt)(Z_l) | cal(F)_(l - 1)]
//   = sans(Q) hat(epsilon.alt)(Z_(l - 1))$.

Substitute @eq:poisson-eq into @eq:W-RR:
$
W^("RR")
  = -frac(1, sqrt(n)) sum_(l = 1)^(n - 1)
    cal(Q)_l^("RR") thin (hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_l)).
$
Peel off $l = 1$:
$
W^("RR")
  = -frac(1, sqrt(n)) cal(Q)_1^("RR") thin hat(epsilon.alt)(Z_1)
    + frac(1, sqrt(n)) cal(Q)_1^("RR") thin sans(Q) hat(epsilon.alt)(Z_1)
    - frac(1, sqrt(n)) M_n^("RR")
    - frac(1, sqrt(n)) sum_(l = 2)^(n - 1)
      cal(Q)_l^("RR") thin (sans(Q) hat(epsilon.alt)(Z_(l - 1)) - sans(Q) hat(epsilon.alt)(Z_l)).
$
Set $g_l := sans(Q) hat(epsilon.alt)(Z_l)$. Abel summation gives
$
sum_(l = 2)^(n - 1) cal(Q)_l^("RR") (g_(l - 1) - g_l)
  = cal(Q)_2^("RR") thin g_1
    - cal(Q)_(n - 1)^("RR") thin g_(n - 1)
    + sum_(l = 2)^(n - 2)
      (cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")) thin g_l.
$
The boundary terms combine as
$
&-cal(Q)_1^("RR") thin hat(epsilon.alt)(Z_1)
+ cal(Q)_1^("RR") thin g_1
- cal(Q)_2^("RR") thin g_1
+ cal(Q)_(n - 1)^("RR") thin g_(n - 1)
- sum_(l = 2)^(n - 2)
    (cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")) thin g_l \
&quad= -cal(Q)_1^("RR") thin hat(epsilon.alt)(Z_1)
       - sum_(l = 1)^(n - 2)
          (cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")) thin g_l
       + cal(Q)_(n - 1)^("RR") thin g_(n - 1).
$
The rightmost boundary vanishes because
$cal(Q)_(n - 1)^("RR") = 2 alpha I - 2 alpha I = 0$. This gives the stated
formula.
// This uses $Q_(n - 1)^((alpha)) = alpha I$ and
// $Q_(n - 1)^((2 alpha)) = 2 alpha I$; the sign is absorbed into
// $D_(2, n)^("RR")$.

For @eq:D2-bound, use the sup-norm Poisson bound, the uniform weight bound, and
the summed-total-variation bound:
$
|| sqrt(n) thin D_(2, n)^("RR") ||
  <= 3 t_"mix" || epsilon.alt ||_infinity (C_(cal(Q)) + C_2 slash a^2).
$
The $L_p$ bound is immediate from the deterministic sup-norm bound. $square$

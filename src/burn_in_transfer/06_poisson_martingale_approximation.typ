#import "../defs.typ": *

== Burned-in Poisson Martingale Approximation

Let $hat(epsilon.alt) := sum_(j = 0)^infinity sans(Q)^j epsilon.alt$ be the
Poisson solution, so
$
hat(epsilon.alt) - sans(Q) hat(epsilon.alt) = epsilon.alt,
quad
|| hat(epsilon.alt) ||_infinity
  <= 3 thin t_"mix" thin || epsilon.alt ||_infinity.
$
The stationary Poisson identity applies because the coefficients
$Q_l^("bRR")$ are deterministic.

#lemma[
  *(Burned-in Poisson martingale decomposition.)*
  Assume *UGE 1* and $pi(epsilon.alt) = 0$. Define, for $2 <= l <= n - 1$,
  $
  Delta M_l^("bRR")
    := Q_l^("bRR") thin
       lr((hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)))) ,
  $
  and set $M_(n,n_0)^("bRR") := sum_(l = 2)^(n - 1) Delta M_l^("bRR")$.
  Then ${Delta M_l^("bRR")}_(l = 2)^(n - 1)$ is a sequence of
  $cal(F)_l$-martingale differences and
  $
  W_(n,n_0)^("RR")
    = -frac(1, sqrt(m)) M_(n,n_0)^("bRR")
      + D_(2,n,n_0)^("bRR"),
  $ <eq:burn-poisson-decomp>
  where
  $
  D_(2,n,n_0)^("bRR")
    := -frac(1, sqrt(m)) lr([
        Q_1^("bRR") thin hat(epsilon.alt)(Z_1)
        + sum_(l = 1)^(n - 2)
            lr((Q_(l + 1)^("bRR") - Q_l^("bRR")))
            thin sans(Q) hat(epsilon.alt)(Z_l)
      ]).
  $
  Moreover, with $C_("burn,Q") := || overline(A)^(-1) || + 6 C_Q$,
  $
  || D_(2,n,n_0)^("bRR") ||_infinity
    <= frac(3 thin t_"mix" thin || epsilon.alt ||_infinity, sqrt(m))
       lr((C_("burn,Q") + frac(C_("burn,V"), a^2))).
  $ <eq:burn-D2-bound>
] <lem:burn-poisson-decomp>

_Proof._ The martingale-difference property follows from the Markov property:
$bb(E)[hat(epsilon.alt)(Z_l) | cal(F)_(l - 1)]
  = sans(Q) hat(epsilon.alt)(Z_(l - 1))$.

Substitute the Poisson equation in @eq:burn-W-RR. The $l = 1$ term is kept as a
left boundary term, and for $l >= 2$ we add and subtract
$sans(Q) hat(epsilon.alt)(Z_(l - 1))$. Abel summation of the telescope gives
exactly @eq:burn-poisson-decomp. The right boundary vanishes because
$Q_(n - 1; n_0, n)^("RR") = 2 alpha I - 2 alpha I = 0$.

For the sup-norm bound, use
$|| sans(Q) hat(epsilon.alt) ||_infinity <= || hat(epsilon.alt) ||_infinity$,
the uniform weight bound $|| Q_l^("bRR") || <= C_("burn,Q")$ from
@eq:burn-post-weight-error and @eq:burn-pre-weight-size, and the total-variation
estimate @eq:burn-weight-variation. $square$

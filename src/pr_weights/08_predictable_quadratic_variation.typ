#import "../defs.typ": *

== Predictable Quadratic Variation Concentration

We control the predictable quadratic variation of $M_n^("RR")$ by applying
@eq:external-markov-conc to a centered quadratic functional of the chain.
// This is the RR analogue of Lemmas 22--23 of Samsonov et al. (2025).

*Predictable quadratic variation.* The increments
$Delta M_l^("RR") = cal(Q)_l^("RR") thin (hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)))$
are $cal(F)_l$-martingale differences. The Markov property gives
$bb(E)[hat(epsilon.alt)(Z_l) | cal(F)_(l - 1)] = sans(Q) hat(epsilon.alt)(Z_(l - 1))$
// Direct computation:
$
bb(E)[Delta M_l^("RR") thin (Delta M_l^("RR"))^top thin | thin cal(F)_(l - 1)]
  // &= cal(Q)_l^("RR")
  //    bb(E)[hat(epsilon.alt)(Z_l) thin hat(epsilon.alt)(Z_l)^top | cal(F)_(l - 1)]
  //    (cal(Q)_l^("RR"))^top \
  // &quad - cal(Q)_l^("RR") thin sans(Q) hat(epsilon.alt)(Z_(l - 1))
  //       thin (sans(Q) hat(epsilon.alt))(Z_(l - 1))^top thin
  //       (cal(Q)_l^("RR"))^top \
  &= cal(Q)_l^("RR") thin cal(V)_(epsilon.alt)(Z_(l - 1)) thin (cal(Q)_l^("RR"))^top,
$
where
$
cal(V)_(epsilon.alt)(z)
  := sans(Q)(hat(epsilon.alt) hat(epsilon.alt)^top)(z)
   - (sans(Q) hat(epsilon.alt))(z) thin (sans(Q) hat(epsilon.alt))(z)^top.
$ <eq:bar-eps-def>
// The cross-terms cancel by the Markov property. This is a matrix-valued
// conditional covariance of the Poisson martingale increment; the vector noise
// remains denoted by $epsilon.alt$.

#lemma[
  *(Poisson covariance identity and absolute convergence.)*
  Assume *UGE 1*, $pi(epsilon.alt) = 0$, and
  $||epsilon.alt||_infinity < infinity$. Then $Sigma_(epsilon.alt)^(("M"))$ is absolutely convergent in operator
  norm, and
  $
  pi(cal(V)_(epsilon.alt)) = Sigma_(epsilon.alt)^(("M")) =: Sigma.
  $ <eq:poisson-cov-identity>
] <lem:poisson-covariance-identity>

_Proof._ Boundedness and UGE give absolute convergence. For $j >= 1$,
$
||bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_j)^top]||
  = ||pi(epsilon.alt thin (sans(Q)^j epsilon.alt)^top)||
  <= ||epsilon.alt||_infinity ||sans(Q)^j epsilon.alt||_infinity
  <= 2 ||epsilon.alt||_infinity^2 (1 slash 4)^(floor(j slash t_"mix")),
$ <eq:SigmaM-abs-one-sided>
and similarly for the transposed covariance. Set $h := hat(epsilon.alt)$ and $r := sans(Q) h = h - epsilon.alt$. Since
$pi sans(Q) = pi$,
$
pi(cal(V)_(epsilon.alt))
  &= pi(h h^top) - pi(r r^top)
  &= pi(epsilon.alt epsilon.alt^top)
     + pi(epsilon.alt r^top) + pi(r epsilon.alt^top).
$
The Poisson series gives $r = sum_(j >= 1) sans(Q)^j epsilon.alt$ in sup-norm:
$
pi(epsilon.alt r^top)
  &= sum_(j >= 1)
     bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_j)^top], \
pi(r epsilon.alt^top)
  &= sum_(j >= 1)
     bb(E)_pi [epsilon.alt(Z_j) epsilon.alt(Z_0)^top].
$
Substitution proves @eq:poisson-cov-identity. $square$
// This is the Poisson-solution covariance identity used by Samsonov et al.
// (2025, Eq. (10)) and by the Markov-chain CLT formulation of
// Douc--Moulines--Priouret--Soulier (2018, Theorem 21.2.5).

Thus
$
chevron.l M^("RR") chevron.r_n
  &:= sum_(l = 2)^(n - 1)
      bb(E)[Delta M_l^("RR") thin (Delta M_l^("RR"))^top | cal(F)_(l - 1)]
  &= sum_(l = 2)^(n - 1)
      cal(Q)_l^("RR") thin cal(V)_(epsilon.alt)(Z_(l - 1))
      thin (cal(Q)_l^("RR"))^top.
$ <eq:M-RR-bracket>
By @lem:poisson-covariance-identity, $pi(cal(V)_(epsilon.alt)) = Sigma$.

*Sup-norm bound on $cal(V)_(epsilon.alt)$.* Since
$|| hat(epsilon.alt) ||_infinity <= 3 thin t_"mix" thin || epsilon.alt ||_infinity$
and $sans(Q)$ is a Markov kernel,
$
|| cal(V)_(epsilon.alt) ||_infinity
  &<= || sans(Q)(hat(epsilon.alt) hat(epsilon.alt)^top) ||_infinity
   + || (sans(Q) hat(epsilon.alt))(sans(Q) hat(epsilon.alt))^top ||_infinity \
  &<= 2 thin || hat(epsilon.alt) ||_infinity^2
   <= 18 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2.
$ <eq:bar-eps-sup>

#lemma[
  *(Stationary predictable-variation concentration.)*
  Assume *UGE 1* and $pi(epsilon.alt) = 0$, with $|| epsilon.alt ||_infinity < infinity$.
  Let $C_(cal(Q))$ be the uniform bound on $|| cal(Q)_l^("RR") ||$ from the
  previous lemma. There exists a universal constant $C_4 > 0$ such that, for
  every $u in bb(R)^d$, every $p >= 2$, every initial distribution $xi$, and every
  $n >= 2$,
  $
  bb(E)_xi^(1 slash p) lr([
    | u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u) |^p
  ])
    <= C_4 thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2
       thin t_"mix"^(5 slash 2) thin sqrt(p thin n).
  $ <eq:M-RR-conc>
] <lem:M-RR-bracket-conc>

_Proof._ Write $h_l(z) := u^top cal(Q)_l^("RR") thin cal(V)_(epsilon.alt)(z) thin (cal(Q)_l^("RR"))^top u$
and $g_l(z) := h_l(z) - pi(h_l)$. Then
$
|h_l(z)|
  <= C_(cal(Q))^2 thin || u ||^2 thin || cal(V)_(epsilon.alt) ||_infinity
  <= 18 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2,
quad
|| g_l ||_infinity <= 2 thin |h_l(z)|
  <= 36 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2.
$
By @eq:M-RR-bracket,
$n thin sigma_n^(2, "RR")(u) = sum_(l = 2)^(n - 1) pi(h_l)$,
$
u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u)
  = sum_(l = 2)^(n - 1) g_l(Z_(l - 1)).
$ <eq:M-RR-conc-decomp>

Set $tilde(g)_i := g_(i + 1)$. Then $pi(tilde(g)_i) = 0$ and
$|| tilde(g)_i ||_infinity <= c := 36 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2$.
Apply @eq:external-markov-conc with $c_i = c$:
$
bb(E)_xi^(1 slash p) lr([
  lr(|sum_(l = 2)^(n - 1) g_l(Z_(l - 1))|)^p
])
  <= C_("MC") sqrt(p thin t_"mix" thin (n - 2) c^2).
$
Substitution gives
$
bb(E)_xi^(1 slash p) lr([
  lr(|sum_(l = 2)^(n - 1) g_l(Z_(l - 1))|)^p
])
  <= C_("MC") dot 36 thin C_(cal(Q))^2 thin || u ||^2
       thin || epsilon.alt ||_infinity^2 thin t_"mix"^(5 slash 2)
       thin sqrt(p thin n).
$
Absorb the universal prefactor into $C_4$. $square$

#corollary[
  *(Stationary asymptotic-variance bracket comparison.)*
  Under the assumptions of the previous lemma, for every $u in bb(R)^d$, every
  $p >= 2$, and every $n >= 2$,
  $
  bb(E)_xi^(1 slash p) lr([
    | u^top chevron.l M^("RR") chevron.r_n u - n thin sigma^2(u) |^p
  ])
    <= C_4 thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2
       thin t_"mix"^(5 slash 2) thin sqrt(p thin n)
       + frac(C_3 thin || u ||^2, alpha thin a),
  $ <eq:M-RR-conc-sigma>
  with $C_3$ the variance-comparison constant of Section 4.5.
] <cor:M-RR-bracket-asymp>

// _Proof._ Use the triangle inequality:
// $
// | u^top chevron.l M^("RR") chevron.r_n u - n thin sigma^2(u) |
//   <= | u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u) |
//    + n thin | sigma_n^(2, "RR")(u) - sigma^2(u) |.
// $
// The first piece is bounded by @eq:M-RR-conc; the second by
// @lem:RR-variance-comparison.
// $square$

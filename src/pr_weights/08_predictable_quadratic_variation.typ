#import "../defs.typ": *

== Predictable Quadratic Variation Concentration

The Berry--Esseen step for the martingale $M_n^("RR")$ requires a quantitative
control of its predictable quadratic variation
$chevron.l M^("RR") chevron.r_n$ around its deterministic asymptotic counterpart
$n thin sigma_n^(2, "RR")(u)$. This subsection produces such a control by
applying the Markov concentration result of Section 2.2 to a suitable centered
quadratic functional of the chain. The result is the RR analogue of Lemmas
22--23 of Samsonov et al. (2025).

*Predictable quadratic variation.* By the Poisson Martingale Approximation
lemma above, the increments
$Delta M_l^("RR") = cal(Q)_l^("RR") thin (hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)))$
are $cal(F)_l$-martingale differences. The Markov property
$bb(E)[hat(epsilon.alt)(Z_l) | cal(F)_(l - 1)] = sans(Q) hat(epsilon.alt)(Z_(l - 1))$
gives, by direct computation,
$
bb(E)[Delta M_l^("RR") thin (Delta M_l^("RR"))^top thin | thin cal(F)_(l - 1)]
  &= cal(Q)_l^("RR")
     bb(E)[hat(epsilon.alt)(Z_l) thin hat(epsilon.alt)(Z_l)^top | cal(F)_(l - 1)]
     (cal(Q)_l^("RR"))^top \
  &quad - cal(Q)_l^("RR") thin sans(Q) hat(epsilon.alt)(Z_(l - 1))
        thin (sans(Q) hat(epsilon.alt))(Z_(l - 1))^top thin
        (cal(Q)_l^("RR"))^top \
  &= cal(Q)_l^("RR") thin cal(V)_(epsilon.alt)(Z_(l - 1)) thin (cal(Q)_l^("RR"))^top,
$
where the cross-terms in the expansion of
$(hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)))(hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)))^top$
cancel by the Markov property and we have set
$
cal(V)_(epsilon.alt)(z)
  := sans(Q)(hat(epsilon.alt) hat(epsilon.alt)^top)(z)
   - (sans(Q) hat(epsilon.alt))(z) thin (sans(Q) hat(epsilon.alt))(z)^top.
$ <eq:bar-eps-def>
This is a matrix-valued conditional covariance of the Poisson martingale
increment; the vector noise remains denoted by $epsilon.alt$.
Summing over $l in {2, dots, n - 1}$ and tracking that
$chevron.l M^("RR") chevron.r_n
:= sum_(l = 2)^(n - 1) bb(E)[Delta M_l^("RR") thin (Delta M_l^("RR"))^top | cal(F)_(l - 1)]$,
$
chevron.l M^("RR") chevron.r_n
  = sum_(l = 2)^(n - 1)
    cal(Q)_l^("RR") thin cal(V)_(epsilon.alt)(Z_(l - 1)) thin (cal(Q)_l^("RR"))^top.
$ <eq:M-RR-bracket>
The conditional covariance function $cal(V)_(epsilon.alt)$ has stationary mean
equal to the long-run noise covariance,
$
pi(cal(V)_(epsilon.alt)) = Sigma_(epsilon.alt)^(("M")) =: Sigma,
$
which is a standard identity for the Poisson solution of a Markov chain
(Samsonov et al. 2025, Eq. (10); see also Douc--Moulines--Priouret--Soulier
2018, Theorem 21.2.5).

*Sup-norm bound on $cal(V)_(epsilon.alt)$.* Since
$|| hat(epsilon.alt) ||_infinity <= 3 thin t_"mix" thin || epsilon.alt ||_infinity$
and $sans(Q)$ is a Markov kernel (so $sans(Q) f$ inherits the sup-norm of $f$),
$
|| cal(V)_(epsilon.alt) ||_infinity
  &<= || sans(Q)(hat(epsilon.alt) hat(epsilon.alt)^top) ||_infinity
   + || (sans(Q) hat(epsilon.alt))(sans(Q) hat(epsilon.alt))^top ||_infinity \
  &<= 2 thin || hat(epsilon.alt) ||_infinity^2
   <= 18 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2.
$ <eq:bar-eps-sup>

#lemma[
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
and $g_l(z) := h_l(z) - pi(h_l)$. Submultiplicativity of the operator norm and
@eq:bar-eps-sup gives
$
|h_l(z)|
  <= C_(cal(Q))^2 thin || u ||^2 thin || cal(V)_(epsilon.alt) ||_infinity
  <= 18 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2,
quad
|| g_l ||_infinity <= 2 thin |h_l(z)|
  <= 36 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2.
$
By @eq:M-RR-bracket and the variance definition
$n thin sigma_n^(2, "RR")(u) = sum_(l = 2)^(n - 1) pi(h_l)$,
$
u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u)
  = sum_(l = 2)^(n - 1) g_l(Z_(l - 1)).
$ <eq:M-RR-conc-decomp>

For the centered sum, set $tilde(g)_i := g_(i + 1)$ for $i in {1, dots, n - 2}$. By
construction $pi(tilde(g)_i) = 0$ and
$|| tilde(g)_i ||_infinity <= c := 36 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2$.
The scalar Markov concentration lemma of Section 2.2, imported from Levin et al.
(2025, Lemma 11) and valid for arbitrary initial law, applied to
$sum_(i = 1)^(n - 2) tilde(g)_i(Z_i)$ yields, for every $t >= 0$,
$
bb(P)_xi lr((
  lr(|sum_(i = 1)^(n - 2) tilde(g)_i(Z_i)|) >= t
))
  <= 2 exp(-frac(t^2, 2 thin u_n^2)),
quad
u_n^2 <= 64 thin t_"mix" thin (n - 2) thin c^2.
$
The sub-Gaussian-to-moment lemma of Section 2.2 then gives, for $p >= 2$,
$
bb(E)_xi^(1 slash p) lr([
  lr(|sum_(l = 2)^(n - 1) g_l(Z_(l - 1))|)^p
])
  <= 2^(1 slash p) thin sqrt(p) thin u_n
  <= 2 sqrt(p) thin u_n
  <= 16 sqrt(2) thin sqrt(p (n - 2)) thin c thin sqrt(t_"mix").
$
Substituting $c$ and bounding $sqrt(n - 2) <= sqrt(n)$,
$
bb(E)_xi^(1 slash p) lr([
  lr(|sum_(l = 2)^(n - 1) g_l(Z_(l - 1))|)^p
])
  <= 16 sqrt(2) dot 36 thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2
       thin t_"mix"^(5 slash 2) thin sqrt(p thin n)
  = 576 sqrt(2) thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2
       thin t_"mix"^(5 slash 2) thin sqrt(p thin n).
$
Rounding the resulting prefactor to $C_4 := 850$ gives the stated bound. $square$

#corollary[
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
]

_Proof._ The triangle inequality and the variance-comparison lemma give
$
| u^top chevron.l M^("RR") chevron.r_n u - n thin sigma^2(u) |
  <= | u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u) |
   + n thin | sigma_n^(2, "RR")(u) - sigma^2(u) |.
$
The first piece is bounded in $L_p$ by @eq:M-RR-conc; the second is the
deterministic bound $n thin C_3 thin || u ||^2 slash (n thin alpha thin a) = C_3 || u ||^2 slash (alpha a)$.
$square$


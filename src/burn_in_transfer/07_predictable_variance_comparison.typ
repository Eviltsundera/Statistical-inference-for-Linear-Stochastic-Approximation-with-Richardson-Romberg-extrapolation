#import "../defs.typ": *

== Burned-in Predictable Variance Comparison

For the martingale in @lem:burn-poisson-decomp define
$
cal(V)_(epsilon.alt)(z)
  := sans(Q)(hat(epsilon.alt) hat(epsilon.alt)^top)(z)
   - (sans(Q) hat(epsilon.alt))(z)
     thin (sans(Q) hat(epsilon.alt))(z)^top.
$ <eq:burn-bar-eps-def>
This is the same matrix-valued conditional covariance function as in the
stationary chapter; it is not the vector noise $epsilon.alt$.
As before, $pi(cal(V)_(epsilon.alt)) = Sigma$ and
$
|| cal(V)_(epsilon.alt) ||_infinity
  <= 18 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2.
$ <eq:burn-bar-eps-sup>
The predictable quadratic variation is
$
chevron.l M^("bRR") chevron.r_(n,n_0)
  := sum_(l = 2)^(n - 1)
      bb(E)[Delta M_l^("bRR") thin (Delta M_l^("bRR"))^top | cal(F)_(l - 1)]
  = sum_(l = 2)^(n - 1)
      Q_l^("bRR") thin cal(V)_(epsilon.alt)(Z_(l - 1)) thin (Q_l^("bRR"))^top.
$ <eq:burn-bracket>

#lemma[
  Assume *UGE 1*, $pi(epsilon.alt) = 0$, and
  $|| epsilon.alt ||_infinity < infinity$. There exists a universal constant
  $C_4 > 0$ such that, for every $u in bb(R)^d$, every $p >= 2$, every initial
  distribution $xi$, every $n >= 2$, and every $0 <= n_0 < n$,
  $
  bb(E)_xi^(1 slash p) lr([
    | u^top chevron.l M^("bRR") chevron.r_(n,n_0) u
      - m thin sigma_(n,n_0)^(2, "bRR")(u) |^p
  ])
    <= C_4 thin C_("burn,Q")^2 thin || u ||^2
       thin || epsilon.alt ||_infinity^2 thin t_"mix"^(5 slash 2)
       thin sqrt(p thin n).
  $ <eq:burn-bracket-conc>
] <lem:burn-bracket-conc>

_Proof._ Set
$
h_l(z) := u^top Q_l^("bRR") thin cal(V)_(epsilon.alt)(z)
  thin (Q_l^("bRR"))^top u,
quad
g_l(z) := h_l(z) - pi(h_l).
$
By @eq:burn-bar-eps-sup and $|| Q_l^("bRR") || <= C_("burn,Q")$,
$
|| g_l ||_infinity
  <= 36 thin C_("burn,Q")^2 thin || u ||^2
     thin t_"mix"^2 thin || epsilon.alt ||_infinity^2.
$
Moreover, @eq:burn-bracket and @eq:burn-variance-proxy imply
$
u^top chevron.l M^("bRR") chevron.r_(n,n_0) u
  - m thin sigma_(n,n_0)^(2, "bRR")(u)
  = sum_(l = 2)^(n - 1) g_l(Z_(l - 1)).
$
The scalar Markov concentration lemma used in the stationary chapter is valid
for arbitrary initial law by Levin et al. (2025, Lemma 11). Applied to the
time-inhomogeneous centered functions $g_l$, it gives the displayed
$sqrt(p n)$ bound. $square$

#corollary[
  Under the assumptions of @lem:burn-bracket-conc,
  $
  bb(E)_xi^(1 slash p) lr([
    | u^top chevron.l M^("bRR") chevron.r_(n,n_0) u
      - m thin sigma^2(u) |^p
  ])
    <= C_4 thin C_("burn,Q")^2 thin || u ||^2
       thin || epsilon.alt ||_infinity^2 thin t_"mix"^(5 slash 2)
       thin sqrt(p thin n)
       + frac(C_("burn,3") || u ||^2, alpha a).
  $ <eq:burn-bracket-asymp>
] <cor:burn-bracket-asymp>

_Proof._ Use the triangle inequality and @eq:burn-scalar-variance-comparison:
$
m thin |sigma_(n,n_0)^(2, "bRR")(u) - sigma^2(u)|
  <= frac(C_("burn,3") || u ||^2, alpha a).
$
The concentration sum still runs over the ambient indices $2, dots, n - 1$,
including pre-burn-in weights, which is why the display keeps
$sqrt(p thin n)$. In the final theorem this is converted to the effective
window scale using $m >= n slash 2$.
$square$


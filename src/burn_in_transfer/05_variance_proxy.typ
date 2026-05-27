#import "../defs.typ": *

== Burned-in Variance Proxy

For the stochastic depth-zero term, write
$Q_l^("bRR") := Q_(l; n_0, n)^("RR")$ and
$Sigma := Sigma_(epsilon.alt)^(("M"))$. We take $n >= 2$ in this subsection.
The martingale part produced by the Poisson decomposition starts at $l = 2$, so
the deterministic variance proxy is
$
Sigma_(n,n_0)^("bRR")
  := frac(1, m) sum_(l = 2)^(n - 1)
      Q_l^("bRR") thin Sigma thin (Q_l^("bRR"))^top,
quad
sigma_(n,n_0)^(2, "bRR")(u)
  := u^top Sigma_(n,n_0)^("bRR") u.
$ <eq:burn-variance-proxy>

#lemma[
  *(Burned-in variance comparison.)*
  Assume the conditions of @lem:burn-rr-weight-bounds and $|| Sigma || < infinity$.
  There exists a constant $C_("burn,3")$, depending only on
  $kappa_Q$, $|| overline(A)^(-1) ||$, $|| overline(A) ||$, and $|| Sigma ||$,
  such that
  $
  || Sigma_(n,n_0)^("bRR") - Sigma_infinity ||
    <= frac(C_("burn,3"), m thin alpha a).
  $ <eq:burn-variance-comparison>
  Consequently,
  $
  | sigma_(n,n_0)^(2, "bRR")(u) - sigma^2(u) |
    <= frac(C_("burn,3") || u ||^2, m thin alpha a).
  $ <eq:burn-scalar-variance-comparison>
] <lem:burn-variance-comparison>

_Proof._ Put
$
I_("post") := {l: 2 <= l <= n - 1, l >= n_0},
quad
I_("pre") := {l: 2 <= l < n_0},
quad
r_m := m - |I_("post")| in {0,1,2}.
$
For $l in I_("post")$, set
$Delta_l := Q_l^("bRR") - overline(A)^(-1)$. Then
$
Q_l^("bRR") Sigma (Q_l^("bRR"))^top - Sigma_infinity
  = Delta_l Sigma overline(A)^(-top)
    + overline(A)^(-1) Sigma Delta_l^top
    + Delta_l Sigma Delta_l^top.
$
Therefore
$
Sigma_(n,n_0)^("bRR") - Sigma_infinity
  &= frac(1, m) sum_(l in I_("post"))
      lr((Q_l^("bRR") Sigma (Q_l^("bRR"))^top - Sigma_infinity))
  &quad + frac(1, m) sum_(l in I_("pre"))
      Q_l^("bRR") Sigma (Q_l^("bRR"))^top
      - frac(r_m, m) Sigma_infinity.
$

The post-burn-in contribution is bounded by
$
&sum_(l in I_("post"))
  ||Q_l^("bRR") Sigma (Q_l^("bRR"))^top - Sigma_infinity||
&quad <=
  2 ||overline(A)^(-1)|| ||Sigma|| sum_(l in I_("post")) ||Delta_l||
  + ||Sigma|| sum_(l in I_("post")) ||Delta_l||^2
&quad <= frac(C, alpha a),
$
where @eq:burn-post-weight-error gives the first geometric sum and
@eq:burn-weight-energy gives the squared sum. For the pre-burn-in part,
@eq:burn-weight-energy gives
$
sum_(l in I_("pre"))
  ||Q_l^("bRR") Sigma (Q_l^("bRR"))^top||
  <= ||Sigma|| sum_(l in I_("pre")) ||Q_l^("bRR")||^2
  <= frac(C, alpha a).
$
Finally,
$
frac(r_m, m) ||Sigma_infinity||
  <= frac(2 ||overline(A)^(-1)||^2 ||Sigma||, m)
  <= frac(C, m alpha a),
$
using $alpha a <= 1$. Combining the last three displays and dividing by $m$
gives @eq:burn-variance-comparison. Also,
$
|sigma_(n,n_0)^(2, "bRR")(u) - sigma^2(u)|
  = |u^top (Sigma_(n,n_0)^("bRR") - Sigma_infinity) u|
  <= ||Sigma_(n,n_0)^("bRR") - Sigma_infinity|| ||u||^2,
$
which gives @eq:burn-scalar-variance-comparison. $square$

Assume $sigma^2(u) > 0$ and impose the burned-in variance lower-bound condition
$
m thin alpha thin a
  >= frac(2 C_("burn,3") || u ||^2, sigma^2(u)).
$ <eq:burn-variance-lb-condition>
Then @eq:burn-scalar-variance-comparison gives
$sigma_(n,n_0)^(2, "bRR")(u) >= sigma^2(u) slash 2$, hence
$sigma_(n,n_0)^("bRR")(u) >= sigma(u) slash sqrt(2)$.
// At the balanced scale $alpha = c thin n^(-1 slash 2)$ with $m >= n slash 2$,
// this condition holds for all sufficiently large $n$ depending on $u$, $c$,
// and the problem constants.

We say that $(n,n_0,p,q,alpha,u)$ is in the *admissible burn-in regime* if
$m := n - n_0$ and
$
n >= 3,
quad
0 <= n_0 < n,
quad
p >= 2,
quad
q >= 2,
quad
p <= q slash 4,
quad
sigma^2(u) > 0,
$
$
m >= n slash 2,
quad
0 < alpha,
quad
2 alpha <= alpha_("burn")(p,q),
quad
m thin alpha thin a
  >= frac(2 C_("burn,3") || u ||^2, sigma^2(u)).
$ <eq:admissible-burn-regime>
The last inequality is @eq:burn-variance-lb-condition.
// This regime is the standing finite-window domain for the burned-in
// Berry--Esseen assembly below. Every later conversion of an $m$-scale rate
// into an $n$-scale rate is made only after invoking $m >= n slash 2$.

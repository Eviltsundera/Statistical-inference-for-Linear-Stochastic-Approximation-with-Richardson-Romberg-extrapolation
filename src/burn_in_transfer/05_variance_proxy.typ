#import "../defs.typ": *

== Burned-in Variance Proxy

For the stochastic depth-zero term, write
$Q_l^("bRR") := Q_(l; n_0, n)^("RR")$ and
$Sigma := Sigma_(epsilon.alt)^(("M"))$. We take $n >= 2$ in this subsection.
The martingale part produced by the
Poisson decomposition below starts at $l = 2$, so the deterministic variance
proxy is
$
Sigma_(n,n_0)^("bRR")
  := frac(1, m) sum_(l = 2)^(n - 1)
      Q_l^("bRR") thin Sigma thin (Q_l^("bRR"))^top,
quad
sigma_(n,n_0)^(2, "bRR")(u)
  := u^top Sigma_(n,n_0)^("bRR") u.
$ <eq:burn-variance-proxy>

#lemma[
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

_Proof._ For post-burn-in indices put
$Delta_l := Q_l^("bRR") - overline(A)^(-1)$. Then
$
Q_l^("bRR") Sigma (Q_l^("bRR"))^top - Sigma_infinity
  = Delta_l Sigma overline(A)^(-top)
    + overline(A)^(-1) Sigma Delta_l^top
    + Delta_l Sigma Delta_l^top.
$
By @eq:burn-post-weight-error,
$
sum_(l >= max(2, n_0)) || Delta_l ||
  <= frac(C, alpha a),
quad
sum_(l >= max(2, n_0)) || Delta_l ||^2
  <= frac(C, alpha a).
$
For pre-burn-in indices there is no comparison with $overline(A)^(-1)$; instead
@eq:burn-weight-energy gives
$
sum_(2 <= l < n_0) || Q_l^("bRR") ||^2 <= frac(C, alpha a).
$
Replacing all post-burn-in weights by $overline(A)^(-1)$ gives at most $m$
copies of $Sigma_infinity$; the martingale index set starts at $l = 2$, so the
possible finite-index mismatch is bounded by two copies of $Sigma_infinity$.
Combining these three estimates and dividing by $m$ yields
@eq:burn-variance-comparison, after absorbing the harmless $2 ||Sigma_infinity|| / m$
term into $C_("burn,3") / (m alpha a)$ using $alpha a <= 1$. The scalar bound
follows from $|u^top H u| <= ||H|| ||u||^2$. $square$

Assume $sigma^2(u) > 0$ and impose the burned-in variance lower-bound condition
$
m thin alpha thin a
  >= frac(2 C_("burn,3") || u ||^2, sigma^2(u)).
$ <eq:burn-variance-lb-condition>
Then @eq:burn-scalar-variance-comparison gives
$sigma_(n,n_0)^(2, "bRR")(u) >= sigma^2(u) slash 2$, hence
$sigma_(n,n_0)^("bRR")(u) >= sigma(u) slash sqrt(2)$. At the balanced scale
$alpha = c thin n^(-1 slash 2)$ with $m >= n slash 2$, this condition holds
for all sufficiently large $n$ depending on $u$, $c$, and the problem
constants.


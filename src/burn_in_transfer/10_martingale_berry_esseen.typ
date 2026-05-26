#import "../defs.typ": *

== Burned-in Martingale Berry--Esseen

The depth-zero martingale in @lem:burn-poisson-decomp has deterministic
coefficients, but its normalization is based on the effective sample size
$m = n - n_0$. We assume $m >= n slash 2$ and impose the mixing-scale burn-in
lower bounds with logarithmic factors, so the inhomogeneous concentration term
obtained from $sqrt(p thin t_"mix" thin sum c_i^2)$ and then written as
$C(u) sqrt(p thin n)$ can be transferred to the $m$ scale.

Fix $u in bb(R)^d$ and put
$X_l^("bRR") := u^top Delta M_l^("bRR")$ for $2 <= l <= n - 1$. Then
$
|X_l^("bRR")| <= kappa_("burn")(u),
quad
kappa_("burn")(u)
  := 6 thin t_"mix" thin C_("burn,Q") thin || epsilon.alt ||_infinity
     thin || u ||.
$ <eq:burn-M-incr>
Indeed, this is the same bounded-increment estimate as in the stationary
chapter, with $C_("burn,Q")$ replacing the full-window weight bound. Set
$
s_(n,n_0)^2(u) := m thin sigma_(n,n_0)^(2, "bRR")(u).
$
Under @eq:burn-variance-lb-condition,
$s_(n,n_0)^2(u) >= m sigma^2(u) slash 2$. Also,
$m >= n slash 2$ and the uniform bound $||Q_l^("bRR")|| <= C_("burn,Q")$
imply the deterministic upper bound
$
s_(n,n_0)^2(u)
  <= 2 m thin C_("burn,Q")^2 thin
     || Sigma_(epsilon.alt)^(("M")) || thin || u ||^2.
$ <eq:burn-s-upper>

#theorem[
  *(Burned-in martingale Berry--Esseen bound.)*
  Assume *UGE 1*, $pi(epsilon.alt) = 0$,
  $|| epsilon.alt ||_infinity < infinity$, $sigma^2(u) > 0$,
  $0 < alpha$, $2 alpha <= alpha_infinity$, $m >= n slash 2$, and the
  burned-in variance lower-bound condition @eq:burn-variance-lb-condition.
  There exist constants $C_("bK,1")(u), C_("bK,2")(u) > 0$, depending only on
  $||u||$, $sigma(u)$, $C_("burn,Q")$, $t_"mix"$,
  $||epsilon.alt||_infinity$, $||Sigma_(epsilon.alt)^(("M"))||$, and the
  universal constants in @eq:imported-bolthausen-fan, such that for every
  $n >= 3$,
  $
  d_K lr((
    frac(u^top M_(n,n_0)^("bRR"),
         sqrt(m) thin sigma_(n,n_0)^("bRR")(u)),
    cal(N)(0, 1)
  ))
    <= frac(C_("bK,1")(u) thin log^(3 slash 4) n, m^(1 slash 4))
     + frac(C_("bK,2")(u) thin log n, sqrt(m)).
  $ <eq:burn-M-BE>
] <thm:burn-M-BE>

_Proof._ Apply @eq:imported-bolthausen-fan to the martingale differences
$X_l^("bRR")$. The bounded-increment input is
@eq:burn-M-incr, and the target variance is $s_(n,n_0)^2(u)$. The first and
third Bolthausen--Fan terms are bounded exactly as before, with $n$ replaced
by $m$ in the denominator and the harmless factor $n slash m <= 2$ absorbed
into the constants. Explicitly, from
$s_(n,n_0)^2(u) >= m sigma^2(u) slash 2$ and $m >= n slash 2$,
$
frac((2 n + 1) log(2 n + 1), s_(n,n_0)^3(u))
  <= C(u) frac(log n, sqrt(m)).
$ <eq:burn-BF-first-nm>
The bounded-increment Lindeberg term is controlled in the same way, using
@eq:burn-s-upper and $s_(n,n_0)^2(u) >= m sigma^2(u) slash 2$. For the
conditional-variance term use
@lem:burn-bracket-conc:
$
bb(E)^(1 slash p) lr([
  | u^top chevron.l M^("bRR") chevron.r_(n,n_0) u
    - s_(n,n_0)^2(u) |^p
])
  <= C thin sqrt(p thin n).
$
Here the constant contains the explicit bracket coefficient
$C_("burn,Q")^2 ||u||^2 ||epsilon.alt||_infinity^2 t_"mix"^(5 slash 2)$
from @lem:burn-bracket-conc; the underlying concentration input is
@eq:imported-markov-conc.
Together with $m >= n slash 2$, @eq:burn-variance-lb-condition, and
@eq:burn-s-upper, this gives
$
"Term II"
  <= C(u) thin p^((3 p + 1) slash (2 (2 p + 1)))
     thin m^(-p slash (2 (2 p + 1))).
$
Taking $p = ceil(log n)$ gives the displayed
$log^(3 slash 4)(n) m^(-1 slash 4)$ bound. The classical Bolthausen and
Lindeberg terms give the second displayed term. $square$

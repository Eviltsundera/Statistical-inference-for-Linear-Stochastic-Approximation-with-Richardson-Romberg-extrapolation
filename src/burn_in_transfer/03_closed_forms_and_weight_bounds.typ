#import "../defs.typ": *

== Closed Forms and Burned-in Weight Bounds

Write $B_w := I - w overline(A)$ for $w in {alpha, 2 alpha}$ and recall
$m = n - n_0$. The burned-in weights have two closed forms. If
$l >= n_0$, then
$
Q_(l; n_0, n)^((w))
  = w sum_(k = l)^(n - 1) B_w^(k - l)
  = overline(A)^(-1) lr((I - B_w^(n - l))).
$ <eq:burn-Q-post-form>
If $l < n_0$, then
$
Q_(l; n_0, n)^((w))
  = w sum_(k = n_0)^(n - 1) B_w^(k - l)
  = overline(A)^(-1) lr((B_w^(n_0 - l) - B_w^(n - l))).
$ <eq:burn-Q-pre-form>
Thus only the post-burn-in weights are close to the asymptotic kernel
$overline(A)^(-1)$. The pre-burn-in weights are instead exponentially small as
one moves backward from $n_0$. Empty sums below are interpreted as zero.

#lemma[
  Assume $alpha, 2 alpha in (0, alpha_infinity]$ and the Lyapunov contraction
  @eq:contraction. Let
  $
  rho_alpha := sqrt(1 - alpha a),
  quad
  C_Q := kappa_Q^(1 slash 2) || overline(A)^(-1) ||,
  quad
  tilde(C)_A := kappa_Q || overline(A) ||.
  $

  *(i) Post-burn-in comparison.* If $l >= n_0$ and $k := n - l$, then
  $
  || Q_(l; n_0, n)^("RR") - overline(A)^(-1) ||
    <= 3 C_Q rho_alpha^k.
  $ <eq:burn-post-weight-error>

  *(ii) Pre-burn-in energy.* If $1 <= l < n_0$ and $r := n_0 - l$, then
  $
  || Q_(l; n_0, n)^("RR") ||
    <= 6 C_Q rho_alpha^r.
  $ <eq:burn-pre-weight-size>

  Consequently, there are constants $C_("burn,E")$ and $C_("burn,V")$,
  depending only on $kappa_Q$, $|| overline(A)^(-1) ||$, and
  $|| overline(A) ||$, such that, uniformly in $n$ and $n_0$,
  $
  sum_(l = 1)^(n_0 - 1) || Q_(l; n_0, n)^("RR") ||^2
  + sum_(l = max(1, n_0))^(n - 1)
      || Q_(l; n_0, n)^("RR") - overline(A)^(-1) ||^2
    <= frac(C_("burn,E"), alpha a),
  $ <eq:burn-weight-energy>
  and
  $
  sum_(l = 1)^(n - 2)
    || Q_(l + 1; n_0, n)^("RR") - Q_(l; n_0, n)^("RR") ||
    <= frac(C_("burn,V"), a^2).
  $ <eq:burn-weight-variation>
] <lem:burn-rr-weight-bounds>

_Proof._ For $l >= n_0$, @eq:burn-Q-post-form gives
$
Q_(l; n_0, n)^("RR") - overline(A)^(-1)
  = -overline(A)^(-1) lr((2 B_alpha^k - B_(2 alpha)^k)).
$
The same contraction argument as in the full-window estimate gives
@eq:burn-post-weight-error.

For $l < n_0$, set $r := n_0 - l$. By @eq:burn-Q-pre-form,
$
Q_(l; n_0, n)^("RR")
  = overline(A)^(-1) lr((
      2 lr((B_alpha^r - B_alpha^(r + m)))
      - lr((B_(2 alpha)^r - B_(2 alpha)^(r + m)))
    )).
$
Taking norms and using $|| B_(2 alpha)^j ||_Q <= rho_alpha^j$ gives
@eq:burn-pre-weight-size. Squaring @eq:burn-post-weight-error and
@eq:burn-pre-weight-size and summing the resulting geometric series gives
@eq:burn-weight-energy.

It remains to bound the total variation. In the post-burn-in region
$l, l + 1 >= n_0$, the full-window identity gives
$
Q_(l + 1; n_0, n)^("RR") - Q_(l; n_0, n)^("RR")
  = -2 alpha lr((B_alpha^s - B_(2 alpha)^s)),
quad s := n - l - 1.
$
The elementary identity
$X^s - Y^s = (X - Y) sum_(i = 1)^s X^(i - 1) Y^(s - i)$ with
$X = B_alpha$ and $Y = B_(2 alpha)$ yields, for $s >= 1$,
$
|| B_alpha^s - B_(2 alpha)^s ||
  <= alpha tilde(C)_A thin s thin rho_alpha^(s - 1).
$ <eq:burn-power-difference>
The left-factored form is legitimate because $B_alpha$ and $B_(2 alpha)$ are
polynomials in the same matrix $overline(A)$, hence commute with each other
and with $overline(A)$.
Thus the post-burn-in contribution is bounded by a constant times
$a^(-2)$, as in the full-window proof.

For $l, l + 1 < n_0$, put $s := n_0 - l - 1$. The pre-burn-in closed form gives
$
Q_(l + 1; n_0, n)^("RR") - Q_(l; n_0, n)^("RR")
  = 2 alpha lr((
      B_alpha^s - B_(2 alpha)^s
      - B_alpha^(s + m) + B_(2 alpha)^(s + m)
    )).
$
Here $s >= 1$. Applying @eq:burn-power-difference to the powers $s$ and
$s + m$ and summing over $s >= 1$ again gives a constant times $a^(-2)$. If
the boundary
$l = n_0 - 1$ exists, then
$
Q_(n_0; n_0, n)^("RR") - Q_(n_0 - 1; n_0, n)^("RR")
  = 2 alpha lr((B_(2 alpha)^m - B_alpha^m)),
$
and @eq:burn-power-difference with $s = m$ bounds this term by the same
$a^(-2)$ scale. Combining the pre-burn-in, boundary, and post-burn-in
contributions proves @eq:burn-weight-variation. $square$


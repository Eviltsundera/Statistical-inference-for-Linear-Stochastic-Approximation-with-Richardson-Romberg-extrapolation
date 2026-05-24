#import "../defs.typ": *

== Closed-Form Identities

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

The geometric-series identity $sum_(j = 0)^(m - 1) B_alpha^j = (alpha overline(A))^(-1)(I - B_alpha^m)$ converts @eq:Q-definition into the closed form
$
Q_l^((alpha))
  = alpha sum_(j = 0)^(n - l - 1) B_alpha^j
  = alpha thin (alpha overline(A))^(-1) thin (I - B_alpha^(n - l))
  = overline(A)^(-1) (I - B_alpha^(n - l)).
$
This makes the convergence $Q_l^((alpha)) -> overline(A)^(-1)$ as $k = n - l -> infinity$ explicit: the only obstruction is the geometric Lyapunov tail $B_alpha^k$. Two immediate consequences are
$
Q_l^((alpha)) - overline(A)^(-1) = - overline(A)^(-1) B_alpha^(n - l),
quad
Q_(l + 1)^((alpha)) - Q_l^((alpha)) = - alpha B_alpha^(n - l - 1).
$
The first is the *asymptotic-weight error* and decays geometrically in $k$. The second is the *discrete derivative* used in Abel summation; note the explicit factor $alpha$ in front.

Applying both identities at $alpha$ and $2 alpha$ and combining yields the basic *RR identities*:
$
cal(Q)_l^("RR") - overline(A)^(-1)
  = - overline(A)^(-1) thin (2 B_alpha^k - B_(2 alpha)^k),
$
$
cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")
  = - 2 alpha thin (B_alpha^(k - 1) - B_(2 alpha)^(k - 1)).
$
These are the exact expressions bounded in the next subsection. Note the structural asymmetry between them:

- The asymptotic-weight error $2 B_alpha^k - B_(2 alpha)^k$ is at the *single-trajectory* rate $(1 - alpha a)^(k slash 2)$ — the triangle inequality $|2 X - Y| <= 2 |X| + |Y|$ does not see the RR coupling, and indeed the leading $-overline(A)^(-1)$ in $cal(Q)_l^("RR")$ is the same as in the single-step weight $Q_l^((alpha))$.
- The discrete derivative $B_alpha^(k - 1) - B_(2 alpha)^(k - 1)$, however, is a *true* difference of contractions evaluated at the same exponent. The elementary identity $X^m - Y^m = (X - Y) sum_(i = 1)^m X^(i - 1) Y^(m - i)$ with $X = B_alpha$, $Y = B_(2 alpha)$, $X - Y = alpha overline(A)$ extracts an extra factor of $alpha$, gaining one full power of $alpha$ over the single-step case $Q_(l + 1)^((alpha)) - Q_l^((alpha)) = -alpha B_alpha^(k - 1)$. The left-factored form is valid because $B_alpha$ and $B_(2 alpha)$ are polynomials in the same matrix $overline(A)$, hence commute with each other and with $overline(A)$.

This local $alpha$-gain in the discrete derivative is the *only* manifestation of Richardson--Romberg cancellation visible at the level of the PR weights.


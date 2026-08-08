#import "../defs.typ": *

== Terminal RR Combination and the $Delta J_n^((0, alpha))$ Term

Applying the deterministic-product decomposition of the previous subsection
separately to step sizes $alpha$ and $2 alpha$, and writing
$B_(2 alpha) := I - 2 alpha overline(A)$, define the chapter-local
terminal Richardson--Romberg object
$
theta_n^(("tRR", alpha)) := 2 theta_n^((alpha)) - theta_n^((2 alpha)).
$
It is not the PR-averaged RR estimator $overline(theta)_n^(("RR", alpha))$
studied in Chapters 4--5. Its deterministic-product decomposition has the form
$
theta_n^(("tRR", alpha)) - theta^*
  = [2 B_alpha^n - B_(2 alpha)^n](theta_0 - theta^*) + [2 J_n^((0, alpha)) - J_n^((0, 2 alpha))] + [2 R_n^((alpha)) - R_n^((2 alpha))].
$
The first bracket is the deterministic transient in this convention. The
second bracket is the linearized stochastic RR difference, and the final one is
the higher-order random-product remainder. In this subsection we focus on the
zeroth-order RR difference, which we denote
$
Delta J_n^((0, alpha)) := 2 J_n^((0, alpha)) - J_n^((0, 2 alpha)),
$
where, by definition,
$
J_n^((0, alpha)) = -alpha sum_(j=1)^n (I - alpha overline(A))^(n-j) epsilon.alt(Z_j).
$

Substituting and using the elementary identity
$X^m - Y^m = (X - Y) sum_(i=1)^m X^(i-1) Y^(m-i)$ with
$X = B_alpha$, $Y = B_(2 alpha)$, and $X - Y = alpha overline(A)$, define
the deterministic kernel
$
H_j^((n))
  := sum_(i=1)^(n-j)
      B_alpha^(i-1) B_(2 alpha)^(n-j-i),
quad 1 <= j <= n - 1,
$
and set $H_n^((n)) := 0$. Then
$
Delta J_n^((0, alpha))
  = -2 alpha sum_(j=1)^n (B_alpha^(n-j) - B_(2 alpha)^(n-j)) epsilon.alt(Z_j)
  = -2 alpha^2 overline(A) sum_(j=1)^(n-1) H_j^((n)) epsilon.alt(Z_j),
$
The left-factored matrix-power identity is legitimate here because
$I - alpha overline(A)$, $I - 2 alpha overline(A)$, and $overline(A)$ are
polynomials in the same matrix $overline(A)$ and therefore commute.
The extra factor $alpha overline(A)$ pulled out front is the source of the additional $alpha$-decay of the RR difference compared to a single LSA trajectory.

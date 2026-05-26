#import "../defs.typ": *

== LSA Error Decomposition

This chapter is a preliminary deterministic-product calculation for the
zero-order RR difference. The final Berry--Esseen assembly uses the full
PR-weight notation and hypotheses introduced in Chapter 4; the present chapter
mainly fixes notation and explains the cancellation mechanism in a simpler
last-iterate setting.

We consider the recursion
$ theta_k = theta_(k-1) - alpha_k (A(Z_k) theta_(k-1) - b(Z_k)), quad alpha_k = alpha = "const". $

Define the transition products
$ Gamma_(m:k) = product_(l=m)^k (I - alpha A(Z_l)). $

Write $B_alpha := I - alpha overline(A)$ and introduce the deterministic-product
linearized term
$
J_k^((0, alpha)) = -alpha sum_(l=1)^k B_alpha^(k-l) thin epsilon.alt(Z_l),
quad J_0^((0, alpha)) = 0.
$
The difference between the exact random-product expansion and this
deterministic-product term is denoted by $R_k^((alpha))$:
$
theta_k^((alpha)) - theta^*
  = J_k^((0, alpha)) + B_alpha^k (theta_0 - theta^*) + R_k^((alpha)).
$
This convention matches the weight decomposition in Section 4.1.

A standard PR-averaged decomposition (cf. Chapter 4 for the full derivation) yields
$ sqrt(n) (overline(theta)_n^((alpha)) - theta^*) = W + D_1, $
where the leading martingale-like term is
$ W = -frac(1, sqrt(n)) sum_(l=1)^(n-1) Q_l thin epsilon.alt(Z_l),
quad
Q_l = alpha sum_(k=l)^(n-1) B_alpha^(k-l), $
and the residual term is
$ D_1 = frac(1, sqrt(n)) sum_(k=0)^(n-1) B_alpha^k (theta_0 - theta^*) + frac(1, sqrt(n)) sum_(k=1)^(n-1) R_k^((alpha)). $

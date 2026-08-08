#import "../defs.typ": *

== Pointwise Bounds for the RR Weights

#lemma[
  *(Pointwise RR weight bounds.)*
  Let $0 < alpha$ and $2 alpha <= alpha_infinity$, set $C_Q := kappa_Q^(1 slash 2) || overline(A)^(-1) ||$ and $tilde(C)_A := kappa_Q || overline(A) ||$, and write $k = n - l$.

  *(i)* For every $1 <= l <= n - 1$,
  $
  || cal(Q)_l^("RR") - overline(A)^(-1) || <= 3 C_Q (1 - alpha a)^(k slash 2).
  $

  *(ii)* For every $1 <= l <= n - 2$,
  $
  || cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR") ||
    <= 2 tilde(C)_A thin alpha^2 thin (k - 1) thin (1 - alpha a)^((k - 2) slash 2).
  $
] <lem:RR-weight-pointwise>

_Proof of (i)._ From the RR identity, norm equivalence, and Lyapunov
contraction,
$
|| 2 B_alpha^k - B_(2 alpha)^k ||_Q
  <= 2 || B_alpha^k ||_Q + || B_(2 alpha)^k ||_Q
  <= 2 (1 - alpha a)^(k slash 2) + (1 - 2 alpha a)^(k slash 2)
  <= 3 (1 - alpha a)^(k slash 2).
$
Thus
$
|| cal(Q)_l^("RR") - overline(A)^(-1) ||
  <= kappa_Q^(1 slash 2) || overline(A)^(-1) || dot || 2 B_alpha^k - B_(2 alpha)^k ||_Q
  <= 3 C_Q (1 - alpha a)^(k slash 2).
$

_Proof of (ii)._ Use
$X^m - Y^m = (X - Y) sum_(i = 1)^m X^(i - 1) Y^(m - i)$
with $X = B_alpha$, $Y = B_(2 alpha)$, and $m = k - 1$:
// The matrices commute because they are polynomials in $overline(A)$, so the
// factor $alpha overline(A)$ may be placed on the left.
$
B_alpha^(k - 1) - B_(2 alpha)^(k - 1)
  = alpha overline(A) sum_(i = 1)^(k - 1)
    B_alpha^(i - 1) thin B_(2 alpha)^(k - 1 - i).
$
Each summand satisfies
$
|| B_alpha^(i - 1) thin B_(2 alpha)^(k - 1 - i) ||
  &<= kappa_Q^(1 slash 2) thin (1 - alpha a)^((i - 1) slash 2) (1 - 2 alpha a)^((k - 1 - i) slash 2) 
  &<= kappa_Q^(1 slash 2) thin (1 - alpha a)^((k - 2) slash 2),
$
Summing over $i$ gives
// In the last displayed line we used $1 - 2 alpha a <= 1 - alpha a$.
$
|| B_alpha^(k - 1) - B_(2 alpha)^(k - 1) ||
  <= alpha kappa_Q || overline(A) || (k - 1) (1 - alpha a)^((k - 2) slash 2).
$
By the discrete-difference identity from the previous section,
$
cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")
  = - 2 alpha thin (B_alpha^(k - 1) - B_(2 alpha)^(k - 1)).
$

$
|| cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR") ||
  &<= 2 alpha thin || B_alpha^(k - 1) - B_(2 alpha)^(k - 1) || 
  // &<= 2 kappa_Q || overline(A) || thin alpha^2 thin (k - 1)
  //     thin (1 - alpha a)^((k - 2) slash 2) 
  &<= 2 tilde(C)_A thin alpha^2 thin (k - 1)
      thin (1 - alpha a)^((k - 2) slash 2),
$
which is the claimed bound. $square$

#import "../defs.typ": *

== Summed Bounds and Comparison with the Single-Step Case

#corollary[
  *(Summed RR weight bounds.)*
  Under the assumptions of the previous lemma, uniformly in $n >= 2$,
  $
  sum_(l = 1)^(n - 1) || cal(Q)_l^("RR") - overline(A)^(-1) ||^2
    <= frac(C_1, alpha a),
  quad
  sum_(l = 1)^(n - 2) || cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR") ||
    <= frac(C_2, a^2),
  $
  with $C_1 = 9 C_Q^2$ and $C_2 = 32 tilde(C)_A$.
] <cor:RR-weight-summed>

// _Proof._ Apply the pointwise bounds and the geometric sums
// $sum_(k >= 1) (1 - alpha a)^k <= 1 / (alpha a)$ and
// $
// sum_(k >= 2) (k - 1) thin (1 - alpha a)^((k - 2) slash 2)
//   = sum_(m >= 0) (m + 1) (1 - alpha a)^(m slash 2)
//   <= frac(1, (1 - sqrt(1 - alpha a))^2)
//   <= frac(4, (alpha a)^2),
// $
// Multiplying by $2 tilde(C)_A alpha^2$ gives the claim.
// The last displayed step uses
// $1 - sqrt(1 - alpha a) >= alpha a / 2$ for $alpha a <= 1 slash 2$.
// $square$
#pagebreak()

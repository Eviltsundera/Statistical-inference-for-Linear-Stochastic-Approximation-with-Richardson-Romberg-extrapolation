#import "../defs.typ": *

== Variance Comparison

Use the deterministic variance proxy
$
Sigma_n^("RR") := frac(1, n) sum_(l = 2)^(n - 1)
  cal(Q)_l^("RR") thin Sigma_(epsilon.alt)^(("M")) thin (cal(Q)_l^("RR"))^top,
quad
sigma_n^(2, "RR")(u) := u^top Sigma_n^("RR") u,
$
where
$
Sigma_(epsilon.alt)^(("M"))
  = bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_0)^top]
    + sum_(j >= 1) lr((
        bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_j)^top]
        + bb(E)_pi [epsilon.alt(Z_j) epsilon.alt(Z_0)^top]
      )).
$
The asymptotic covariance is
$
Sigma_infinity = overline(A)^(-1) Sigma_(epsilon.alt)^(("M")) overline(A)^(-top),
quad
sigma^2(u) = u^top Sigma_infinity u.
$
// After the Poisson decomposition the $l = 1$ noise sample is absorbed into the
// Poisson boundary remainder, while the martingale increments are indexed by
// $l in {2, dots, n - 1}$. The deterministic variance proxy uses the same index
// set.

#lemma[
  *(Stationary finite-window variance comparison.)*
  Let $0 < alpha$ and $2 alpha <= alpha_infinity$, \
  set $Sigma := Sigma_(epsilon.alt)^(("M"))$,
  and assume $|| Sigma || < infinity$. Then
  $
  || Sigma_n^("RR") - Sigma_infinity ||
    <= frac(C_3, n thin alpha a),
  $
  with
  $C_3 = 12 thin C_Q thin || overline(A)^(-1) || thin || Sigma ||
    + 9 thin C_Q^2 thin || Sigma ||
    + 2 thin || Sigma_infinity ||$.
  Consequently, for every $u in bb(R)^d$,
  $
  | sigma_n^(2, "RR")(u) - sigma^2(u) |
    <= frac(C_3 thin || u ||^2, n thin alpha a).
  $
  // At the working scale $alpha = c thin n^(- 1 slash 2)$ this is $O(n^(- 1 slash 2))$.
] <lem:RR-variance-comparison>

_Proof._ Write $Delta_l := cal(Q)_l^("RR") - overline(A)^(-1)$ and expand
$
cal(Q)_l^("RR") thin Sigma thin (cal(Q)_l^("RR"))^top - Sigma_infinity
  = R_(1,l) + R_(2,l),
$
where
$
R_(1,l)
  := Delta_l Sigma overline(A)^(-top)
     + overline(A)^(-1) Sigma Delta_l^top,
quad
R_(2,l) := Delta_l Sigma Delta_l^top.
$
The pointwise bounds are
$
|| R_(1, l) || <= 2 thin || overline(A)^(-1) || thin || Sigma || thin || Delta_l ||,
quad
|| R_(2, l) || <= || Sigma || thin || Delta_l ||^2.
$
The linear part satisfies
// Use part (i) of the previous lemma and
// $sum_(k >= 1) (1 - alpha a)^(k slash 2)
//   <= 1 / (1 - sqrt(1 - alpha a)) <= 2 / (alpha a)$.
$
sum_(l = 1)^(n - 1) || Delta_l ||
  <= 3 C_Q sum_(k = 1)^(n - 1) (1 - alpha a)^(k slash 2)
  <= frac(6 C_Q, alpha a).
$
The quadratic part satisfies
$
sum_(l = 1)^(n - 1) || Delta_l ||^2 <= frac(9 C_Q^2, alpha a).
$
Combining, including the missing two boundary terms in the $l = 2$ to $n - 1$
sum,
// Since the sum starts at $l = 2$, replacing every weight by
// $overline(A)^(-1)$ gives $((n - 2) slash n) Sigma_infinity$, not
// $Sigma_infinity$. Hence there is an additional deterministic finite-sum
// boundary term $2 Sigma_infinity slash n$.
$
|| Sigma_n^("RR") - Sigma_infinity ||
  &<= frac(2 || Sigma_infinity ||, n)
      + frac(1, n) sum_(l = 2)^(n - 1) (|| R_(1, l) || + || R_(2, l) ||) \
  &<= frac(1, n) (2 thin || overline(A)^(-1) || thin || Sigma || dot frac(6 C_Q, alpha a)
                + || Sigma || dot frac(9 C_Q^2, alpha a)
                + 2 thin || Sigma_infinity || slash (alpha a))
  = frac(C_3, n thin alpha a)
$
$square$

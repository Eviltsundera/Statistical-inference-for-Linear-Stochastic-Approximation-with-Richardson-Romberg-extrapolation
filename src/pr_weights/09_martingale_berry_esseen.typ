#import "../defs.typ": *

== Martingale Berry--Esseen Step

We apply the martingale Berry--Esseen theorem to the scalar martingale
$u^top M_n^("RR")$.
// The Poisson decomposition writes
// $W^("RR") = -n^(-1 slash 2) M_n^("RR") + D_(2, n)^("RR")$, with
// $D_(2, n)^("RR") = O(n^(-1 slash 2))$. The bracket concentration gives
// $|u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u)|$
// in $L_p$ at scale $B(u) sqrt(p thin n)$.

*Bounded increments.* For $X_l := u^top Delta M_l^("RR")$,
$
|X_l| <= kappa.alt(u),
quad
kappa.alt(u) := 6 thin t_"mix" thin C_(cal(Q)) thin || epsilon.alt ||_infinity thin || u ||.
$ <eq:M-RR-incr>
The conditional variances sum to
$
V_n^2 := sum_(l = 2)^(n - 1) sigma_l^2 = u^top chevron.l M^("RR") chevron.r_n u.
$

*Variance lower bound.* Set $s_n^2 := n thin sigma_n^(2, "RR")(u)$. If
$
n thin alpha thin a >= frac(2 thin C_3 thin || u ||^2, sigma^2(u))
$ <eq:variance-lb-condition>
then $s_n^2 >= n sigma^2(u) slash 2$.
// At $alpha = c thin n^(-1 slash 2)$, @eq:variance-lb-condition is satisfied
// for $n >= (2 C_3 || u ||^2 slash (c thin a thin sigma^2(u)))^2$.
The trivial upper bound
$sigma_n^(2, "RR")(u) <= C_(cal(Q))^2 || Sigma_(epsilon.alt)^(("M")) || || u ||^2$
gives $s_n^2 <= K^2(u) thin n$ with
$K(u) := C_(cal(Q)) thin || Sigma_(epsilon.alt)^(("M")) ||^(1 slash 2) thin || u ||$.

We apply @lem:external-martingale-be to this scalar martingale; the number of
increments is at most $n$, and the increment bound is $kappa.alt(u)$.

#theorem[
  *(Stationary martingale Berry--Esseen bound.)*
  Assume *UGE 1*, $pi(epsilon.alt) = 0$, $|| epsilon.alt ||_infinity < infinity$,
  $sigma^2(u) > 0$, $0 < alpha$, and $2 alpha <= alpha_infinity$. There exist
  constants $C_(K, 1)(u), C_(K, 2)(u) > 0$ depending only on $|| u ||$,
  $sigma(u)$, $C_(cal(Q))$, $t_"mix"$, $|| epsilon.alt ||_infinity$,
  $|| Sigma_(epsilon.alt)^(("M")) ||$, and the constants
  $L_B(kappa.alt(u)), C_1, C_2$ of @lem:external-martingale-be, such that for every
  $n >= 3$ satisfying
  the variance lower-bound condition @eq:variance-lb-condition,
  $
  d_K lr((
    frac(u^top M_n^("RR"), sqrt(n) thin sigma_n^("RR")(u)),
    cal(N)(0, 1)
  ))
    <= frac(C_(K, 1)(u) thin log^(3 slash 4) n, n^(1 slash 4))
     + frac(C_(K, 2)(u) thin log n, sqrt(n)).
  $ <eq:M-RR-BE>
] <thm:M-RR-BE>

_Proof._ Apply @lem:external-martingale-be to
$(X_l)_(l = 2)^(n - 1)$ with partial sum $u^top M_n^("RR")$,
deterministic scale $s_n^2 = n thin sigma_n^(2, "RR")(u)$, increment bound
$kappa.alt(u)$, and $p = ceil(log n)$. This martingale array has at most $n$
increments, so the first term in @lem:external-martingale-be is bounded using
$(2 n + 1) log(2 n + 1)$.
// Use $s_n^2 in [n thin sigma^2(u) slash 2, thin K^2(u) thin n]$ from the
// variance lower and upper bounds.

*Term I.*
$
L_B(kappa.alt(u)) thin frac((2 n + 1) log(2 n + 1), s_n^3)
  <= frac(6 sqrt(2) thin L_B(kappa.alt(u)), sigma^3(u)) thin frac(log n, sqrt(n))
  =: frac(C^("(I)")(u) thin log n, sqrt(n)).
$ <eq:term-I>

*Term III.* With $a_p := 2 p slash (2 p + 1)$,
$
s_n^(1 slash (2 p + 1)) <= max(1, K(u))^(1 slash (2 p + 1)) thin n^(1 slash (2 (2 p + 1))).
$
For $p >= log n$,
$
C_2 thin s_n^(- a_p) thin p thin kappa.alt(u)^(a_p)
  <= sqrt(2) thin e^(3 slash 4) thin frac(C_2 thin max(1, kappa.alt(u)), sigma(u)) thin frac(p, sqrt(n))
  <= frac(C^("(III)")(u) thin log n, sqrt(n)),
$ <eq:term-III>
// The constant absorbs the bounded powers of $K(u)$ and $kappa.alt(u)$.

*Term II.* By bracket concentration,
$
(bb(E) | V_n^2 - s_n^2 |^p)^(1 slash p)
  <= B(u) thin sqrt(p thin n),
quad
B(u) := C_4 thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2 thin t_"mix"^(5 slash 2),
$ <eq:Bu-def>
The conditional-variance term is bounded by
$
&C_1 thin sqrt(p) thin s_n^(- a_p) thin (bb(E) | V_n^2 - s_n^2 |^p)^(1 slash (2 p + 1)) \
&quad<= C thin C_1 thin sigma^(-1)(u)
        thin max(1, K(u))^(1 slash (2 p + 1))
        thin max(1, B(u))^(p slash (2 p + 1)) \
&quad quad times p^((3 p + 1) slash (2 (2 p + 1)))
        n^(- p slash (2 (2 p + 1))).
$ <eq:term-II-1>

For $p = ceil(log n)$,
$p^((3 p + 1) slash (2 (2 p + 1))) <= C log^(3 slash 4) n$ and
$n^(- p slash (2 (2 p + 1))) <= C n^(-1 slash 4)$. Thus
// The $K(u)$- and $B(u)$-factors are absorbed into the direction-dependent
// constant.
$
"Term II"
  <= frac(C^("(II)")(u) thin log^(3 slash 4) n, n^(1 slash 4)).
$ <eq:term-II>

Adding the three bounds and setting
$C_(K, 1)(u) := C^("(II)")(u)$, $C_(K, 2)(u) := C^("(I)")(u) + C^("(III)")(u)$
proves the martingale Berry--Esseen bound. $square$

#corollary[
  *(Stationary martingale asymptotic-normalization bound.)*
  Under the hypotheses of the previous theorem,
  $
  d_K lr((
    frac(u^top M_n^("RR"), sqrt(n) thin sigma(u)),
    cal(N)(0, 1)
  ))
    <= frac(C_(K, 1)(u) thin log^(3 slash 4) n, n^(1 slash 4))
     + frac(C_(K, 2)(u) thin log n, sqrt(n))
     + frac(C_3 thin || u ||^2, n thin alpha thin a thin sigma^2(u)).
  $ <eq:M-RR-BE-sigma>
  // At $alpha = c thin n^(-1 slash 2)$ the last term is $O(n^(-1 slash 2))$, hence
  // absorbed into $C_(K, 2)(u) thin log n slash sqrt(n)$ up to a constant.
] <cor:M-RR-BE-sigma>

_Proof._ Set $r := sigma_n^("RR")(u) slash sigma(u)$ and
$W := u^top M_n^("RR") slash (sqrt(n) sigma_n^("RR")(u))$. Then the statistic
normalised by $sigma(u)$ is $W r$. Under the variance lower-bound condition @eq:variance-lb-condition,
$r >= 1 slash sqrt(2)$, while the trivial upper bound on
$sigma_n^(2, "RR")(u)$ gives $r <= r_max(u) < infinity$. On this compact
interval the standard normal cdf satisfies
$
sup_x |Phi(x slash r) - Phi(x)| <= C_Phi thin |r - 1|,
quad
C_Phi := sqrt(2) slash sqrt(pi e).
$
Therefore
$
d_K lr((W r, cal(N)(0, 1)))
  <= d_K lr((W, cal(N)(0, 1))) + C_Phi thin |r - 1|.
$
The variance comparison of Section 4.5 gives
$
|r - 1|
  = frac(|sigma_n^(2, "RR")(u) - sigma^2(u)|,
         sigma(u) thin (sigma_n^("RR")(u) + sigma(u)))
  <= frac(C_3 thin || u ||^2,
          n thin alpha thin a thin sigma^2(u)).
$
Adding this perturbation to the martingale Berry--Esseen bound proves the
stated claim after absorbing $C_Phi$ into the constants. $square$

// This theorem concerns only the martingale term; the stationary assembly below
// adds the Poisson remainder and the Levin depth-two misadjustment.

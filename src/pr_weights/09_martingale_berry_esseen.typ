#import "../defs.typ": *

== Martingale Berry--Esseen Step

The Poisson decomposition (Section 4.6) writes
$W^("RR") = -n^(-1 slash 2) M_n^("RR") + D_(2, n)^("RR")$, with the remainder
$D_(2, n)^("RR")$ controlled in sup-norm at the order $n^(-1 slash 2)$. The
quadratic variation concentration (Section 4.7) bounds
$|u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u)|$ in
$L_p$ by $C thin sqrt(p thin n)$. This subsection assembles those two inputs
into a Berry--Esseen rate for the *scalar* martingale $u^top M_n^("RR")$,
the leading contribution to the final RR Berry--Esseen bound.

*Bounded increments.* Fix $u in bb(R)^d$ and set
$X_l := u^top Delta M_l^("RR")$ for $2 <= l <= n - 1$. The $X_l$ are
$cal(F)_l$-martingale differences (Section 4.6), and bounding the three
factors $|| u ||$, $|| cal(Q)_l^("RR") || <= C_(cal(Q))$, and
$|| hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)) || <= 2 || hat(epsilon.alt) ||_infinity <= 6 t_"mix" || epsilon.alt ||_infinity$
yields the deterministic bound
$
|X_l| <= kappa.alt(u),
quad
kappa.alt(u) := 6 thin t_"mix" thin C_(cal(Q)) thin || epsilon.alt ||_infinity thin || u ||.
$ <eq:M-RR-incr>
The conditional variances $sigma_l^2 := bb(E)[X_l^2 | cal(F)_(l - 1)]$ sum to
the predictable quadratic variation along $u$,
$
V_n^2 := sum_(l = 2)^(n - 1) sigma_l^2 = u^top chevron.l M^("RR") chevron.r_n u.
$

*Variance lower bound.* Set $s_n^2 := n thin sigma_n^(2, "RR")(u)$. The
variance comparison lemma (Section 4.5) and assumption $sigma^2(u) > 0$ give
$|sigma_n^(2, "RR")(u) - sigma^2(u)| <= C_3 || u ||^2 slash (n alpha a)$, so
whenever
$
n thin alpha thin a >= frac(2 thin C_3 thin || u ||^2, sigma^2(u))
$ <eq:variance-lb-condition>
one has $sigma_n^(2, "RR")(u) >= sigma^2(u) slash 2$, i.e.
$s_n^2 >= n sigma^2(u) slash 2$. At the working scale
$alpha = c thin n^(-1 slash 2)$, the variance lower-bound condition @eq:variance-lb-condition is
satisfied for $n >= (2 C_3 || u ||^2 slash (c thin a thin sigma^2(u)))^2$.
The trivial upper bound
$sigma_n^(2, "RR")(u) <= C_(cal(Q))^2 || Sigma_(epsilon.alt)^(("M")) || || u ||^2$
gives $s_n^2 <= K^2(u) thin n$ with
$K(u) := C_(cal(Q)) thin || Sigma_(epsilon.alt)^(("M")) ||^(1 slash 2) thin || u ||$.

*Bolthausen--Fan inequality.* We apply the martingale Berry--Esseen input
@eq:imported-bolthausen-fan with $N = n$ and bounded-increment parameter
$kappa.alt(u)$.

#theorem[
  Assume *UGE 1*, $pi(epsilon.alt) = 0$, $|| epsilon.alt ||_infinity < infinity$,
  $sigma^2(u) > 0$, and $alpha, 2 alpha in (0, alpha_infinity]$. There exist
  constants $C_(K, 1)(u), C_(K, 2)(u) > 0$ depending only on $|| u ||$,
  $sigma(u)$, $C_(cal(Q))$, $t_"mix"$, $|| epsilon.alt ||_infinity$,
  $|| Sigma_(epsilon.alt)^(("M")) ||$, and the constants
  $L_B(kappa.alt(u)), C_1, C_2$ of @eq:imported-bolthausen-fan, such that for every
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

_Proof._ Apply @eq:imported-bolthausen-fan with $S_n = u^top M_n^("RR")$, the
increment bound $kappa.alt = kappa.alt(u)$ from @eq:M-RR-incr,
$s_n^2 = n thin sigma_n^(2, "RR")(u)$, and $p = ceil(log n)$ (so
$log n <= p <= log n + 1$). Bound the three terms in turn, using
$s_n^2 in [n thin sigma^2(u) slash 2, thin K^2(u) thin n]$ from the variance
lower and upper bounds.

*Term I (classical Bolthausen).* From $s_n^3 >= (n thin sigma^2(u) slash 2)^(3 slash 2)$
and $(2 n + 1) log(2 n + 1) <= 3 thin n thin log n$ for $n >= 3$,
$
L_B(kappa.alt(u)) thin frac((2 n + 1) log(2 n + 1), s_n^3)
  <= frac(6 sqrt(2) thin L_B(kappa.alt(u)), sigma^3(u)) thin frac(log n, sqrt(n))
  =: frac(C^("(I)")(u) thin log n, sqrt(n)).
$ <eq:term-I>

*Term III (Lindeberg).* Write $a_p := 2 p slash (2 p + 1) = 1 - 1 slash (2 p + 1)$.
Using $s_n^(- a_p) = s_n^(-1) thin s_n^(1 slash (2 p + 1))$ and
$s_n <= K(u) thin sqrt(n)$,
$
s_n^(1 slash (2 p + 1)) <= max(1, K(u))^(1 slash (2 p + 1)) thin n^(1 slash (2 (2 p + 1))).
$
For $p >= log n$ one has $n^(1 slash (2 (2 p + 1))) <= n^(1 slash (4 log n)) = e^(1 slash 4)$;
similarly $max(1, K(u))^(1 slash (2 p + 1)) <= e^(1 slash 2)$ once
$n >= max(1, K(u))$, which is absorbed into $C^("(III)")(u)$. Bounding
$kappa.alt(u)^(a_p) <= max(1, kappa.alt(u))$ and $s_n^(-1) <= sqrt(2) slash (sigma(u) sqrt(n))$,
$
C_2 thin s_n^(- a_p) thin p thin kappa.alt(u)^(a_p)
  <= sqrt(2) thin e^(3 slash 4) thin frac(C_2 thin max(1, kappa.alt(u)), sigma(u)) thin frac(p, sqrt(n))
  <= frac(C^("(III)")(u) thin log n, sqrt(n)),
$ <eq:term-III>
with $C^("(III)")(u)$ depending on $|| u ||, sigma(u), C_(cal(Q)),
|| Sigma_(epsilon.alt)^(("M")) ||, t_"mix", || epsilon.alt ||_infinity$ and on
$C_2$.

*Term II (conditional-variance concentration).* By the bracket concentration
bound above, for every $p >= 2$,
$
(bb(E) | V_n^2 - s_n^2 |^p)^(1 slash p)
  <= B(u) thin sqrt(p thin n),
quad
B(u) := C_4 thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2 thin t_"mix"^(5 slash 2),
$ <eq:Bu-def>
Using $s_n^(-a_p) = s_n^(-1) s_n^(1 slash (2 p + 1))$, the variance lower
bound $s_n^2 >= n sigma^2(u) / 2$, and the upper bound
$s_n <= K(u) sqrt(n)$, the second Bolthausen--Fan term is bounded by
$
&C_1 thin sqrt(p) thin s_n^(- a_p) thin (bb(E) | V_n^2 - s_n^2 |^p)^(1 slash (2 p + 1)) \
&quad<= C thin C_1 thin sigma^(-1)(u)
        thin max(1, K(u))^(1 slash (2 p + 1))
        thin max(1, B(u))^(p slash (2 p + 1)) \
&quad quad times p^((3 p + 1) slash (2 (2 p + 1)))
        n^(- p slash (2 (2 p + 1))).
$ <eq:term-II-1>

For $p = ceil(log n)$ and $n$ large enough depending on $u$,
$p^((3 p + 1) slash (2 (2 p + 1))) <= C log^(3 slash 4) n$ and
$n^(- p slash (2 (2 p + 1))) <= C n^(-1 slash 4)$; the remaining
$K(u)$- and $B(u)$-factors are absorbed into the direction-dependent constant.
Thus
$
"Term II"
  <= frac(C^("(II)")(u) thin log^(3 slash 4) n, n^(1 slash 4)).
$ <eq:term-II>

Adding the three term bounds and setting
$C_(K, 1)(u) := C^("(II)")(u)$, $C_(K, 2)(u) := C^("(I)")(u) + C^("(III)")(u)$
proves the martingale Berry--Esseen bound. $square$

#corollary[
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
  At $alpha = c thin n^(-1 slash 2)$ the last term is $O(n^(-1 slash 2))$, hence
  absorbed into $C_(K, 2)(u) thin log n slash sqrt(n)$ up to a constant.
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
d_K(W r, cal(N)(0, 1)) <= d_K(W, cal(N)(0, 1)) + C_Phi thin |r - 1|.
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

This theorem concerns only the martingale term; the stationary assembly below
adds the Poisson remainder and the Levin depth-two misadjustment.


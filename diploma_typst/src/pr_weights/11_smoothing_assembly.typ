#import "../defs.typ": *

== Smoothing Assembly

We now assemble the stationary martingale bound and the composite remainder via
the Bobkov--Götze smoothing inequality.
// Inputs: the Poisson decomposition, bracket concentration, martingale
// Berry--Esseen bound, and Levin depth-two misadjustment estimate. Together
// they yield a Kolmogorov bound for the stationary $n_0 = 0$ augmented-chain RR
// statistic.

*Stationary assembled statistic.* Define
$
S_(n, "stat")^("RR")(u)
  := -frac(u^top M_n^("RR"), sqrt(n)) + u^top cal(R)_(n, "stat")^("RR"),
$ <eq:full-decomp>
$
cal(R)_(n, "stat")^("RR")
  := D_(2, n)^("RR") + R_n^("mis, RR"),
$
where both terms are read under the stationary augmented-chain convention.
Dividing by $sigma_n^("RR")(u)$,
// The deterministic-start transient and random initial-product discrepancy are
// not part of this stationary result; they are handled in the burn-in transfer
// theorem.
$
frac(S_(n, "stat")^("RR")(u), sigma_n^("RR")(u))
  = X_n + Y_n,
$ <eq:XY-split>
$
X_n := -frac(u^top M_n^("RR"), sqrt(n) thin sigma_n^("RR")(u)),
quad
Y_n := frac(u^top cal(R)_(n, "stat")^("RR"), sigma_n^("RR")(u)).
$
The martingale bound applies to $X_n$ with direction $-u$.
// The bounded-increment constant and predictable variance are unchanged:
// $kappa.alt(-u) = kappa.alt(u)$ and
// $sigma_n^(2, "RR")(-u) = sigma_n^(2, "RR")(u)$.

*Smoothing inequality.* For any $t > 0$,
$
d_K (X + Y, cal(N)(0, 1))
  <= d_K (X, cal(N)(0, 1))
   + bb(P)(|Y| > t)
   + frac(t, sqrt(2 pi)),
$ <eq:smoothing>
Markov's inequality with $t = e thin ||Y||_(L_p)$ gives
$
d_K (X + Y, cal(N)(0, 1))
  <= d_K (X, cal(N)(0, 1))
   + frac(e thin || Y ||_(L_p), sqrt(2 pi))
   + e^(-p),
$ <eq:smoothing-Lp>
// The third term uses the Lipschitz constant of the standard normal cdf. The
// trailing $e^(-p)$ is $O(1 slash n)$ when $p >= log n$.

*$L_p$-bound on the composite remainder.* The two stationary remainder pieces
contribute additively.
// The finite-start deterministic transient is deliberately excluded: setting
// $theta_0 = theta^*$ removes only that deterministic term, not the startup
// discrepancy between zero-start perturbation variables and the stationary
// augmented chain.

#lemma[
  *(Stationary composite remainder bound.)*
  Assume the hypotheses and step-size restrictions of the stationary
  PR-averaged RR misadjustment bound above. Then for every $u in bb(R)^d$,
  every $p >= 2$, and every $q >= 2$ satisfying $p <= q slash 2$ and
  $2 alpha <= alpha_("stat")(q)$,
  $
  || u^top cal(R)_(n, "stat")^("RR") ||_(L_p)
    &<= frac(C_("D2") thin || u ||, sqrt(n))
     + || u || thin C_("mis") thin lr((
        sqrt(n) thin alpha^2
        + (1 + d^(1 slash q)) thin p^(7 slash 2) thin t_"mix"^(5 slash 2)
            thin sqrt(n) thin alpha^(3 slash 2) thin log^(3 slash 2)(1 slash (alpha a))
      \
        & quad quad quad quad
        + p^(3 slash 2) thin sqrt(alpha)
        + p^3 thin (alpha n)^(-1 slash 2) thin log^(1 slash 2)(1 slash (alpha a))
        + Phi_+(p, alpha) thin n^(-1 slash 2)
      )),
  $
  with constants
  $
  C_("D2") := 3 thin t_"mix" thin || epsilon.alt ||_infinity
              thin (C_(cal(Q)) + C_2 slash a^2),
  $
  and $C_("mis")$ the constant in the misadjustment theorem.
] <lem:R-bound>

_Proof._ Combine the sup-norm bound for $D_(2,n)^("RR")$ with
@thm:misadjustment. $square$

#theorem[
  *(Stationary augmented-chain Berry--Esseen assembly for the RR comparison statistic.)*
  Assume *UGE 1*, $pi(tilde(A)) = 0$, $pi(epsilon.alt) = 0$,
  $|| epsilon.alt ||_infinity < infinity$, $sigma^2(u) > 0$,
  $0 < alpha$, and the stationary augmented-chain
  convention above. Use the external inputs summarized in
  @sec:external-inputs. Set
  $p = max(2, ceil(log n))$ and $q = max(2 p, ceil(log(e thin d)), 2)$.
  Then $p >= 2$, $q >= 2$, and $p <= q slash 2$. If $n >= 3$ satisfies the
  small-step and variance lower-bound conditions
  $
  2 alpha <= alpha_("stat")(q),
  quad
  n thin alpha thin a >= frac(2 thin C_3 thin || u ||^2, sigma^2(u)),
  $ <eq:RR-BE-master-conditions>
  where the second inequality is @eq:variance-lb-condition, then
  $
  d_K lr((
    frac(S_(n, "stat")^("RR")(u), sigma_n^("RR")(u)),
    cal(N)(0, 1)
  ))
    &<= frac(C_(K, 1)(u) thin log^(3 slash 4) n, n^(1 slash 4))
     + frac(C_(K, 2)(u) thin log n, sqrt(n)) \
    &quad + frac(e thin || u^top cal(R)_(n, "stat")^("RR") ||_(L_p),
                 sqrt(2 pi) thin sigma_n^("RR")(u))
     + frac(e, n),
  $ <eq:RR-BE-master>
  with $|| u^top cal(R)_(n, "stat")^("RR") ||_(L_p)$ bounded by the preceding
  lemma, and $C_(K, 1)(u), C_(K, 2)(u)$ the constants of the martingale
  Berry--Esseen theorem.
] <thm:RR-BE>

_Proof._ Apply @eq:smoothing-Lp with $X = X_n$ and $Y = Y_n$. The martingale
terms come from @thm:M-RR-BE, and the remainder term is
$e thin || Y_n ||_(L_p) slash sqrt(2 pi) = e thin || u^top cal(R)_(n, "stat")^("RR")
||_(L_p) slash (sqrt(2 pi) thin sigma_n^("RR")(u))$. Since $p >= log n$,
$e^(-p) <= n^(-1)$. $square$

// #corollary[
//   *(Stationary triangular-array admissible-step CLT.)*
//   Assume the stationary augmented-chain hypotheses of @thm:RR-BE, including
//   $sigma^2(u) > 0$. Let $alpha = alpha_n$ be a step-size sequence and set
//   $p_n := max(2, ceil(log n))$, $q_n := max(2 p_n, ceil(log(e thin d)), 2)$,
//   and $Lambda_n := log(1 slash (alpha_n a))$. Then $p_n <= q_n slash 2$.
//   Assume that, for all sufficiently large $n$,
//   $
//   2 alpha_n <= alpha_("stat")(q_n),
//   quad
//   n thin alpha_n thin a >= frac(2 thin C_3 thin || u ||^2, sigma^2(u)).
//   $
//   If
//   $
//   p_n^3 (n alpha_n)^(-1 slash 2) Lambda_n^(1 slash p_n) -> 0,
//   quad
//   p_n^(7 slash 2) sqrt(n) thin alpha_n^(3 slash 2) Lambda_n^(3 slash 2) -> 0,
//   $ <eq:alpha-window>
//   then
//   $
//   d_K lr((
//     frac(S_(n, "stat")^("RR")(u), sigma_n^("RR")(u)),
//     cal(N)(0, 1)
//   )) -> 0.
//   $
//   In particular, for a power scale $alpha_n = c thin n^(-gamma)$, up to
//   logarithmic factors, the admissible window is
//   $
//   frac(1, 3) < gamma < 1.
//   $
// ] <cor:RR-BE-admissible-alpha>

// _Proof._ Apply @thm:RR-BE and @lem:R-bound with
// $p = p_n$, $q = q_n$, and $alpha = alpha_n$. The displayed conditions are the
// remaining nontrivial remainder constraints. For $alpha_n = c n^(-gamma)$ they
// reduce, modulo logarithms, to
// $n^(-(1 - gamma) slash 2) -> 0$ and
// $n^(1 slash 2 - 3 gamma slash 2) -> 0$, i.e. $gamma < 1$ and
// $gamma > 1 slash 3$. $square$

#corollary[
  *(Stationary balanced-scale augmented-chain Berry--Esseen bound.)*
  At the balanced scale $alpha = c thin n^(-1 slash 2)$ with $c > 0$, put
  $p = max(2, ceil(log n))$ and $q = max(2 p, ceil(log(e thin d)), 2)$.
  Then $p <= q slash 2$. Assume the stationary augmented-chain hypotheses of
  @thm:RR-BE, including $sigma^2(u) > 0$. If $n >= 3$ satisfies
  $
  2 alpha <= alpha_("stat")(q),
  quad
  n thin alpha thin a >= frac(2 thin C_3 thin || u ||^2, sigma^2(u)),
  $
  the bound
  @eq:RR-BE-master reduces to
  $
  d_K lr((
    frac(S_(n, "stat")^("RR")(u), sigma_n^("RR")(u)),
    cal(N)(0, 1)
  ))
    <= frac(C(u) thin "polylog"(n), n^(1 slash 4)),
  $
  where $C(u)$ depends on $|| u ||$, $sigma(u)$, $C_(cal(Q))$,
  $|| overline(A) ||$, $|| overline(A)^(-1) ||$, $kappa_Q$, $C_A$,
  $|| epsilon.alt ||_infinity$, $|| Sigma_(epsilon.alt)^(("M")) ||$,
  $t_"mix"$, $a$, $alpha_infinity$, $c$, and the universal constants of
  the smoothing inequality, @lem:external-martingale-be, and
  @eq:external-markov-conc.
] <cor:RR-BE-working>

// _Proof._ At $alpha = c thin n^(-1 slash 2)$,
// $C_("D2") || u || slash sqrt(n) = O(n^(-1 slash 2))$;
// $sqrt(n) alpha^2 = c^2 thin n^(-1 slash 2)$;
// $p^(7 slash 2) sqrt(n) alpha^(3 slash 2) log^(3 slash 2)(1 slash (alpha a))
// = O("polylog"(n) n^(-1 slash 4))$;
// $p^(3 slash 2) sqrt(alpha) = O("polylog"(n) n^(-1 slash 4))$;
// $p^3 (alpha n)^(-1 slash 2) log^(1 slash 2)(1 slash (alpha a))
// = O("polylog"(n) n^(-1 slash 4))$; and
// $Phi_+(p, alpha) n^(-1 slash 2) = O("polylog"(n) n^(-1 slash 2))$.
// After division by $sigma_n^("RR")(u) >= sigma(u) slash sqrt(2)$, all remainder
// terms are dominated by $"polylog"(n) n^(-1 slash 4)$. Combine with the
// martingale Berry--Esseen term.
// $square$

// #corollary[
//   *(Stationary asymptotic-normalization version.)*
//   Assume the stationary augmented-chain hypotheses of @thm:RR-BE for the same
//   $n,p,q,alpha,u$. In particular, assume $p <= q slash 2$, $sigma^2(u) > 0$,
//   and
//   $
//   2 alpha <= alpha_("stat")(q),
//   quad
//   n thin alpha thin a >= frac(2 thin C_3 thin || u ||^2, sigma^2(u)).
//   $
//   Then the same finite-$n$ bound with asymptotic normalization contains one
//   additional variance-comparison term:
//   $
//   d_K lr((
//     frac(S_(n, "stat")^("RR")(u), sigma(u)),
//     cal(N)(0, 1)
//   ))
//     &<= frac(C_(K, 1)(u) thin log^(3 slash 4) n, n^(1 slash 4))
//      + frac(C_(K, 2)(u) thin log n, sqrt(n)) \
//     &quad + frac(e thin || u^top cal(R)_(n, "stat")^("RR") ||_(L_p),
//                  sqrt(2 pi) thin sigma_n^("RR")(u))
//      + frac(e, n)
//      + frac(C thin || u ||^2, n thin alpha thin a thin sigma^2(u)).
//   $ <eq:RR-BE-sigma-master>
//   Consequently, under the balanced-scale hypotheses above,
//   $
//   d_K lr((
//     frac(S_(n, "stat")^("RR")(u), sigma(u)),
//     cal(N)(0, 1)
//   ))
//     <= frac(C'(u) thin "polylog"(n), n^(1 slash 4)).
//   $
// ] <cor:RR-BE-sigma>

// _Proof._ Set $r := sigma_n^("RR")(u) slash sigma(u)$ and write
// $W := S_(n, "stat")^("RR")(u) slash sigma_n^("RR")(u)$, so
// $W r = S_(n, "stat")^("RR")(u) slash sigma(u)$. Under
// the variance lower-bound condition @eq:variance-lb-condition, $r in [1 slash sqrt(2), thin r_max]$ for some
// finite $r_max$ (from the trivial upper bound
// $sigma_n^(2, "RR")(u) <= C_(cal(Q))^2 thin || Sigma || thin || u ||^2$).
// For $r$ in this compact interval,
// $
// sup_x | Phi(x slash r) - Phi(x) | <= C_Phi thin |r - 1|,
// quad
// C_Phi := sqrt(2) slash sqrt(pi e),
// $
// Hence
// // The displayed Lipschitz constant follows from
// // $sup_x |x thin phi(x)| = 1 slash sqrt(2 pi e)$.
// $
// d_K (W r, cal(N)(0, 1))
//   <= d_K (W, cal(N)(0, 1)) + C_Phi thin |r - 1|.
// $
// The variance comparison of Section 4.5 gives
// $
// |r - 1|
//   = frac(|sigma_n^(2, "RR")(u) - sigma^2(u)|,
//          sigma(u) thin (sigma_n^("RR")(u) + sigma(u)))
//   <= frac(C_3 thin || u ||^2,
//           n thin alpha thin a thin sigma^2(u)),
// $
// using $sigma_n^("RR")(u) + sigma(u) >= sigma(u)$. At
// $alpha = c n^(-1 slash 2)$ the final term is $O(n^(-1 slash 2))$. $square$

// This is an $n_0 = 0$ stationary augmented-chain statement; deterministic starts
// are transferred in the burn-in chapter.

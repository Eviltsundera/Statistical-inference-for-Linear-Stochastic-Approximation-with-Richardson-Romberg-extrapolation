#import "../defs.typ": *

== Smoothing Assembly

The previous sections produced the four ingredients of the stationary
$n_0 = 0$ Berry--Esseen program for the PR-averaged Richardson--Romberg
statistic:

+ the *Poisson decomposition* $W^("RR") = -n^(-1 slash 2) M_n^("RR") + D_(2, n)^("RR")$
  with deterministic sup-norm control of $D_(2, n)^("RR")$ at the order
  $t_"mix" slash (a^2 sqrt(n))$;
+ the *predictable quadratic variation concentration* of
  $u^top chevron.l M^("RR") chevron.r_n u$ around $n thin sigma_n^(2, "RR")(u)$
  in $L_p$, obtained from the Markov concentration scale
  $sqrt(p thin t_"mix" thin sum c_i^2)$ and then written as
  $B(u) sqrt(p thin n)$;
+ the *martingale Berry--Esseen* for
  $u^top M_n^("RR") slash (sqrt(n) thin sigma_n^("RR")(u))$ at rate
  $log^(3 slash 4)(n) thin n^(-1 slash 4)$;
+ the *Levin depth-two misadjustment bound* on $R_n^("mis, RR")$ controlling
  the non-martingale residual at the same $log^c(n) thin n^(-1 slash 4)$
  rate.

The smoothing inequality of Bobkov--Götze (Samsonov et al. 2025,
Proposition 12) assembles these four into a single Kolmogorov bound on
the stationary $n_0 = 0$ augmented-chain RR statistic.

*Stationary assembled statistic.* Combining the depth-one identity in
Section 4.1 with the Poisson decomposition of Section 4.6 gives a finite-start
identity. Since the transfer from a zero-start recursion to the stationary
augmented chain is not proved here, the bound below is stated for the
stationary $n_0 = 0$ assembled scalar statistic
$
S_(n, "stat")^("RR")(u)
  := -frac(u^top M_n^("RR"), sqrt(n)) + u^top cal(R)_(n, "stat")^("RR"),
$ <eq:full-decomp>
$
cal(R)_(n, "stat")^("RR")
  := D_(2, n)^("RR") + R_n^("mis, RR"),
$
where $D_(2, n)^("RR")$ is the Poisson boundary/Abel remainder of
Section 4.6 and $R_n^("mis, RR")$ is the Levin depth-two misadjustment
defined in @eq:R-mis-def, both read under the stationary augmented-chain
convention. The deterministic-start transient
$D_("tr")^("RR") := 2 thin D_("tr")^((alpha)) - D_("tr")^((2 alpha))$ and the
random initial-product discrepancy
$2 D_(op("init"))^((alpha)) - D_(op("init"))^((2 alpha))$ are not part of this
stationary result; handling them together with the accumulated startup error
belongs to the finite-start/burn-in transfer theorem. Dividing by
$sigma_n^("RR")(u)$,
$
frac(S_(n, "stat")^("RR")(u), sigma_n^("RR")(u))
  = X_n + Y_n,
$ <eq:XY-split>
$
X_n := -frac(u^top M_n^("RR"), sqrt(n) thin sigma_n^("RR")(u)),
quad
Y_n := frac(u^top cal(R)_(n, "stat")^("RR"), sigma_n^("RR")(u)).
$
The martingale Berry--Esseen bound controls $d_K(X_n, cal(N)(0, 1))$ by
applying @thm:M-RR-BE to the signed direction $-u$. Equivalently, use the
martingale differences $-u^top Delta M_l^("RR")$; their bounded-increment
constant and predictable variance are the same as for $u$, because
$kappa.alt(-u) = kappa.alt(u)$ and
$sigma_n^(2, "RR")(-u) = sigma_n^(2, "RR")(u)$.

*Smoothing inequality.* For random variables $X, Y$ on the same probability
space and any $t > 0$,
$
d_K (X + Y, cal(N)(0, 1))
  <= d_K (X, cal(N)(0, 1))
   + bb(P)(|Y| > t)
   + frac(t, sqrt(2 pi)),
$ <eq:smoothing>
where the third term uses the $1 slash sqrt(2 pi)$ Lipschitz constant of
the standard normal cdf (Bobkov--Götze; see Samsonov et al. 2025,
Proposition 12 for an LSA-tailored statement). Bounding
$bb(P)(|Y| > t) <= || Y ||_(L_p)^p slash t^p$ via Markov's inequality and
choosing $t = e thin || Y ||_(L_p)$ gives the *working form*
$
d_K (X + Y, cal(N)(0, 1))
  <= d_K (X, cal(N)(0, 1))
   + frac(e thin || Y ||_(L_p), sqrt(2 pi))
   + e^(-p),
$ <eq:smoothing-Lp>
in which the trailing tail probability $e^(-p)$ is absorbed into
$O(1 slash n)$ as soon as $p >= log n$.

*$L_p$-bound on the composite remainder.* The two pieces of
$cal(R)_(n, "stat")^("RR")$ contribute additively. The finite-start
deterministic transient displayed in Section 4.1 is deliberately excluded:
setting $theta_0 = theta^*$ would remove only that deterministic term, but not
the startup discrepancy between zero-start perturbation variables and the
stationary augmented chain.

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

_Proof._ Triangle inequality on $u^top cal(R)_(n, "stat")^("RR")$:

(a) The deterministic sup-norm bound for $D_(2, n)^("RR")$
yields $|| u^top D_(2, n)^("RR") ||_(L_p) <= || u || thin || D_(2, n)^("RR") ||_infinity
<= C_("D2") thin || u || slash sqrt(n)$ for every $p >= 1$.

(b) The misadjustment theorem gives
the remaining summands directly. $square$

#theorem[
  *(Stationary augmented-chain Berry--Esseen assembly for the RR comparison statistic.)*
  Assume *UGE 1*, $pi(tilde(A)) = 0$, $pi(epsilon.alt) = 0$,
  $|| epsilon.alt ||_infinity < infinity$, $sigma^2(u) > 0$,
  $0 < alpha$, and the stationary augmented-chain
  convention above. Use the imported inputs summarized in
  @sec:imported-inputs. Set
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

_Proof._ Apply the smoothing inequality with $X = X_n$ and $Y = Y_n$ from the
displayed decomposition above. The first two terms come from @thm:M-RR-BE
applied to the signed scalar martingale with increments
$-u^top Delta M_l^("RR")$; the constants are unchanged since the theorem's
direction-dependent quantities are invariant under $u -> -u$.
The remainder term is
$e thin || Y_n ||_(L_p) slash sqrt(2 pi) = e thin || u^top cal(R)_(n, "stat")^("RR")
||_(L_p) slash (sqrt(2 pi) thin sigma_n^("RR")(u))$. Since
$p = max(2, ceil(log n)) >= log n$, the trailing $e^(-p)$ is at most
$n^(-1)$. $square$

#corollary[
  *(Stationary triangular-array admissible-step CLT.)*
  Assume the stationary augmented-chain hypotheses of @thm:RR-BE, including
  $sigma^2(u) > 0$. Let $alpha = alpha_n$ be a step-size sequence and set
  $p_n := max(2, ceil(log n))$, $q_n := max(2 p_n, ceil(log(e thin d)), 2)$,
  and $Lambda_n := log(1 slash (alpha_n a))$. Then $p_n <= q_n slash 2$.
  Assume that, for all sufficiently large $n$,
  $
  2 alpha_n <= alpha_("stat")(q_n),
  quad
  n thin alpha_n thin a >= frac(2 thin C_3 thin || u ||^2, sigma^2(u)).
  $
  If
  $
  p_n^3 (n alpha_n)^(-1 slash 2) Lambda_n^(1 slash p_n) -> 0,
  quad
  p_n^(7 slash 2) sqrt(n) thin alpha_n^(3 slash 2) Lambda_n^(3 slash 2) -> 0,
  $ <eq:alpha-window>
  then
  $
  d_K lr((
    frac(S_(n, "stat")^("RR")(u), sigma_n^("RR")(u)),
    cal(N)(0, 1)
  )) -> 0.
  $
  In particular, for a power scale $alpha_n = c thin n^(-gamma)$, up to
  logarithmic factors, the admissible window is
  $
  frac(1, 3) < gamma < 1.
  $
] <cor:RR-BE-admissible-alpha>

_Proof._ Apply the stationary $n_0 = 0$ Berry--Esseen bound and the preceding
remainder lemma with
$p = p_n$, $q = q_n$, and $alpha = alpha_n$. The Poisson boundary term is
$O(n^(-1 slash 2))$. The bias term $sqrt(n) alpha_n^2$ is smaller than
$sqrt(n) alpha_n^(3 slash 2)$ for all sufficiently large $n$, because the
admissibility conditions imply $alpha_n -> 0$. The terms
$p_n^(3 slash 2) sqrt(alpha_n)$ and $Phi_+(p_n, alpha_n) n^(-1 slash 2)$
are also negligible under the second admissibility condition. The remaining
nontrivial terms are exactly the two quantities displayed in the statement.
The martingale Berry--Esseen terms
$log^(3 slash 4)(n) n^(-1 slash 4)$ and $log(n) n^(-1 slash 2)$ vanish
independently of $alpha_n$. For $alpha_n = c n^(-gamma)$, the two displayed
conditions reduce, modulo logarithms, to
$n^(-(1 - gamma) slash 2) -> 0$ and
$n^(1 slash 2 - 3 gamma slash 2) -> 0$, i.e. $gamma < 1$ and
$gamma > 1 slash 3$. $square$

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
  the smoothing inequality, @eq:imported-bolthausen-fan, and
  @eq:imported-markov-conc.
] <cor:RR-BE-working>

_Proof._ The scale $alpha = c thin n^(-1 slash 2)$ satisfies
@eq:alpha-window, but gives the sharper balanced rate. Indeed,
$C_("D2") || u || slash sqrt(n) = O(n^(-1 slash 2))$;
$sqrt(n) alpha^2 = c^2 thin n^(-1 slash 2)$;
$p^(7 slash 2) sqrt(n) alpha^(3 slash 2) log^(3 slash 2)(1 slash (alpha a))
= O("polylog"(n) n^(-1 slash 4))$;
$p^(3 slash 2) sqrt(alpha) = O("polylog"(n) n^(-1 slash 4))$;
$p^3 (alpha n)^(-1 slash 2) log^(1 slash 2)(1 slash (alpha a))
= O("polylog"(n) n^(-1 slash 4))$; and
$Phi_+(p, alpha) n^(-1 slash 2) = O("polylog"(n) n^(-1 slash 2))$.
After division by $sigma_n^("RR")(u) >= sigma(u) slash sqrt(2)$, these
terms are dominated by $"polylog"(n) n^(-1 slash 4)$. Combining with the
martingale Berry--Esseen term proves the claim.
$square$

#corollary[
  *(Stationary asymptotic-normalization version.)*
  Assume the stationary augmented-chain hypotheses of @thm:RR-BE for the same
  $n,p,q,alpha,u$. In particular, assume $p <= q slash 2$, $sigma^2(u) > 0$,
  and
  $
  2 alpha <= alpha_("stat")(q),
  quad
  n thin alpha thin a >= frac(2 thin C_3 thin || u ||^2, sigma^2(u)).
  $
  Then the same finite-$n$ bound with asymptotic normalisation contains one
  additional variance-comparison term:
  $
  d_K lr((
    frac(S_(n, "stat")^("RR")(u), sigma(u)),
    cal(N)(0, 1)
  ))
    &<= frac(C_(K, 1)(u) thin log^(3 slash 4) n, n^(1 slash 4))
     + frac(C_(K, 2)(u) thin log n, sqrt(n)) \
    &quad + frac(e thin || u^top cal(R)_(n, "stat")^("RR") ||_(L_p),
                 sqrt(2 pi) thin sigma_n^("RR")(u))
     + frac(e, n)
     + frac(C thin || u ||^2, n thin alpha thin a thin sigma^2(u)).
  $ <eq:RR-BE-sigma-master>
  Consequently, under the balanced-scale hypotheses above,
  $
  d_K lr((
    frac(S_(n, "stat")^("RR")(u), sigma(u)),
    cal(N)(0, 1)
  ))
    <= frac(C'(u) thin "polylog"(n), n^(1 slash 4)).
  $
] <cor:RR-BE-sigma>

_Proof._ Set $r := sigma_n^("RR")(u) slash sigma(u)$ and write
$W := S_(n, "stat")^("RR")(u) slash sigma_n^("RR")(u)$, so
$W r = S_(n, "stat")^("RR")(u) slash sigma(u)$. Under
the variance lower-bound condition @eq:variance-lb-condition, $r in [1 slash sqrt(2), thin r_max]$ for some
finite $r_max$ (from the trivial upper bound
$sigma_n^(2, "RR")(u) <= C_(cal(Q))^2 thin || Sigma || thin || u ||^2$).
For $r$ in this compact interval,
$
sup_x | Phi(x slash r) - Phi(x) | <= C_Phi thin |r - 1|,
quad
C_Phi := sqrt(2) slash sqrt(pi e),
$
because $sup_x |x thin phi(x)| = 1 slash sqrt(2 pi e)$. Hence
$
d_K (W r, cal(N)(0, 1))
  <= d_K (W, cal(N)(0, 1)) + C_Phi thin |r - 1|.
$
The variance comparison of Section 4.5 gives
$
|r - 1|
  = frac(|sigma_n^(2, "RR")(u) - sigma^2(u)|,
         sigma(u) thin (sigma_n^("RR")(u) + sigma(u)))
  <= frac(C_3 thin || u ||^2,
          n thin alpha thin a thin sigma^2(u)),
$
using $sigma_n^("RR")(u) + sigma(u) >= sigma(u)$. This proves the displayed
asymptotic-normalisation bound. At $alpha = c n^(-1 slash 2)$ the final term is
$O(n^(-1 slash 2))$, hence it is absorbed into the balanced-scale bound of
the preceding corollary. $square$

The bound above is therefore an $n_0 = 0$ stationary augmented-chain statement
for $S_(n, "stat")^("RR")(u)$, not the deterministic-start RR average itself.
A deterministic-start theorem with mixing-scale burn-in and logarithmic factors
requires a separate transfer theorem with $Q_(l,n_0)^((alpha))$ and the
corresponding Poisson, variance-comparison, and misadjustment bounds.

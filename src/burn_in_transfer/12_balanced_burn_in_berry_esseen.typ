#import "../defs.typ": *

== Balanced Burn-in Berry--Esseen Bound

The finite-window bound uses $sigma_(n,n_0)^("bRR")(u)$. The final inference
statement uses $sigma(u)$, via the following burned-in analogue of
@cor:RR-BE-sigma.

#lemma[
  *(Burned-in normalization transfer.)*
  Assume $sigma^2(u) > 0$ and the burned-in variance lower-bound condition
  @eq:burn-variance-lb-condition. Put
  $
  r_(n,n_0)(u)
    := frac(sigma_(n,n_0)^("bRR")(u), sigma(u)).
  $
  Then
  $
  frac(1, sqrt(2)) <= r_(n,n_0)(u) <= sqrt(3 slash 2),
  $
  and, for any real random variable $W$,
  $
  d_K lr((r_(n,n_0)(u) W, cal(N)(0, 1)))
    <= d_K lr((W, cal(N)(0, 1)))
     + frac(C_("norm") C_("burn,3") ||u||^2,
            m thin alpha thin a thin sigma^2(u)),
  $ <eq:burn-normalization-transfer>
  where $C_("norm")$ is a universal constant.
] <lem:burn-normalization-transfer>

_Proof._ The variance lower-bound condition and
@eq:burn-scalar-variance-comparison give
$
|sigma_(n,n_0)^(2, "bRR")(u) - sigma^2(u)| <= sigma^2(u) slash 2.
$
Hence $r_(n,n_0)(u) in [1 slash sqrt(2), sqrt(3 slash 2)]$. For $r$ in this
compact interval,
$
sup_x | Phi(x slash r) - Phi(x) | <= C_("norm") thin |r - 1|,
$
because $sup_x |x phi(x)| < infinity$. Therefore
$
d_K (r W, cal(N)(0, 1))
  <= d_K (W, cal(N)(0, 1)) + C_("norm") thin |r - 1|.
$
Finally,
$
|r_(n,n_0)(u) - 1|
  = frac(|sigma_(n,n_0)^(2, "bRR")(u) - sigma^2(u)|,
         sigma(u) thin (sigma_(n,n_0)^("bRR")(u) + sigma(u)))
  <= frac(C_("burn,3") ||u||^2,
          m thin alpha thin a thin sigma^2(u)),
$
using @eq:burn-scalar-variance-comparison and
$sigma_(n,n_0)^("bRR")(u) + sigma(u) >= sigma(u)$. $square$

#theorem[
  *(Balanced-scale deterministic-start burned-in PR-averaged RR Berry--Esseen bound.)*
  Assume Assumptions 1--3 from @sec:assumptions. Assume also the Lyapunov
  contraction @eq:contraction for the two step sizes used below and the
  external inputs and local extensions summarized in @sec:imported-inputs.
  Fix $c > 0$ and set
  $
  alpha := c thin n^(-1 slash 2),
  quad
  m := n - n_0,
  quad
  p := max(2, ceil(log n)),
  quad
  q := max(4 p, ceil(log(e thin d)), 2).
  $
  Then $p >= 2$, $q >= 2$, and $p <= q slash 4$.
  Use the deterministic-start admissibility threshold
  $alpha_("burn")(p,q)$ defined in @eq:alpha-admissibility-thresholds.
  Suppose $(n,n_0,p,q,alpha,u)$ is in the admissible burn-in regime
  @eq:admissible-burn-regime.
  Assume also that the burn-in satisfies the explicit mixing-scale conditions
  with logarithmic factors
  $
  n_0 >= frac(2, alpha a) log n,
  quad
  n_0 >= frac(p, c_("init") alpha a) log n,
  quad
  n_0 >= frac(p, c_("st") alpha a) log n.
  $ <eq:burn-final-log-conditions>
  Then there exists a finite constant $C_("burn,final")(u,c,theta_0)$,
  depending only on $u$, $c$, $||theta_0 - theta^*||$, and the problem and
  universal constants in the preceding bounds, but not on $xi$, such that
  $
  d_K lr((Xi_(n,n_0)^("bRR")(u), cal(N)(0, 1)))
    <= frac(C_("burn,final")(u,c,theta_0) thin "polylog"(n),
             n^(1 slash 4)),
  $ <eq:burn-final-finite-window>
  and
  $
  d_K lr((Xi_(n,n_0)^("asy,RR")(u), cal(N)(0, 1)))
    <= frac(C_("burn,final")(u,c,theta_0) thin "polylog"(n),
             n^(1 slash 4)).
  $ <eq:burn-final-asymptotic>
] <thm:burn-final-balanced>

_Proof._ Apply the finite-window assembly theorem @thm:burn-RR-BE-master. Since
$m >= n slash 2$, polynomial prefactors in $n$ may be written on the $m$ scale
up to universal constants; for instance $sqrt(n) slash m <= sqrt(2) slash
sqrt(m)$. In particular, the martingale terms satisfy
$
frac(log^(3 slash 4) n, m^(1 slash 4))
  + frac(log n, sqrt(m))
  <= C thin frac("polylog"(n), n^(1 slash 4)).
$
It remains to bound the composite remainder in @lem:burn-R-bound. The first
condition in @eq:burn-final-log-conditions is @eq:burn-log-condition with
$beta = 1$; by @cor:burn-log-transient and
$alpha = c n^(-1 slash 2)$, $m >= n slash 2$, the deterministic transient is
$O(n^(-1))$. The second condition is @eq:burn-log-init-condition with
$beta = 1$, so @cor:burn-log-initial-product makes the random initial-product
term $O("polylog"(n) n^(-1))$. The Poisson Abel remainder is
$O(m^(-1 slash 2))$. The third condition in @eq:burn-final-log-conditions is
the startup condition @eq:burn-log-startup-condition with $beta = 1$, so
@cor:burn-misadjustment-rate gives
$
|| R_(n,n_0, op("fin"))^("mis,RR") ||_(L_p)
  <= C thin "polylog"(n) thin n^(-1 slash 4).
$
Together with
$sigma_(n,n_0)^("bRR")(u) >= sigma(u) slash sqrt(2)$, this makes the smoothing
remainder in @eq:burn-RR-BE-master of order
$"polylog"(n) n^(-1 slash 4)$. The smoothing tail $e slash n$ is lower order,
so @eq:burn-final-finite-window follows.

For the asymptotic normalization, write
$
Xi_(n,n_0)^("asy,RR")(u)
  = r_(n,n_0)(u) thin Xi_(n,n_0)^("bRR")(u).
$
By @lem:burn-normalization-transfer, the additional cost is at most
$
frac(C thin ||u||^2, m thin alpha thin a thin sigma^2(u))
  = O(n^(-1 slash 2)),
$
because $m >= n slash 2$ and $alpha = c n^(-1 slash 2)$. This is absorbed
into the balanced finite-window rate, proving @eq:burn-final-asymptotic.
$square$

The admissible burn-in regime @eq:admissible-burn-regime collects the
finite-$n$ admissibility requirements. The inequality
$2 alpha <= alpha_("burn")(p,q)$ enforces the Lyapunov small-step ceiling, the
local inverse ceiling, the Levin depth-two admissibility threshold, the
random-product stability estimate
@lem:burn-product-stability, and the full-state startup contraction
@lem:burn-full-startup. Since
$alpha = c n^(-1 slash 2)$ and $m >= n slash 2$, the elementary step-size
constraints and @eq:burn-variance-lb-condition hold automatically for all
sufficiently large $n$. The remaining non-elementary large-$n$ requirement is
$
2 c n^(-1 slash 2)
  <= alpha_("burn")(p,q),
$ <eq:burn-final-levin-eventual>
with $p,q$ as in the theorem. Under this Levin/startup admissibility condition,
the large-$n$ reading of the theorem keeps only $m >= n slash 2$ and the
burn-in lower bounds in @eq:burn-final-log-conditions.

#corollary[
  *($sqrt(n)$-normalization for the burned-in RR statistic.)*
  Under the assumptions of @thm:burn-final-balanced and the admissible
  burn-in regime @eq:admissible-burn-regime, in particular
  $m = n - n_0 >= n slash 2$, $p <= q slash 4$, $sigma^2(u) > 0$,
  $2 alpha <= alpha_("burn")(p,q)$, and the burned-in variance lower bound
  @eq:burn-variance-lb-condition, define the final scalar statistic
  $
  T_(n,n_0)^("RR,n")(u)
    := sqrt(n) thin u^top lr((
        overline(theta)_(n,n_0)^(("RR", alpha)) - theta^*
      ))
    = sqrt(n slash m) thin T_(n,n_0)^("RR")(u),
  $
  and its asymptotic normalization
  $
  Xi_(n,n_0)^("n,RR")(u)
    := frac(T_(n,n_0)^("RR,n")(u), sigma(u)).
  $
  For the Berry--Esseen moment choice $p = max(2, ceil(log n))$ in
  @thm:burn-final-balanced, the lower burn-in conditions
  @eq:burn-final-log-conditions are implied by
  $
  n_0 >= C_- thin (alpha a)^(-1) log^2 n
  $
  with any fixed $C_-$ large enough. At the balanced scale
  $alpha = c n^(-1 slash 2)$ this is
  $n_0 = O(n^(1 slash 2) log^2 n)$, not a purely logarithmic burn-in in $n$.
  If, in addition, the burn-in window stays in the same mixing-scale window
  with logarithmic-square factor,
  $
  C_- thin (alpha a)^(-1) log^2 n
    <= n_0
    <= C_0 thin (alpha a)^(-1) log^2 n
  $ <eq:burn-log-window>
  for some finite $C_0$ and such a fixed $C_-$, then there exists a finite constant
  $C_("burn,n")(u,c,theta_0,C_0,C_-)$, independent of $xi$, such that
  $
  d_K lr((Xi_(n,n_0)^("n,RR")(u), cal(N)(0, 1)))
    <= frac(C_("burn,n")(u,c,theta_0,C_0,C_-) thin "polylog"(n),
             n^(1 slash 4)).
  $ <eq:burn-sqrt-n-final>
] <cor:burn-sqrt-n-transfer>

_Proof._ Put $s_(n,n_0) := sqrt(n slash m)$. Since
@eq:admissible-burn-regime gives $m >= n slash 2$,
$
0 <= s_(n,n_0) - 1
  = frac(n_0, m thin (s_(n,n_0) + 1))
  <= frac(2 n_0, n).
$
For every real random variable $W$ and every $s in [1, sqrt(2)]$,
$
d_K lr((s W, cal(N)(0, 1)))
  <= d_K lr((W, cal(N)(0, 1))) + C thin |s - 1|,
$
by the same scaling argument as in @lem:burn-normalization-transfer. Applying
this with $W = Xi_(n,n_0)^("asy,RR")(u)$ and using
@thm:burn-final-balanced gives the already proved
$"polylog"(n) n^(-1 slash 4)$ term. The upper burn-in bound
in @eq:burn-log-window and $alpha = c n^(-1 slash 2)$ give
$
|s_(n,n_0) - 1|
  <= frac(2 C_0, c a) frac(log^2 n, n^(1 slash 2)),
$
which is lower order and is absorbed into the same balanced-scale rate.
$square$

The lower side of @eq:burn-log-window is the logarithmic-square mixing-scale
burn-in needed by the current $L_p$ startup contraction; the upper side keeps
the $sqrt(n slash m)$ rescaling lower order. At
$alpha = c n^(-1 slash 2)$ this window has order
$n^(1 slash 2) log^2 n$.

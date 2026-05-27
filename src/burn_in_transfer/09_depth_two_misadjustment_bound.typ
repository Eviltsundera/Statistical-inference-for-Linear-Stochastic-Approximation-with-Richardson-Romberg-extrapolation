#import "../defs.typ": *

== Burned-in Depth-Two Misadjustment Bound

Define the finite-start burned-in RR misadjustment by
$
R_(n,n_0, op("fin"))^("mis,RR")
  := frac(1, sqrt(m)) sum_(k = n_0)^(n - 1)
    lr((
      2 R_(k, op("fin"))^((alpha))
        - R_(k, op("fin"))^((2 alpha))
    )).
$ <eq:burn-mis-fin-def>
For comparison, let
$
R_(m, op("aug"))^("mis,RR")
  := frac(1, sqrt(m)) sum_(j = 0)^(m - 1)
    lr((
      2 R_(j, op("aug"))^((alpha))
        - R_(j, op("aug"))^((2 alpha))
    ))
$ <eq:burn-mis-aug-def>
be the stationary augmented-chain depth-two misadjustment over a window of
length $m$.
// By stationarity, the same distribution is obtained if the sum in
// @eq:burn-mis-aug-def is taken over $j = n_0, dots, n - 1$.

#theorem[
  *(Burned-in PR-averaged RR misadjustment bound.)*
  Assume *UGE 1*, $pi(epsilon.alt) = 0$, $|| epsilon.alt ||_infinity < infinity$,
  and $0 < alpha$. Set
  $
  Phi_+(p, alpha) := 1 + p^(3 slash 2) thin t_"mix"^(1 slash 2) slash a
                   + p^(1 slash 2) thin t_"mix"^(3 slash 2) sqrt(alpha slash a).
  $
  There exists a constant $C_("burn,mis")$ depending only on the stationary
  misadjustment constants, the startup-contraction constants, and the problem
  constants such that, for every $p >= 2$, every $q >= 2$ satisfying
  $p <= q slash 4$ and $2 alpha <= alpha_("burn")(p,q)$, and every $m >= 2$,
  $
  || R_(n,n_0, op("fin"))^("mis,RR") ||_(L_p)
    &<= C_("burn,mis") sqrt(m) thin alpha^2 \
    &quad + C_("burn,mis") (1 + d^(1 slash q)) p^(7 slash 2)
       t_"mix"^(5 slash 2) sqrt(m) thin alpha^(3 slash 2)
       log^(3 slash 2)(1 slash (alpha a)) \
    &quad + C_("burn,mis") p^(3 slash 2) sqrt(alpha) \
    &quad + C_("burn,mis") p^3 (alpha m)^(-1 slash 2)
       log^(1 slash 2)(1 slash (alpha a)) \
    &quad + C_("burn,mis") Phi_+(p, alpha) thin m^(-1 slash 2) \
    &quad + frac(C_("burn,mis") p thin A_("st")(p,q,alpha),
                  alpha a sqrt(m))
       exp(-c_("st") alpha a n_0 slash p).
  $ <eq:burn-mis-bound>
] <thm:burn-misadjustment>

_Proof._ Couple the finite-start and stationary augmented-chain remainders as
in @lem:burn-full-startup, and define the stationary window on the same
indices by
$
tilde(R)_(n,n_0, op("aug"))^("mis,RR")
  := frac(1, sqrt(m)) sum_(k = n_0)^(n - 1)
    lr((
      2 R_(k, op("aug"))^((alpha))
        - R_(k, op("aug"))^((2 alpha))
    )).
$
Then, with the notation of @eq:burn-startup-discrepancy,
$
R_(n,n_0, op("fin"))^("mis,RR")
  = tilde(R)_(n,n_0, op("aug"))^("mis,RR")
    + cal(U)_(n,n_0)^("start,RR")
$
under this coupling. Therefore
$
|| R_(n,n_0, op("fin"))^("mis,RR") ||_(L_p)
  <= || tilde(R)_(n,n_0, op("aug"))^("mis,RR") ||_(L_p)
     + || cal(U)_(n,n_0)^("start,RR") ||_(L_p).
$
By stationarity,
$|| tilde(R)_(n,n_0, op("aug"))^("mis,RR") ||_(L_p)
  = || R_(m, op("aug"))^("mis,RR") ||_(L_p)$.
Apply @thm:misadjustment with $n$ replaced by $m$ and
@lem:burn-startup-transfer. $square$

#corollary[
  *(Balanced-scale burned-in misadjustment rate.)*
  Assume the hypotheses of @thm:burn-misadjustment. Let
  $alpha = c thin n^(-1 slash 2)$, $m >= n slash 2$,
  $p = max(2, ceil(log n))$, and
  $q = max(4 p, ceil(log(e thin d)), 2)$. If, for some fixed $beta > 0$,
  $
  n_0 >= frac(beta p, c_("st") alpha a) log n,
  $
  and $n$ is large enough that $2 alpha <= alpha_("burn")(p,q)$, then
  $
  || R_(n,n_0, op("fin"))^("mis,RR") ||_(L_p)
    <= C_("burn,mis") thin "polylog"(n) thin n^(-1 slash 4).
  $ <eq:burn-mis-rate>
] <cor:burn-misadjustment-rate>

_Proof._ Apply @cor:misadjustment-rate with $n$ replaced by $m$, use
$m >= n slash 2$, and control the startup term by @cor:burn-log-startup.
$square$
// The startup term is $"polylog"(n) thin n^(-1 slash 4 - beta)$ at this scale.

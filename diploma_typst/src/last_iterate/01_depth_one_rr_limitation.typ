#import "../defs.typ": *

== A Depth-One RR Misadjustment Bound and Its Limitation

This subsection records why the depth-one route is insufficient. The final
Berry--Esseen assembly uses the depth-two Levin transfer.

A depth-one route would control
$
D_1^("mis, RR")
  = frac(sqrt(n), n - n_0) sum_(k=n_0)^(n-1)
    (2 J_k^((1, alpha)) - J_k^((1, 2 alpha))),
$
with centered part
$
D_(1, "c")^("mis, RR")
  := D_1^("mis, RR") - bb(E) D_1^("mis, RR")
$

Assume $2 alpha <= alpha_infinity$ and $2 alpha <= alpha_("inv")$.

By @lem:levin-prop-2,
$ bb(E)_pi lr([J_infinity^((1, alpha))]) = alpha Delta + O(alpha^2), $
and the linear term cancels in the RR combination. Hence
$
||bb(E) D_1^("mis, RR")|| <= C sqrt(n) thin alpha^2.
$

Define
$
Phi(p, alpha) := p^(3 slash 2) thin t_"mix"^(1 slash 2) / a
                + p^(1 slash 2) thin t_"mix"^(3 slash 2) sqrt(alpha slash a).
$
For $w in {alpha, 2 alpha}$, @lem:last-shifted-first-order gives
$
||u^top (T_n^((1, w)) - bb(E) T_n^((1, w)))||_(L_p)
  <= C ||u|| thin alpha thin Phi(p, alpha),
$
after increasing $C$. Since
$
u^top J_k^((1, w))
  = ((I - w overline(A))^(-top) u)^top T_k^((1, w)).
$
the inverse bound yields, uniformly in $k$,
$
||u^top lr((2 J_k^((1, alpha)) - J_k^((1, 2 alpha)))
     - bb(E) (2 J_k^((1, alpha)) - J_k^((1, 2 alpha))))||_(L_p)
  <= C ||u|| thin alpha thin Phi(p, alpha),
$
and PR averaging gives
$
||u^top D_(1, "c")^("mis, RR")||_(L_p)
  <= C ||u|| thin sqrt(n) thin alpha thin Phi(p, alpha)
  = O(sqrt(n) thin alpha).
$
Together with the bias estimate,
$
||u^top D_1^("mis, RR")||_(L_p)
  <= C ||u|| thin sqrt(n) thin alpha thin Phi(p, alpha)
    + C ||u|| thin sqrt(n) thin alpha^2.
$
At $alpha asymp n^(-1 slash 2)$ the centered term is $O(1)$, so this route does
not yield the target $n^(-1 slash 4)$ Berry--Esseen remainder.

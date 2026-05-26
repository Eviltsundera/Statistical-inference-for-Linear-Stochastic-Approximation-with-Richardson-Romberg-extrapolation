#import "../defs.typ": *

== A Depth-One RR Misadjustment Bound and Its Limitation

This subsection records the natural depth-one attempt and explains why it is
not used in the final Berry--Esseen assembly. The actual stationary and
burned-in theorems use the depth-two Levin transfer developed in the next
chapter.

The PR-averaged Richardson--Romberg expansion produces, after applying the
first deterministic-product perturbation step underlying Samsonov et al.
(2025, Proposition 9) separately at step sizes $alpha$ and $2 alpha$, a
depth-one "misadjustment" remainder
$
D_1^("mis, RR")
  = frac(sqrt(n), n - n_0) sum_(k=n_0)^(n-1)
    (2 J_k^((1, alpha)) - J_k^((1, 2 alpha))),
$
whose centered part must be controlled to feed into a Berry--Esseen statement
by this route. We write
$
D_(1, "c")^("mis, RR")
  := D_1^("mis, RR") - bb(E) D_1^("mis, RR")
$
for this centered statistic.

In this subsection assume explicitly that
$2 alpha <= alpha_infinity$ and $2 alpha <= alpha_("inv")$. The first
condition makes the one-stepsize last-iterate lemma admissible at both RR
levels; the second makes the shifted-to-unshifted transfer
$T_k^((1,w)) = (I - w overline(A)) J_k^((1,w))$ uniformly invertible for
$w in {alpha, 2 alpha}$.

The stationary bias is smaller than the fluctuation term. By the working form
@lem:levin-prop-2,
$ bb(E)_pi lr([J_infinity^((1, alpha))]) = alpha Delta + O(alpha^2), $
so the linear term $alpha Delta$ cancels in the RR-combination and the
per-iterate stationary RR bias is $O(alpha^2)$. Therefore the PR-scaled
stationary bias satisfies
$
||bb(E) D_1^("mis, RR")|| <= C sqrt(n) thin alpha^2.
$
What remains is the centered fluctuation.

Define
$
Phi(p, alpha) := p^(3 slash 2) thin t_"mix"^(1 slash 2) / a
                + p^(1 slash 2) thin t_"mix"^(3 slash 2) sqrt(alpha slash a).
$
Fix a deterministic direction $u$. The lemma applied at $alpha$ and at
$2 alpha$ gives, separately,
$
||u^top (T_n^((1, alpha)) - bb(E) T_n^((1, alpha)))||_(L_p)
  <= C ||u|| thin alpha thin Phi(p, alpha),
quad
||u^top (T_n^((1, 2 alpha)) - bb(E) T_n^((1, 2 alpha)))||_(L_p)
  <= C ||u|| thin alpha thin Phi(p, 2 alpha)
  <= C' ||u|| thin alpha thin Phi(p, alpha),
$
where $C' = sqrt(2) C$ absorbs the $sqrt(2)$-factor coming from the $2 alpha$ scaling. Combining the two by the triangle inequality and using the index-shift identity $T_k^((1, w)) = (I - w overline(A)) thin J_k^((1, w))$ gives
$
u^top J_k^((1, w))
  = ((I - w overline(A))^(-top) u)^top T_k^((1, w)).
$
The local inverse bound $|| (I - w overline(A))^(-1) || <= 2$ therefore holds
for $w in {alpha, 2 alpha}$ and yields
$
||u^top lr((2 J_k^((1, alpha)) - J_k^((1, 2 alpha)))
     - bb(E) (2 J_k^((1, alpha)) - J_k^((1, 2 alpha))))||_(L_p)
  <= C ||u|| thin alpha thin Phi(p, alpha),
$
uniformly in $k$. PR-averaging through $sqrt(n) / (n - n_0)$ and absorbing the constant therefore yields
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
At the optimal scale $alpha asymp n^(-1 slash 2)$ the centered-fluctuation
term is $O(1)$, whereas the stationary bias is $O(n^(-1 slash 2))$. Hence this
depth-one route still does not yield a useful Berry--Esseen remainder of order
$n^(-1 slash 4)$: the centered misadjustment must be controlled more sharply
to be subleading.

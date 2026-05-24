#import "../defs.typ": *

== Startup Transfer for Augmented-Chain Remainders

After the deterministic transient is removed, a startup discrepancy remains:
the finite-start perturbation variables $J^((ell,w))$ and $H^((2,w))$ start
from zero, whereas the stationary theorem uses the invariant law of the
augmented chain. This discrepancy is summed over the post-burn-in window.

For $w in {alpha, 2 alpha}$ write
$
Y_k^((w)) := (Z_(k + 1), J_k^((0,w)), J_k^((1,w)), J_k^((2,w))).
$
For $y = (z, j_0, j_1, j_2)$ and
$y' = (z', j'_0, j'_1, j'_2)$ define the depth-two augmented-chain cost
used by Levin et al. (2025, Appendix B.2, Eq. (49)):
$
c_(J,2)^((w))(y,y')
  &:= ||j_0 - j'_0|| + ||j_1 - j'_1|| + ||j_2 - j'_2|| \
  &quad + lr((||j_0|| + ||j'_0|| + ||j_1|| + ||j'_1||
        + ||j_2|| + ||j'_2|| + sqrt(w a) ||epsilon.alt||_infinity))
        thin 1_(z != z').
$ <eq:levin-depth-two-cost>
Levin et al. (2025, Appendix B.2, Proposition 5, with constants defined in
their Eq. (55)) prove the Wasserstein contraction for this cost. The
componentwise estimates in its proof imply that, under their Proposition-5
step-size restriction, two copies of $Y_k^((w))$ started from deterministic
states $y,y'$ can be coupled so that, for $ell = 0,1,2$,
$
|| J_k^((ell,w))(y) - J_k^((ell,w))(y') ||_(L_p)
  <= C_W thin p^(7 slash 2) thin t_"mix"^(5 slash 2)
     thin log^(3 slash 2)(1 slash (w a))
     thin exp(-w a k slash (12 p))
     thin c_(J,2)^((w))(y,y').
$ <eq:levin-depth-two-component-contraction>
Their Corollary 4 gives the corresponding invariant law
$Pi_(J,2,w)$ for $Y_k^((w))$.

The contraction above does not include $H^((2,w))$. Levin et al. (2025,
Appendix D.1, Proposition 9) prove the required one-trajectory moment bound
for $H^((2,w))$, using the representation
$
H_k^((2,w))
  = - w sum_(l = 1)^k Gamma_(l + 1:k)^((w))
        tilde(A)(Z_l) J_(l - 1)^((2,w)).
$ <eq:levin-H2-representation>
The invariant law for the full augmented state is obtained by a finite-past
construction. On a two-sided stationary version of the driving chain, start
the $J^((0,w))$, $J^((1,w))$, $J^((2,w))$, and $H^((2,w))$ recursions from
zero at time $-m$ and evaluate them at time $0$. Levin Proposition 5 gives the
$L_p$ Cauchy property for the $J$ coordinates. For the $H$ coordinate, use the
finite-past version of @eq:levin-H2-representation,
$
H_(0,m)^((2,w))
  := - w sum_(l = -m + 1)^0 Gamma_(l + 1:0)^((w))
        tilde(A)(Z_l) J_(l - 1,m)^((2,w)).
$
Levin Proposition 9 and the random-product stability estimate below make this
sequence Cauchy in $L_p$ for each fixed admissible $p$; its limit is denoted
$H_0^((2,w))$. Shifting the construction defines the stationary two-sided
process
$(Z_(k + 1), J_k^((0,w)), J_k^((1,w)), J_k^((2,w)), H_k^((2,w)))_(k in ZZ)$.
This is the full augmented-chain invariant law used below.

#lemma[
  *(Conditional product stability at a coupling time.)*
  Assume the hypotheses of @lem:burn-product-stability and let $T$ be an
  exact-coupling time for two copies of the base chain satisfying
  $bb(P)(T > r) <= C_T exp(-c_T r)$. Then, after decreasing the exponential
  constant, the following two estimates hold uniformly in $k$ and
  $w in {alpha, 2 alpha}$.

  *(i) Random restart.* If $(V_s)_(s >= 0)$ is adapted and
  $sup_s ||V_s||_(L_(2p)) <= B$, then
  $
  || Gamma_(T + 1:k)^((w)) V_T 1_(T <= k) ||_(L_p)
    <= C frac(p, w a) B exp(-c w a k slash p).
  $ <eq:burn-random-time-product>

  *(ii) Stable convolution.* If $(U_l)_(l >= 0)$ is adapted and
  $||U_l||_(L_(2p)) <= B exp(-c_0 w a l slash p)$ for all $l$, then
  $
  w sum_(l = 1)^k
    || Gamma_(l + 1:k)^((w)) U_l ||_(L_p)
    <= C frac(p, a) B exp(-c w a k slash p).
  $ <eq:burn-stable-convolution>
] <lem:burn-random-time-product>

_Proof._ The proof of Levin's product-stability estimate is conditional on
the past at the starting time, hence @eq:burn-product-stability may be applied
on each event $T = s$ with the same constants. For the first estimate,
Minkowski's inequality gives
$
|| Gamma_(T + 1:k)^((w)) V_T 1_(T <= k) ||_(L_p)
  <= C sum_(s = 0)^k
    exp(-c w a (k - s) slash p)
    || V_s 1_(T = s) ||_(L_p).
$
Holder's inequality and the exponential tail of $T$ give
$||V_s 1_(T=s)||_(L_p) <= B thin bb(P)(T=s)^(1 slash (2p))
 <= C B exp(-c_T s slash (2p))$. Since $w a$ is bounded above by the
admissible small-step constant, the convolution of
$exp(-c w a (k-s) slash p)$ and $exp(-c_T s slash (2p))$ is bounded by
$C (p slash (w a)) exp(-c w a k slash p)$ after decreasing $c$.
This proves @eq:burn-random-time-product.

For the second estimate, apply @eq:burn-product-stability at each deterministic
time $l$ and use the assumed decay of $U_l$:
$
w sum_(l = 1)^k
  exp(-c w a (k - l) slash p) B exp(-c_0 w a l slash p)
  <= C frac(p, a) B exp(-c' w a k slash p).
$
Renaming $c'$ as $c$ proves @eq:burn-stable-convolution. $square$

For startup transfer, however, one must compare two copies of this display.
The following lemma records the resulting full-state contraction. It is a
technical extension of the Levin Proposition-5 coupling: the $J^((0))$,
$J^((1))$, and $J^((2))$ coordinates use Levin Appendix B.2 directly, while
the $H^((2))$ coordinate is controlled by applying the random-product stability
estimate used inside Levin Appendix D.1, Proposition 9, to the difference of
two representations @eq:levin-H2-representation.

The threshold $alpha_("st")(p)$ was fixed above as the common ceiling for the
Levin depth-two startup contraction and the random-product stability estimate.
For $w in {alpha, 2 alpha}$ write
$
R_(k, op("fin"))^((w))
  := J_(k, op("fin"))^((1,w))
     + J_(k, op("fin"))^((2,w))
     + H_(k, op("fin"))^((2,w))
$
for the finite-start depth-two remainder, and define
$R_(k, op("aug"))^((w))$ analogously for a copy initialized from the invariant
law of the full augmented chain
$(Z_(k + 1), J_k^((0,w)), J_k^((1,w)), J_k^((2,w)), H_k^((2,w)))$.

#lemma[
  *(Full-state startup contraction for the depth-two augmented remainder.)*
  Assume *UGE 1*, $pi(epsilon.alt)=0$, $||epsilon.alt||_infinity < infinity$,
  the step-size restrictions of the Levin depth-two moment bounds, and
  $2 alpha <= alpha_("st")(p)$. There exist constants
  $c_("st") > 0$ and $C_("st") < infinity$, depending only on the problem
  constants and on the constants in @eq:levin-depth-two-component-contraction
  and Levin Proposition 9, such that for every $p >= 2$, every admissible
  $q >= 2$, every initial distribution $xi$ of the base chain with finite-start
  perturbation coordinates initialized at zero, every $w in {alpha, 2 alpha}$,
  and every $k >= 0$, the finite-start and stationary augmented remainders can
  be coupled so that
  $
  || R_(k, op("fin"))^((w)) - R_(k, op("aug"))^((w)) ||_(L_p)
    <= A_("st")(p,q,w) exp(-c_("st") w a k slash p),
  $ <eq:burn-startup-pointwise>
  where
  $
  A_("st")(p,q,w)
    := C_("st") (1 + d^(1 slash q)) p^8
       t_"mix"^5 frac(1, a) sqrt(w slash a)
       log^3(1 slash (w a)).
  $
] <lem:burn-full-startup>

_Proof._ Work on the exact-coupling probability space used in Levin Appendix
B.2. Let $T$ be the coupling time of the two base chains. By UGE,
$bb(P)(T > r) <= C rho^r$, and Proposition 5 gives, for
$ell in {0,1,2}$,
$
|| J_(k, op("fin"))^((ell,w)) - J_(k, op("aug"))^((ell,w)) ||_(L_p)
  <= C p^(7 slash 2) t_"mix"^(5 slash 2)
     log^(3 slash 2)(1 slash (w a))
     e^(-c w a k slash p)
     bb(E)^(1 slash p) [c_(J,2)^((w))(Y_0^("fin"),Y_0^("aug"))^p].
$ <eq:burn-J-startup-proof>
The finite-start $J$ coordinates are zero. The invariant copy is the
finite-past limit described above, so the elementary $J^((0,w))$ and
$J^((1,w))$ estimates and Levin Proposition 8 for $J^((2,w))$ give
$
bb(E)^(1 slash p) [c_(J,2)^((w))(Y_0^("fin"),Y_0^("aug"))^p]
  <= C (1 + d^(1 slash q)) p^(7 slash 2) t_"mix"^(5 slash 2)
     sqrt(w slash a) log^(3 slash 2)(1 slash (w a)).
$ <eq:burn-initial-cost-proof>
Thus the $J^((1,w))$ and $J^((2,w))$ parts of $R_k^((w))$ satisfy the
pointwise bound with
$
A_("J")(p,q,w)
  := C (1 + d^(1 slash q)) p^7 t_"mix"^5 sqrt(w slash a)
     log^3(1 slash (w a)).
$ <eq:burn-startup-J-scale>
Since $A_("J")(p,q,w) <= A_("st")(p,q,w)$ after enlarging $C_("st")$, these
parts already satisfy @eq:burn-startup-pointwise.

It remains to control $H^((2,w))$. We shall repeatedly use the one-trajectory
bound, valid for the finite-start and finite-past stationary copies,
$
|| H_s^((2,w)) ||_(L_(2p))
  <= B_H(p,q,w)
  := C (1 + d^(1 slash q)) p^(7 slash 2) t_"mix"^(5 slash 2)
        w^(3 slash 2) log^(3 slash 2)(1 slash (w a)),
$ <eq:burn-H2-one-trajectory-scale>
where $C$ is allowed to change when $p$ is replaced by $2p$. On the event
$T <= k$, the base chains are identical after time $T$, and subtracting
@eq:levin-H2-representation from time $T$ onward gives
$
Delta H_k^((2,w))
  = Gamma_(T + 1:k)^((w)) Delta H_T^((2,w))
    - w sum_(l = T + 1)^k Gamma_(l + 1:k)^((w))
        tilde(A)(Z_l) Delta J_(l - 1)^((2,w)),
$ <eq:burn-H2-coupled-diff>
where $Delta$ denotes finite-start minus stationary.

On the bad event $T > k$, Holder inequality gives the explicit reduction
$
|| Delta H_k^((2,w)) 1_(T > k) ||_(L_p)
  <= lr((
       || H_(k, op("fin"))^((2,w)) ||_(L_(2p))
       + || H_(k, op("aug"))^((2,w)) ||_(L_(2p))
     )) thin bb(P)(T > k)^(1 slash (2p)).
$
The bound @eq:burn-H2-one-trajectory-scale controls both one-trajectory terms
uniformly in $k$, and UGE gives
$bb(P)(T > k)^(1 slash (2p)) <= C e^(-c k slash p)$. Therefore
$
|| Delta H_k^((2,w)) 1_(T > k) ||_(L_p)
  <= C B_H(p,q,w)
       e^(-c k slash p).
$ <eq:burn-H2-bad-event>
Since $w a <= 1$, this is bounded by the right-hand side of
@eq:burn-startup-pointwise after absorbing a fixed power of $a^(-1)$ into
$C_("st")$.

For the first term in @eq:burn-H2-coupled-diff, apply
@lem:burn-random-time-product with $V_s = Delta H_s^((2,w))$. The input
$sup_s ||V_s||_(L_(2p)) <= C B_H(p,q,w)$ follows from
@eq:burn-H2-one-trajectory-scale for the two coupled copies. Hence
$
|| Gamma_(T + 1:k)^((w)) Delta H_T^((2,w)) 1_(T <= k) ||_(L_p)
  <= C frac(p, w a) B_H(p,q,w) e^(-c w a k slash p)
  <= C A_("st")(p,q,w) e^(-c w a k slash p).
$ <eq:burn-H2-initial-term>
For the convolution term, set
$U_l := tilde(A)(Z_l) Delta J_(l - 1)^((2,w))$. Applying
@eq:burn-J-startup-proof with moment order $2p$ gives
$
|| U_l ||_(L_(2p))
  <= C A_("J")(2p,q,w) exp(-c w a l slash p),
$
where replacing $l - 1$ by $l$ only changes the constant. The stable
convolution estimate @eq:burn-stable-convolution yields
$
& w sum_(l = 1)^k
  || Gamma_(l + 1:k)^((w)) tilde(A)(Z_l)
     Delta J_(l - 1)^((2,w)) ||_(L_p) \
&quad <= C frac(p, a) A_("J")(2p,q,w) e^(-c w a k slash p) \
&quad <= C A_("st")(p,q,w) e^(-c w a k slash p).
$ <eq:burn-H2-convolution-term>
The additional factor $p slash a$ is the reason for the enlarged definition of
$A_("st")$ relative to the pure $J$-coordinate scale $A_("J")$.
Combining
@eq:burn-H2-bad-event, @eq:burn-H2-initial-term, and
@eq:burn-H2-convolution-term gives the same startup bound for
$H^((2,w))$. Adding the already controlled $J^((1,w))$ and $J^((2,w))$ pieces
proves @eq:burn-startup-pointwise. $square$

Define the RR startup discrepancy accumulated over the burned-in window by
$
cal(U)_(n,n_0)^("start,RR")
  := frac(1, sqrt(m)) sum_(k = n_0)^(n - 1)
    lr([
      2 lr((R_(k, op("fin"))^((alpha))
             - R_(k, op("aug"))^((alpha))))
      - lr((R_(k, op("fin"))^((2 alpha))
             - R_(k, op("aug"))^((2 alpha))))
    ]).
$ <eq:burn-startup-discrepancy>

#lemma[
  *(Accumulated startup transfer.)*
  Under @lem:burn-full-startup, if
  $alpha, 2 alpha in (0, alpha_infinity]$ and $alpha a <= 1 slash 4$, then
  for every $p >= 2$ and admissible $q >= 2$,
  $
  || cal(U)_(n,n_0)^("start,RR") ||_(L_p)
    <= frac(C_("start,RR") p thin A_("st")(p,q,alpha),
             alpha a sqrt(m))
       exp(-c_("st") alpha a n_0 slash p).
  $ <eq:burn-startup-transfer>
] <lem:burn-startup-transfer>

_Proof._ Apply @eq:burn-startup-pointwise at $w = alpha$ and $w = 2 alpha$,
then use the triangle inequality:
$
|| cal(U)_(n,n_0)^("start,RR") ||_(L_p)
  <= frac(1, sqrt(m)) sum_(k = n_0)^(n - 1)
    lr((2 A_("st")(p,q,alpha) e^(-c_("st") alpha a k slash p)
      + A_("st")(p,q,2 alpha) e^(-2 c_("st") alpha a k slash p))).
$
Since $alpha a <= 1 slash 4$,
$A_("st")(p,q,2 alpha) <= C A_("st")(p,q,alpha)$ and
$1 - exp(-c_("st") alpha a slash p) >= C^(-1) alpha a slash p$. Extending the geometric
sum to infinity gives @eq:burn-startup-transfer. $square$

#corollary[
  *(Mixing-scale burn-in removes the startup discrepancy.)*
  If, for some $beta > 0$,
  $
  n_0 >= frac(beta p, c_("st") alpha a) log n,
  $ <eq:burn-log-startup-condition>
  then
  $
  || cal(U)_(n,n_0)^("start,RR") ||_(L_p)
    <= frac(C_("start,RR") p thin A_("st")(p,q,alpha),
             alpha a sqrt(m)) n^(-beta).
  $ <eq:burn-log-startup-bound>
  At the balanced scale $alpha = c n^(-1 slash 2)$, with
  $m >= n slash 2$ and $p, q$ logarithmic in $n$, this is
  $"polylog"(n) thin n^(-1 slash 4 - beta)$.
  Thus the Berry--Esseen choice $p asymp log n$ requires a mixing-scale
  burn-in with logarithmic-square factor,
  $n_0$ of order $(alpha a)^(-1) log^2 n$, under this $L_p$ contraction.
] <cor:burn-log-startup>

_Proof._ Substitute @eq:burn-log-startup-condition into
@eq:burn-startup-transfer. For the balanced scale,
$A_("st")(p,q,alpha) = "polylog"(n) thin alpha^(1 slash 2)$, hence
$A_("st")(p,q,alpha) slash (alpha sqrt(m))
= O("polylog"(n) n^(-1 slash 4))$. $square$


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
The direct $J$-coordinate contraction, its cost
@eq:levin-depth-two-cost, and the invariant $J$ law are recorded in
@lem:levin-prop-5-component and @lem:levin-invariant-depth-two-law. The
$H^((2,w))$ one-trajectory bound and representation
@eq:levin-H2-representation are recorded in @lem:levin-prop-9.
The invariant law for the full augmented state is the finite-past limit
constructed in @lem:finite-past-full-augmented-state. In particular, for every
$w in {alpha, 2 alpha}$ satisfying the admissibility condition, the stationary
copy
$
(Z_(k + 1), J_k^((0,w)), J_k^((1,w)), J_k^((2,w)), H_k^((2,w)))_(k in ZZ)
$
exists, the $H^((2,w))$ coordinate is obtained as an $L_p$ limit of finite-past
truncations, and the limiting coordinate inherits @lem:levin-prop-9. This is
the full augmented-chain invariant law
used below.

#lemma[
  *(Conditional product stability at a coupling time.)*
  Work on a coupling space with joint filtration $(cal(G)_s)_(s >= 0)$ for two
  copies of the base chain and their perturbation variables. Let $T$ be an
  integer-valued $cal(G)_s$-stopping time such that the two base chains are
  exactly coupled from time $T$ onward and, uniformly over the initial states
  used below,
  $
  bb(P)(T > r) <= C_T exp(-c_T r),
  quad r >= 0.
  $ <eq:coupling-time-tail>
  Products are taken along the post-coupling chain; we use the convention
  $Gamma_(k + 1:k)^((w)) = I$ and empty sums are zero. Assume the hypotheses of
  @lem:burn-product-stability. Then, after increasing constants and decreasing
  the exponential constant, the following two estimates hold for every
  $p >= 2$, every $k >= 0$, and every $w in {alpha, 2 alpha}$ for which the
  product-stability input is admissible and $w a <= 1$. In the applications
  below this small-step condition follows from $2 alpha <= alpha_("burn")(p,q)$.
  The constants depend only on the product-stability constants and on
  $C_T,c_T$ (and on $c_0$ in part (ii)), but not on $k,p,w,B$.

  *(i) Random restart.* If $(V_s)_(s >= 0)$ is $cal(G)_s$-adapted and
  $sup_s ||V_s||_(L_(2p)) <= B$, then $V_T 1_(T <= k)$ is well-defined by
  $sum_(s = 0)^k V_s 1_(T = s)$ and
  $
  || Gamma_(T + 1:k)^((w)) V_T 1_(T <= k) ||_(L_p)
    <= C frac(p, w a) B exp(-c w a k slash p).
  $ <eq:burn-random-time-product>

  *(ii) Stable convolution.* If $c_0 > 0$, $(U_l)_(l >= 0)$ is
  $cal(G)_l$-adapted, and
  $||U_l||_(L_(2p)) <= B exp(-c_0 w a l slash p)$ for all $l$, then
  $
  w sum_(l = 1)^k
    || Gamma_(l + 1:k)^((w)) U_l ||_(L_p)
    <= C frac(p, a) B exp(-c w a k slash p).
  $ <eq:burn-stable-convolution>
] <lem:burn-random-time-product>

_Proof._ The conditional form needed here is part of the technical input
@lem:burn-product-stability. Fix deterministic times $s <= k$ and a
$cal(G)_s$-measurable vector $W_s$. Conditional on $cal(G)_s$, the future of
either coupled base chain starts from its current state, and the
product-stability constants are uniform over that state. Applying
@eq:burn-product-stability-conditional gives
$
bb(E) lr([
  || Gamma_(s + 1:k)^((w)) W_s ||^p mid cal(G)_s
])^(1 slash p)
  <= C_("prod") exp(-c_("prod") w a (k - s) slash p) || W_s ||.
$ <eq:conditional-product-pointwise>
Integrating this conditional inequality gives
$
|| Gamma_(s + 1:k)^((w)) W_s ||_(L_p)
  <= C_("prod") exp(-c_("prod") w a (k - s) slash p)
     || W_s ||_(L_p).
$ <eq:conditional-product-stability>
For $s = k$ this is interpreted with the empty product
$Gamma_(k + 1:k)^((w)) = I$; after increasing $C_("prod")$ the same display is
trivial.

Since $T$ is a stopping time, $1_(T = s)$ is $cal(G)_s$-measurable. Thus
$V_s 1_(T = s)$ is an admissible input in
@eq:conditional-product-stability. By Minkowski's inequality and the
decomposition
$Gamma_(T + 1:k)^((w)) V_T 1_(T <= k)
  = sum_(s = 0)^k Gamma_(s + 1:k)^((w)) V_s 1_(T = s)$,
$
|| Gamma_(T + 1:k)^((w)) V_T 1_(T <= k) ||_(L_p)
  <= C sum_(s = 0)^k
    exp(-c w a (k - s) slash p)
    || V_s 1_(T = s) ||_(L_p).
$
Holder's inequality gives
$
|| V_s 1_(T = s) ||_(L_p)
  <= ||V_s||_(L_(2p)) bb(P)(T = s)^(1 slash (2p))
  <= B thin bb(P)(T = s)^(1 slash (2p)).
$
The tail bound @eq:coupling-time-tail implies, for $s >= 1$,
$bb(P)(T = s) <= bb(P)(T > s - 1) <= C_T e^(-c_T (s - 1))$, while the case
$s = 0$ is absorbed into the same bound after changing the constant. Since
$p >= 2$, $C_T^(1 slash (2p))$ is bounded by a constant depending only on
$C_T$, and therefore
$
bb(P)(T = s)^(1 slash (2p)) <= C exp(-c_T s slash (4 p)).
$
The small-step condition $w a <= 1$ gives, after decreasing $c_T$,
$exp(-c_T s slash (4 p)) <= exp(-c_T w a s slash (4 p))$. Hence
$
sum_(s = 0)^k
  exp(-c w a (k - s) slash p)
  exp(-c_T w a s slash (4 p))
  <= C frac(p, w a) exp(-c' w a k slash p).
$
Combining the last three displays proves @eq:burn-random-time-product after
renaming $c'$ as $c$.

For the second estimate, each $U_l$ is $cal(G)_l$-measurable, so
@eq:conditional-product-stability gives, including the empty-product case
$l = k$,
$
|| Gamma_(l + 1:k)^((w)) U_l ||_(L_p)
  <= C exp(-c w a (k - l) slash p)
     || U_l ||_(L_p)
  <= C exp(-c w a (k - l) slash p)
     B exp(-c_0 w a l slash p).
$
Therefore
$
w sum_(l = 1)^k
  || Gamma_(l + 1:k)^((w)) U_l ||_(L_p)
  <= C w B sum_(l = 1)^k
     exp(-c w a (k - l) slash p)
     exp(-c_0 w a l slash p).
$
The geometric convolution is bounded by
$C (p slash (w a)) exp(-c' w a k slash p)$, with $C,c'$ depending only on
$c,c_0$. Multiplication by $w$ gives the factor $p slash a$.
Renaming $c'$ as $c$ proves @eq:burn-stable-convolution. $square$

For startup transfer, however, one must compare two copies of this display.
The following lemma records the resulting full-state contraction. It is a
local technical extension of @lem:levin-prop-5-component: the $J^((0))$,
$J^((1))$, and $J^((2))$ coordinates use the imported $J$ contraction, while
the $H^((2))$ coordinate is controlled by applying the local random-product
stability estimate @lem:burn-random-time-product to the difference of two
representations @eq:levin-H2-representation.

The threshold $alpha_("st")(p)$ was fixed above as the common ceiling for the
local depth-two startup contraction and the random-product stability estimate.
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
  and $0 < alpha$. There exist constants
  $c_("st") > 0$ and $C_("st") < infinity$, depending only on the problem
  constants and on the constants in @eq:levin-depth-two-component-contraction
  and @lem:levin-prop-9, such that for every $p >= 2$, every
  $q >= 2$ satisfying $p <= q slash 4$ and
  $2 alpha <= alpha_("burn")(p,q)$, every initial distribution $xi$ of the
  base chain with finite-start perturbation coordinates initialized at zero,
  every $w in {alpha, 2 alpha}$, and every $k >= 0$, the finite-start and
  stationary augmented remainders can be coupled so that
  $
  || R_(k, op("fin"))^((w)) - R_(k, op("aug"))^((w)) ||_(L_p)
    <= A_("st")(p,q,w) exp(-c_("st") w a k slash p),
  $ <eq:burn-startup-pointwise>
  where
  $
  A_("st")(p,q,w)
    := C_("st") (1 + d^(1 slash q))
       lr(p^7 + frac(p^8, a))
       t_"mix"^5 sqrt(w slash a)
       log^3(1 slash (w a)).
  $
] <lem:burn-full-startup>

_Proof._ Couple a finite-start copy, whose base chain has initial law $xi$ and
whose perturbation coordinates are zero, with an augmented stationary copy
drawn from the full invariant law constructed in
@lem:finite-past-full-augmented-state. Let
$
Delta J_k^((ell,w))
  := J_(k, op("fin"))^((ell,w)) - J_(k, op("aug"))^((ell,w)),
quad
Delta H_k^((2,w))
  := H_(k, op("fin"))^((2,w)) - H_(k, op("aug"))^((2,w)).
$
Use the exact-coupling probability space behind
@lem:levin-prop-5-component for the two base chains, and let $T$ be their
coupling time. UGE gives the uniform tail
$bb(P)(T > r) <= C_T exp(-c_T r)$, with constants independent of $xi$. All
constants below are uniform over the initial law because
@eq:levin-depth-two-component-contraction and @lem:burn-product-stability are
uniform over the current base state.

*The $J^((1)) + J^((2))$ part.* Conditional on the initial augmented states,
@lem:levin-prop-5-component and the coordinate projection of the cost
@eq:levin-depth-two-cost give, for $ell in {0,1,2}$,
$
|| J_(k, op("fin"))^((ell,w)) - J_(k, op("aug"))^((ell,w)) ||_(L_p)
  <= C p^(7 slash 2) t_"mix"^(5 slash 2)
     log^(3 slash 2)(1 slash (w a))
     e^(-c w a k slash p)
     bb(E)^(1 slash p) [c_(J,2)^((w))(Y_0^("fin"),Y_0^("aug"))^p].
$ <eq:burn-J-startup-proof>
The finite-start $J$ coordinates are zero. For the augmented stationary copy,
the finite-past construction of @lem:finite-past-full-augmented-state, the
elementary stationary bounds for $J^((0,w))$ and $J^((1,w))$, and
@lem:levin-prop-8 for $J^((2,w))$ give
$
bb(E)^(1 slash p) [c_(J,2)^((w))(Y_0^("fin"),Y_0^("aug"))^p]
  <= C (1 + d^(1 slash q)) p^(7 slash 2) t_"mix"^(5 slash 2)
     sqrt(w slash a) log^(3 slash 2)(1 slash (w a)).
$ <eq:burn-initial-cost-proof>
Here the factor $1 + d^(1 slash q)$ also covers the indicator term in
@eq:levin-depth-two-cost, and the estimate is uniform over the initial law
$xi$ because only the stationary copy contributes to the initial cost.
Combining @eq:burn-J-startup-proof and @eq:burn-initial-cost-proof gives
$
A_("J")(p,q,w)
  := C (1 + d^(1 slash q)) p^7 t_"mix"^5 sqrt(w slash a)
     log^3(1 slash (w a)).
$ <eq:burn-startup-J-scale>
Thus
$
|| Delta J_k^((1,w)) ||_(L_p) + || Delta J_k^((2,w)) ||_(L_p)
  <= A_("J")(p,q,w) exp(-c w a k slash p).
$ <eq:burn-J12-startup-bound>
The $p^7$ summand in $A_("st")$ is exactly this pure $J$-coordinate scale.

*The $H^((2))$ part.* This is the local extension beyond
@lem:levin-prop-5-component.
The proof uses $L_(2p)$ estimates; this is why the statement assumes
$p <= q slash 4$. With moment order $2p$, @lem:levin-prop-9 and Fatou's
lemma for the stationary finite-past limit give, for both finite-start and
augmented stationary copies,
$
|| H_s^((2,w)) ||_(L_(2p))
  <= B_H(p,q,w)
  := C (1 + d^(1 slash q)) p^(7 slash 2) t_"mix"^(5 slash 2)
        w^(3 slash 2) log^(3 slash 2)(1 slash (w a)),
$ <eq:burn-H2-one-trajectory-scale>
where the replacement of $p$ by $2p$ is absorbed into $C$ and the displayed
power of $p$.
We record the scale comparisons used below. Since $w a <= 1$, $p >= 2$,
$t_"mix" >= 1$, and $log(1 slash (w a)) >= log 2$, the constants convention
for fixed powers of $a^(-1)$ gives
$
&B_H(p,q,w) <= C A_("J")(p,q,w), \
&frac(p, w a) B_H(p,q,w) <= C A_("J")(p,q,w), \
&frac(p, a) A_("J")(2p,q,w)
  <= C (1 + d^(1 slash q)) frac(p^8, a)
     t_"mix"^5 sqrt(w slash a) log^3(1 slash (w a)).
$ <eq:burn-startup-scale-audit>
Thus $A_("st")$ is the sum of the $J$ scale $A_("J")$ and the convolution
scale $(p slash a) A_("J")(2p,q,w)$; the bad-coupling and random
restart $H^((2))$ terms do not introduce additional powers of
$p$, $t_"mix"$, $w$, or $a^(-1)$.

On the event $T <= k$, the two base chains are identical after time $T$.
Restarting the representation @eq:levin-H2-representation at time $T$ gives
for each copy
$
H_k^((2,w))
  = Gamma_(T + 1:k)^((w)) H_T^((2,w))
    - w sum_(l = T + 1)^k Gamma_(l + 1:k)^((w))
        tilde(A)(Z_l) J_(l - 1)^((2,w)),
$
where the product is the identity and the sum is empty when $T = k$.
Subtracting the two restarted representations gives
$
Delta H_k^((2,w))
  = Gamma_(T + 1:k)^((w)) Delta H_T^((2,w))
    - w sum_(l = T + 1)^k Gamma_(l + 1:k)^((w))
        tilde(A)(Z_l) Delta J_(l - 1)^((2,w)),
$ <eq:burn-H2-coupled-diff>
on $\{T <= k\}$.

On the bad event $\{T > k\}$, Holder's inequality gives
$
|| Delta H_k^((2,w)) 1_(T > k) ||_(L_p)
  <= lr((
       || H_(k, op("fin"))^((2,w)) ||_(L_(2p))
       + || H_(k, op("aug"))^((2,w)) ||_(L_(2p))
     )) thin bb(P)(T > k)^(1 slash (2p)).
$
The bound @eq:burn-H2-one-trajectory-scale controls both one-trajectory terms
uniformly in $k$, while @eq:coupling-time-tail gives
$bb(P)(T > k)^(1 slash (2p)) <= C e^(-c k slash p)$. Therefore
$
|| Delta H_k^((2,w)) 1_(T > k) ||_(L_p)
  <= C B_H(p,q,w)
       e^(-c k slash p).
$ <eq:burn-H2-bad-event>
Since $w a <= 1$, $e^(-c k slash p) <= e^(-c w a k slash p)$, and
@eq:burn-startup-scale-audit puts this term at the target scale.

For the restart term on $\{T <= k\}$, apply the random-time product estimate
@lem:burn-random-time-product with the adapted process
$V_s = Delta H_s^((2,w))$. The required input is exactly where the
$L_(2p)$ norm is used:
$
sup_s || V_s ||_(L_(2p))
  <= || H_(s, op("fin"))^((2,w)) ||_(L_(2p))
     + || H_(s, op("aug"))^((2,w)) ||_(L_(2p))
  <= C B_H(p,q,w).
$
Thus
$
|| Gamma_(T + 1:k)^((w)) Delta H_T^((2,w)) 1_(T <= k) ||_(L_p)
  <= C frac(p, w a) B_H(p,q,w) e^(-c w a k slash p)
  <= C A_("st")(p,q,w) e^(-c w a k slash p).
$ <eq:burn-H2-initial-term>
For the convolution term in @eq:burn-H2-coupled-diff, set
$U_l := tilde(A)(Z_l) Delta J_(l - 1)^((2,w))$. This process is adapted to
the joint filtration at time $l$. Applying @eq:burn-J-startup-proof with
moment order $2p$ is allowed by $p <= q slash 4$ and gives
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
The last line is the only place where the $p^8 slash a$ summand in
$A_("st")$ is needed.
Combining the bad-event, restart, and convolution estimates gives
$
|| Delta H_k^((2,w)) ||_(L_p)
  <= C A_("st")(p,q,w) e^(-c w a k slash p).
$ <eq:burn-H2-startup-bound>
Finally,
$
R_(k, op("fin"))^((w)) - R_(k, op("aug"))^((w))
  = Delta J_k^((1,w)) + Delta J_k^((2,w)) + Delta H_k^((2,w)).
$
The triangle inequality, @eq:burn-J12-startup-bound, and
@eq:burn-H2-startup-bound prove @eq:burn-startup-pointwise after renaming the
constant and the exponential rate. The displayed $A_("st")$ is the only
allowed dependence on $p,q,w$; all other factors are fixed problem constants,
uniformly over $xi$. $square$

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
  $0 < alpha$, then
  for every $p >= 2$ and every $q >= 2$ satisfying $p <= q slash 4$ and
  $2 alpha <= alpha_("burn")(p,q)$,
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

#import "../defs.typ": *

== Startup Transfer for Augmented-Chain Remainders

The remaining startup discrepancy is the difference between the finite-start
and stationary augmented-chain remainders.
// The finite-start perturbation variables $J^((ell,w))$ and $H^((2,w))$ start
// from zero, whereas the stationary theorem uses the invariant law of the
// augmented chain. This discrepancy is summed over the post-burn-in window.

For $w in {alpha, 2 alpha}$ write
$
Y_k^((w)) := (Z_(k + 1), J_k^((0,w)), J_k^((1,w)), J_k^((2,w))).
$
For every admissible $w in {alpha, 2 alpha}$, let
$
(Z_(k + 1), J_k^((0,w)), J_k^((1,w)), J_k^((2,w)), H_k^((2,w)))_(k in ZZ)
$
be the stationary copy from @lem:finite-past-full-augmented-state.
// The direct $J$-coordinate contraction, its cost @eq:levin-depth-two-cost, and
// the invariant $J$ law are recorded in @lem:levin-prop-5-component and
// @lem:levin-invariant-depth-two-law. The $H^((2,w))$ one-trajectory bound and
// representation @eq:levin-H2-representation are recorded in @lem:levin-prop-9.
// The $H^((2,w))$ coordinate is obtained as an $L_p$ limit of finite-past
// truncations, and the limiting coordinate inherits @lem:levin-prop-9.

#lemma[
  *(Conditional product stability at a coupling time.)*
  On a coupling space with joint filtration $(cal(G)_s)_(s >= 0)$, let $T$ be
  the integer-valued exact-coupling stopping time of two base chains. Assume,
  uniformly over the initial states used below,
  $
  bb(P)(T > r) <= C_T exp(-c_T r),
  quad r >= 0.
  $ <eq:coupling-time-tail>
  Products are taken along the post-coupling chain, with
  $Gamma_(k + 1:k)^((w)) = I$. Then, for every $p >= 2$, $k >= 0$, and
  $w in {alpha, 2 alpha}$ for which @lem:burn-product-stability applies and
  $w a <= 1$:
  // In the applications below this small-step condition follows from
  // $2 alpha <= alpha_("burn")(p,q)$. The constants depend only on the
  // product-stability constants and on $C_T,c_T$ (and on $c_0$ in part (ii)),
  // but not on $k,p,w,B$.

  *(i) Random restart.* If $(V_s)_(s >= 0)$ is $cal(G)_s$-adapted and
  $sup_s ||V_s||_(L_(2p)) <= B$, then
  $
  || Gamma_(T + 1:k)^((w)) V_T 1_(T <= k) ||_(L_p)
    <= C frac(p, w a) B
       exp(- frac(c w a k, p)).
  $ <eq:burn-random-time-product>

  *(ii) Stable convolution.* If $c_0 > 0$, $(U_l)_(l >= 0)$ is
  $cal(G)_l$-adapted, and, for all $l$,
  $
  ||U_l||_(L_(2p))
    <= B exp(- frac(c_0 w a l, p)),
  $
  then
  $
  w sum_(l = 1)^k
    || Gamma_(l + 1:k)^((w)) U_l ||_(L_p)
    <= C frac(p, a) B
       exp(- frac(c w a k, p)).
  $ <eq:burn-stable-convolution>
] <lem:burn-random-time-product>

_Proof._ For $s <= k$ and $cal(G)_s$-measurable $W_s$,
$
bb(E) lr([
  || Gamma_(s + 1:k)^((w)) W_s ||^p | cal(G)_s
])^(frac(1, p))
  <= C_("prod") exp(- frac(c_("prod") w a (k - s), p))
     || W_s ||.
$ <eq:conditional-product-pointwise>
and therefore
$
|| Gamma_(s + 1:k)^((w)) W_s ||_(L_p)
  <= C_("prod") exp(- frac(c_("prod") w a (k - s), p))
     || W_s ||_(L_p).
$ <eq:conditional-product-stability>
For $V_s 1_(T = s)$, which is $cal(G)_s$-measurable,
$Gamma_(T + 1:k)^((w)) V_T 1_(T <= k)
  = sum_(s = 0)^k Gamma_(s + 1:k)^((w)) V_s 1_(T = s)$,
$
|| Gamma_(T + 1:k)^((w)) V_T 1_(T <= k) ||_(L_p)
  <= C sum_(s = 0)^k
    exp(- frac(c w a (k - s), p))
    || V_s 1_(T = s) ||_(L_p).
$
$
|| V_s 1_(T = s) ||_(L_p)
  <= ||V_s||_(L_(2p)) bb(P)(T = s)^(1 slash (2p))
  <= B thin bb(P)(T = s)^(1 slash (2p)).
$
$
bb(P)(T = s)^(1 slash (2p))
  <= C exp(- frac(c_T s, 4 p)).
$
$
sum_(s = 0)^k
  exp(- frac(c w a (k - s), p))
  exp(- frac(c_T w a s, 4 p))
  <= C frac(p, w a)
     exp(- frac(c' w a k, p)).
$
This proves @eq:burn-random-time-product.

For (ii), @eq:conditional-product-stability gives
$
|| Gamma_(l + 1:k)^((w)) U_l ||_(L_p)
  <= C exp(- frac(c w a (k - l), p))
     || U_l ||_(L_p)
  <= C exp(- frac(c w a (k - l), p))
     B exp(- frac(c_0 w a l, p)).
$
$
w sum_(l = 1)^k
  || Gamma_(l + 1:k)^((w)) U_l ||_(L_p)
  <= C w B sum_(l = 1)^k
     exp(- frac(c w a (k - l), p))
     exp(- frac(c_0 w a l, p)).
$
The last sum is bounded by
$
C frac(p, w a) exp(- frac(c' w a k, p)),
$
hence
@eq:burn-stable-convolution. $square$

// The following lemma records the resulting full-state contraction.
// This is a local technical extension of @lem:levin-prop-5-component: the
// $J^((0))$, $J^((1))$, and $J^((2))$ coordinates use the $J$
// contraction, while the $H^((2))$ coordinate is controlled by applying the
// local random-product stability estimate @lem:burn-random-time-product to the
// difference of two representations @eq:levin-H2-representation.

// The product part of $alpha_("burn")(p,q)$ is used below through
// $alpha_("st")(p)$; the Levin depth-two ceilings enter through
// $alpha_("stat")(q)$.
For $w in {alpha, 2 alpha}$ write
$
R_(k, op("fin"))^((w))
  := J_(k, op("fin"))^((1,w))
     + J_(k, op("fin"))^((2,w))
     + H_(k, op("fin"))^((2,w))
$
and
$
R_(k, op("aug"))^((w))
  := J_(k, op("aug"))^((1,w))
     + J_(k, op("aug"))^((2,w))
     + H_(k, op("aug"))^((2,w)).
$
Here the augmented copy is initialized from the invariant law of
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

_Proof._ Couple the finite-start copy to the invariant full augmented chain of
@lem:finite-past-full-augmented-state. Let
$
Delta J_k^((ell,w))
  := J_(k, op("fin"))^((ell,w)) - J_(k, op("aug"))^((ell,w)),
quad
Delta H_k^((2,w))
  := H_(k, op("fin"))^((2,w)) - H_(k, op("aug"))^((2,w)).
$
Let $T$ be the exact-coupling time from @lem:levin-prop-5-component. UGE gives
$bb(P)(T > r) <= C_T exp(-c_T r)$, uniformly in $xi$.
// All constants below are uniform over the initial law because
// @eq:levin-depth-two-component-contraction and @lem:burn-product-stability are
// uniform over the current base state.

*The $J^((1)) + J^((2))$ part.* By @lem:levin-prop-5-component and
@eq:levin-depth-two-cost, for $ell in {0,1,2}$,
$
|| J_(k, op("fin"))^((ell,w)) - J_(k, op("aug"))^((ell,w)) ||_(L_p)
  <= C p^(7 slash 2) t_"mix"^(5 slash 2)
     log^(3 slash 2)(1 slash (w a))
     e^(-c w a k slash p)
     bb(E)^(1 slash p) [c_(J,2)^((w))(Y_0^("fin"),Y_0^("aug"))^p].
$ <eq:burn-J-startup-proof>
The finite-start $J$ coordinates are zero, and the stationary copy satisfies
$
bb(E)^(1 slash p) [c_(J,2)^((w))(Y_0^("fin"),Y_0^("aug"))^p]
  <= C (1 + d^(1 slash q)) p^(7 slash 2) t_"mix"^(5 slash 2)
     sqrt(w slash a) log^(3 slash 2)(1 slash (w a)).
$ <eq:burn-initial-cost-proof>
// This uses @lem:finite-past-full-augmented-state, the elementary stationary
// bounds for $J^((0,w))$ and $J^((1,w))$, and @lem:levin-prop-8 for
// $J^((2,w))$.
// Here the factor $1 + d^(1 slash q)$ also covers the indicator term in
// @eq:levin-depth-two-cost, and the estimate is uniform over the initial law
// $xi$ because only the stationary copy contributes to the initial cost.
Put
$
A_("J")(p,q,w)
  := C (1 + d^(1 slash q)) p^7 t_"mix"^5 sqrt(w slash a)
     log^3(1 slash (w a)).
$ <eq:burn-startup-J-scale>
$
|| Delta J_k^((1,w)) ||_(L_p) + || Delta J_k^((2,w)) ||_(L_p)
  <= A_("J")(p,q,w) exp(-c w a k slash p).
$ <eq:burn-J12-startup-bound>
// The $p^7$ summand in $A_("st")$ is exactly this pure $J$-coordinate scale.

*The $H^((2))$ part.* By @lem:levin-prop-9 and Fatou,
$
|| H_s^((2,w)) ||_(L_(2p))
  <= B_H(p,q,w)
  := C (1 + d^(1 slash q)) p^(7 slash 2) t_"mix"^(5 slash 2)
        w^(3 slash 2) log^(3 slash 2)(1 slash (w a)),
$ <eq:burn-H2-one-trajectory-scale>
// The proof uses $L_(2p)$ estimates; this is why the statement assumes
// $p <= q slash 4$.
$
&B_H(p,q,w) <= C A_("J")(p,q,w), \
&frac(p, w a) B_H(p,q,w)
  <= C frac(p, a) A_("J")(2p,q,w), \
&frac(p, a) A_("J")(2p,q,w)
  <= C (1 + d^(1 slash q)) frac(p^8, a)
     t_"mix"^5 sqrt(w slash a) log^3(1 slash (w a)).
$ <eq:burn-startup-scale-audit>
// Thus $A_("st")$ is the sum of the pure $J$ scale $A_("J")$ and the
// $H^((2))$ restart/convolution scale $(p slash a) A_("J")(2p,q,w)$.

On $\{T <= k\}$, @eq:levin-H2-representation gives
$
H_k^((2,w))
  = Gamma_(T + 1:k)^((w)) H_T^((2,w))
    - w sum_(l = T + 1)^k Gamma_(l + 1:k)^((w))
        tilde(A)(Z_l) J_(l - 1)^((2,w)),
$
$
Delta H_k^((2,w))
  = Gamma_(T + 1:k)^((w)) Delta H_T^((2,w))
    - w sum_(l = T + 1)^k Gamma_(l + 1:k)^((w))
        tilde(A)(Z_l) Delta J_(l - 1)^((2,w)),
$ <eq:burn-H2-coupled-diff>

$
|| Delta H_k^((2,w)) 1_(T > k) ||_(L_p)
  <= lr((
       || H_(k, op("fin"))^((2,w)) ||_(L_(2p))
       + || H_(k, op("aug"))^((2,w)) ||_(L_(2p))
     )) thin bb(P)(T > k)^(1 slash (2p)).
$
$
|| Delta H_k^((2,w)) 1_(T > k) ||_(L_p)
  <= C B_H(p,q,w)
       e^(-c k slash p).
$ <eq:burn-H2-bad-event>

Set $V_s := Delta H_s^((2,w))$. Then
$
sup_s || V_s ||_(L_(2p))
  <= || H_(s, op("fin"))^((2,w)) ||_(L_(2p))
     + || H_(s, op("aug"))^((2,w)) ||_(L_(2p))
  <= C B_H(p,q,w).
$
By @eq:burn-random-time-product,
$
|| Gamma_(T + 1:k)^((w)) Delta H_T^((2,w)) 1_(T <= k) ||_(L_p)
  <= C frac(p, w a) B_H(p,q,w) e^(-c w a k slash p)
  // <= C frac(p, a) A_("J")(2p,q,w) e^(-c w a k slash p)
  <= C A_("st")(p,q,w) e^(-c w a k slash p).
$ <eq:burn-H2-initial-term>
For $U_l := tilde(A)(Z_l) Delta J_(l - 1)^((2,w))$,
$
|| U_l ||_(L_(2p))
  <= C A_("J")(2p,q,w) exp(-c w a l slash p),
$
// This is @eq:burn-J-startup-proof at moment order $2p$; replacing $l - 1$ by
// $l$ changes only the constant.
$
& w sum_(l = 1)^k
  || Gamma_(l + 1:k)^((w)) tilde(A)(Z_l)
     Delta J_(l - 1)^((2,w)) ||_(L_p)
// &quad <= C frac(p, a) A_("J")(2p,q,w) e^(-c w a k slash p)
&quad <= C A_("st")(p,q,w) e^(-c w a k slash p).
$ <eq:burn-H2-convolution-term>
// The last line is the only place where the $p^8 slash a$ summand in
// $A_("st")$ is needed.
$
|| Delta H_k^((2,w)) ||_(L_p)
  <= C A_("st")(p,q,w) e^(-c w a k slash p).
$ <eq:burn-H2-startup-bound>
$
R_(k, op("fin"))^((w)) - R_(k, op("aug"))^((w))
  = Delta J_k^((1,w)) + Delta J_k^((2,w)) + Delta H_k^((2,w)).
$
Together with @eq:burn-J12-startup-bound this proves
@eq:burn-startup-pointwise. $square$
// The displayed $A_("st")$ is the only allowed dependence on $p,q,w$; all other
// factors are fixed problem constants, uniformly over $xi$.

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

_Proof._ By @eq:burn-startup-pointwise at $w = alpha, 2 alpha$,
$
|| cal(U)_(n,n_0)^("start,RR") ||_(L_p)
  <= frac(1, sqrt(m)) sum_(k = n_0)^(n - 1)
    lr((2 A_("st")(p,q,alpha) e^(-c_("st") alpha a k slash p)
      + A_("st")(p,q,2 alpha) e^(-2 c_("st") alpha a k slash p))).
$
$
A_("st")(p,q,2 alpha) <= C A_("st")(p,q,alpha),
quad
1 - exp(-c_("st") alpha a slash p) >= C^(-1) alpha a slash p.
$
The geometric sum gives @eq:burn-startup-transfer. $square$

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
  // Thus the Berry--Esseen choice $p asymp log n$ requires a mixing-scale
  // burn-in with logarithmic-square factor, $n_0$ of order
  // $(alpha a)^(-1) log^2 n$, under this $L_p$ contraction.
] <cor:burn-log-startup>

// _Proof._ Substitute @eq:burn-log-startup-condition into
// @eq:burn-startup-transfer. For the balanced scale,
// $A_("st")(p,q,alpha) = "polylog"(n) thin alpha^(1 slash 2)$, hence
// $A_("st")(p,q,alpha) slash (alpha sqrt(m))
// = O("polylog"(n) n^(-1 slash 4))$. $square$

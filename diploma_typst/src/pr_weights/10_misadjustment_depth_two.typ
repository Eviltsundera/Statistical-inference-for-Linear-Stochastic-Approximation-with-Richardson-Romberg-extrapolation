#import "../defs.typ": *

== Depth-Two Misadjustment Bound

The remaining stationary non-martingale contribution is
$
R_n^("mis, RR") := frac(1, sqrt(n)) sum_(k = 0)^(n - 1)
                    (2 R_k^((alpha)) - R_k^((2 alpha))),
$ <eq:R-mis-def>
where $R_k^((alpha)) := J_k^((1, alpha)) + H_k^((1, alpha))$.
// A direct kernel-difference route only gives
// $O(sqrt(n) thin alpha)$, which is too crude at
// $alpha asymp n^(-1 slash 2)$. The depth-two route below uses
// @lem:levin-prop-2, @lem:levin-cor-6, @lem:levin-prop-8, and
// @lem:levin-prop-9 to recover the target
// $n^(-1 slash 4) thin "polylog"(n)$ rate.

*Depth-two refinement.* For $ell >= 1$ define
$
J_n^((ell, alpha)) := (I - alpha overline(A)) thin J_(n - 1)^((ell, alpha))
                       - alpha tilde(A)(Z_n) thin J_(n - 1)^((ell - 1, alpha)),
$
$
H_n^((ell, alpha)) := (I - alpha A(Z_n)) thin H_(n - 1)^((ell, alpha))
                       - alpha tilde(A)(Z_n) thin J_(n - 1)^((ell, alpha)),
$
with $J_0^((ell, alpha)) = H_0^((ell, alpha)) = 0$ and the $ell = 0$
processes as in @sec:last_iterate. Then
$
H_n^((ell, alpha)) = J_n^((ell + 1, alpha)) + H_n^((ell + 1, alpha)),
quad ell >= 0.
$ <eq:depth-recursion>
Applying @eq:depth-recursion with $ell = 1$ refines $R_k^((alpha))$ to
$
R_k^((alpha)) = J_k^((1, alpha)) + J_k^((2, alpha)) + H_k^((2, alpha)),
$
so
$
R_n^("mis, RR") = T_n^((1)) + T_n^((2)) + T_n^((H)),
$ <eq:R-mis-split>
$
T_n^((j)) := frac(1, sqrt(n)) sum_(k = 0)^(n - 1)
              lr((2 J_k^((j, alpha)) - J_k^((j, 2 alpha)))),
quad j in {1, 2},
$
with $T_n^((H))$ defined identically with $H^((2))$ in place of $J^((j))$.
Each piece is bounded separately.
// The direct Levin working forms used in this section are collected in
// @sec:external-direct-inputs.

*Finite-past truncation.* Fix $q >= 2$, $2 <= p <= q slash 2$, and
$0 < w <= alpha_("stat")(q)$. Let $(Z_t)_(t in ZZ)$ be a two-sided stationary
version of the driving Markov chain, and write $B_w := I - w overline(A)$.
For $m >= 1$ start at time $-m$ from zero,
$
J_(-m,m)^((0,w)) = J_(-m,m)^((1,w)) = J_(-m,m)^((2,w))
  = H_(-m,m)^((2,w)) = 0,
$
and evolve, for $r = -m + 1, dots, 0$,
$
J_(r,m)^((0,w))
  = B_w J_(r - 1,m)^((0,w)) - w epsilon.alt(Z_r),
$
$
J_(r,m)^((ell,w))
  = B_w J_(r - 1,m)^((ell,w))
    - w tilde(A)(Z_r) J_(r - 1,m)^((ell - 1,w)),
quad ell in {1, 2},
$
$
H_(r,m)^((2,w))
  = (I - w A(Z_r)) H_(r - 1,m)^((2,w))
    - w tilde(A)(Z_r) J_(r - 1,m)^((2,w)).
$
#lemma[
  *(Stationary full depth-two augmented state.)*
  There exist constants $C_("fp"), c_("fp") > 0$, depending only on the fixed
  problem parameters allowed by the chapter convention, on $C_W$, $D_J$, $D_H$
  from @lem:levin-prop-5-component, @lem:levin-prop-8, @lem:levin-prop-9, and on
  $C_("prod")$, $c_("prod")$ from @lem:burn-product-stability, such that, with
  $
  A_("fp")(p,q,w)
    := C_("fp") (1 + d^(1 slash q)) p^8 t_"mix"^5
       frac(1, a) sqrt(w slash a) log^3(1 slash (w a)),
  $
  the terminal vector of the finite-past truncation above,
  $
  (J_(0,m)^((0,w)), J_(0,m)^((1,w)), J_(0,m)^((2,w)),
   H_(0,m)^((2,w)))
  $
  is Cauchy in $L_p$. More precisely, for $m' >= m >= 1$,
  $
  || H_(0,m')^((2,w)) - H_(0,m)^((2,w)) ||_(L_p)
    <= A_("fp")(p,q,w) exp(-c_("fp") w a m slash p),
  $ <eq:finite-past-H2-cauchy>
  The $J$ coordinates converge to the invariant law
  $Pi_(J,2,w)$ of @lem:levin-invariant-depth-two-law, and the $L_p$ limit
  $
  H_0^((2,w)) := lim_(m -> infinity) H_(0,m)^((2,w))
  $
  satisfies the one-trajectory bound of @lem:levin-prop-9.
] <lem:finite-past-full-augmented-state>

_Proof._ First control the $J$-coordinates. Compare the truncation started at
$-m'$ with the one started at $-m$ by applying the component contraction
@eq:levin-depth-two-component-contraction after time $-m$. The contraction
constant is $C_W$ and its exponential rate is $1 slash 12$. The initial
depth-two cost at time $-m$ is bounded by the elementary moment bounds for
$J^((0,w))$ and $J^((1,w))$, together with @lem:levin-prop-8 for
$J^((2,w))$. Write
$
Y_(r,m)^((w)) := (Z_(r + 1), J_(r,m)^((0,w)), J_(r,m)^((1,w)), J_(r,m)^((2,w))).
$
Thus there is a constant $C_("cost")$, depending only on the
fixed problem parameters and on $D_J$, such that
$
|| c_(J,2)^((w))(Y_(-m,m')^((w)), Y_(-m,m)^((w))) ||_(L_p)
  <= C_("cost") (1 + d^(1 slash q)) p^(7 slash 2) t_"mix"^(5 slash 2)
     sqrt(w slash a) log^(3 slash 2)(1 slash (w a)).
$ <eq:finite-past-cost-bound>
Consequently, for $r >= -m$ and $ell in {0,1,2}$,
$
|| J_(r,m')^((ell,w)) - J_(r,m)^((ell,w)) ||_(L_p)
  <= A_J(p,q,w) exp(-w a (r + m) slash (12 p)),
$ <eq:finite-past-J-cauchy>
where
$
A_J(p,q,w)
  := C_W C_("cost") (1 + d^(1 slash q)) p^7 t_"mix"^5
     sqrt(w slash a) log^3(1 slash (w a)).
$ <eq:finite-past-AJ>

Now subtract the two $H^((2,w))$ finite-past recursions. With
$Delta H_r := H_(r,m')^((2,w)) - H_(r,m)^((2,w))$ and similarly for
$Delta J_r^((2,w))$,
$
Delta H_0
  = Gamma_(-m + 1:0)^((w)) H_(-m,m')^((2,w))
    - w sum_(l = -m + 1)^0 Gamma_(l + 1:0)^((w))
        tilde(A)(Z_l) Delta J_(l - 1)^((2,w)).
$ <eq:finite-past-H2-difference>

For the initial term, use the conditional product-stability estimate
@eq:burn-product-stability-conditional and @lem:levin-prop-9:
$
|| Gamma_(-m + 1:0)^((w)) H_(-m,m')^((2,w)) ||_(L_p)
  &<= C_("prod") exp(-c_("prod") w a m slash p)
       || H_(-m,m')^((2,w)) ||_(L_p) \
  &<= C_("prod") D_H d^(1 slash q) t_"mix"^(5 slash 2) p^(7 slash 2)
       w^(3 slash 2) log^(3 slash 2)(1 slash (w a))
       exp(-c_("prod") w a m slash p).
$ <eq:finite-past-H-initial-term>

For the convolution term, combine @eq:burn-product-stability-conditional,
$||tilde(A)(z)|| <= C_A$, and @eq:finite-past-J-cauchy:
$
&w sum_(l = -m + 1)^0
  || Gamma_(l + 1:0)^((w)) tilde(A)(Z_l) Delta J_(l - 1)^((2,w)) ||_(L_p) \
&quad <= w C_("prod") C_A A_J(p,q,w)
   sum_(l = -m + 1)^0
      exp(-c_("prod") w a (-l) slash p)
      exp(-w a (l - 1 + m) slash (12 p)).
$ <eq:finite-past-H-convolution-raw>
The elementary convolution bound
$
sum_(l = -m + 1)^0
      exp(-c_("prod") w a (-l) slash p)
      exp(-w a (l - 1 + m) slash (12 p))
  <= C_("conv") frac(p, w a)
     exp(-c_("conv") w a m slash p)
$ <eq:finite-past-exp-convolution>
holds with $C_("conv"), c_("conv") > 0$ depending only on $c_("prod")$ and
the numerical rate $1 slash 12$. Therefore the convolution term is bounded by
$
C_("prod") C_A C_("conv") frac(p, a) A_J(p,q,w)
  exp(-c_("conv") w a m slash p).
$ <eq:finite-past-H-convolution-bound>

Choose
$
c_("fp") := frac(1,2) min(c_("prod"), c_("conv"))
$
and choose $C_("fp")$ large enough to dominate the constants in
@eq:finite-past-H-initial-term and @eq:finite-past-H-convolution-bound after
substituting @eq:finite-past-AJ. Since $w <= alpha_infinity$ implies
$w a <= 1 slash 2$, the initial-term factor $w^(3 slash 2)$ is absorbed by
$a^(-1) sqrt(w slash a)$, and the powers
$p^(7 slash 2) t_"mix"^(5 slash 2) log^(3 slash 2)(1 slash (w a))$ are
absorbed by the displayed $p^8 t_"mix"^5 log^3(1 slash (w a))$ in
$A_("fp")(p,q,w)$. This gives @eq:finite-past-H2-cauchy.

The $J$-bounds and @eq:finite-past-H2-cauchy show that the full terminal vector
is Cauchy in $L_p$. Passing to the $L_p$ limit gives the stationary full
augmented-chain law.
// The limits are measurable functions of the two-sided stationary driving
// chain. Passing to the limit in the finite-past recursions is justified in
// $L_p$, stationarity follows from shift-covariance, and the limiting
// $H^((2,w))$ bound follows from @lem:levin-prop-9 and Fatou's lemma.
$square$

Shifting the finite-past construction before taking the limit defines a
two-sided stationary process
$
(Z_(k + 1), J_k^((0,w)), J_k^((1,w)), J_k^((2,w)), H_k^((2,w)))_(k in ZZ).
$
It solves the full depth-two recursions, and its law is the stationary full
augmented-chain law used below.

*Joint RR stationary construction.* Apply
@lem:finite-past-full-augmented-state at $w = alpha$ and $w = 2 alpha$ on the
same two-sided stationary base chain.
The joint limit
$
lr((
  Z_(k + 1),
  J_k^((0, alpha)), J_k^((1, alpha)), J_k^((2, alpha)),
  H_k^((2, alpha)),
  J_k^((0, 2 alpha)), J_k^((1, 2 alpha)), J_k^((2, 2 alpha)),
  H_k^((2, 2 alpha))
))_(k in ZZ)
$
is the stationary RR augmented state used below.
// All RR differences below, such as
// $2 J_k^((j, alpha)) - J_k^((j, 2 alpha))$ and
// $2 H_k^((2, alpha)) - H_k^((2, 2 alpha))$, are read under this joint
// construction.

*Stationary augmented-chain convention.* The estimates below use the stationary
versions constructed above. A zero-start full average is not obtained by a
single terminal contraction: summing startup contractions yields
$
frac(1, sqrt(n)) sum_(k >= 0) rho_alpha^k
  asymp frac(1, sqrt(n) thin alpha a)
$
when $rho_alpha approx exp(-c alpha a)$. This term is deferred to the burn-in
chapter.
// The one-step shift in
// $(Z_(t + 1), J_t^((0, w)), J_t^((1, w)), J_t^((2, w)), H_t^((2, w)))$ is
// intentional: $J_t$ is the perturbation state after the observation
// $Z_(t + 1)$ has updated the recursion, while base-chain covariance formulas
// may still use $Z_0$ by stationarity. A practical arbitrary-start theorem
// requires either burned-in weights $Q_(l,n_0)^((alpha))$ or a separate transfer
// lemma with the accumulated startup dependence.

For the shifted-to-unshifted first-order transfer we use the local inverse
ceiling $alpha_("inv")$ defined in @eq:alpha-inv.

#lemma[
  *(Stationary-limit transfer for the centered first-order iterate.)*
  Fix $w in (0, alpha_infinity]$ and set $B_w := I - w overline(A)$. Assume
  the stationary augmented-chain convention above, *UGE 1*,
  $pi(tilde(A)) = 0$, and $|| epsilon.alt ||_infinity < infinity$. Let
  $(J_t^((0, w)), J_t^((1, w)))_(t in ZZ)$ be the stationary two-sided
  solution of the first two perturbation recursions, and set
  $T_t^((1, w)) := B_w thin J_t^((1, w))$. Then, with
  $
  Phi_+(p, w) := 1 + p^(3 slash 2) thin t_"mix"^(1 slash 2) slash a
               + p^(1 slash 2) thin t_"mix"^(3 slash 2) sqrt(w slash a),
  $
  there exists a constant $C_("stat,1")$, depending only on the constants in
  the zero-start centered shifted first-order bound in @lem:last-shifted-first-order,
  such that
  for every $p >= 2$ and every $t in ZZ$,
  $
  || T_t^((1, w)) - bb(E)_pi T_t^((1, w)) ||_(L_p)
    <= C_("stat,1") thin w thin Phi_+(p, w).
  $
  Consequently, whenever $w <= alpha_("inv")$,
  $
  || J_t^((1, w)) - bb(E)_pi J_t^((1, w)) ||_(L_p)
    <= 2 C_("stat,1") thin w thin Phi_+(p, w).
  $
] <lem:stationary-limit-J1>

_Proof._ Use finite-past zero-start versions
$J_(t, m)^((0, w))$ and $J_(t, m)^((1, w))$. By stationarity and
@lem:last-shifted-first-order,
$
|| B_w J_(t, m)^((1, w)) - bb(E) B_w J_(t, m)^((1, w)) ||_(L_p)
  <= C_("stat,1") thin w thin Phi_+(p, w).
$
The deterministic-product expansions give
$
J_(t, m)^((0, w))
  = -w sum_(r = 0)^(m - 1) B_w^r epsilon.alt(Z_(t - r)),
$
and the corresponding $J_(t,m)^((1,w))$ series is Cauchy for fixed admissible
$w$. Passing to the limit gives the bound for $T_t^((1,w))$. The bound for
$J_t^((1,w))$ follows from
$J_t^((1, w)) - bb(E)_pi J_t^((1, w))
  = B_w^(-1) lr((T_t^((1, w)) - bb(E)_pi T_t^((1, w))))$.
$square$

// The finite-past domination is fixed-$w$; later estimates keep the displayed
// $w$-dependence through $Phi_+(p,w)$ and use constants independent of $n$.

*Telescoping identity for $J^((1))$.* Summing
$J_k^((1, alpha)) = (I - alpha overline(A)) thin J_(k - 1)^((1, alpha))
  - alpha tilde(A)(Z_k) thin J_(k - 1)^((0, alpha))$
from $k = 1$ to $n$ and rearranging gives, for an arbitrary initial value,
$
overline(A) sum_(k = 0)^(n - 1) J_k^((1, alpha))
  = -sum_(k = 1)^n tilde(A)(Z_k) thin J_(k - 1)^((0, alpha))
    + frac(1, alpha) thin lr((J_0^((1, alpha)) - J_n^((1, alpha)))).
$
In stationarity,
$bb(E)_pi [tilde(A)(Z_1) thin J_0^((0, alpha))]
= -overline(A) thin bb(E)_pi [J_infinity^((1, alpha))]$.
// Under the stationary augmented-chain convention,
// $(J_(k - 1)^((0, w)), Z_k)$ has the same law as
// $(J_0^((0, w)), Z_1)$, so $overline(psi)_w$ is centered for each summand.
Subtracting $n thin bb(E)_pi [J_infinity^((1, alpha))]$ from both sides and
applying $overline(A)^(-1)$ gives the *centered telescoping identity*
$
sum_(k = 0)^(n - 1) lr((J_k^((1, alpha)) - bb(E)_pi [J_infinity^((1, alpha))]))
  = -overline(A)^(-1) sum_(k = 1)^n
      overline(psi)_alpha (J_(k - 1)^((0, alpha)), Z_k)
    + frac(1, alpha) thin overline(A)^(-1)
      lr((J_0^((1, alpha)) - J_n^((1, alpha)))).
$ <eq:J1-telescope>
// This identity links the PR-average of $J^((1))$ to the centered bilinear sum
// bounded in @lem:levin-cor-6.

#lemma[
  *(Stationary first-order misadjustment bound.)*
  Assume the stationary augmented-chain convention above, *UGE 1*,
  $pi(tilde(A)) = 0$, $pi(epsilon.alt) = 0$,
  $|| epsilon.alt ||_infinity < infinity$, and
  $0 < alpha$, $2 alpha <= alpha_infinity$, and
  $2 alpha <= alpha_("inv")$. Set
  $
  Phi_+(p, alpha) := 1 + p^(3 slash 2) thin t_"mix"^(1 slash 2) slash a
                   + p^(1 slash 2) thin t_"mix"^(3 slash 2) sqrt(alpha slash a).
  $
  There exists a constant $C_("mis,1")$ depending on
  $|| overline(A) ||, || overline(A)^(-1) ||, kappa_Q, C_A,
   || epsilon.alt ||_infinity, t_"mix"$, the universals $c_(W, 1), c_(W, 2)$
  of @lem:levin-cor-6, and the constant of the centered bound in
  @lem:last-shifted-first-order, such that for every $p >= 2$ and every
  $n >= 2$,
  $
  || T_n^((1)) ||_(L_p)
    &<= C_("mis,1") sqrt(n) thin alpha^2
     + C_("mis,1") thin p^(3 slash 2) sqrt(alpha) \
    &quad + C_("mis,1") thin p^3 thin (alpha n)^(-1 slash 2) thin log^(1 slash 2)(1 slash (alpha a))
     + C_("mis,1") thin Phi_+(p, alpha) thin n^(-1 slash 2).
  $
] <lem:T1-bound>

_Proof._ Decompose $T_n^((1)) = T_n^("(1, b)") + T_n^("(1, c)")$ via the
bias-fluctuation split
$
T_n^("(1, b)") := sqrt(n) thin lr((
                    2 thin bb(E)_pi [J_infinity^((1, alpha))]
                    - bb(E)_pi [J_infinity^((1, 2 alpha))]
                  )),
$
$
T_n^("(1, c)") := frac(1, sqrt(n)) sum_(k = 0)^(n - 1) sum_(w in {alpha, 2 alpha}) c_w
                   lr((J_k^((1, w)) - bb(E)_pi [J_infinity^((1, w))])),
quad c_alpha = 2, thin c_(2 alpha) = -1.
$
// The bias is deterministic; the centered piece carries the $L_p$ fluctuation.

*Bias.* By @lem:levin-prop-2,
$bb(E)_pi [J_infinity^((1, w))] = w thin Delta + R(w)$ with
$|| R(w) || <= 12 thin || overline(A)^(-1) || thin C_A^2 thin t_"mix"^2 thin w^2 thin || epsilon.alt ||_infinity$.
The leading $w thin Delta$ cancels, leaving
$
|| 2 thin bb(E)_pi [J_infinity^((1, alpha))] - bb(E)_pi [J_infinity^((1, 2 alpha))] ||
  <= 2 thin || R(alpha) || + || R(2 alpha) ||
  <= 72 thin || overline(A)^(-1) || thin C_A^2 thin t_"mix"^2 thin alpha^2 thin || epsilon.alt ||_infinity.
$
Multiplying by $sqrt(n)$ gives the first term.

*Centered.* Apply @eq:J1-telescope at $w in {alpha, 2 alpha}$:
$
|| T_n^("(1, c)") ||_(L_p)
  &<= frac(1, sqrt(n)) sum_(w in {alpha, 2 alpha}) |c_w| lr((
       || overline(A)^(-1) || dot
       || sum_(k = 1)^n overline(psi)_w (J_(k - 1)^((0, w)), Z_k) ||_(L_p) \
    &quad quad quad quad
       + frac(1, w) thin || overline(A)^(-1) || dot
         || J_0^((1, w)) - J_n^((1, w)) ||_(L_p)
     )).
$
For the bilinear sum,
$
|| sum_(k = 1)^n overline(psi)_w ||_(L_p)
  <= c_(W, 1) thin p^(3 slash 2) sqrt(w n)
   + c_(W, 2) thin p^3 thin w^(-1 slash 2) thin log^(1 slash 2)(1 slash (w a)),
$
so
$
frac(1, sqrt(n)) || sum overline(psi)_w ||_(L_p)
  <= sqrt(2) c_(W, 1) thin p^(3 slash 2) sqrt(alpha)
   + sqrt(2) c_(W, 2) thin p^3 thin (alpha n)^(-1 slash 2) thin log^(1 slash 2)(1 slash (alpha a)).
$
For the boundary term, @lem:stationary-limit-J1 gives
$
|| J_n^((1, w)) - bb(E) J_n^((1, w)) ||_(L_p) <= C w thin Phi_+(p, w),
$
and @lem:levin-prop-2 gives $|| bb(E) J_n^((1, w)) || <= C w$, hence
$
|| J_n^((1, w)) ||_(L_p) <= C w thin Phi_+(p, w).
$
The same bound holds for $J_0^((1, w))$, so
$w^(-1) || J_0^((1, w)) - J_n^((1, w)) ||_(L_p)
<= C thin Phi_+(p, w) <= sqrt(2) thin C thin Phi_+(p, alpha)$ after absorbing
the factor $2$ into $C$.
Dividing by $sqrt(n)$ gives the last term. $square$

#lemma[
  *(Stationary second-order and depth-two remainder bound.)*
  Under the assumptions of the previous lemma, for every $p >= 2$ and every
  $q >= 2$ satisfying $p <= q slash 2$ and
  $2 alpha <= alpha_("stat")(q)$,
  $
  || T_n^((2)) ||_(L_p) + || T_n^((H)) ||_(L_p)
    <= C_("mis,2") thin (1 + d^(1 slash q)) thin p^(7 slash 2) thin t_"mix"^(5 slash 2)
       thin sqrt(n) thin alpha^(3 slash 2) thin log^(3 slash 2)(1 slash (alpha a)),
  $
  with $C_("mis,2") := 6 thin sqrt(2)^3 thin (D_J + D_H)$.
] <lem:T2H-bound>

_Proof._ By the triangle inequality and @lem:levin-prop-8,
$
|| T_n^((2)) ||_(L_p)
  &<= frac(1, sqrt(n)) sum_(k = 0)^(n - 1) sum_(w in {alpha, 2 alpha}) |c_w| thin || J_k^((2, w)) ||_(L_p) \
  &<= 3 thin sqrt(n) thin sup_(w in {alpha, 2 alpha}) || J_k^((2, w)) ||_(L_p) \
  &<= 3 thin D_J thin (2 alpha)^(3 slash 2)
      thin t_"mix"^(5 slash 2) thin p^(7 slash 2) thin sqrt(n)
      thin log^(3 slash 2)(1 slash (alpha a)),
$
The same argument with @lem:levin-prop-9 controls $T_n^((H))$. $square$
// Here $|c_alpha| + |c_(2 alpha)| = 3$ absorbs the RR combination and
// $(2 alpha)^(3 slash 2)$ is the worse bound at the larger step.

#theorem[
  *(Stationary PR-averaged RR misadjustment bound.)*
  Assume *UGE 1*, $pi(tilde(A)) = 0$, $pi(epsilon.alt) = 0$,
  $|| epsilon.alt ||_infinity < infinity$, $0 < alpha$, and
  the stationary augmented-chain convention above. There exists a constant $C$
  depending on the universal and problem constants of the previous two lemmas
  such that for every
  $p >= 2$, every $q >= 2$ satisfying
  $p <= q slash 2$ and $2 alpha <= alpha_("stat")(q)$, and every $n >= 2$,
  $
  || R_n^("mis, RR") ||_(L_p)
    &<= C sqrt(n) thin alpha^2
     + C thin (1 + d^(1 slash q)) thin p^(7 slash 2) thin t_"mix"^(5 slash 2)
         thin sqrt(n) thin alpha^(3 slash 2) thin log^(3 slash 2)(1 slash (alpha a)) \
    &quad + C thin p^(3 slash 2) sqrt(alpha)
     + C thin p^3 thin (alpha n)^(-1 slash 2) thin log^(1 slash 2)(1 slash (alpha a))
     + C thin Phi_+(p, alpha) thin n^(-1 slash 2).
  $
] <thm:misadjustment>

_Proof._ From @eq:R-mis-split,
$|| R_n^("mis, RR") ||_(L_p) <= || T_n^((1)) ||_(L_p) + || T_n^((2)) ||_(L_p) + || T_n^((H)) ||_(L_p)$.
The result follows from @lem:T1-bound and @lem:T2H-bound. $square$

#corollary[
  *(Stationary balanced-scale misadjustment rate.)*
  At the working scale $alpha = c thin n^(-1 slash 2)$, with
  $p = max(2, ceil(log n))$ and $q = max(2 p, ceil(log(e thin d)), 2)$, and for $n$
  large enough that $2 alpha <= alpha_("stat")(q)$,
  $
  || R_n^("mis, RR") ||_(L_p) <= C thin "polylog"(n) thin n^(-1 slash 4),
  $
  of the same polynomial order as the leading martingale Berry--Esseen rate,
  up to logarithmic factors.
] <cor:misadjustment-rate>

_Proof._ At $alpha = c thin n^(-1 slash 2)$: $sqrt(n) alpha^2 = c^2 thin n^(-1 slash 2)$,
$sqrt(n) alpha^(3 slash 2) = c^(3 slash 2) thin n^(-1 slash 4)$,
$sqrt(alpha) = c^(1 slash 2) thin n^(-1 slash 4)$,
$(alpha n)^(-1 slash 2) = c^(-1 slash 2) thin n^(-1 slash 4)$, and
$Phi_+(p, alpha) thin n^(-1 slash 2) = O(p^(3 slash 2) thin n^(-1 slash 2))$.
With $p asymp log n$ and $q >= log(e d)$, the displayed bound is
$"polylog"(n) thin n^(-1 slash 4)$. $square$

This is a stationary $n_0 = 0$ augmented-chain bound; the deterministic-start
version is handled in the burn-in transfer chapter.

#import "../defs.typ": *

== Misadjustment via Levin Depth-Two

The Berry--Esseen assembly so far controls the leading martingale piece
(Section 4.8) and the Poisson boundary remainder $D_(2, n)^("RR")$
(Section 4.6). The remaining non-martingale contribution to
$sqrt(n) thin u^top (overline(theta)_n^(("RR", alpha)) - theta^*)$ is the
*PR-averaged misadjustment*
$
R_n^("mis, RR") := frac(1, sqrt(n)) sum_(k = 0)^(n - 1)
                    (2 R_k^((alpha)) - R_k^((2 alpha))),
$ <eq:R-mis-def>
where $R_k^((alpha)) := J_k^((1, alpha)) + H_k^((1, alpha))$ is the depth-one
remainder of @eq:depth-one. A direct kernel-difference route only bounds
$R_n^("mis, RR")$ at order $O(sqrt(n) thin alpha) = O(1)$ at the working scale
$alpha asymp n^(-1 slash 2)$ — too crude for a Berry--Esseen rate
$n^(-1 slash 4)$. The depth-two route below uses the imported working forms
@lem:levin-prop-2, @lem:levin-cor-6, @lem:levin-prop-8, and
@lem:levin-prop-9 to recover the target
$n^(-1 slash 4) thin "polylog"(n)$ rate.

*Depth-two refinement.* The deterministic-product expansion from the
depth-two setup extends the depth-one decomposition by one more level. For
$ell >= 1$ define
$
J_n^((ell, alpha)) := (I - alpha overline(A)) thin J_(n - 1)^((ell, alpha))
                       - alpha tilde(A)(Z_n) thin J_(n - 1)^((ell - 1, alpha)),
$
$
H_n^((ell, alpha)) := (I - alpha A(Z_n)) thin H_(n - 1)^((ell, alpha))
                       - alpha tilde(A)(Z_n) thin J_(n - 1)^((ell, alpha)),
$
with $J_0^((ell, alpha)) = H_0^((ell, alpha)) = 0$ and the $ell = 0$
processes as in @sec:last_iterate. Substituting
$A(Z_n) = overline(A) + tilde(A)(Z_n)$ and grouping terms gives, by
induction on $n$, the recursive identity
$
H_n^((ell, alpha)) = J_n^((ell + 1, alpha)) + H_n^((ell + 1, alpha)),
quad ell >= 0.
$ <eq:depth-recursion>
Applying @eq:depth-recursion with $ell = 1$ refines $R_k^((alpha))$ to
$
R_k^((alpha)) = J_k^((1, alpha)) + J_k^((2, alpha)) + H_k^((2, alpha)),
$
which splits the misadjustment into three structurally different pieces:
$
R_n^("mis, RR") = T_n^((1)) + T_n^((2)) + T_n^((H)),
$ <eq:R-mis-split>
$
T_n^((j)) := frac(1, sqrt(n)) sum_(k = 0)^(n - 1)
              lr((2 J_k^((j, alpha)) - J_k^((j, 2 alpha)))),
quad j in {1, 2},
$
with $T_n^((H))$ defined identically with $H^((2))$ in place of $J^((j))$.
Each piece is bounded separately below. The direct Levin working forms used in
this section are collected in @sec:external-direct-inputs:
@lem:levin-prop-2, @lem:levin-cor-6, @lem:levin-prop-8, and
@lem:levin-prop-9.

#lemma[
  *(Finite-past construction of the stationary full depth-two augmented state.)*
  Fix $q >= 2$, $2 <= p <= q slash 2$, and an admissible step size
  $0 < w <= alpha_("stat")(q)$. Let $(Z_t)_(t in ZZ)$ be a two-sided
  stationary version of the driving Markov chain, and write
  $B_w := I - w overline(A)$. For $m >= 1$ define the finite-past truncation
  by starting at time $-m$ from
  $
  J_(-m,m)^((0,w)) = J_(-m,m)^((1,w)) = J_(-m,m)^((2,w))
    = H_(-m,m)^((2,w)) = 0
  $
  and evolving, for $r = -m + 1, dots, 0$,
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
  Then the vector
  $
  (J_(0,m)^((0,w)), J_(0,m)^((1,w)), J_(0,m)^((2,w)),
   H_(0,m)^((2,w)))
  $
  is Cauchy in $L_p$. More precisely, for $m' >= m >= 1$,
  $
  || H_(0,m')^((2,w)) - H_(0,m)^((2,w)) ||_(L_p)
    <= A_("fp")(p,q,w) exp(-c_("fp") w a m slash p),
  $ <eq:finite-past-H2-cauchy>
  where
  $
  A_("fp")(p,q,w)
    := C_("fp") (1 + d^(1 slash q)) p^8 t_"mix"^5
       frac(1, a) sqrt(w slash a) log^3(1 slash (w a)),
  $
  and $C_("fp"), c_("fp") > 0$ depend only on fixed problem constants and the
  Levin constants. The $J$ coordinates converge to the invariant law
  $Pi_(J,2,w)$ of @lem:levin-invariant-depth-two-law, and the $L_p$ limit
  $
  H_0^((2,w)) := lim_(m -> infinity) H_(0,m)^((2,w))
  $
  satisfies the one-trajectory bound of @lem:levin-prop-9. Shifting the
  construction in time defines a two-sided stationary process
  $
  (Z_(k + 1), J_k^((0,w)), J_k^((1,w)), J_k^((2,w)), H_k^((2,w)))_(k in ZZ)
  $
  which solves the full depth-two recursions and whose law is the stationary
  full augmented-chain law used below.
] <lem:finite-past-full-augmented-state>

_Proof._ The $J$-coordinate construction is @lem:levin-invariant-depth-two-law
together with the component contraction @lem:levin-prop-5-component for
$Y_k^((w)) = (Z_(k + 1), J_k^((0,w)), J_k^((1,w)), J_k^((2,w)))$. On the
same two-sided driving chain, the finite-past truncations started at $-m'$ and
$-m$ agree in the driving coordinate after time $-m$, while their
$J$-coordinates at time $-m$ differ by the older finite-past state. The
componentwise contraction @eq:levin-depth-two-component-contraction therefore gives, for
$r >= -m$ and $ell in {0,1,2}$,
$
|| J_(r,m')^((ell,w)) - J_(r,m)^((ell,w)) ||_(L_p)
  <= A_J(p,q,w) exp(-c w a (r + m) slash p),
$ <eq:finite-past-J-cauchy>
with
$
A_J(p,q,w)
  := C (1 + d^(1 slash q)) p^7 t_"mix"^5
     sqrt(w slash a) log^3(1 slash (w a)).
$
Here the initial finite-past cost is controlled by the elementary
$J^((0,w))$, $J^((1,w))$ moment bounds and @lem:levin-prop-8 for
$J^((2,w))$; the constants are uniform in $m,m'$.

It remains to justify the extra $H^((2,w))$ coordinate. Subtract the two
finite-past versions of the representation
@eq:levin-H2-representation from time $-m$ to time $0$. With
$Delta H_r := H_(r,m')^((2,w)) - H_(r,m)^((2,w))$ and similarly for
$Delta J_r^((2,w))$,
$
Delta H_0
  = Gamma_(-m + 1:0)^((w)) H_(-m,m')^((2,w))
    - w sum_(l = -m + 1)^0 Gamma_(l + 1:0)^((w))
        tilde(A)(Z_l) Delta J_(l - 1)^((2,w)).
$ <eq:finite-past-H2-difference>
The product-stability working form @lem:burn-product-stability, applied at the
deterministic time $-m$, and the one-trajectory bound @lem:levin-prop-9 give
$
|| Gamma_(-m + 1:0)^((w)) H_(-m,m')^((2,w)) ||_(L_p)
  <= C B_H(p,q,w) exp(-c w a m slash p),
$
where
$
B_H(p,q,w)
  := (1 + d^(1 slash q)) p^(7 slash 2) t_"mix"^(5 slash 2)
     w^(3 slash 2) log^(3 slash 2)(1 slash (w a)).
$
For the convolution term, combine product stability with
@eq:finite-past-J-cauchy:
$
w sum_(l = -m + 1)^0
  || Gamma_(l + 1:0)^((w)) tilde(A)(Z_l) Delta J_(l - 1)^((2,w)) ||_(L_p)
  <= C w sum_(l = -m + 1)^0
     e^(-c w a (-l) slash p)
     A_J(p,q,w) e^(-c w a (l + m) slash p).
$
The last convolution is bounded by
$C (p slash a) A_J(p,q,w) exp(-c w a m slash (2 p))$. Enlarging the
constant and decreasing the exponential rate gives @eq:finite-past-H2-cauchy.
This proves the Cauchy property for the full vector.

The limits are measurable functions of the two-sided stationary driving
chain. Passing to the limit in the finite-past recursions is justified in
$L_p$, so the limiting process solves the same recursions. Stationarity follows
because the finite-past construction is shift-covariant on the two-sided
stationary chain. The uniform one-trajectory bound for the limiting
$H^((2,w))$ follows from @lem:levin-prop-9 and Fatou's lemma. $square$

*Stationary augmented-chain convention.* The recursions above are displayed in
finite-time notation with $J_0^((ell, alpha)) = H_0^((ell, alpha)) = 0$, while
@lem:levin-prop-2 is a statement about the stationary augmented chain. The
misadjustment estimates below use the stationary versions constructed in
@lem:finite-past-full-augmented-state:
$(Z_(t + 1), J_t^((0, w)), J_t^((1, w)), J_t^((2, w)), H_t^((2, w)))$,
$w in {alpha, 2 alpha}$, as deterministic centers and should be read under
this stationary augmented-chain convention. The one-step shift is intentional:
$J_t$ is the perturbation state after the observation $Z_(t + 1)$ has updated
the recursion, while covariance formulas for the base chain may still use
$Z_0$ by stationarity. We keep the same symbols
$J_t^((ell, w))$ and $H_t^((ell, w))$ to avoid duplicating notation.
For a zero-start full average, the transfer from finite-start sums to
stationary centered sums is not a single terminal $rho^n$ term: summing
pointwise contractions over $k = 0, dots, n - 1$ produces a startup
contribution of order
$
frac(1, sqrt(n)) sum_(k >= 0) rho_alpha^k
  asymp frac(1, sqrt(n) thin alpha a)
$
when $rho_alpha approx exp(-c alpha a)$. This term is not part of the theorem
proved here. A practical arbitrary-start theorem requires either a genuine
burn-in average with the burned-in weights $Q_(l,n_0)^((alpha))$ of
Section 4.1 or a separate transfer lemma with the correct accumulated
startup dependence.

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

_Proof._ For $m >= 1$, run the same recursions on the stationary driving
chain over the finite window $\{t - m + 1, dots, t\}$ with zero initial
conditions at time $t - m$; denote the resulting variables by
$J_(t, m)^((0, w))$ and $J_(t, m)^((1, w))$. Stationarity of the driving chain
implies that $B_w J_(t, m)^((1, w))$ has the same law as the zero-start
$T_m^((1, w))$ in @lem:last-shifted-first-order. Therefore the centered
last-iterate
bound gives, uniformly in $m$ and $t$,
$
|| B_w J_(t, m)^((1, w)) - bb(E) B_w J_(t, m)^((1, w)) ||_(L_p)
  <= C_("stat,1") thin w thin Phi_+(p, w).
$
The deterministic-product expansions give
$
J_(t, m)^((0, w))
  = -w sum_(r = 0)^(m - 1) B_w^r epsilon.alt(Z_(t - r)),
$
and express $J_(t, m)^((1, w))$ as a bounded double series whose summands are
dominated by
$C w^2 s (1 - w a)^(s slash 2) ||epsilon.alt||_infinity$ at total lag $s$.
For the fixed admissible $w$ in the lemma,
$sum_(s >= 0) s (1 - w a)^(s slash 2) < infinity$, so the finite-past
approximations are Cauchy in every $L_p$ and converge to the stationary
two-sided solution. This domination is not asserted uniformly as $w -> 0$;
the later estimates retain the resulting step-size dependence through
$Phi_+(p,w)$. Passing to the limit in the preceding uniform bound yields the
displayed estimate for $T_t^((1, w))$. The estimate for $J_t^((1, w))$ then
follows from
$J_t^((1, w)) - bb(E)_pi J_t^((1, w))
  = B_w^(-1) lr((T_t^((1, w)) - bb(E)_pi T_t^((1, w))))$.
$square$

The lemma is therefore compatible with the triangular-array substitution
$w = alpha_n$: the finite-past domination is fixed-$w$, while every later
bound keeps the displayed $w$-dependence through $Phi_+(p,w)$ and uses
constants independent of $n$.

*Telescoping identity for $J^((1))$.* Summing the recursion
$J_k^((1, alpha)) = (I - alpha overline(A)) thin J_(k - 1)^((1, alpha))
  - alpha tilde(A)(Z_k) thin J_(k - 1)^((0, alpha))$
from $k = 1$ to $n$ and rearranging gives, for an arbitrary initial value,
$
overline(A) sum_(k = 0)^(n - 1) J_k^((1, alpha))
  = -sum_(k = 1)^n tilde(A)(Z_k) thin J_(k - 1)^((0, alpha))
    + frac(1, alpha) thin lr((J_0^((1, alpha)) - J_n^((1, alpha)))).
$
The stationary version of the same recursion yields
$bb(E)_pi [tilde(A)(Z_1) thin J_0^((0, alpha))]
= -overline(A) thin bb(E)_pi [J_infinity^((1, alpha))]$.
Under the stationary augmented-chain convention, for every $k$ the pair
$(J_(k - 1)^((0, w)), Z_k)$ has the same law as
$(J_0^((0, w)), Z_1)$. Thus the centered function
$overline(psi)_w$ in @lem:levin-cor-6 is centered for each summand in the
telescoping sum below; the index shift is only notational.
Subtracting $n thin bb(E)_pi [J_infinity^((1, alpha))]$ from both sides and
applying $overline(A)^(-1)$ gives the *centered telescoping identity*
$
sum_(k = 0)^(n - 1) lr((J_k^((1, alpha)) - bb(E)_pi [J_infinity^((1, alpha))]))
  = -overline(A)^(-1) sum_(k = 1)^n
      overline(psi)_alpha (J_(k - 1)^((0, alpha)), Z_k)
    + frac(1, alpha) thin overline(A)^(-1)
      lr((J_0^((1, alpha)) - J_n^((1, alpha)))).
$ <eq:J1-telescope>
This identity is the bridge between the (vector-valued) PR-average of
$J^((1))$ and the centered bilinear sum bounded in @lem:levin-cor-6.

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
    &quad + C_("mis,1") thin p^3 thin (alpha n)^(-1 slash 2) thin log^(1 slash p)(1 slash (alpha a))
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
The bias is deterministic; the centered piece carries all the $L_p$
fluctuation.

*Bias.* By @lem:levin-prop-2 applied at $w in {alpha, 2 alpha}$,
$bb(E)_pi [J_infinity^((1, w))] = w thin Delta + R(w)$ with
$|| R(w) || <= 12 thin || overline(A)^(-1) || thin C_A^2 thin t_"mix"^2 thin w^2 thin || epsilon.alt ||_infinity$.
The leading $w thin Delta$ cancels in the RR combination,
$2 alpha thin Delta - 2 alpha thin Delta = 0$, leaving
$
|| 2 thin bb(E)_pi [J_infinity^((1, alpha))] - bb(E)_pi [J_infinity^((1, 2 alpha))] ||
  <= 2 thin || R(alpha) || + || R(2 alpha) ||
  <= 72 thin || overline(A)^(-1) || thin C_A^2 thin t_"mix"^2 thin alpha^2 thin || epsilon.alt ||_infinity.
$
Multiplying by $sqrt(n)$ produces the first term of the bound.

*Centered.* Apply @eq:J1-telescope at $w in {alpha, 2 alpha}$, take $L_p$
norms, and combine via the triangle inequality:
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
For the bilinear sum apply @lem:levin-cor-6 with $r = n$,
$
|| sum_(k = 1)^n overline(psi)_w ||_(L_p)
  <= c_(W, 1) thin p^(3 slash 2) sqrt(w n)
   + c_(W, 2) thin p^3 thin w^(-1 slash 2) thin log^(1 slash p)(1 slash (w a)),
$
and divide by $sqrt(n)$, using $w in [alpha, 2 alpha]$ to upper-bound by the
$alpha$-form,
$
frac(1, sqrt(n)) || sum overline(psi)_w ||_(L_p)
  <= sqrt(2) c_(W, 1) thin p^(3 slash 2) sqrt(alpha)
   + sqrt(2) c_(W, 2) thin p^3 thin (alpha n)^(-1 slash 2) thin log^(1 slash p)(1 slash (alpha a)).
$
For the boundary term, apply the stationary-limit transfer
@lem:stationary-limit-J1 to
$T_n^((1, w)) = (I - w overline(A)) thin J_n^((1, w))$ with
$w <= alpha_("inv")$. This gives
$
|| J_n^((1, w)) - bb(E) J_n^((1, w)) ||_(L_p) <= C w thin Phi_+(p, w),
$
and $|| bb(E) J_n^((1, w)) || <= w thin || Delta || + || R(w) || <= C w$ by
@lem:levin-prop-2; combining,
$
|| J_n^((1, w)) ||_(L_p) <= C w thin Phi_+(p, w).
$
The same bound holds for $J_0^((1, w))$ under stationarity, so
$w^(-1) || J_0^((1, w)) - J_n^((1, w)) ||_(L_p)
<= C thin Phi_+(p, w) <= sqrt(2) thin C thin Phi_+(p, alpha)$ after absorbing
the factor $2$ into $C$.
Dividing by $sqrt(n)$ produces the last term. Summing the three RR-combined
pieces and absorbing universal factors into a single $C_("mis,1")$ completes
the proof. $square$

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

_Proof._ Triangle inequality on the $n$ summands of $T_n^((2))$, then
@lem:levin-prop-8 and @lem:levin-prop-9 applied uniformly in $k$ and
$w in {alpha, 2 alpha}$:
$
|| T_n^((2)) ||_(L_p)
  &<= frac(1, sqrt(n)) sum_(k = 0)^(n - 1) sum_(w in {alpha, 2 alpha}) |c_w| thin || J_k^((2, w)) ||_(L_p) \
  &<= 3 thin sqrt(n) thin sup_(w in {alpha, 2 alpha}) || J_k^((2, w)) ||_(L_p) \
  &<= 3 thin D_J thin (2 alpha)^(3 slash 2) thin t_"mix"^(5 slash 2) thin p^(7 slash 2) thin sqrt(n) thin log^(3 slash 2)(1 slash (alpha a)),
$
where $|c_alpha| + |c_(2 alpha)| = 3$ absorbs the RR combination and
$(2 alpha)^(3 slash 2)$ the worse bound at the larger step. Identical
argument for $T_n^((H))$, with the additional $d^(1 slash q)$ factor of
@lem:levin-prop-9. Adding the two bounds gives the lemma. $square$

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
     + C thin p^3 thin (alpha n)^(-1 slash 2) thin log^(1 slash p)(1 slash (alpha a))
     + C thin Phi_+(p, alpha) thin n^(-1 slash 2).
  $
] <thm:misadjustment>

_Proof._ Triangle inequality on @eq:R-mis-split:
$|| R_n^("mis, RR") ||_(L_p) <= || T_n^((1)) ||_(L_p) + || T_n^((2)) ||_(L_p) + || T_n^((H)) ||_(L_p)$.
Combine the centered $T^((1))$ bound and the raw $T^((2)) + T^((H))$ bound. $square$

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
  With $p asymp log n$ and $q >= log(e d)$, the factor $d^(1 slash q)$ is
  bounded by a universal constant, and the dominant order in
  the stationary misadjustment bound is $"polylog"(n) thin n^(-1 slash 4)$. $square$

The theorem just proved is only a stationary augmented-chain misadjustment
bound at $n_0 = 0$. A finite-start theorem with burn-in requires the burned-in
weights $Q_(l,n_0)^((alpha))$ and is not used in the assembly below.

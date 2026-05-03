#import "defs.typ": *

== PR-Averaged Error Decomposition and the RR Weight

The starting point for all subsequent estimates is the *explicit*
representation of $sqrt(n) (overline(theta)_n^(("RR", alpha)) - theta^*)$
as a noise-weighted sum with a deterministic kernel. This subsection
derives that representation step by step.

*Depth-one expansion.* Unfolding the error recursion of Chapter 1
gives, for every $k >= 1$,
$
theta_k^((alpha)) - theta^*
  = -alpha sum_(l = 1)^k Gamma_(l + 1 : k)^((alpha)) epsilon.alt(Z_l)
    + Gamma_(1 : k)^((alpha)) (theta_0 - theta^*),
quad
Gamma_(l + 1 : k)^((alpha))
  := product_(j = l + 1)^k (I - alpha A(Z_j)).
$
Replacing each random product $Gamma_(l + 1 : k)^((alpha))$ by its
deterministic counterpart $B_alpha^(k - l) := (I - alpha overline(A))^(k - l)$
and absorbing the difference into a higher-order remainder yields the
*depth-one decomposition* (Samsonov et al., 2025, Proposition 9):
$
theta_k^((alpha)) - theta^*
  = -alpha sum_(l = 1)^k B_alpha^(k - l) epsilon.alt(Z_l)
    + B_alpha^k (theta_0 - theta^*)
    + R_k^((alpha)),
$ <eq:depth-one>
where $R_k^((alpha)) := J_k^((1, alpha)) + H_k^((1, alpha))$ collects all
contributions of order $alpha^(3 slash 2)$ or higher in $L_p$. The first
sum is the *depth-zero* term and is the only piece that carries the
limiting Gaussian.

*PR averaging produces $Q_l^((alpha))$.* Recall the PR average
$overline(theta)_n^((alpha)) = (n - n_0)^(-1) sum_(k = n_0)^(n - 1) theta_k^((alpha))$.
For notational clarity we set $n_0 = 0$ from this point on (the burn-in
adds only an exponentially small transient). Subtracting $theta^*$,
substituting the depth-one decomposition above, and *exchanging the
order of summation* in the depth-zero piece,
$
sum_(k = 0)^(n - 1) sum_(l = 1)^k B_alpha^(k - l) epsilon.alt(Z_l)
  = sum_(l = 1)^(n - 1)
      lr((sum_(k = l)^(n - 1) B_alpha^(k - l)))
      epsilon.alt(Z_l),
$
isolates each noise sample $epsilon.alt(Z_l)$ together with the deterministic
*PR weight*
$
Q_l^((alpha))
  := alpha sum_(k = l)^(n - 1) B_alpha^(k - l)
   = alpha sum_(j = 0)^(n - l - 1) B_alpha^j.
$ <eq:Q-definition>
Operationally, $Q_l^((alpha))$ collects the cumulative contribution of
$epsilon.alt(Z_l)$ to all averaged iterates $theta_l, theta_(l + 1), dots,
theta_(n - 1)$. Multiplying the resulting identity by $sqrt(n)$ gives the
PR-averaged decomposition
$
sqrt(n) thin (overline(theta)_n^((alpha)) - theta^*) = W^((alpha)) + D^((alpha)),
$
where the *leading martingale-like sum* is
$
W^((alpha)) := -frac(1, sqrt(n)) sum_(l = 1)^(n - 1) Q_l^((alpha)) thin epsilon.alt(Z_l),
$ <eq:W-alpha>
and the remainder $D^((alpha))$ packs the deterministic transient $D_(op("tr"))^((alpha))$
and the higher-order stochastic part $D_R^((alpha))$:
$
D^((alpha)) := D_(op("tr"))^((alpha)) + D_R^((alpha)),
quad
D_(op("tr"))^((alpha))
  := frac(1, sqrt(n)) sum_(k = 0)^(n - 1) B_alpha^k (theta_0 - theta^*),
quad
D_R^((alpha))
  := frac(1, sqrt(n)) sum_(k = 0)^(n - 1) R_k^((alpha)).
$
The first sum is a deterministic geometric tail and becomes exponentially
small after a logarithmic burn-in $n_0 asymp log(n) slash (alpha a)$
(Lyapunov contraction $|| B_alpha^(n_0) ||_Q <= e^(-1)$). The second is
the source of the leading non-Gaussian correction in the Berry--Esseen
rate; its RR-combination $2 D_R^((alpha)) - D_R^((2 alpha))$ is exactly
the *misadjustment* $D_1^("mis, RR")$ controlled in Chapters 2 and 3.

*RR combination produces $cal(Q)_l^("RR")$.* The Richardson--Romberg
iterate $overline(theta)_n^(("RR", alpha)) := 2 overline(theta)_n^((alpha)) - overline(theta)_n^((2 alpha))$
inherits the PR decomposition *by linearity*: applying the previous
display at step sizes $alpha$ and $2 alpha$ separately and combining,
$
sqrt(n) thin (overline(theta)_n^(("RR", alpha)) - theta^*)
  = W^("RR") + D^("RR"),
$
$
W^("RR") := 2 W^((alpha)) - W^((2 alpha))
        = -frac(1, sqrt(n)) sum_(l = 1)^(n - 1) cal(Q)_l^("RR") thin epsilon.alt(Z_l),
$ <eq:W-RR>
where the *RR weight* is
$
cal(Q)_l^("RR") := 2 Q_l^((alpha)) - Q_l^((2 alpha)).
$ <eq:Q-RR-definition>
Crucially, both PR averages share the *same* noise realization
${Z_k}$, so the bracket $cal(Q)_l^("RR")$ is a single deterministic
matrix kernel: all the stochastic content of $W^("RR")$ now lives in
$epsilon.alt(Z_l)$ alone.

*Why $cal(Q)_l^("RR")$ is the right object.* Two questions about
$W^("RR")$ drive the rest of the chapter, and both answer themselves
in terms of $cal(Q)_l^("RR")$:

+ *Variance comparison.* The CLT identifies the limiting covariance of
  $W^("RR")$ as $Sigma_infinity = overline(A)^(-1) Sigma_epsilon.alt^(("M")) overline(A)^(-top)$
  (the Markov-chain CLT covariance), which is exactly what one obtains
  if every $cal(Q)_l^("RR")$ is replaced by the asymptotic weight
  $overline(A)^(-1)$. The finite-$n$ deviation is therefore controlled by
  $sum_l ||cal(Q)_l^("RR") - overline(A)^(-1)||^2$ — see Section 4.5.

+ *Poisson-equation / Abel-summation remainder.* The standard
  Berry--Esseen route for Markov-chain noise solves the Poisson equation
  $hat(epsilon.alt) - sans(Q) hat(epsilon.alt) = epsilon.alt$, replaces
  $epsilon.alt(Z_l)$ by $hat(epsilon.alt)(Z_l) - hat(epsilon.alt)(Z_(l + 1))$
  (up to a martingale increment), and Abel-sums against the weight
  sequence $cal(Q)_l^("RR")$. The resulting remainder has norm bounded
  by the *total variation* $sum_l ||cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")||$.

Both quantities are estimated in Sections 4.3--4.4 below. The closed-form
identities of Section 4.2 reduce them to elementary operator-norm
calculations on $B_alpha^m - B_(2 alpha)^m$.

== Closed-Form Identities

Throughout this chapter write
$
B_alpha := I - alpha overline(A),
quad
B_(2 alpha) := I - 2 alpha overline(A),
quad
k := n - l.
$
We assume $alpha, 2 alpha in (0, alpha_infinity]$, so that the Lyapunov contraction
$|| B_alpha^m ||_Q^2 <= (1 - alpha a)^m$ and $|| B_(2 alpha)^m ||_Q^2 <= (1 - 2 alpha a)^m$ hold (and we use freely $1 - 2 alpha a <= 1 - alpha a$).

The geometric-series identity $sum_(j = 0)^(m - 1) B_alpha^j = (alpha overline(A))^(-1)(I - B_alpha^m)$ converts <eq:Q-definition> into the closed form
$
Q_l^((alpha))
  = alpha sum_(j = 0)^(n - l - 1) B_alpha^j
  = alpha thin (alpha overline(A))^(-1) thin (I - B_alpha^(n - l))
  = overline(A)^(-1) (I - B_alpha^(n - l)).
$
This makes the convergence $Q_l^((alpha)) -> overline(A)^(-1)$ as $k = n - l -> infinity$ explicit: the only obstruction is the geometric Lyapunov tail $B_alpha^k$. Two immediate consequences are
$
Q_l^((alpha)) - overline(A)^(-1) = - overline(A)^(-1) B_alpha^(n - l),
quad
Q_(l + 1)^((alpha)) - Q_l^((alpha)) = - alpha B_alpha^(n - l - 1).
$
The first is the *asymptotic-weight error* and decays geometrically in $k$. The second is the *discrete derivative* used in Abel summation; note the explicit factor $alpha$ in front.

Applying both identities at $alpha$ and $2 alpha$ and combining yields the basic *RR identities*:
$
cal(Q)_l^("RR") - overline(A)^(-1)
  = - overline(A)^(-1) thin (2 B_alpha^k - B_(2 alpha)^k),
$
$
cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")
  = - 2 alpha thin (B_alpha^(k - 1) - B_(2 alpha)^(k - 1)).
$
These are the exact expressions bounded in the next subsection. Note the structural asymmetry between them:

- The asymptotic-weight error $2 B_alpha^k - B_(2 alpha)^k$ is at the *single-trajectory* rate $(1 - alpha a)^(k slash 2)$ — the triangle inequality $|2 X - Y| <= 2 |X| + |Y|$ does not see the RR coupling, and indeed the leading $-overline(A)^(-1)$ in $cal(Q)_l^("RR")$ is the same as in the single-step weight $Q_l^((alpha))$.
- The discrete derivative $B_alpha^(k - 1) - B_(2 alpha)^(k - 1)$, however, is a *true* difference of contractions evaluated at the same exponent. The elementary identity $X^m - Y^m = (X - Y) sum_(i = 1)^m X^(i - 1) Y^(m - i)$ with $X = B_alpha$, $Y = B_(2 alpha)$, $X - Y = alpha overline(A)$ extracts an extra factor of $alpha$, gaining one full power of $alpha$ over the single-step case $Q_(l + 1)^((alpha)) - Q_l^((alpha)) = -alpha B_alpha^(k - 1)$.

This local $alpha$-gain in the discrete derivative is the *only* manifestation of Richardson--Romberg cancellation visible at the level of the PR weights.

== Pointwise Bounds for the RR Weights

#lemma[
  Let $alpha, 2 alpha in (0, alpha_infinity]$, set $C_Q := kappa_Q^(1 slash 2) || overline(A)^(-1) ||$ and $tilde(C)_A := kappa_Q || overline(A) ||$, and write $k = n - l$.

  *(i)* For every $1 <= l <= n - 1$,
  $
  || cal(Q)_l^("RR") - overline(A)^(-1) || <= 3 C_Q (1 - alpha a)^(k slash 2).
  $

  *(ii)* For every $1 <= l <= n - 2$,
  $
  || cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR") ||
    <= 2 tilde(C)_A thin alpha^2 thin (k - 1) thin (1 - alpha a)^((k - 2) slash 2).
  $
]

_Proof of (i)._ Take norms in the basic RR identity above. Using the equivalence $|| dot || <= kappa_Q^(1 slash 2) || dot ||_Q$ and the Lyapunov contraction at both step sizes,
$
|| 2 B_alpha^k - B_(2 alpha)^k ||_Q
  <= 2 || B_alpha^k ||_Q + || B_(2 alpha)^k ||_Q
  <= 2 (1 - alpha a)^(k slash 2) + (1 - 2 alpha a)^(k slash 2)
  <= 3 (1 - alpha a)^(k slash 2).
$
Submultiplicativity in $|| dot ||_Q$ followed by norm equivalence then gives
$
|| cal(Q)_l^("RR") - overline(A)^(-1) ||
  <= kappa_Q^(1 slash 2) || overline(A)^(-1) || dot || 2 B_alpha^k - B_(2 alpha)^k ||_Q
  <= 3 C_Q (1 - alpha a)^(k slash 2).
$

_Proof of (ii)._ Apply the elementary identity
$X^m - Y^m = (X - Y) sum_(i = 1)^m X^(i - 1) Y^(m - i)$
with $X = B_alpha$, $Y = B_(2 alpha)$, $X - Y = alpha overline(A)$, and $m = k - 1$:
$
B_alpha^(k - 1) - B_(2 alpha)^(k - 1)
  = alpha overline(A) sum_(i = 1)^(k - 1)
    B_alpha^(i - 1) thin B_(2 alpha)^(k - 1 - i).
$
Each summand obeys, by submultiplicativity in $|| dot ||_Q$ and Lyapunov contraction,
$
|| B_alpha^(i - 1) thin B_(2 alpha)^(k - 1 - i) ||
  &<= kappa_Q^(1 slash 2) thin (1 - alpha a)^((i - 1) slash 2) (1 - 2 alpha a)^((k - 1 - i) slash 2) \
  &<= kappa_Q^(1 slash 2) thin (1 - alpha a)^((k - 2) slash 2),
$
where in the last step we used $1 - 2 alpha a <= 1 - alpha a$. Summing over $i in {1, dots, k - 1}$ and absorbing constants,
$
|| B_alpha^(k - 1) - B_(2 alpha)^(k - 1) ||
  <= alpha kappa_Q || overline(A) || (k - 1) (1 - alpha a)^((k - 2) slash 2).
$
Inserting this bound into the discrete-difference identity finishes the proof. $square$

== Summed Bounds and Comparison with the Single-Step Case

#corollary[
  Under the assumptions of the previous lemma, uniformly in $n >= 2$,
  $
  sum_(l = 1)^(n - 1) || cal(Q)_l^("RR") - overline(A)^(-1) ||^2
    <= frac(C_1, alpha a),
  quad
  sum_(l = 1)^(n - 2) || cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR") ||
    <= frac(C_2, a^2),
  $
  with $C_1 = 9 C_Q^2$ and $C_2 = 32 tilde(C)_A$.
]

_Proof._ For the first sum apply (i) and the geometric series $sum_(k >= 1) (1 - alpha a)^k <= 1 / (alpha a)$. For the second, apply (ii) and
$
sum_(k >= 2) (k - 1) thin (1 - alpha a)^((k - 2) slash 2)
  = sum_(m >= 0) (m + 1) (1 - alpha a)^(m slash 2)
  <= frac(1, (1 - sqrt(1 - alpha a))^2)
  <= frac(4, (alpha a)^2),
$
where the last step uses $1 - sqrt(1 - alpha a) >= alpha a / 2$ for $alpha a <= 1 slash 2$. Multiplying by $2 tilde(C)_A alpha^2$ gives the claim, up to the stated universal constant. $square$

#remark[
  Comparison with the single-step case. The same identities for $Q_l^((alpha))$ give
  $|| Q_l^((alpha)) - overline(A)^(-1) || <= C_Q (1 - alpha a)^(k slash 2)$ and
  $|| Q_(l + 1)^((alpha)) - Q_l^((alpha)) || <= alpha kappa_Q^(1 slash 2) (1 - alpha a)^((k - 1) slash 2)$.
  Part (i) of the lemma above is therefore the same order as the single-step bound: the leading
  $- overline(A)^(-1)$ term of $cal(Q)_(n - 1)^("RR")$ is exactly $- overline(A)^(-1)$ (since
  $cal(Q)_(n - 1)^("RR") = 0$ identically), so no factor of $alpha$ is gained on the boundary.
  The Richardson--Romberg cancellation is visible in the successive difference of
  (ii): for fixed lag $k$ the discrete derivative gains an extra factor of $alpha$.
  This local gain comes with the factor $k - 1$, so after summing over the geometric
  tail it should not be read as a global extra factor of $alpha$ in the Abel term.
  What is needed for the Poisson-equation remainder is the uniform total-variation
  bound in the corollary. On the variance side the corollary gives
  $sum || cal(Q)_l^("RR") - overline(A)^(-1) ||^2 = O(1 / (alpha a))$, which yields
  the finite-horizon comparison error $O(1 / (n alpha a))$.
]

== Variance Comparison

After the Poisson decomposition the leading martingale $W^("RR")$ has predictable
quadratic variation that converges to
$
Sigma_n^("RR") := frac(1, n) sum_(l = 1)^(n - 1)
  cal(Q)_l^("RR") thin Sigma_(epsilon.alt)^(("M")) thin (cal(Q)_l^("RR"))^top,
quad
sigma_n^(2, "RR")(u) := u^top Sigma_n^("RR") u,
$
with $Sigma_(epsilon.alt)^(("M"))$ the long-run noise covariance of the Markov chain,
$Sigma_(epsilon.alt)^(("M")) = bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_0)^top]
  + 2 sum_(j >= 1) bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_j)^top]$.
The asymptotic covariance is
$
Sigma_infinity = overline(A)^(-1) Sigma_(epsilon.alt)^(("M")) overline(A)^(-top),
quad
sigma^2(u) = u^top Sigma_infinity u.
$
The bounds of the previous section give a quantitative comparison between
$Sigma_n^("RR")$ and $Sigma_infinity$.

#lemma[
  Let $alpha, 2 alpha in (0, alpha_infinity]$, set $Sigma := Sigma_(epsilon.alt)^(("M"))$,
  and assume $|| Sigma || < infinity$. Then
  $
  || Sigma_n^("RR") - Sigma_infinity ||
    <= frac(C_3, n thin alpha a),
  $
  with $C_3 = 12 thin C_Q thin || overline(A)^(-1) || thin || Sigma || + 9 thin C_Q^2 thin || Sigma ||$.
  Consequently, for every $u in RR^d$,
  $
  | sigma_n^(2, "RR")(u) - sigma^2(u) |
    <= frac(C_3 thin || u ||^2, n thin alpha a).
  $
  At the working scale $alpha = c thin n^(- 1 slash 2)$ this is $O(n^(- 1 slash 2))$.
]

_Proof._ Write $Delta_l := cal(Q)_l^("RR") - overline(A)^(-1)$ and expand
$
cal(Q)_l^("RR") thin Sigma thin (cal(Q)_l^("RR"))^top - Sigma_infinity
  = underbrace(
      Delta_l thin Sigma thin overline(A)^(-top)
      + overline(A)^(-1) thin Sigma thin Delta_l^top,
      R_(1, l)
    )
  + underbrace(Delta_l thin Sigma thin Delta_l^top, R_(2, l)).
$
Submultiplicativity of the operator norm yields the pointwise bounds
$
|| R_(1, l) || <= 2 thin || overline(A)^(-1) || thin || Sigma || thin || Delta_l ||,
quad
|| R_(2, l) || <= || Sigma || thin || Delta_l ||^2.
$
Summing the linear part with the bound from part (i) of the previous lemma and the
geometric series $sum_(k >= 1) (1 - alpha a)^(k slash 2) <= 1 / (1 - sqrt(1 - alpha a)) <= 2 / (alpha a)$
(valid for $alpha a <= 1 slash 2$),
$
sum_(l = 1)^(n - 1) || Delta_l ||
  <= 3 C_Q sum_(k = 1)^(n - 1) (1 - alpha a)^(k slash 2)
  <= frac(6 C_Q, alpha a).
$
The quadratic part is bounded directly by the previous corollary,
$
sum_(l = 1)^(n - 1) || Delta_l ||^2 <= frac(9 C_Q^2, alpha a).
$
Combining,
$
|| Sigma_n^("RR") - Sigma_infinity ||
  &<= frac(1, n) sum_(l = 1)^(n - 1) (|| R_(1, l) || + || R_(2, l) ||) \
  &<= frac(1, n) (2 thin || overline(A)^(-1) || thin || Sigma || dot frac(6 C_Q, alpha a)
                + || Sigma || dot frac(9 C_Q^2, alpha a))
  = frac(C_3, n thin alpha a),
$
which proves the operator-norm bound. The scalar bound on
$| sigma_n^(2, "RR")(u) - sigma^2(u) | = | u^top (Sigma_n^("RR") - Sigma_infinity) u |$
follows from the Cauchy--Schwarz inequality. $square$

#remark[
  The constant $C_3$ is independent of $alpha$ and $n$; only $a$ enters through the
  Lyapunov contraction. The bound is tight in the geometric scale: the dominant
  contribution comes from the near-boundary region $l approx n$, where
  $|| Delta_l ||$ is bounded below by a positive constant (see the boundary identity
  $cal(Q)_(n - 1)^("RR") = 0$). The same calculation for the single-step PR weights
  yields $|| Sigma_n^("PR") - Sigma_infinity || <= C / (n alpha a)$, so
  Richardson--Romberg does not improve the variance comparison rate, but it also does
  not degrade it: the leading-order variance is preserved.
]

== Poisson Martingale Approximation

The variance comparison of the previous section identifies the limiting
covariance of $W^("RR")$, but it does not by itself produce a martingale.
This subsection converts $W^("RR")$ into a martingale plus a quantitatively
small remainder via the Poisson equation for the Markov chain ${Z_l}$. The
deterministic kernel bounds of Sections 4.3--4.4 enter once and for all in
the boundary/Abel control of the remainder.

*Poisson kernel.* Let $sans(Q)$ denote the one-step Markov transition kernel
of $(Z_k)_(k >= 1)$, acting on bounded measurable functions $f : Z -> bb(R)^d$ by
$
(sans(Q) f)(z) = bb(E) lr([f(Z_(k + 1)) | Z_k = z]).
$
Under UGE 1 with mixing time $t_"mix"$, the geometric Dobrushin bound
$|| sans(Q)^k epsilon.alt ||_infinity <= 2 || epsilon.alt ||_infinity (1 slash 4)^(floor(k slash t_"mix"))$
(valid for centered $epsilon.alt$, $pi(epsilon.alt) = 0$) makes the Poisson
series
$
hat(epsilon.alt) := sum_(k = 0)^infinity sans(Q)^k epsilon.alt
$
absolutely convergent in sup-norm, with
$
|| hat(epsilon.alt) ||_infinity
  <= 2 || epsilon.alt ||_infinity sum_(k = 0)^infinity (1 slash 4)^(floor(k slash t_"mix"))
  <= 3 thin t_"mix" thin || epsilon.alt ||_infinity,
quad
|| sans(Q) hat(epsilon.alt) ||_infinity
  <= || hat(epsilon.alt) ||_infinity,
$
the second inequality because $sans(Q)$ is a Markov kernel and contracts
the sup-norm. By construction $hat(epsilon.alt)$ solves the *Poisson equation*
$
hat(epsilon.alt) - sans(Q) hat(epsilon.alt) = epsilon.alt.
$ <eq:poisson-eq>

*Conditional centering.* Let $cal(F)_l := sigma(Z_1, dots, Z_l)$. Substituting
<eq:poisson-eq> into the noise sample and adding and subtracting the
conditional mean,
$
epsilon.alt(Z_l)
  = underbrace([hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1))], "martingale increment")
  + underbrace([sans(Q) hat(epsilon.alt)(Z_(l - 1)) - sans(Q) hat(epsilon.alt)(Z_l)], "telescope"),
quad l >= 2,
$
where the first bracket is centered conditionally on $cal(F)_(l - 1)$ since
the Markov property gives $bb(E)[hat(epsilon.alt)(Z_l) | cal(F)_(l - 1)] = sans(Q) hat(epsilon.alt)(Z_(l - 1))$.
The $l = 1$ term has no past state to condition on; we treat it directly via
<eq:poisson-eq> as $epsilon.alt(Z_1) = hat(epsilon.alt)(Z_1) - sans(Q) hat(epsilon.alt)(Z_1)$.
The decomposition therefore separates $W^("RR")$ into a martingale piece and
a deterministic-coefficient telescope; Abel summation against the kernel
sequence ${cal(Q)_l^("RR")}$ converts the telescope into boundary terms plus
a sum against the discrete derivative $cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")$,
which is exactly the object bounded in the Corollary of Section 4.4.

#lemma[
  Assume *UGE 1* and $pi(epsilon.alt) = 0$. Set
  $
  Delta M_l^("RR")
    := cal(Q)_l^("RR") thin (hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1))),
  quad 2 <= l <= n - 1,
  $
  and let $M_n^("RR") := sum_(l = 2)^(n - 1) Delta M_l^("RR")$. Then
  ${Delta M_l^("RR")}_(l = 2)^(n - 1)$ is a sequence of $cal(F)_l$-martingale
  differences, and
  $
  W^("RR")
    = -frac(1, sqrt(n)) M_n^("RR") + D_(2, n)^("RR"),
  $ <eq:poisson-decomp>
  with the *Poisson boundary/Abel remainder*
  $
  D_(2, n)^("RR")
    := -frac(1, sqrt(n)) lr([
        cal(Q)_1^("RR") thin hat(epsilon.alt)(Z_1)
        + sum_(l = 1)^(n - 2)
            (cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")) thin sans(Q) hat(epsilon.alt)(Z_l)
      ]).
  $
  (The right boundary $cal(Q)_(n - 1)^("RR") thin sans(Q) hat(epsilon.alt)(Z_(n - 1))$
  vanishes identically because $cal(Q)_(n - 1)^("RR") = 0$.) Moreover, with
  $C_(cal(Q)) := || overline(A)^(-1) || + 3 C_Q$ a uniform bound on
  $|| cal(Q)_l^("RR") ||$, and $C_2$ the constant from the Corollary of
  Section 4.4,
  $
  || D_(2, n)^("RR") ||_infinity
    <= frac(3 thin t_"mix" thin || epsilon.alt ||_infinity, sqrt(n))
       lr((C_(cal(Q)) + frac(C_2, a^2))).
  $ <eq:D2-bound>
  Consequently, for every $p >= 1$,
  $
  || D_(2, n)^("RR") ||_(L_p)
    <= frac(C thin t_"mix" thin || epsilon.alt ||_infinity, a^2 thin sqrt(n)),
  $
  with a constant $C$ depending only on $|| overline(A)^(-1) ||$, $kappa_Q$,
  and $|| overline(A) ||$.
]

_Proof._ The increments $Delta M_l^("RR")$ are $cal(F)_l$-martingale
differences because $cal(Q)_l^("RR")$ is deterministic and the Markov property
gives $bb(E)[hat(epsilon.alt)(Z_l) | cal(F)_(l - 1)] = sans(Q) hat(epsilon.alt)(Z_(l - 1))$.

For the decomposition, substitute <eq:poisson-eq> into the definition <eq:W-RR>:
$
W^("RR")
  = -frac(1, sqrt(n)) sum_(l = 1)^(n - 1)
    cal(Q)_l^("RR") thin (hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_l)).
$
Peel off $l = 1$ and apply the conditional-centering identity above to the
remaining $l in {2, dots, n - 1}$:
$
W^("RR")
  = -frac(1, sqrt(n)) cal(Q)_1^("RR") thin hat(epsilon.alt)(Z_1)
    + frac(1, sqrt(n)) cal(Q)_1^("RR") thin sans(Q) hat(epsilon.alt)(Z_1)
    - frac(1, sqrt(n)) M_n^("RR")
    - frac(1, sqrt(n)) sum_(l = 2)^(n - 1)
      cal(Q)_l^("RR") thin (sans(Q) hat(epsilon.alt)(Z_(l - 1)) - sans(Q) hat(epsilon.alt)(Z_l)).
$
Set $g_l := sans(Q) hat(epsilon.alt)(Z_l)$ and Abel-sum the last sum:
$
sum_(l = 2)^(n - 1) cal(Q)_l^("RR") (g_(l - 1) - g_l)
  = cal(Q)_2^("RR") thin g_1
    - cal(Q)_(n - 1)^("RR") thin g_(n - 1)
    + sum_(l = 2)^(n - 2)
      (cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")) thin g_l.
$
Adding the contribution $cal(Q)_1^("RR") thin g_1$ from the peeled $l = 1$
term combines with $-cal(Q)_2^("RR") thin g_1$ to yield
$-(cal(Q)_2^("RR") - cal(Q)_1^("RR")) thin g_1$, which folds the boundary
$l = 1$ summand back into the discrete-derivative sum:
$
&-cal(Q)_1^("RR") thin hat(epsilon.alt)(Z_1)
+ cal(Q)_1^("RR") thin g_1
- cal(Q)_2^("RR") thin g_1
+ cal(Q)_(n - 1)^("RR") thin g_(n - 1)
- sum_(l = 2)^(n - 2)
    (cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")) thin g_l \
&quad= -cal(Q)_1^("RR") thin hat(epsilon.alt)(Z_1)
       - sum_(l = 1)^(n - 2)
          (cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")) thin g_l
       + cal(Q)_(n - 1)^("RR") thin g_(n - 1).
$
The rightmost boundary $cal(Q)_(n - 1)^("RR") thin g_(n - 1)$ vanishes because
$cal(Q)_(n - 1)^("RR") = 2 alpha I - 2 alpha I = 0$ identically (from
$Q_(n - 1)^((alpha)) = alpha I$ and $Q_(n - 1)^((2 alpha)) = 2 alpha I$).
Multiplying by $1 slash sqrt(n)$ and absorbing the sign into $D_(2, n)^("RR")$
produces exactly the stated formula.

For the sup-norm bound <eq:D2-bound>, use $|| g_l ||_infinity <= || hat(epsilon.alt) ||_infinity <= 3 t_"mix" || epsilon.alt ||_infinity$,
the uniform bound $|| cal(Q)_1^("RR") || <= C_(cal(Q))$ on the left boundary,
and the summed-total-variation bound
$sum_(l = 1)^(n - 2) || cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR") || <= C_2 slash a^2$
on the Abel sum:
$
|| sqrt(n) thin D_(2, n)^("RR") ||
  <= 3 t_"mix" || epsilon.alt ||_infinity (C_(cal(Q)) + C_2 slash a^2).
$
The $L_p$ bound is immediate from the deterministic sup-norm bound. $square$

#remark[
  *Two structural features.* First, the bound is *deterministic*: no
  martingale concentration or moment inequality is invoked, only pointwise
  bounds on $cal(Q)_l^("RR")$, on its discrete derivative, and on
  $hat(epsilon.alt)$. The dependence on $p$ promised in the working plan
  ($"poly"(p, t_"mix")$) is therefore absorbed into the constant; the
  effective scaling is $|| D_(2, n)^("RR") ||_(L_p) <= C thin t_"mix" slash (a^2 sqrt(n))$.
  Second, the $1 slash a^2$ factor is *exactly* the cost of the discrete
  derivative summed Abel-style: it is the same $C_2 slash a^2$ that appears
  in the second part of the Corollary of Section 4.4. No further $1 slash alpha$
  blow-up enters, because the kernel-difference bound of Section 4.3 already
  carries the cancelling $alpha^2 (k - 1)$ prefactor.

  *Boundary asymmetry.* The right-end boundary vanishes because
  $cal(Q)_(n - 1)^("RR") = 0$ — a Richardson--Romberg cancellation that does
  not occur for the single-step PR weight $Q_(n - 1)^((alpha)) = alpha I$,
  which contributes a (small) $alpha t_"mix" slash sqrt(n)$ residue. RR
  therefore *removes* one of the two Poisson-boundary terms outright.

  *Comparison with martingale scale.* The leading martingale piece
  $M_n^("RR") slash sqrt(n)$ has $L_2$-norm of order $1$ (it is the natural
  $sqrt(n)$-scaled object whose CLT variance equals $sigma_n^(2, "RR")(u)$ at
  leading order, by the variance-comparison lemma). The remainder
  $D_(2, n)^("RR")$ is therefore $O(n^(-1 slash 2))$ relative to it, well
  below the target Berry--Esseen scale $n^(-1 slash 4)$.
]

== Predictable Quadratic Variation Concentration

The Berry--Esseen step for the martingale $M_n^("RR")$ requires a quantitative
control of its predictable quadratic variation
$chevron.l M^("RR") chevron.r_n$ around its deterministic asymptotic counterpart
$n thin sigma_n^(2, "RR")(u)$. This subsection produces such a control by
applying the Markov concentration result of Section 2.2 to a suitable centered
quadratic functional of the chain. The result is the RR analogue of Lemmas
22--23 of Samsonov et al. (2025).

*Predictable quadratic variation.* By the Poisson Martingale Approximation
lemma above, the increments
$Delta M_l^("RR") = cal(Q)_l^("RR") thin (hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)))$
are $cal(F)_l$-martingale differences. The Markov property
$bb(E)[hat(epsilon.alt)(Z_l) | cal(F)_(l - 1)] = sans(Q) hat(epsilon.alt)(Z_(l - 1))$
gives, by direct computation,
$
bb(E)[Delta M_l^("RR") thin (Delta M_l^("RR"))^top thin | thin cal(F)_(l - 1)]
  &= cal(Q)_l^("RR")
     bb(E)[hat(epsilon.alt)(Z_l) thin hat(epsilon.alt)(Z_l)^top | cal(F)_(l - 1)]
     (cal(Q)_l^("RR"))^top \
  &quad - cal(Q)_l^("RR") thin sans(Q) hat(epsilon.alt)(Z_(l - 1))
        thin (sans(Q) hat(epsilon.alt))(Z_(l - 1))^top thin
        (cal(Q)_l^("RR"))^top \
  &= cal(Q)_l^("RR") thin overline(epsilon.alt)(Z_(l - 1)) thin (cal(Q)_l^("RR"))^top,
$
where the cross-terms in the expansion of
$(hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)))(hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)))^top$
cancel by the Markov property and we have set
$
overline(epsilon.alt)(z)
  := sans(Q)(hat(epsilon.alt) hat(epsilon.alt)^top)(z)
   - (sans(Q) hat(epsilon.alt))(z) thin (sans(Q) hat(epsilon.alt))(z)^top.
$ <eq:bar-eps-def>
Summing over $l in {2, dots, n - 1}$ and tracking that
$chevron.l M^("RR") chevron.r_n
:= sum_(l = 2)^(n - 1) bb(E)[Delta M_l^("RR") thin (Delta M_l^("RR"))^top | cal(F)_(l - 1)]$,
$
chevron.l M^("RR") chevron.r_n
  = sum_(l = 2)^(n - 1)
    cal(Q)_l^("RR") thin overline(epsilon.alt)(Z_(l - 1)) thin (cal(Q)_l^("RR"))^top.
$ <eq:M-RR-bracket>
The function $overline(epsilon.alt)$ has stationary mean equal to the long-run
noise covariance,
$
pi(overline(epsilon.alt)) = Sigma_(epsilon.alt)^(("M")) =: Sigma,
$
which is a standard identity for the Poisson solution of a Markov chain
(Samsonov et al. 2025, Eq. (10); see also Douc--Moulines--Priouret--Soulier
2018, Theorem 21.2.5).

*Sup-norm bound on $overline(epsilon.alt)$.* Since
$|| hat(epsilon.alt) ||_infinity <= 3 thin t_"mix" thin || epsilon.alt ||_infinity$
and $sans(Q)$ is a Markov kernel (so $sans(Q) f$ inherits the sup-norm of $f$),
$
|| overline(epsilon.alt) ||_infinity
  &<= || sans(Q)(hat(epsilon.alt) hat(epsilon.alt)^top) ||_infinity
   + || (sans(Q) hat(epsilon.alt))(sans(Q) hat(epsilon.alt))^top ||_infinity \
  &<= 2 thin || hat(epsilon.alt) ||_infinity^2
   <= 18 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2.
$ <eq:bar-eps-sup>

#lemma[
  Assume *UGE 1* and $pi(epsilon.alt) = 0$, with $|| epsilon.alt ||_infinity < infinity$.
  Let $C_(cal(Q))$ be the uniform bound on $|| cal(Q)_l^("RR") ||$ from the
  previous lemma. There exists a universal constant $C_4 > 0$ such that, for
  every $u in bb(R)^d$, every $p >= 2$, every initial distribution $xi$, and every
  $n >= 2$,
  $
  bb(E)_xi^(1 slash p) lr([
    | u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u) |^p
  ])
    <= C_4 thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2
       thin t_"mix"^(5 slash 2) thin sqrt(p thin n).
  $ <eq:M-RR-conc>
] <lem:M-RR-bracket-conc>

_Proof._ Write $h_l(z) := u^top cal(Q)_l^("RR") thin overline(epsilon.alt)(z) thin (cal(Q)_l^("RR"))^top u$
and $g_l(z) := h_l(z) - pi(h_l)$. Submultiplicativity of the operator norm and
<eq:bar-eps-sup> give
$
|h_l(z)|
  <= C_(cal(Q))^2 thin || u ||^2 thin || overline(epsilon.alt) ||_infinity
  <= 18 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2,
quad
|| g_l ||_infinity <= 2 thin |h_l(z)|
  <= 36 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2.
$
By <eq:M-RR-bracket> and the variance definition
$n thin sigma_n^(2, "RR")(u) = sum_(l = 1)^(n - 1) pi(h_l)$,
$
u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u)
  = sum_(l = 2)^(n - 1) g_l(Z_(l - 1)) - pi(h_1).
$ <eq:M-RR-conc-decomp>
The boundary scalar is deterministic and obeys
$|pi(h_1)| <= C_(cal(Q))^2 thin || u ||^2 thin || Sigma ||
       <= 18 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2$,
where the last step uses $|| Sigma || = || pi(overline(epsilon.alt)) ||
<= || overline(epsilon.alt) ||_infinity$ and <eq:bar-eps-sup>.

For the centered sum, set $tilde(g)_i := g_(i + 1)$ for $i in {1, dots, n - 2}$. By
construction $pi(tilde(g)_i) = 0$ and
$|| tilde(g)_i ||_infinity <= c := 36 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2$.
The Markov concentration lemma of Section 2.2 (Levin et al. 2025, Lemma 11) applied to
$sum_(i = 1)^(n - 2) tilde(g)_i(Z_i)$ yields, for every $t >= 0$,
$
bb(P)_xi lr((
  lr(|sum_(i = 1)^(n - 2) tilde(g)_i(Z_i)|) >= t
))
  <= 2 exp(-frac(t^2, 2 thin u_n^2)),
quad
u_n^2 <= 64 thin t_"mix" thin (n - 2) thin c^2.
$
The sub-Gaussian-to-moment lemma of Section 2.2 then gives, for $p >= 2$,
$
bb(E)_xi^(1 slash p) lr([
  lr(|sum_(l = 2)^(n - 1) g_l(Z_(l - 1))|)^p
])
  <= 2^(1 slash p) thin sqrt(p) thin u_n
  <= 2 sqrt(p) thin u_n
  <= 16 sqrt(2) thin sqrt(p (n - 2)) thin c thin sqrt(t_"mix").
$
Substituting $c$ and bounding $sqrt(n - 2) <= sqrt(n)$,
$
bb(E)_xi^(1 slash p) lr([
  lr(|sum_(l = 2)^(n - 1) g_l(Z_(l - 1))|)^p
])
  <= 16 sqrt(2) dot 36 thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2
       thin t_"mix"^(5 slash 2) thin sqrt(p thin n)
  = 576 sqrt(2) thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2
       thin t_"mix"^(5 slash 2) thin sqrt(p thin n).
$
The boundary contribution satisfies
$|pi(h_1)| <= 18 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2
<= 18 thin C_(cal(Q))^2 thin || u ||^2 thin t_"mix"^(5 slash 2) thin || epsilon.alt ||_infinity^2 thin sqrt(p thin n)$
for every $p, n >= 1$ (using $t_"mix" >= 1$ and $sqrt(p thin n) >= 1$). Adding
the two pieces in <eq:M-RR-conc-decomp> by the triangle inequality and rounding
the resulting prefactor to $C_4 := 850$ gives <eq:M-RR-conc>. $square$

#corollary[
  Under the assumptions of the previous lemma, for every $u in bb(R)^d$, every
  $p >= 2$, and every $n >= 2$,
  $
  bb(E)_xi^(1 slash p) lr([
    | u^top chevron.l M^("RR") chevron.r_n u - n thin sigma^2(u) |^p
  ])
    <= C_4 thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2
       thin t_"mix"^(5 slash 2) thin sqrt(p thin n)
       + frac(C_3 thin || u ||^2, alpha thin a),
  $ <eq:M-RR-conc-sigma>
  with $C_3$ the variance-comparison constant of Section 4.5.
]

_Proof._ The triangle inequality and the variance-comparison lemma give
$
| u^top chevron.l M^("RR") chevron.r_n u - n thin sigma^2(u) |
  <= | u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u) |
   + n thin | sigma_n^(2, "RR")(u) - sigma^2(u) |.
$
The first piece is bounded in $L_p$ by <eq:M-RR-conc>; the second is the
deterministic bound $n thin C_3 thin || u ||^2 slash (n thin alpha thin a) = C_3 || u ||^2 slash (alpha a)$.
$square$

#remark[
  *Working scale and Berry--Esseen consequence.* The bound
  $|| u^top chevron.l M^("RR") chevron.r_n u - n sigma_n^(2, "RR")(u) ||_(L_p)
   <= C thin t_"mix"^(5 slash 2) thin sqrt(p thin n)$
  is exactly what feeds into the martingale Berry--Esseen of
  Bolthausen--Fan type (cf. Lemma 21 / Proposition 13 of Samsonov et al. 2025): the
  conditional-variance term in the Bolthausen--Fan inequality is
  $sqrt(p) thin s_n^(- 2 p slash (2 p + 1)) thin
   bb(E)^(1 slash (2 p + 1))[|sum sigma_l^2 - s_n^2|^p]$,
  which with $s_n^2 = n thin sigma_n^(2, "RR")(u) asymp n$ (uniformly in
  $alpha$ at the working scale) and the exponent
  $sqrt(p n) -> n^(p slash (2 p + 1))$ gives the standard $n^(- 1 slash 4)$
  rate after taking $p = ceil(log n)$.

  *RR vs single-step structure.* The bound is structurally identical to the
  single-step PR case (Samsonov et al. 2025, Lemma 23): the only RR-specific
  input is the uniform sup bound $|| cal(Q)_l^("RR") || <= C_(cal(Q))$. No extra
  $1 slash a$ or $1 slash alpha$ factor enters, because the Markov concentration
  is applied to a *bounded* function and the weights $cal(Q)_l^("RR")$ enter
  only through their sup-norm. Variance comparison (Section 4.5) is the only
  place where $1 slash (alpha thin a)$ appears, and at the working scale
  $alpha = c thin n^(- 1 slash 2)$ this contributes $O(sqrt(n))$ — the same
  order as the leading concentration term.

  *Why we centred against $sigma_n^(2, "RR")(u)$.* The natural normalisation in
  the Bolthausen--Fan martingale BE step is $s_n^2 = sum_l sigma_l^2$, which is
  precisely $n thin sigma_n^(2, "RR")(u)$ for the increments
  $Delta M_l^("RR")$. Centering against $n thin sigma^2(u)$ (corollary above) is
  used in the corollary that converts the BE statement from the
  $sigma_n^("RR")(u)$ normalisation to the asymptotic $sigma(u)$ normalisation.
]

== Martingale Berry--Esseen Step

The Poisson decomposition (Section 4.6) writes
$W^("RR") = -n^(-1 slash 2) M_n^("RR") + D_(2, n)^("RR")$, with the remainder
$D_(2, n)^("RR")$ controlled in sup-norm at the order $n^(-1 slash 2)$. The
quadratic variation concentration (Section 4.7) bounds
$|u^top chevron.l M^("RR") chevron.r_n u - n thin sigma_n^(2, "RR")(u)|$ in
$L_p$ by $C thin sqrt(p thin n)$. This subsection assembles those two inputs
into a Berry--Esseen rate for the *scalar* martingale $u^top M_n^("RR")$,
the leading contribution to the final RR Berry--Esseen bound.

*Bounded increments.* Fix $u in bb(R)^d$ and set
$X_l := u^top Delta M_l^("RR")$ for $2 <= l <= n - 1$. The $X_l$ are
$cal(F)_l$-martingale differences (Section 4.6), and bounding the three
factors $|| u ||$, $|| cal(Q)_l^("RR") || <= C_(cal(Q))$, and
$|| hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)) || <= 2 || hat(epsilon.alt) ||_infinity <= 6 t_"mix" || epsilon.alt ||_infinity$
yields the deterministic bound
$
|X_l| <= kappa.alt(u),
quad
kappa.alt(u) := 6 thin t_"mix" thin C_(cal(Q)) thin || epsilon.alt ||_infinity thin || u ||.
$ <eq:M-RR-incr>
The conditional variances $sigma_l^2 := bb(E)[X_l^2 | cal(F)_(l - 1)]$ sum to
the predictable quadratic variation along $u$,
$
V_n^2 := sum_(l = 2)^(n - 1) sigma_l^2 = u^top chevron.l M^("RR") chevron.r_n u.
$

*Variance lower bound.* Set $s_n^2 := n thin sigma_n^(2, "RR")(u)$. The
variance comparison lemma (Section 4.5) and assumption $sigma^2(u) > 0$ give
$|sigma_n^(2, "RR")(u) - sigma^2(u)| <= C_3 || u ||^2 slash (n alpha a)$, so
whenever
$
n thin alpha thin a >= frac(2 thin C_3 thin || u ||^2, sigma^2(u))
$ <eq:variance-lb-condition>
one has $sigma_n^(2, "RR")(u) >= sigma^2(u) slash 2$, i.e.
$s_n^2 >= n sigma^2(u) slash 2$. At the working scale
$alpha = c thin n^(-1 slash 2)$, condition <eq:variance-lb-condition> is
satisfied for $n >= (2 C_3 || u ||^2 slash (c thin a thin sigma^2(u)))^2$.
The trivial upper bound
$sigma_n^(2, "RR")(u) <= C_(cal(Q))^2 || Sigma_(epsilon.alt)^(("M")) || || u ||^2$
gives $s_n^2 <= K^2(u) thin n$ with
$K(u) := C_(cal(Q)) thin || Sigma_(epsilon.alt)^(("M")) ||^(1 slash 2) thin || u ||$.

*Bolthausen--Fan inequality.* We apply Lemma 21 of Samsonov et al. (2025),
which combines Bolthausen (1982) with Fan (2019). For any sequence of
$cal(F)_l$-martingale differences ${X_l}$ with $|X_l| <= kappa.alt$, any
$p >= 1$,
$
d_K (S_n slash s_n, cal(N)(0, 1))
  &<= L_B thin frac((2 n + 1) log(2 n + 1) thin kappa.alt^3, s_n^3) \
  &quad + C_1 thin sqrt(p) thin s_n^(- 2 p slash (2 p + 1)) thin
       (bb(E) | V_n^2 - s_n^2 |^p)^(1 slash (2 p + 1)) \
  &quad + C_2 thin s_n^(- 2 p slash (2 p + 1)) thin p thin
       kappa.alt^(2 p slash (2 p + 1)),
$ <eq:bolthausen-fan>
with $S_n := sum_l X_l$, $L_B$ the universal Bolthausen constant, and
$C_1, C_2$ universal.

#theorem[
  Assume *UGE 1*, $pi(epsilon.alt) = 0$, $|| epsilon.alt ||_infinity < infinity$,
  $sigma^2(u) > 0$, and $alpha, 2 alpha in (0, alpha_infinity]$. There exist
  constants $C_(K, 1)(u), C_(K, 2)(u) > 0$ depending only on $|| u ||$,
  $sigma(u)$, $C_(cal(Q))$, $t_"mix"$, $|| epsilon.alt ||_infinity$,
  $|| Sigma_(epsilon.alt)^(("M")) ||$, and the universal $L_B, C_1, C_2$ of
  <eq:bolthausen-fan>, such that for every $n >= 3$ satisfying
  <eq:variance-lb-condition>,
  $
  d_K lr((
    frac(u^top M_n^("RR"), sqrt(n) thin sigma_n^("RR")(u)),
    cal(N)(0, 1)
  ))
    <= frac(C_(K, 1)(u) thin log^(3 slash 4) n, n^(1 slash 4))
     + frac(C_(K, 2)(u) thin log n, sqrt(n)).
  $ <eq:M-RR-BE>
] <thm:M-RR-BE>

_Proof._ Apply <eq:bolthausen-fan> with $S_n = u^top M_n^("RR")$, the
increment bound $kappa.alt = kappa.alt(u)$ from <eq:M-RR-incr>,
$s_n^2 = n thin sigma_n^(2, "RR")(u)$, and $p = ceil(log n)$ (so
$log n <= p <= log n + 1$). Bound the three terms in turn, using
$s_n^2 in [n thin sigma^2(u) slash 2, thin K^2(u) thin n]$ from the variance
lower and upper bounds.

*Term I (classical Bolthausen).* From $s_n^3 >= (n thin sigma^2(u) slash 2)^(3 slash 2)$
and $(2 n + 1) log(2 n + 1) <= 3 thin n thin log n$ for $n >= 3$,
$
L_B thin frac((2 n + 1) log(2 n + 1) thin kappa.alt(u)^3, s_n^3)
  <= frac(6 sqrt(2) thin L_B thin kappa.alt(u)^3, sigma^3(u)) thin frac(log n, sqrt(n))
  =: frac(C^("(I)")(u) thin log n, sqrt(n)).
$ <eq:term-I>

*Term III (Lindeberg).* Write $a_p := 2 p slash (2 p + 1) = 1 - 1 slash (2 p + 1)$.
Using $s_n^(- a_p) = s_n^(-1) thin s_n^(1 slash (2 p + 1))$ and
$s_n <= K(u) thin sqrt(n)$,
$
s_n^(1 slash (2 p + 1)) <= max(1, K(u))^(1 slash (2 p + 1)) thin n^(1 slash (2 (2 p + 1))).
$
For $p >= log n$ one has $n^(1 slash (2 (2 p + 1))) <= n^(1 slash (4 log n)) = e^(1 slash 4)$;
similarly $max(1, K(u))^(1 slash (2 p + 1)) <= e^(1 slash 2)$ once
$n >= max(1, K(u))$, which is absorbed into $C^("(III)")(u)$. Bounding
$kappa.alt(u)^(a_p) <= max(1, kappa.alt(u))$ and $s_n^(-1) <= sqrt(2) slash (sigma(u) sqrt(n))$,
$
C_2 thin s_n^(- a_p) thin p thin kappa.alt(u)^(a_p)
  <= sqrt(2) thin e^(3 slash 4) thin frac(C_2 thin max(1, kappa.alt(u)), sigma(u)) thin frac(p, sqrt(n))
  <= frac(C^("(III)")(u) thin log n, sqrt(n)),
$ <eq:term-III>
with $C^("(III)")(u)$ depending on $|| u ||, sigma(u), C_(cal(Q)),
|| Sigma_(epsilon.alt)^(("M")) ||, t_"mix", || epsilon.alt ||_infinity$ and on
$C_2$.

*Term II (conditional-variance concentration).* By Lemma 4.7
(equation <eq:M-RR-conc>), for every $p >= 2$,
$
(bb(E) | V_n^2 - s_n^2 |^p)^(1 slash p)
  <= B(u) thin sqrt(p thin n),
quad
B(u) := C_4 thin C_(cal(Q))^2 thin || u ||^2 thin || epsilon.alt ||_infinity^2 thin t_"mix"^(5 slash 2),
$ <eq:Bu-def>
hence $(bb(E) | V_n^2 - s_n^2 |^p)^(1 slash (2 p + 1)) <= B(u)^(p slash (2 p + 1)) thin (p thin n)^(p slash (2 (2 p + 1)))$.
Combining with $s_n^(-1) <= sqrt(2 slash (sigma^2(u) thin n))$ and
$s_n^(1 slash (2 p + 1)) <= max(1, K(u))^(1 slash (2 p + 1)) thin n^(1 slash (2 (2 p + 1)))$,
$
&C_1 thin sqrt(p) thin s_n^(- a_p) thin (bb(E) | V_n^2 - s_n^2 |^p)^(1 slash (2 p + 1)) \
&quad<= C_1 thin sqrt(p) thin sqrt(2 slash (sigma^2(u) n))
        thin max(1, K(u))^(1 slash (2 p + 1)) thin n^(1 slash (2 (2 p + 1))) \
&quad quad times B(u)^(p slash (2 p + 1)) thin p^(p slash (2 (2 p + 1))) thin n^(p slash (2 (2 p + 1))) \
&quad= C_1 thin sqrt(2 slash sigma^2(u)) thin max(1, K(u))^(1 slash (2 p + 1)) thin
        B(u)^(p slash (2 p + 1)) thin p^((3 p + 1) slash (2 (2 p + 1))) thin
        n^(- p slash (2 (2 p + 1))).
$ <eq:term-II-1>
At $p = ceil(log n) >= log n$, three elementary bounds hold:
$
&p^((3 p + 1) slash (2 (2 p + 1))) <= p^(3 slash 4) <= 2^(3 slash 4) thin (log n)^(3 slash 4),
$
$
&n^(- p slash (2 (2 p + 1))) = n^(- 1 slash 4 + 1 slash (4 (2 p + 1))) <= e^(1 slash 8) thin n^(- 1 slash 4),
$
$
&B(u)^(p slash (2 p + 1)) <= max(1, sqrt(B(u))) thin e^(1 slash 2),
$
where the first uses $(3 p + 1) slash (2 (2 p + 1)) <= 3 slash 4$ (equivalent to
$4(3p + 1) <= 6(2 p + 1)$, true), the second uses
$log(n^(1 slash (4 (2 p + 1)))) = log(n) slash (4 (2 p + 1)) <= 1 slash 8$ for
$p >= log n$, and the third uses $|p slash (2 p + 1) - 1 slash 2| <= 1 slash (2 (2 p + 1))$
together with $log B(u) slash (2 p + 1) <= 1$ for $n >= max(1, B(u))$
(absorbed into $C^("(II)")(u)$). Likewise
$max(1, K(u))^(1 slash (2 p + 1)) <= e^(1 slash 2)$ for $n >= K(u)$. Substituting,
$
"Term II"
  &<= C_1 thin sqrt(2 slash sigma^2(u)) thin e^(1 slash 2) thin max(1, sqrt(B(u))) thin e^(1 slash 2) thin
       2^(3 slash 4) thin (log n)^(3 slash 4) thin e^(1 slash 8) thin n^(- 1 slash 4) \
  &<= 2^(3 slash 4) thin e^(9 slash 8) thin C_1 thin
       sqrt(2 max(1, B(u)) slash sigma^2(u)) thin
       frac(log^(3 slash 4) n, n^(1 slash 4))
  =: frac(C^("(II)")(u) thin log^(3 slash 4) n, n^(1 slash 4)).
$ <eq:term-II>

Adding <eq:term-I>, <eq:term-II>, <eq:term-III> and setting
$C_(K, 1)(u) := C^("(II)")(u)$, $C_(K, 2)(u) := C^("(I)")(u) + C^("(III)")(u)$
proves <eq:M-RR-BE>. $square$

#corollary[
  Under the hypotheses of the previous theorem,
  $
  d_K lr((
    frac(u^top M_n^("RR"), sqrt(n) thin sigma(u)),
    cal(N)(0, 1)
  ))
    <= frac(C_(K, 1)(u) thin log^(3 slash 4) n, n^(1 slash 4))
     + frac(C_(K, 2)(u) thin log n, sqrt(n))
     + frac(C_3 thin || u ||^2, n thin alpha thin a thin sigma^2(u)).
  $ <eq:M-RR-BE-sigma>
  At $alpha = c thin n^(-1 slash 2)$ the last term is $O(n^(-1 slash 2))$, hence
  absorbed into $C_(K, 2)(u) thin log n slash sqrt(n)$ up to a constant.
] <cor:M-RR-BE-sigma>

_Proof._ Set $r := sigma(u) slash sigma_n^("RR")(u)$. Under
<eq:variance-lb-condition>, $r^2 in [1, 2]$ and
$|r - 1| = |r^2 - 1| slash (r + 1) <= |r^2 - 1| slash 2 <= C_3 || u ||^2 slash (2 thin n thin alpha thin a thin sigma^2(u))$
by variance comparison (Section 4.5). For any random variable $X$ and $r > 0$,
$d_K(X slash r, cal(N)) <= d_K(X, cal(N)) + |r - 1| thin (2 pi)^(-1 slash 2)$
(uniformly in $X$, by the $1 slash sqrt(2 pi)$ Lipschitz constant of the
standard normal cdf evaluated against multiplicative perturbations of the
argument). Applying this to $X = u^top M_n^("RR") slash (sqrt(n) sigma_n^("RR")(u))$
and noting $X slash r = u^top M_n^("RR") slash (sqrt(n) sigma(u))$ gives the
stated bound after absorbing the universal $(2 pi)^(-1 slash 2)$ into the
constants. $square$

#remark[
  *Working scale and rate.* At $alpha = c thin n^(-1 slash 2)$ the leading
  rate $log^(3 slash 4)(n) thin n^(-1 slash 4)$ matches the single-step PR
  Berry--Esseen of Samsonov et al. (2025, Proposition 13). Richardson--Romberg
  neither improves nor degrades the martingale Berry--Esseen rate; the
  cancellation acts on the *misadjustment* remainder, not on the leading
  martingale.

  *Constants.* The dominant $C_(K, 1)(u)$ scales as
  $sqrt(B(u) slash sigma^2(u))$ with $B(u)$ the conditional-variance
  concentration constant of <eq:Bu-def>: explicitly $C_(K, 1)(u) asymp
  C_(cal(Q)) || u ||^2 thin t_"mix"^(5 slash 4) thin || epsilon.alt ||_infinity slash sigma(u)$.
  The $sigma(u)^(-1)$ blow-up at degenerate directions is genuine and
  reflects the assumption $sigma^2(u) > 0$.

  *Scope.* This theorem bounds the Kolmogorov distance for the martingale
  $u^top M_n^("RR")$ alone, not for the full PR-averaged RR iterate
  $sqrt(n) thin u^top (overline(theta)_n^(("RR", alpha)) - theta^*)$. The
  full Berry--Esseen bound additionally requires $L_p$-control of
  $D_(2, n)^("RR")$ (Section 4.6, already done), the transient
  $D_("tr")^("RR")$, and the misadjustment $R_n^("mis,RR")$ via the Levin
  depth-two transfer (still to be written), assembled by the smoothing
  inequality of Samsonov et al. (2025, Proposition 12).
]

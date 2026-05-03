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

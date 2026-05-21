#import "defs.typ": *

The previous chapter proves a stationary $n_0 = 0$ Berry--Esseen bound for
$S_(n, "stat")^("RR")(u)$. This chapter transfers that bound to the
deterministic-start Richardson--Romberg average after burn-in. Burn-in changes
the deterministic PR kernels, so the result is not obtained by simply inserting
$n_0 > 0$ into the stationary theorem.

Throughout this chapter, constants with a `"burn"` subscript are independent
of $n$, $n_0$, $m$, $alpha$, $p$, and $q$ unless these variables appear as
arguments. Dependencies on fixed problem constants and imported Levin/startup
constants are absorbed into named constants; powers of $a$, $t_"mix"$, $p$,
$q$, and $d$ are kept explicit when they affect the final rate.

== Target Statistic and Normalization

Let $0 <= n_0 < n$ and set $m := n - n_0$. The burned-in PR average is
$
overline(theta)_(n,n_0)^((alpha))
  := frac(1, m) sum_(k = n_0)^(n - 1) theta_k^((alpha)),
$
and the two-level RR average is
$
overline(theta)_(n,n_0)^(("RR", alpha))
  := 2 overline(theta)_(n,n_0)^((alpha))
     - overline(theta)_(n,n_0)^((2 alpha)).
$ <eq:burn-rr-average>
Both stepsizes are run from the same deterministic initial point $theta_0$ and
on the same Markov trajectory. This coupling is part of the RR statistic; if
the two levels use different paths, the deterministic-weight decomposition
below is a different object.

The finite-start burned-in vector statistic is
$
cal(T)_(n,n_0)^("RR")
  := sqrt(m) thin lr((
      overline(theta)_(n,n_0)^(("RR", alpha)) - theta^*
    )).
$ <eq:burn-vector-target>
Its scalar projection in direction $u in bb(R)^d$ is
$
T_(n,n_0)^("RR")(u)
  := u^top cal(T)_(n,n_0)^("RR")
  = sqrt(m) thin u^top lr((
      overline(theta)_(n,n_0)^(("RR", alpha)) - theta^*
    )).
$ <eq:burn-target>
The stationary augmented-chain statistic from the previous chapter is used only
as a comparison object after the startup transfer.

There are two normalizations. The finite-window normalization is
$
sigma_(n,n_0)^("bRR")(u)
  := sqrt(sigma_(n,n_0)^(2, "bRR")(u)),
quad
Xi_(n,n_0)^("bRR")(u)
  := frac(T_(n,n_0)^("RR")(u), sigma_(n,n_0)^("bRR")(u)),
$ <eq:burn-finite-normalization>
where $sigma_(n,n_0)^(2, "bRR")(u)$ is the deterministic variance proxy
defined in @eq:burn-variance-proxy. The asymptotic normalization is
$
sigma(u) := sqrt(u^top Sigma_infinity u),
quad
Xi_(n,n_0)^("asy,RR")(u)
  := frac(T_(n,n_0)^("RR")(u), sigma(u)).
$ <eq:burn-asymptotic-normalization>
We first control $Xi_(n,n_0)^("bRR")(u)$ and then pass to
$Xi_(n,n_0)^("asy,RR")(u)$. A final corollary converts the $sqrt(m)$ statistic
to the thesis-facing $sqrt(n)$ statistic when the burn-in window is
logarithmic.

== Burned-in Deterministic Weights

With the $sqrt(m)$ normalization, the depth-zero linearized term has the
weights
$
Q_(l; n_0, n)^((alpha))
  := alpha sum_(k = max(n_0, l))^(n - 1) B_alpha^(k - l),
quad
1 <= l <= n - 1,
$ <eq:burn-Q-alpha>
and $Q_(l; n_0, n)^((alpha)) = 0$ for $l >= n$. The RR weight is
$
Q_(l; n_0, n)^("RR")
  := 2 Q_(l; n_0, n)^((alpha))
     - Q_(l; n_0, n)^((2 alpha)).
$ <eq:burn-Q-RR>
Thus the leading burned-in sum is
$
W_(n,n_0)^("RR")
  := -frac(1, sqrt(m)) sum_(l = 1)^(n - 1)
      Q_(l; n_0, n)^("RR") epsilon.alt(Z_l).
$ <eq:burn-W-RR>

For $l >= n_0$ these weights are full-window weights with horizon $n-l$. For
$l < n_0$, the lower summation limit is $n_0$ rather than $l$, so the weight
comparison, Poisson decomposition, and variance proxy must be restated for
$Q_(l; n_0, n)^("RR")$.

== Closed Forms and Burned-in Weight Bounds

Write $B_w := I - w overline(A)$ for $w in {alpha, 2 alpha}$ and recall
$m = n - n_0$. The burned-in weights have two closed forms. If
$l >= n_0$, then
$
Q_(l; n_0, n)^((w))
  = w sum_(k = l)^(n - 1) B_w^(k - l)
  = overline(A)^(-1) lr((I - B_w^(n - l))).
$ <eq:burn-Q-post-form>
If $l < n_0$, then
$
Q_(l; n_0, n)^((w))
  = w sum_(k = n_0)^(n - 1) B_w^(k - l)
  = overline(A)^(-1) lr((B_w^(n_0 - l) - B_w^(n - l))).
$ <eq:burn-Q-pre-form>
Thus only the post-burn-in weights are close to the asymptotic kernel
$overline(A)^(-1)$. The pre-burn-in weights are instead exponentially small as
one moves backward from $n_0$. Empty sums below are interpreted as zero.

#lemma[
  Assume $alpha, 2 alpha in (0, alpha_infinity]$ and the Lyapunov contraction
  @eq:contraction. Let
  $
  rho_alpha := sqrt(1 - alpha a),
  quad
  C_Q := kappa_Q^(1 slash 2) || overline(A)^(-1) ||,
  quad
  tilde(C)_A := kappa_Q || overline(A) ||.
  $

  *(i) Post-burn-in comparison.* If $l >= n_0$ and $k := n - l$, then
  $
  || Q_(l; n_0, n)^("RR") - overline(A)^(-1) ||
    <= 3 C_Q rho_alpha^k.
  $ <eq:burn-post-weight-error>

  *(ii) Pre-burn-in energy.* If $1 <= l < n_0$ and $r := n_0 - l$, then
  $
  || Q_(l; n_0, n)^("RR") ||
    <= 6 C_Q rho_alpha^r.
  $ <eq:burn-pre-weight-size>

  Consequently, there are constants $C_("burn,E")$ and $C_("burn,V")$,
  depending only on $kappa_Q$, $|| overline(A)^(-1) ||$, and
  $|| overline(A) ||$, such that, uniformly in $n$ and $n_0$,
  $
  sum_(l = 1)^(n_0 - 1) || Q_(l; n_0, n)^("RR") ||^2
  + sum_(l = max(1, n_0))^(n - 1)
      || Q_(l; n_0, n)^("RR") - overline(A)^(-1) ||^2
    <= frac(C_("burn,E"), alpha a),
  $ <eq:burn-weight-energy>
  and
  $
  sum_(l = 1)^(n - 2)
    || Q_(l + 1; n_0, n)^("RR") - Q_(l; n_0, n)^("RR") ||
    <= frac(C_("burn,V"), a^2).
  $ <eq:burn-weight-variation>
] <lem:burn-rr-weight-bounds>

_Proof._ For $l >= n_0$, @eq:burn-Q-post-form gives
$
Q_(l; n_0, n)^("RR") - overline(A)^(-1)
  = -overline(A)^(-1) lr((2 B_alpha^k - B_(2 alpha)^k)).
$
The same contraction argument as in the full-window estimate gives
@eq:burn-post-weight-error.

For $l < n_0$, set $r := n_0 - l$. By @eq:burn-Q-pre-form,
$
Q_(l; n_0, n)^("RR")
  = overline(A)^(-1) lr((
      2 lr((B_alpha^r - B_alpha^(r + m)))
      - lr((B_(2 alpha)^r - B_(2 alpha)^(r + m)))
    )).
$
Taking norms and using $|| B_(2 alpha)^j ||_Q <= rho_alpha^j$ gives
@eq:burn-pre-weight-size. Squaring @eq:burn-post-weight-error and
@eq:burn-pre-weight-size and summing the resulting geometric series gives
@eq:burn-weight-energy.

It remains to bound the total variation. In the post-burn-in region
$l, l + 1 >= n_0$, the full-window identity gives
$
Q_(l + 1; n_0, n)^("RR") - Q_(l; n_0, n)^("RR")
  = -2 alpha lr((B_alpha^s - B_(2 alpha)^s)),
quad s := n - l - 1.
$
The elementary identity
$X^s - Y^s = (X - Y) sum_(i = 1)^s X^(i - 1) Y^(s - i)$ with
$X = B_alpha$ and $Y = B_(2 alpha)$ yields, for $s >= 1$,
$
|| B_alpha^s - B_(2 alpha)^s ||
  <= alpha tilde(C)_A thin s thin rho_alpha^(s - 1).
$ <eq:burn-power-difference>
Thus the post-burn-in contribution is bounded by a constant times
$a^(-2)$, as in the full-window proof.

For $l, l + 1 < n_0$, put $s := n_0 - l - 1$. The pre-burn-in closed form gives
$
Q_(l + 1; n_0, n)^("RR") - Q_(l; n_0, n)^("RR")
  = 2 alpha lr((
      B_alpha^s - B_(2 alpha)^s
      - B_alpha^(s + m) + B_(2 alpha)^(s + m)
    )).
$
Here $s >= 1$. Applying @eq:burn-power-difference to the powers $s$ and
$s + m$ and summing over $s >= 1$ again gives a constant times $a^(-2)$. If
the boundary
$l = n_0 - 1$ exists, then
$
Q_(n_0; n_0, n)^("RR") - Q_(n_0 - 1; n_0, n)^("RR")
  = 2 alpha lr((B_(2 alpha)^m - B_alpha^m)),
$
and @eq:burn-power-difference with $s = m$ bounds this term by the same
$a^(-2)$ scale. Combining the pre-burn-in, boundary, and post-burn-in
contributions proves @eq:burn-weight-variation. $square$

== Deterministic Transient After Burn-in

For a generic step size $w in {alpha, 2 alpha}$ define the deterministic
transient
$
D_(op("tr"), n, n_0)^((w))
  := frac(1, sqrt(m)) sum_(k = n_0)^(n - 1)
      B_w^k (theta_0 - theta^*),
quad
B_w := I - w overline(A).
$ <eq:burn-single-transient>
The Richardson--Romberg transient entering @eq:burn-target is
$
D_(op("tr"), n, n_0)^("RR")(u)
  := u^top lr((
      2 D_(op("tr"), n, n_0)^((alpha))
      - D_(op("tr"), n, n_0)^((2 alpha))
    )).
$ <eq:burn-RR-transient>

#lemma[
  Assume $alpha, 2 alpha in (0, alpha_infinity]$ and the Lyapunov contraction
  @eq:contraction. Then, for each $w in {alpha, 2 alpha}$,
  $
  || D_(op("tr"), n, n_0)^((w)) ||
    <= frac(2 sqrt(kappa_Q), w a sqrt(m))
       (1 - w a)^(n_0 slash 2) || theta_0 - theta^* ||.
  $ <eq:burn-single-transient-bound>
  Consequently,
  $
  | D_(op("tr"), n, n_0)^("RR")(u) |
    <= frac(5 sqrt(kappa_Q) || u || || theta_0 - theta^* ||,
             alpha a sqrt(m))
       (1 - alpha a)^(n_0 slash 2).
  $ <eq:burn-RR-transient-bound>
] <lem:burn-deterministic-transient>

_Proof._ Let $r_w := sqrt(1 - w a)$. By @eq:contraction and equivalence of the
Euclidean and $Q$-norms,
$
|| B_w^k (theta_0 - theta^*) ||
  <= sqrt(kappa_Q) r_w^k || theta_0 - theta^* ||.
$
Therefore
$
|| D_(op("tr"), n, n_0)^((w)) ||
  <= frac(sqrt(kappa_Q) || theta_0 - theta^* ||, sqrt(m))
     sum_(k = n_0)^(n - 1) r_w^k
  <= frac(sqrt(kappa_Q) || theta_0 - theta^* ||, sqrt(m))
     frac(r_w^(n_0), 1 - r_w).
$
Since $1 - sqrt(1 - x) >= x slash 2$ for $0 <= x <= 1$, the last display gives
@eq:burn-single-transient-bound. Applying this estimate at $w = alpha$ and
$w = 2 alpha$, using $1 - 2 alpha a <= 1 - alpha a$, and taking the
triangle inequality in @eq:burn-RR-transient gives
$
| D_(op("tr"), n, n_0)^("RR")(u) |
  <= || u || lr((
       2 || D_(op("tr"), n, n_0)^((alpha)) ||
       + || D_(op("tr"), n, n_0)^((2 alpha)) ||
     ))
  <= frac(5 sqrt(kappa_Q) || u || || theta_0 - theta^* ||,
           alpha a sqrt(m))
       (1 - alpha a)^(n_0 slash 2).
$
$square$

#corollary[
  *(Logarithmic burn-in removes the deterministic transient.)*
  If, for some $beta > 0$,
  $
  n_0 >= frac(2 beta, alpha a) log n,
  $ <eq:burn-log-condition>
  then
  $
  | D_(op("tr"), n, n_0)^("RR")(u) |
    <= frac(5 sqrt(kappa_Q) || u || || theta_0 - theta^* ||,
             alpha a sqrt(m)) n^(-beta).
  $ <eq:burn-log-transient-bound>
  In particular, at the balanced scale $alpha = c n^(-1 slash 2)$ and
  $m >= n slash 2$, this is $O(n^(-beta))$. Taking $beta = 1$ makes the
  deterministic transient negligible relative to the stationary
  $n^(-1 slash 4)$ Berry--Esseen rate.
] <cor:burn-log-transient>

_Proof._ The elementary inequality
$(1 - alpha a)^(n_0 slash 2) <= exp(-alpha a n_0 slash 2)$ and
@eq:burn-log-condition imply
$(1 - alpha a)^(n_0 slash 2) <= n^(-beta)$. Substitute this into
@eq:burn-RR-transient-bound. If $alpha = c n^(-1 slash 2)$ and $m >= n slash 2$,
then $(alpha sqrt(m))^(-1) <= sqrt(2) slash c$, so the displayed
$O(n^(-beta))$ bound follows. $square$

The deterministic transient is not the full initial-condition contribution:
the exact recursion contains the random product
$Gamma_(1:k)^((w)) := product_(j = 1)^k (I - w A(Z_j))$. The difference
between this product and $B_w^k$ is another finite-start term. We use the same
random-product stability input that appears in Levin et al. (2025,
Appendix D.1, Proposition 9). Let $alpha_("st")(p)$ denote the minimum of the
Levin depth-two startup ceiling and the product-stability ceiling at moment
order $2p$; the startup section below uses the same threshold.

#lemma[
  *(Imported random-product stability.)*
  Under the stability and bounded-noise assumptions used in Levin et al.
  (2025, Appendix D.1, Proposition 9), if $2 alpha <= alpha_("st")(p)$, then
  there exist constants $C_("prod") < infinity$ and $c_("prod") > 0$ such
  that, for every $p >= 2$, every $w in {alpha, 2 alpha}$, every $0 <= s < k$,
  and every $cal(F)_s$-measurable vector $V_s$,
  $
  || Gamma_(s + 1:k)^((w)) V_s ||_(L_p)
    <= C_("prod") exp(-c_("prod") w a (k - s) slash p)
       || V_s ||_(L_(2p)).
  $ <eq:burn-product-stability>
] <lem:burn-product-stability>

Define the accumulated RR random initial-product discrepancy by
$
cal(I)_(n,n_0)^("init,RR")(u)
  := frac(1, sqrt(m)) sum_(k = n_0)^(n - 1)
      u^top lr([
        2 lr((Gamma_(1:k)^((alpha)) - B_alpha^k))
        - lr((Gamma_(1:k)^((2 alpha)) - B_(2 alpha)^k))
      ]) (theta_0 - theta^*),
$ <eq:burn-random-init-discrepancy>
where the empty product at $k = 0$ is the identity.

#lemma[
  *(Burned-in random initial-product transient.)*
  Assume the hypotheses of @lem:burn-product-stability,
  $alpha, 2 alpha in (0, alpha_infinity]$, $alpha a <= 1 slash 4$, and
  the Lyapunov contraction @eq:contraction. Then, for every $p >= 2$,
  $
  || cal(I)_(n,n_0)^("init,RR")(u) ||_(L_p)
    <= frac(C_("init,RR") ||u|| ||theta_0 - theta^*|| p,
             alpha a sqrt(m))
       exp(-c_("init") alpha a n_0 slash p),
  $ <eq:burn-random-init-bound>
  where $C_("init,RR") < infinity$ and $c_("init") > 0$ depend only on
  $C_("prod")$, $c_("prod")$, and the Lyapunov norm-equivalence constant.
] <lem:burn-random-initial-product>

_Proof._ Let $e_0 := theta_0 - theta^*$. By @lem:burn-product-stability with
$s = 0$ and $V_0 = e_0$,
$
|| Gamma_(1:k)^((w)) e_0 ||_(L_p)
  <= C_("prod") exp(-c_("prod") w a k slash p) ||e_0||.
$
The deterministic Lyapunov contraction gives
$
|| B_w^k e_0 ||
  <= sqrt(kappa_Q) (1 - w a)^(k slash 2) ||e_0||
  <= sqrt(kappa_Q) exp(-w a k slash (2 p)) ||e_0||,
$
since $p >= 1$. Hence, after decreasing the exponential constant,
$
|| lr((Gamma_(1:k)^((w)) - B_w^k)) e_0 ||_(L_p)
  <= C thin exp(-c_("init") w a k slash p) ||e_0||.
$
Apply this estimate at $w = alpha$ and $w = 2 alpha$, use the triangle
inequality in @eq:burn-random-init-discrepancy, and extend the geometric sum
to infinity:
$
|| cal(I)_(n,n_0)^("init,RR")(u) ||_(L_p)
  <= frac(C ||u|| ||e_0||, sqrt(m))
     sum_(k = n_0)^infinity exp(-c_("init") alpha a k slash p).
$
Because $alpha a <= 1 slash 4$ and $p >= 2$,
$1 - exp(-c_("init") alpha a slash p) >= C^(-1) alpha a slash p$, which gives
@eq:burn-random-init-bound. $square$

#corollary[
  *(Logarithmic burn-in removes the random initial-product discrepancy.)*
  If, for some $beta > 0$,
  $
  n_0 >= frac(beta p, c_("init") alpha a) log n,
  $ <eq:burn-log-init-condition>
  then
  $
  || cal(I)_(n,n_0)^("init,RR")(u) ||_(L_p)
    <= frac(C_("init,RR") ||u|| ||theta_0 - theta^*|| p,
             alpha a sqrt(m)) n^(-beta).
  $ <eq:burn-log-init-bound>
  At the balanced scale $alpha = c n^(-1 slash 2)$, with
  $m >= n slash 2$ and $p$ logarithmic in $n$, this is
  $"polylog"(n) thin n^(-beta)$.
] <cor:burn-log-initial-product>

_Proof._ Substitute @eq:burn-log-init-condition into
@eq:burn-random-init-bound. At the balanced scale,
$(alpha sqrt(m))^(-1) = O(1)$, so the remaining factor is logarithmic. $square$

== Burned-in Variance Proxy

For the stochastic depth-zero term, write
$Q_l^("bRR") := Q_(l; n_0, n)^("RR")$ and
$Sigma := Sigma_(epsilon.alt)^(("M"))$. We take $n >= 2$ in this subsection.
The martingale part produced by the
Poisson decomposition below starts at $l = 2$, so the deterministic variance
proxy is
$
Sigma_(n,n_0)^("bRR")
  := frac(1, m) sum_(l = 2)^(n - 1)
      Q_l^("bRR") thin Sigma thin (Q_l^("bRR"))^top,
quad
sigma_(n,n_0)^(2, "bRR")(u)
  := u^top Sigma_(n,n_0)^("bRR") u.
$ <eq:burn-variance-proxy>

#lemma[
  Assume the conditions of @lem:burn-rr-weight-bounds and $|| Sigma || < infinity$.
  There exists a constant $C_("burn,3")$, depending only on
  $kappa_Q$, $|| overline(A)^(-1) ||$, $|| overline(A) ||$, and $|| Sigma ||$,
  such that
  $
  || Sigma_(n,n_0)^("bRR") - Sigma_infinity ||
    <= frac(C_("burn,3"), m thin alpha a).
  $ <eq:burn-variance-comparison>
  Consequently,
  $
  | sigma_(n,n_0)^(2, "bRR")(u) - sigma^2(u) |
    <= frac(C_("burn,3") || u ||^2, m thin alpha a).
  $ <eq:burn-scalar-variance-comparison>
] <lem:burn-variance-comparison>

_Proof._ For post-burn-in indices put
$Delta_l := Q_l^("bRR") - overline(A)^(-1)$. Then
$
Q_l^("bRR") Sigma (Q_l^("bRR"))^top - Sigma_infinity
  = Delta_l Sigma overline(A)^(-top)
    + overline(A)^(-1) Sigma Delta_l^top
    + Delta_l Sigma Delta_l^top.
$
By @eq:burn-post-weight-error,
$
sum_(l >= max(2, n_0)) || Delta_l ||
  <= frac(C, alpha a),
quad
sum_(l >= max(2, n_0)) || Delta_l ||^2
  <= frac(C, alpha a).
$
For pre-burn-in indices there is no comparison with $overline(A)^(-1)$; instead
@eq:burn-weight-energy gives
$
sum_(2 <= l < n_0) || Q_l^("bRR") ||^2 <= frac(C, alpha a).
$
Replacing all post-burn-in weights by $overline(A)^(-1)$ gives at most $m$
copies of $Sigma_infinity$; the martingale index set starts at $l = 2$, so the
possible finite-index mismatch is bounded by two copies of $Sigma_infinity$.
Combining these three estimates and dividing by $m$ yields
@eq:burn-variance-comparison, after absorbing the harmless $2 ||Sigma_infinity|| / m$
term into $C_("burn,3") / (m alpha a)$ using $alpha a <= 1$. The scalar bound
follows from $|u^top H u| <= ||H|| ||u||^2$. $square$

Assume $sigma^2(u) > 0$ and impose the burned-in variance lower-bound condition
$
m thin alpha thin a
  >= frac(2 C_("burn,3") || u ||^2, sigma^2(u)).
$ <eq:burn-variance-lb-condition>
Then @eq:burn-scalar-variance-comparison gives
$sigma_(n,n_0)^(2, "bRR")(u) >= sigma^2(u) slash 2$, hence
$sigma_(n,n_0)^("bRR")(u) >= sigma(u) slash sqrt(2)$. At the balanced scale
$alpha = c thin n^(-1 slash 2)$ with $m >= n slash 2$, this condition holds
for all sufficiently large $n$ depending on $u$, $c$, and the problem
constants.

== Burned-in Poisson Martingale Approximation

Let $hat(epsilon.alt) := sum_(j = 0)^infinity sans(Q)^j epsilon.alt$ be the
Poisson solution, so
$
hat(epsilon.alt) - sans(Q) hat(epsilon.alt) = epsilon.alt,
quad
|| hat(epsilon.alt) ||_infinity
  <= 3 thin t_"mix" thin || epsilon.alt ||_infinity.
$
The stationary Poisson identity applies because the coefficients
$Q_l^("bRR")$ are deterministic.

#lemma[
  Assume *UGE 1* and $pi(epsilon.alt) = 0$. Define, for $2 <= l <= n - 1$,
  $
  Delta M_l^("bRR")
    := Q_l^("bRR") thin
       lr((hat(epsilon.alt)(Z_l) - sans(Q) hat(epsilon.alt)(Z_(l - 1)))) ,
  $
  and set $M_(n,n_0)^("bRR") := sum_(l = 2)^(n - 1) Delta M_l^("bRR")$.
  Then ${Delta M_l^("bRR")}_(l = 2)^(n - 1)$ is a sequence of
  $cal(F)_l$-martingale differences and
  $
  W_(n,n_0)^("RR")
    = -frac(1, sqrt(m)) M_(n,n_0)^("bRR")
      + D_(2,n,n_0)^("bRR"),
  $ <eq:burn-poisson-decomp>
  where
  $
  D_(2,n,n_0)^("bRR")
    := -frac(1, sqrt(m)) lr([
        Q_1^("bRR") thin hat(epsilon.alt)(Z_1)
        + sum_(l = 1)^(n - 2)
            lr((Q_(l + 1)^("bRR") - Q_l^("bRR")))
            thin sans(Q) hat(epsilon.alt)(Z_l)
      ]).
  $
  Moreover, with $C_("burn,Q") := || overline(A)^(-1) || + 6 C_Q$,
  $
  || D_(2,n,n_0)^("bRR") ||_infinity
    <= frac(3 thin t_"mix" thin || epsilon.alt ||_infinity, sqrt(m))
       lr((C_("burn,Q") + frac(C_("burn,V"), a^2))).
  $ <eq:burn-D2-bound>
] <lem:burn-poisson-decomp>

_Proof._ The martingale-difference property follows from the Markov property:
$bb(E)[hat(epsilon.alt)(Z_l) | cal(F)_(l - 1)]
  = sans(Q) hat(epsilon.alt)(Z_(l - 1))$.

Substitute the Poisson equation in @eq:burn-W-RR. The $l = 1$ term is kept as a
left boundary term, and for $l >= 2$ we add and subtract
$sans(Q) hat(epsilon.alt)(Z_(l - 1))$. Abel summation of the telescope gives
exactly @eq:burn-poisson-decomp. The right boundary vanishes because
$Q_(n - 1; n_0, n)^("RR") = 2 alpha I - 2 alpha I = 0$.

For the sup-norm bound, use
$|| sans(Q) hat(epsilon.alt) ||_infinity <= || hat(epsilon.alt) ||_infinity$,
the uniform weight bound $|| Q_l^("bRR") || <= C_("burn,Q")$ from
@eq:burn-post-weight-error and @eq:burn-pre-weight-size, and the total-variation
estimate @eq:burn-weight-variation. $square$

== Burned-in Predictable Variance Comparison

For the martingale in @lem:burn-poisson-decomp define
$
cal(V)_(epsilon.alt)(z)
  := sans(Q)(hat(epsilon.alt) hat(epsilon.alt)^top)(z)
   - (sans(Q) hat(epsilon.alt))(z)
     thin (sans(Q) hat(epsilon.alt))(z)^top.
$ <eq:burn-bar-eps-def>
This is the same matrix-valued conditional covariance function as in the
stationary chapter; it is not the vector noise $epsilon.alt$.
As before, $pi(cal(V)_(epsilon.alt)) = Sigma$ and
$
|| cal(V)_(epsilon.alt) ||_infinity
  <= 18 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2.
$ <eq:burn-bar-eps-sup>
The predictable quadratic variation is
$
chevron.l M^("bRR") chevron.r_(n,n_0)
  := sum_(l = 2)^(n - 1)
      bb(E)[Delta M_l^("bRR") thin (Delta M_l^("bRR"))^top | cal(F)_(l - 1)]
  = sum_(l = 2)^(n - 1)
      Q_l^("bRR") thin cal(V)_(epsilon.alt)(Z_(l - 1)) thin (Q_l^("bRR"))^top.
$ <eq:burn-bracket>

#lemma[
  Assume *UGE 1*, $pi(epsilon.alt) = 0$, and
  $|| epsilon.alt ||_infinity < infinity$. There exists a universal constant
  $C_4 > 0$ such that, for every $u in bb(R)^d$, every $p >= 2$, every initial
  distribution $xi$, every $n >= 2$, and every $0 <= n_0 < n$,
  $
  bb(E)_xi^(1 slash p) lr([
    | u^top chevron.l M^("bRR") chevron.r_(n,n_0) u
      - m thin sigma_(n,n_0)^(2, "bRR")(u) |^p
  ])
    <= C_4 thin C_("burn,Q")^2 thin || u ||^2
       thin || epsilon.alt ||_infinity^2 thin t_"mix"^(5 slash 2)
       thin sqrt(p thin n).
  $ <eq:burn-bracket-conc>
] <lem:burn-bracket-conc>

_Proof._ Set
$
h_l(z) := u^top Q_l^("bRR") thin cal(V)_(epsilon.alt)(z)
  thin (Q_l^("bRR"))^top u,
quad
g_l(z) := h_l(z) - pi(h_l).
$
By @eq:burn-bar-eps-sup and $|| Q_l^("bRR") || <= C_("burn,Q")$,
$
|| g_l ||_infinity
  <= 36 thin C_("burn,Q")^2 thin || u ||^2
     thin t_"mix"^2 thin || epsilon.alt ||_infinity^2.
$
Moreover, @eq:burn-bracket and @eq:burn-variance-proxy imply
$
u^top chevron.l M^("bRR") chevron.r_(n,n_0) u
  - m thin sigma_(n,n_0)^(2, "bRR")(u)
  = sum_(l = 2)^(n - 1) g_l(Z_(l - 1)).
$
The scalar Markov concentration lemma used in the stationary chapter is valid
for arbitrary initial law by Levin et al. (2025, Lemma 11). Applied to the
time-inhomogeneous centered functions $g_l$, it gives the displayed
$sqrt(p n)$ bound. $square$

#corollary[
  Under the assumptions of @lem:burn-bracket-conc,
  $
  bb(E)_xi^(1 slash p) lr([
    | u^top chevron.l M^("bRR") chevron.r_(n,n_0) u
      - m thin sigma^2(u) |^p
  ])
    <= C_4 thin C_("burn,Q")^2 thin || u ||^2
       thin || epsilon.alt ||_infinity^2 thin t_"mix"^(5 slash 2)
       thin sqrt(p thin n)
       + frac(C_("burn,3") || u ||^2, alpha a).
  $ <eq:burn-bracket-asymp>
] <cor:burn-bracket-asymp>

_Proof._ Use the triangle inequality and @eq:burn-scalar-variance-comparison:
$
m thin |sigma_(n,n_0)^(2, "bRR")(u) - sigma^2(u)|
  <= frac(C_("burn,3") || u ||^2, alpha a).
$
$square$

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
law of the augmented chain
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
The finite-start coordinates are zero. The invariant copy satisfies the
finite-past limits of the elementary $J^((0,w))$, $J^((1,w))$ estimates and of
Levin Propositions 8 and 9; in particular
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

It remains to control $H^((2,w))$. On the event $T <= k$, the base chains are
identical after time $T$, and subtracting @eq:levin-H2-representation gives
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
Levin Proposition 9 bounds both one-trajectory terms uniformly in $k$, and UGE
gives $bb(P)(T > k)^(1 slash (2p)) <= C e^(-c k slash p)$. Therefore
$
|| Delta H_k^((2,w)) 1_(T > k) ||_(L_p)
  <= C (1 + d^(1 slash q)) p^(7 slash 2) t_"mix"^(5 slash 2)
       w^(3 slash 2) log^(3 slash 2)(1 slash (w a))
       e^(-c k slash p).
$ <eq:burn-H2-bad-event>
Since $w a <= 1$, this is bounded by the right-hand side of
@eq:burn-startup-pointwise after absorbing a fixed power of $a^(-1)$ into
$C_("st")$.

For the first term in @eq:burn-H2-coupled-diff, apply
@lem:burn-product-stability conditionally with $s = T$ and
$V_T = Delta H_T^((2,w))$, then use the same one-trajectory moment bound and
the exponential moment of $T$:
$
|| Gamma_(T + 1:k)^((w)) Delta H_T^((2,w)) 1_(T <= k) ||_(L_p)
  <= C (1 + d^(1 slash q)) p^(7 slash 2) t_"mix"^(5 slash 2)
       w^(3 slash 2) log^(3 slash 2)(1 slash (w a))
       e^(-c w a k slash p).
$ <eq:burn-H2-initial-term>
For the convolution term, apply @lem:burn-product-stability term by term and
then use the $J^((2,w))$ contraction @eq:burn-J-startup-proof:
$
& w sum_(l = 1)^k
  || Gamma_(l + 1:k)^((w)) tilde(A)(Z_l)
     Delta J_(l - 1)^((2,w)) ||_(L_p) \
&quad <= C w A_("J")(p,q,w) sum_(l = 1)^k
     e^(-c w a (k - l) slash p)
     e^(-c w a l slash p) \
&quad <= C A_("st")(p,q,w) e^(-c w a k slash p).
$ <eq:burn-H2-convolution-term>
Indeed, with $r = k - l$ and $p >= 2$,
$
w sum_(l = 1)^k
  e^(-c w a (k - l) slash p) e^(-c w a l slash p)
  <= C frac(p, a) e^(-c' w a k slash p),
$
after decreasing the exponential constant from $c$ to $c'$. The additional
factor $p slash a$ is the reason for the enlarged definition of $A_("st")$.
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
  *(Logarithmic burn-in removes the startup discrepancy.)*
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
  Thus the Berry--Esseen choice $p asymp log n$ requires
  $n_0$ of order $(alpha a)^(-1) log^2 n$ under this $L_p$ contraction.
] <cor:burn-log-startup>

_Proof._ Substitute @eq:burn-log-startup-condition into
@eq:burn-startup-transfer. For the balanced scale,
$A_("st")(p,q,alpha) = "polylog"(n) thin alpha^(1 slash 2)$, hence
$A_("st")(p,q,alpha) slash (alpha sqrt(m))
= O("polylog"(n) n^(-1 slash 4))$. $square$

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
length $m$. By stationarity, the same distribution is obtained if the sum in
@eq:burn-mis-aug-def is taken over $j = n_0, dots, n - 1$.

#theorem[
  *(Burned-in PR-averaged RR misadjustment bound.)*
  Assume *UGE 1*, $pi(epsilon.alt) = 0$, $|| epsilon.alt ||_infinity < infinity$,
  $alpha, 2 alpha in (0, alpha_infinity]$,
  $2 alpha <= alpha_("inv")$,
  $alpha a <= 1 slash 4$, and the step-size restrictions of the Levin
  depth-two and startup-contraction bounds, where $alpha_("inv")$ is defined
  in @eq:alpha-inv. Set
  $
  Phi(p, alpha) := p^(3 slash 2) thin t_"mix"^(1 slash 2) slash a
                   + p^(1 slash 2) thin t_"mix"^(3 slash 2) sqrt(alpha slash a).
  $
  There exists a constant $C_("burn,mis")$ depending only on the stationary
  misadjustment constants, the startup-contraction constants, and the problem
  constants such that, for every $p >= 2$, every $q >= 2$ satisfying
  $p <= q slash 2$, every $2 alpha <= alpha_*(q, t_"mix")$ and
  $2 alpha <= alpha_("st")(p)$, and every $m >= 2$,
  $
  || R_(n,n_0, op("fin"))^("mis,RR") ||_(L_p)
    &<= C_("burn,mis") sqrt(m) thin alpha^2 \
    &quad + C_("burn,mis") (1 + d^(1 slash q)) p^(7 slash 2)
       t_"mix"^(5 slash 2) sqrt(m) thin alpha^(3 slash 2)
       log^(3 slash 2)(1 slash (alpha a)) \
    &quad + C_("burn,mis") p^(3 slash 2) sqrt(alpha) \
    &quad + C_("burn,mis") p^3 (alpha m)^(-1 slash 2)
       log^(1 slash p)(1 slash (alpha a)) \
    &quad + C_("burn,mis") Phi(p, alpha) thin m^(-1 slash 2) \
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
  = || R_(m, op("aug"))^("mis,RR") ||_(L_p)$, and the latter is exactly the
stationary augmented-chain misadjustment bound @thm:misadjustment with $n$
replaced by $m$. The second term is @lem:burn-startup-transfer. Combining the
two estimates gives @eq:burn-mis-bound. $square$

#corollary[
  *(Balanced-scale burned-in misadjustment rate.)*
  Assume the hypotheses of @thm:burn-misadjustment. Let
  $alpha = c thin n^(-1 slash 2)$, $m >= n slash 2$,
  $p = max(2, ceil(log n))$, and
  $q = max(2 p, ceil(log(e thin d)), 2)$. If, for some fixed $beta > 0$,
  $
  n_0 >= frac(beta p, c_("st") alpha a) log n,
  $
  and $n$ is large enough that $2 alpha <= alpha_*(q, t_"mix")$ and
  $2 alpha <= alpha_("st")(p)$ and $2 alpha <= alpha_("inv")$, then
  $
  || R_(n,n_0, op("fin"))^("mis,RR") ||_(L_p)
    <= C_("burn,mis") thin "polylog"(n) thin n^(-1 slash 4).
  $ <eq:burn-mis-rate>
] <cor:burn-misadjustment-rate>

_Proof._ The stationary part of @eq:burn-mis-bound is the bound of
@cor:misadjustment-rate with $n$ replaced by $m$, and $m >= n slash 2$ keeps
the rate unchanged. The startup term is controlled by
@cor:burn-log-startup and is
$"polylog"(n) thin n^(-1 slash 4 - beta)$ at this scale. $square$

== Burned-in Martingale Berry--Esseen

The depth-zero martingale in @lem:burn-poisson-decomp has deterministic
coefficients, but its normalization is based on the effective sample size
$m = n - n_0$. We assume $m >= n slash 2$, the logarithmic burn-in regime, so
the inhomogeneous concentration term of order $sqrt(p n)$ can be written on the
$m$ scale.

Fix $u in bb(R)^d$ and put
$X_l^("bRR") := u^top Delta M_l^("bRR")$ for $2 <= l <= n - 1$. Then
$
|X_l^("bRR")| <= kappa_("burn")(u),
quad
kappa_("burn")(u)
  := 6 thin t_"mix" thin C_("burn,Q") thin || epsilon.alt ||_infinity
     thin || u ||.
$ <eq:burn-M-incr>
Indeed, this is the same bounded-increment estimate as in the stationary
chapter, with $C_("burn,Q")$ replacing the full-window weight bound. Set
$
s_(n,n_0)^2(u) := m thin sigma_(n,n_0)^(2, "bRR")(u).
$
Under @eq:burn-variance-lb-condition,
$s_(n,n_0)^2(u) >= m sigma^2(u) slash 2$. Also,
$m >= n slash 2$ and the uniform bound $||Q_l^("bRR")|| <= C_("burn,Q")$
imply the deterministic upper bound
$
s_(n,n_0)^2(u)
  <= 2 m thin C_("burn,Q")^2 thin
     || Sigma_(epsilon.alt)^(("M")) || thin || u ||^2.
$ <eq:burn-s-upper>

#theorem[
  *(Burned-in martingale Berry--Esseen bound.)*
  Assume *UGE 1*, $pi(epsilon.alt) = 0$,
  $|| epsilon.alt ||_infinity < infinity$, $sigma^2(u) > 0$,
  $alpha, 2 alpha in (0, alpha_infinity]$, $m >= n slash 2$, and the
  burned-in variance lower-bound condition @eq:burn-variance-lb-condition.
  There exist constants $C_("bK,1")(u), C_("bK,2")(u) > 0$, depending only on
  $||u||$, $sigma(u)$, $C_("burn,Q")$, $t_"mix"$,
  $||epsilon.alt||_infinity$, $||Sigma_(epsilon.alt)^(("M"))||$, and the
  universal Bolthausen--Fan constants, such that for every $n >= 3$,
  $
  d_K lr((
    frac(u^top M_(n,n_0)^("bRR"),
         sqrt(m) thin sigma_(n,n_0)^("bRR")(u)),
    cal(N)(0, 1)
  ))
    <= frac(C_("bK,1")(u) thin log^(3 slash 4) n, m^(1 slash 4))
     + frac(C_("bK,2")(u) thin log n, sqrt(m)).
  $ <eq:burn-M-BE>
] <thm:burn-M-BE>

_Proof._ Apply the Bolthausen--Fan inequality used in the stationary chapter
to the martingale differences $X_l^("bRR")$. The bounded-increment input is
@eq:burn-M-incr, and the target variance is $s_(n,n_0)^2(u)$. The first and
third Bolthausen--Fan terms are bounded exactly as before, with $n$ replaced
by $m$ in the denominator and the harmless factor $n slash m <= 2$ absorbed
into the constants. For the conditional-variance term use
@lem:burn-bracket-conc:
$
bb(E)^(1 slash p) lr([
  | u^top chevron.l M^("bRR") chevron.r_(n,n_0) u
    - s_(n,n_0)^2(u) |^p
])
  <= C thin sqrt(p thin n).
$
Together with $m >= n slash 2$, @eq:burn-variance-lb-condition, and
@eq:burn-s-upper, this gives
$
"Term II"
  <= C(u) thin p^((3 p + 1) slash (2 (2 p + 1)))
     thin m^(-p slash (2 (2 p + 1))).
$
Taking $p = ceil(log n)$ gives the displayed
$log^(3 slash 4)(n) m^(-1 slash 4)$ bound. The classical Bolthausen and
Lindeberg terms give the second displayed term. $square$

== Finite-Window Smoothing Assembly

The finite-start scalar statistic decomposes as
$
T_(n,n_0)^("RR")(u)
  = -frac(u^top M_(n,n_0)^("bRR"), sqrt(m))
    + cal(R)_(n,n_0, op("fin"))^("bRR")(u),
$ <eq:burn-full-decomp>
where the composite non-Gaussian remainder is
$
cal(R)_(n,n_0, op("fin"))^("bRR")(u)
  := D_(op("tr"), n, n_0)^("RR")(u)
     + cal(I)_(n,n_0)^("init,RR")(u)
     + u^top D_(2,n,n_0)^("bRR")
     + u^top R_(n,n_0, op("fin"))^("mis,RR").
$ <eq:burn-composite-remainder>
The terms are controlled by @lem:burn-deterministic-transient,
@lem:burn-random-initial-product, @lem:burn-poisson-decomp, and
@thm:burn-misadjustment, respectively.

Let $cal(B)_("mis")(m,n_0,p,q,alpha)$ denote the right-hand side of
@eq:burn-mis-bound.

#lemma[
  Assume the hypotheses of @lem:burn-deterministic-transient,
  @lem:burn-random-initial-product, @lem:burn-poisson-decomp, and
  @thm:burn-misadjustment. Then, for every $p >= 2$ and admissible $q >= 2$,
  $
  || cal(R)_(n,n_0, op("fin"))^("bRR")(u) ||_(L_p)
    &<= frac(5 sqrt(kappa_Q) || u || || theta_0 - theta^* ||,
             alpha a sqrt(m))
        (1 - alpha a)^(n_0 slash 2) \
    &quad + frac(C_("init,RR") ||u|| ||theta_0 - theta^*|| p,
             alpha a sqrt(m))
       exp(-c_("init") alpha a n_0 slash p) \
    &quad + frac(C_("burn,D2") thin || u ||, sqrt(m))
     + || u || thin cal(B)_("mis")(m,n_0,p,q,alpha),
  $ <eq:burn-R-bound>
  where
  $
  C_("burn,D2")
    := 3 thin t_"mix" thin || epsilon.alt ||_infinity
       thin lr((C_("burn,Q") + frac(C_("burn,V"), a^2))).
  $
] <lem:burn-R-bound>

_Proof._ Apply the triangle inequality in @eq:burn-composite-remainder. The
deterministic part is @eq:burn-RR-transient-bound. The random initial-product
term is @eq:burn-random-init-bound. The Poisson boundary term is bounded by
$
|| u^top D_(2,n,n_0)^("bRR") ||_(L_p)
  <= ||u|| thin ||D_(2,n,n_0)^("bRR")||_infinity
  <= frac(C_("burn,D2") ||u||, sqrt(m))
$
using @eq:burn-D2-bound. The last term is exactly @eq:burn-mis-bound. $square$

Dividing @eq:burn-full-decomp by the finite-window normalization gives
$
Xi_(n,n_0)^("bRR")(u) = X_(n,n_0)^("bRR") + Y_(n,n_0)^("bRR"),
$ <eq:burn-XY-split>
where
$
X_(n,n_0)^("bRR")
  := -frac(u^top M_(n,n_0)^("bRR"),
           sqrt(m) thin sigma_(n,n_0)^("bRR")(u)),
quad
Y_(n,n_0)^("bRR")
  := frac(cal(R)_(n,n_0, op("fin"))^("bRR")(u),
          sigma_(n,n_0)^("bRR")(u)).
$

#theorem[
  *(Finite-window burned-in PR-averaged RR Berry--Esseen bound.)*
  Assume the hypotheses of @thm:burn-M-BE, @lem:burn-random-initial-product,
  and @thm:burn-misadjustment. Let
  $p = max(2, ceil(log n))$ and
  $q = max(2 p, ceil(log(e thin d)), 2)$. If
  $2 alpha <= alpha_*(q, t_"mix")$ and $2 alpha <= alpha_("st")(p)$, then
  $
  d_K lr((Xi_(n,n_0)^("bRR")(u), cal(N)(0, 1)))
    &<= frac(C_("bK,1")(u) thin log^(3 slash 4) n, m^(1 slash 4))
     + frac(C_("bK,2")(u) thin log n, sqrt(m)) \
    &quad + frac(e thin
        || cal(R)_(n,n_0, op("fin"))^("bRR")(u) ||_(L_p),
        sqrt(2 pi) thin sigma_(n,n_0)^("bRR")(u))
     + frac(e, n),
  $ <eq:burn-RR-BE-master>
  with the composite remainder bounded by @lem:burn-R-bound.
] <thm:burn-RR-BE-master>

_Proof._ Apply the smoothing inequality @eq:smoothing-Lp to the split
@eq:burn-XY-split. The martingale Berry--Esseen term is @thm:burn-M-BE; the
minus sign in $X_(n,n_0)^("bRR")$ is irrelevant because the standard normal
law is symmetric. Since $p = max(2, ceil(log n))$, the smoothing tail
$e^(-p)$ is at most $e slash n$. The $L_p$ norm of the perturbation is
$
||Y_(n,n_0)^("bRR")||_(L_p)
  = frac(||cal(R)_(n,n_0, op("fin"))^("bRR")(u)||_(L_p),
         sigma_(n,n_0)^("bRR")(u)),
$
which gives @eq:burn-RR-BE-master. $square$

== Balanced Burn-in Berry--Esseen Bound

The finite-window bound uses $sigma_(n,n_0)^("bRR")(u)$. The final inference
statement uses $sigma(u)$, via the following burned-in analogue of
@cor:RR-BE-sigma.

#lemma[
  Assume $sigma^2(u) > 0$ and the burned-in variance lower-bound condition
  @eq:burn-variance-lb-condition. Put
  $
  r_(n,n_0)(u)
    := frac(sigma_(n,n_0)^("bRR")(u), sigma(u)).
  $
  Then
  $
  frac(1, sqrt(2)) <= r_(n,n_0)(u) <= sqrt(3 slash 2),
  $
  and, for any real random variable $W$,
  $
  d_K lr((r_(n,n_0)(u) W, cal(N)(0, 1)))
    <= d_K lr((W, cal(N)(0, 1)))
     + frac(C_("norm") C_("burn,3") ||u||^2,
            m thin alpha thin a thin sigma^2(u)),
  $ <eq:burn-normalization-transfer>
  where $C_("norm")$ is a universal constant.
] <lem:burn-normalization-transfer>

_Proof._ The variance lower-bound condition and
@eq:burn-scalar-variance-comparison give
$
|sigma_(n,n_0)^(2, "bRR")(u) - sigma^2(u)| <= sigma^2(u) slash 2.
$
Hence $r_(n,n_0)(u) in [1 slash sqrt(2), sqrt(3 slash 2)]$. For $r$ in this
compact interval,
$
sup_x | Phi(x slash r) - Phi(x) | <= C_("norm") thin |r - 1|,
$
because $sup_x |x phi(x)| < infinity$. Therefore
$
d_K (r W, cal(N)(0, 1))
  <= d_K (W, cal(N)(0, 1)) + C_("norm") thin |r - 1|.
$
Finally,
$
|r_(n,n_0)(u) - 1|
  = frac(|sigma_(n,n_0)^(2, "bRR")(u) - sigma^2(u)|,
         sigma(u) thin (sigma_(n,n_0)^("bRR")(u) + sigma(u)))
  <= frac(C_("burn,3") ||u||^2,
          m thin alpha thin a thin sigma^2(u)),
$
using @eq:burn-scalar-variance-comparison and
$sigma_(n,n_0)^("bRR")(u) + sigma(u) >= sigma(u)$. $square$

#theorem[
  *(Balanced-scale burned-in PR-averaged RR Berry--Esseen bound.)*
  Assume Assumptions 1--3 from @sec:assumptions. Assume also the Lyapunov
  contraction @eq:contraction for the two step sizes used below, the Levin
  depth-two stationary moment and misadjustment bounds used in
  @thm:misadjustment, and the non-degeneracy condition $sigma^2(u) > 0$.
  Fix $c > 0$ and set
  $
  alpha := c thin n^(-1 slash 2),
  quad
  m := n - n_0,
  quad
  p := max(2, ceil(log n)),
  quad
  q := max(2 p, ceil(log(e thin d)), 2).
  $
  Put
  $
  alpha_("adm")(p,q)
    := min (
      alpha_infinity,
      alpha_("inv"),
      frac(1, 2 a),
      alpha_*(q, t_"mix"),
      alpha_("st")(p)
    ).
  $ <eq:burn-final-alpha-adm>
  Here $alpha_("inv")$ is the local inverse ceiling from @eq:alpha-inv,
  $alpha_*(q,t_"mix")$ collects the Levin depth-two moment admissibility
  restrictions, and $alpha_("st")(p)$ is the common product-stability and
  full-state startup threshold in @lem:burn-product-stability and
  @lem:burn-full-startup.
  Suppose $n >= 3$ is such that
  $
  m >= n slash 2,
  quad
  2 alpha <= alpha_("adm")(p,q),
  quad
  m thin alpha thin a
    >= frac(2 C_("burn,3") || u ||^2, sigma^2(u)),
  $ <eq:burn-final-step-conditions>
  and the burn-in satisfies the explicit logarithmic conditions
  $
  n_0 >= frac(2, alpha a) log n,
  quad
  n_0 >= frac(p, c_("init") alpha a) log n,
  quad
  n_0 >= frac(p, c_("st") alpha a) log n.
  $ <eq:burn-final-log-conditions>
  Then there exists a finite constant $C_("burn,final")(u,c,theta_0)$,
  depending only on $u$, $c$, $||theta_0 - theta^*||$, and the problem and
  universal constants in the preceding bounds, such that
  $
  d_K lr((Xi_(n,n_0)^("bRR")(u), cal(N)(0, 1)))
    <= frac(C_("burn,final")(u,c,theta_0) thin "polylog"(n),
             n^(1 slash 4)),
  $ <eq:burn-final-finite-window>
  and
  $
  d_K lr((Xi_(n,n_0)^("asy,RR")(u), cal(N)(0, 1)))
    <= frac(C_("burn,final")(u,c,theta_0) thin "polylog"(n),
             n^(1 slash 4)).
  $ <eq:burn-final-asymptotic>
] <thm:burn-final-balanced>

_Proof._ Apply the finite-window assembly theorem @thm:burn-RR-BE-master. Since
$m >= n slash 2$, the martingale terms satisfy
$
frac(log^(3 slash 4) n, m^(1 slash 4))
  + frac(log n, sqrt(m))
  <= C thin frac("polylog"(n), n^(1 slash 4)).
$
It remains to bound the composite remainder in @lem:burn-R-bound. The first
condition in @eq:burn-final-log-conditions is @eq:burn-log-condition with
$beta = 1$; by @cor:burn-log-transient and
$alpha = c n^(-1 slash 2)$, $m >= n slash 2$, the deterministic transient is
$O(n^(-1))$. The second condition is @eq:burn-log-init-condition with
$beta = 1$, so @cor:burn-log-initial-product makes the random initial-product
term $O("polylog"(n) n^(-1))$. The Poisson Abel remainder is
$O(m^(-1 slash 2))$. The third condition in @eq:burn-final-log-conditions is
the startup condition @eq:burn-log-startup-condition with $beta = 1$, so
@cor:burn-misadjustment-rate gives
$
|| R_(n,n_0, op("fin"))^("mis,RR") ||_(L_p)
  <= C thin "polylog"(n) thin n^(-1 slash 4).
$
Together with
$sigma_(n,n_0)^("bRR")(u) >= sigma(u) slash sqrt(2)$, this makes the smoothing
remainder in @eq:burn-RR-BE-master of order
$"polylog"(n) n^(-1 slash 4)$. The smoothing tail $e slash n$ is lower order,
so @eq:burn-final-finite-window follows.

For the asymptotic normalization, write
$
Xi_(n,n_0)^("asy,RR")(u)
  = r_(n,n_0)(u) thin Xi_(n,n_0)^("bRR")(u).
$
By @lem:burn-normalization-transfer, the additional cost is at most
$
frac(C thin ||u||^2, m thin alpha thin a thin sigma^2(u))
  = O(n^(-1 slash 2)),
$
because $m >= n slash 2$ and $alpha = c n^(-1 slash 2)$. This is absorbed
into the balanced finite-window rate, proving @eq:burn-final-asymptotic.
$square$

Condition @eq:burn-final-step-conditions collects the finite-$n$ admissibility
requirements. The inequality $2 alpha <= alpha_("adm")(p,q)$ enforces the
Lyapunov small-step ceiling, the local inverse ceiling, the Levin depth-two
admissibility threshold, the random-product stability estimate
@lem:burn-product-stability, and the full-state startup contraction
@lem:burn-full-startup. Since
$alpha = c n^(-1 slash 2)$ and $m >= n slash 2$, the elementary step-size
constraints and @eq:burn-variance-lb-condition hold automatically for all
sufficiently large $n$. The remaining non-elementary large-$n$ requirement is
$
2 c n^(-1 slash 2)
  <= min (alpha_*(q, t_"mix"), alpha_("st")(p)),
$ <eq:burn-final-levin-eventual>
with $p,q$ as in the theorem. Under this Levin/startup admissibility condition,
the large-$n$ reading of the theorem keeps only $m >= n slash 2$ and the
burn-in lower bounds in @eq:burn-final-log-conditions.

#corollary[
  *($sqrt(n)$-normalization for the burned-in RR statistic.)*
  Under the assumptions of @thm:burn-final-balanced, define the thesis-facing
  scalar statistic
  $
  T_(n,n_0)^("RR,n")(u)
    := sqrt(n) thin u^top lr((
        overline(theta)_(n,n_0)^(("RR", alpha)) - theta^*
      ))
    = sqrt(n slash m) thin T_(n,n_0)^("RR")(u),
  $
  and its asymptotic normalization
  $
  Xi_(n,n_0)^("n,RR")(u)
    := frac(T_(n,n_0)^("RR,n")(u), sigma(u)).
  $
  For the logarithmic choice $p = max(2, ceil(log n))$ in
  @thm:burn-final-balanced, the lower burn-in conditions
  @eq:burn-final-log-conditions are implied by
  $
  n_0 >= C_- thin (alpha a)^(-1) log^2 n
  $
  with any fixed $C_-$ large enough. If, in addition, the burn-in window stays
  in the same logarithmic scale,
  $
  C_- thin (alpha a)^(-1) log^2 n
    <= n_0
    <= C_0 thin (alpha a)^(-1) log^2 n
  $ <eq:burn-log-window>
  for some finite $C_0$ and such a fixed $C_-$, then there exists a finite constant
  $C_("burn,n")(u,c,theta_0,C_0,C_-)$ such that
  $
  d_K lr((Xi_(n,n_0)^("n,RR")(u), cal(N)(0, 1)))
    <= frac(C_("burn,n")(u,c,theta_0,C_0,C_-) thin "polylog"(n),
             n^(1 slash 4)).
  $ <eq:burn-sqrt-n-final>
] <cor:burn-sqrt-n-transfer>

_Proof._ Put $s_(n,n_0) := sqrt(n slash m)$. Since
@eq:burn-final-step-conditions gives $m >= n slash 2$,
$
0 <= s_(n,n_0) - 1
  = frac(n_0, m thin (s_(n,n_0) + 1))
  <= frac(2 n_0, n).
$
For every real random variable $W$ and every $s in [1, sqrt(2)]$,
$
d_K lr((s W, cal(N)(0, 1)))
  <= d_K lr((W, cal(N)(0, 1))) + C thin |s - 1|,
$
by the same scaling argument as in @lem:burn-normalization-transfer. Applying
this with $W = Xi_(n,n_0)^("asy,RR")(u)$ and using
@thm:burn-final-balanced gives the already proved
$"polylog"(n) n^(-1 slash 4)$ term. The upper burn-in bound
in @eq:burn-log-window and $alpha = c n^(-1 slash 2)$ give
$
|s_(n,n_0) - 1|
  <= frac(2 C_0, c a) frac(log^2 n, n^(1 slash 2)),
$
which is lower order and is absorbed into the same balanced-scale rate.
$square$

The lower side of @eq:burn-log-window is the logarithmic-square burn-in needed
by the current $L_p$ startup contraction; the upper side keeps the
$sqrt(n slash m)$ rescaling lower order.

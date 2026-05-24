#import "../defs.typ": *

== PR-Averaged Error Decomposition and the RR Weight

The deterministic weight estimates start from the finite-start representation
of $sqrt(n) (overline(theta)_n^(("RR", alpha)) - theta^*)$ as a noise-weighted
sum with a deterministic kernel. The stationary $n_0 = 0$ Berry--Esseen bound
proved below is stated for the augmented-chain assembly built from the same
full-window weights.

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
Replacing the random products in the noise sum by their deterministic
counterparts $B_alpha^(k - l) := (I - alpha overline(A))^(k - l)$ and keeping
the initial-product discrepancy separate yields the *depth-one decomposition*
(Samsonov et al., 2025, Proposition 9):
$
R_(k, op("init"))^((alpha))
  := lr((Gamma_(1 : k)^((alpha)) - B_alpha^k)) (theta_0 - theta^*),
$ <eq:init-product-remainder>
$
theta_k^((alpha)) - theta^*
  = -alpha sum_(l = 1)^k B_alpha^(k - l) epsilon.alt(Z_l)
    + B_alpha^k (theta_0 - theta^*)
    + R_(k, op("init"))^((alpha))
    + R_k^((alpha)),
$ <eq:depth-one>
where $R_k^((alpha)) := J_k^((1, alpha)) + H_k^((1, alpha))$ is the
first noise-driven misadjustment remainder. The term
$R_(k, op("init"))^((alpha))$ vanishes when $theta_0 = theta^*$; otherwise it
is a finite-start transient and is handled in the burn-in transfer theorem,
not by the stationary augmented-chain misadjustment bound. The leading
component $J_k^((1, alpha))$ has a stationary bias of order $alpha$, so it is
not treated as an $alpha^(3 slash 2)$ term. The Berry--Esseen proof below
controls this noise-driven remainder by refining it to depth two, where
$J^((2)) + H^((2))$ carries the $alpha^(3 slash 2)$ moment scale. The first
sum is the *depth-zero* term and is the only piece that carries the limiting
Gaussian.

*PR averaging produces $Q_l^((alpha))$.* Recall the PR average
$overline(theta)_n^((alpha)) = (n - n_0)^(-1) sum_(k = n_0)^(n - 1) theta_k^((alpha))$.
For notational clarity, and for the stationary result proved in this chapter,
we set $n_0 = 0$ from this point on. A burned-in average has a different
deterministic weight,
$
Q_(l,n_0)^((alpha))
  = frac(n, n - n_0) alpha sum_(k = max(n_0, l))^(n - 1)
      B_alpha^(k - l),
$
when the statistic is normalized as
$- n^(-1 slash 2) sum_l Q_(l,n_0)^((alpha)) epsilon.alt(Z_l)$. The
burned-in non-stationary theorem therefore requires separate weight, Poisson,
and variance-comparison arguments. Subtracting $theta^*$, substituting the
depth-one decomposition above, and *exchanging the order of summation* in the
depth-zero piece,
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
Multiplying the resulting identity by $sqrt(n)$ gives the PR-averaged
decomposition
$
sqrt(n) thin (overline(theta)_n^((alpha)) - theta^*) = W^((alpha)) + D^((alpha)),
$
where the *leading martingale-like sum* is
$
W^((alpha)) := -frac(1, sqrt(n)) sum_(l = 1)^(n - 1) Q_l^((alpha)) thin epsilon.alt(Z_l),
$ <eq:W-alpha>
and the remainder $D^((alpha))$ contains the deterministic transient
$D_(op("tr"))^((alpha))$, the random initial-product transient
$D_(op("init"))^((alpha))$, and the higher-order noise-driven stochastic part
$D_R^((alpha))$:
$
D^((alpha))
  := D_(op("tr"))^((alpha))
     + D_(op("init"))^((alpha))
     + D_R^((alpha)),
quad
D_(op("tr"))^((alpha))
  := frac(1, sqrt(n)) sum_(k = 0)^(n - 1) B_alpha^k (theta_0 - theta^*),
quad
D_(op("init"))^((alpha))
  := frac(1, sqrt(n)) sum_(k = 0)^(n - 1)
      R_(k, op("init"))^((alpha)),
quad
D_R^((alpha))
  := frac(1, sqrt(n)) sum_(k = 0)^(n - 1) R_k^((alpha)).
$
The first sum is the deterministic initial-condition transient. Under the
full-average convention it is only $O((sqrt(n) alpha a)^(-1))$ in general, so
it must either be retained explicitly or removed by a centered initialization
$theta_0 = theta^*$. The second sum is stochastic but still purely
finite-start: it is controlled after burn-in by random-product stability. The
third is the source of the leading non-Gaussian correction in the
Berry--Esseen rate. Its RR-combination
$2 D_R^((alpha)) - D_R^((2 alpha))$ is the *misadjustment*
$D_1^("mis, RR")$ controlled below by the Levin depth-two transfer.

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
Since both PR averages share the *same* noise realization
${Z_k}$, the bracket $cal(Q)_l^("RR")$ is a single deterministic
matrix kernel: all the stochastic content of $W^("RR")$ now lives in
$epsilon.alt(Z_l)$ alone.

*Two deterministic estimates.* The rest of the martingale approximation uses
two quantities determined by $cal(Q)_l^("RR")$:

+ *Variance comparison.* The target CLT covariance of $W^("RR")$ is
  $Sigma_infinity = overline(A)^(-1) Sigma_epsilon.alt^(("M")) overline(A)^(-top)$
  (the Markov-chain CLT covariance). Replacing each $cal(Q)_l^("RR")$ by the
  asymptotic weight $overline(A)^(-1)$ gives this covariance, so the
  finite-$n$ deviation is controlled by
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


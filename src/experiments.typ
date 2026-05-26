#import "defs.typ": *

== Goals and scope

The purpose of this chapter is to compare the finite-sample behavior of the
constant-stepsize Richardson--Romberg estimator with standard alternatives
for Markovian linear stochastic approximation. The experiments are not used
in the proofs of @thm:RR-BE or @thm:burn-final-balanced. Instead, they serve
three complementary roles:

+ They check whether stepsize Richardson--Romberg extrapolation reduces the
  point-estimator bias at the sample sizes where confidence intervals are
  constructed.
+ They compare the resulting confidence intervals with diminishing-stepsize
  and Polyak--Ruppert baselines.
+ They isolate the effect of practical long-run variance estimators, in
  particular overlapping batch means (OBM) and its lugsail variant.

The last point is deliberately experimental in this thesis. The theoretical
chapters above prove Berry--Esseen and burn-in transfer results for the
RR-averaged estimator itself. A detailed non-asymptotic theory for OBM and
lugsail covariance estimation along RR-averaged constant-stepsize LSA
trajectories is left for future work.

== Experimental setup

The main comparison experiment follows the finite-state Markovian LSA setup
used in Huo, Chen, and Xie (2024). For each randomly generated problem, the
Markov chain has 10 states and the parameter dimension is $d=5$. The reported
numbers aggregate 100 independent problems, with 100 Monte Carlo trajectories
per problem and trajectory length $T=10^6$.

#table(
  columns: (1.25fr, 2.75fr),
  inset: 4pt,
  [*Quantity*], [*Value used in the main comparison*],
  [Problem class], [Finite-state Markovian linear stochastic approximation],
  [Number of problems], [$100$ independently generated problems],
  [Trajectories per problem], [$100$],
  [Dimension and states], [$d=5$ and 10 Markov states],
  [Trajectory length], [$T=10^6$],
  [Problem generation], [`eig_min = 0.25`, `eig_max = 0.60`,
    `noise_target = 0.35`, `A_norm` cap `1.0`],
  [Diminishing-stepsize schedule],
  [$c_0=200$, $k_0=20000$, $gamma=0.65$],
  [Constant stepsizes for RR],
  [$alpha in {0.2, 0.02}$],
  [Projection for confidence intervals],
  [One random unit direction is sampled for each problem; coverage is reported
   for this scalar direction, not for the first coordinate.],
)

The methods compared in the main table differ in two ways: the estimator used
for the center of the interval and the estimator used for the long-run
variance.

#table(
  columns: (1.35fr, 2.65fr),
  inset: 4pt,
  [*Method*], [*Description*],
  [$alpha=0.2$ constant],
  [Constant-stepsize averaged LSA with the larger stepsize branch.],
  [$alpha=0.02$ constant],
  [Constant-stepsize averaged LSA with the smaller stepsize branch.],
  [RR constant stepsizes],
  [Richardson--Romberg combination of the two constant-stepsize averages,
   using $alpha in {0.2, 0.02}$.],
  [Diminishing $0.2 slash sqrt(k)$],
  [Classical diminishing-stepsize baseline with $alpha_k = 0.2 slash sqrt(k)$.],
  [PR + OBM],
  [Polyak--Ruppert averaging with OBM long-run variance estimation.],
  [RR + OBM],
  [RR point estimator with OBM long-run variance estimation.],
  [RR + OBM-RR],
  [RR point estimator with lugsail/OBM-RR long-run variance estimation.],
)

The main comparison reports three metrics. The L2 error is the Euclidean norm
of the final estimation error, aggregated over problems; it reflects both
finite-sample variance and bias. The CI width is the length of the two-sided
scalar confidence interval in the sampled projection direction. Coverage is
the empirical probability that this scalar interval contains the projected
target $u^top theta^*$. Thus the coverage numbers are one-dimensional
random-direction coverages; they are not coordinatewise coverages and not
simultaneous multivariate coverages. L2 errors and CI widths are shown in
units of $10^(-3)$.

== Long-run variance estimation: OBM and lugsail

The main theorems above are stated either with a deterministic finite-window
variance proxy or with the asymptotic variance
$sigma^2(u)=u^top Sigma_infinity u$. In an actual confidence interval this
quantity must be estimated from the observed dependent trajectory. The naive
sample variance of the iterates estimates the marginal variance
$op("Var")(Y_t)$, not the long-run variance $sigma^2(u)$, and therefore
misses the serial-correlation correction. OBM is used precisely because it
targets this long-run variance.

For a scalar projected trajectory $Y_t = u^top theta_t$, the overlapping batch
means (OBM) estimator with block size $b$ is
$
hat(sigma)_(op("OBM"))^2(b)
  = frac(b, T - b + 1) sum_(s=0)^(T-b)
      lr((overline(Y)_(s,b) - overline(Y)_T))^2,
quad
overline(Y)_(s,b) = frac(1, b) sum_(j=s)^(s+b-1) Y_j.
$ <eq:obm-experimental-estimator>
Equivalently, OBM is the batch-means form of the Bartlett-window estimator of
the spectral density at frequency zero. This is the appropriate object for
estimating the time-average covariance in Markov chain CLTs; see Flegal and
Jones (2010) and Liu, Vats, and Flegal (2022).

The price is a window bias. For Bartlett/OBM estimators the standard
asymptotic template is
$
bb(E) hat(sigma)_(op("OBM"))^2(b)
  = sigma^2(u) + frac(c_1(u), b) + "lower order terms",
$ <eq:obm-bias-template>
and the MSE has the characteristic bias--variance tradeoff
$c_1(u)^2 slash b^2 + C(u) b slash T$. For positively correlated output, the
leading bias is typically negative, so confidence intervals based on the raw
OBM estimator can be too narrow.

Lugsail windows are designed to reduce this leading window bias of the
variance estimator (Vats and Flegal, 2022). In the implementation used here,
the lugsail version is the block-size Richardson--Romberg combination
$
hat(sigma)_(op("OBM-RR"))^2(b, lambda)
  = frac(lambda, lambda - 1) hat(sigma)_(op("OBM"))^2(lambda b)
    - frac(1, lambda - 1) hat(sigma)_(op("OBM"))^2(b),
quad lambda > 1.
$ <eq:obm-rr-estimator>
For the Bartlett kernel, $lambda=2$ cancels the leading $1 slash b$ bias term.
This correction is conceptually separate from Richardson--Romberg
extrapolation in the stepsize. RR in $alpha$ targets the stochastic
approximation bias of the point estimator
$overline(theta)^(("RR", alpha))$; lugsail or OBM-RR in $b$ targets the window
bias of the long-run variance estimator. Thus the two Richardson--Romberg
uses operate on different quantities and are not substitutes for each other.

The lugsail estimator is a signed linear combination of two OBM estimators.
Consequently, in finite samples the scalar estimate can be negative, and the
matrix version can fail to be positive semidefinite. Any implementation that
uses lugsail covariance matrices must therefore report how often this occurs
or specify a clipping/projection convention.

In this thesis, OBM and lugsail are investigated experimentally only. A
theorem for OBM/lugsail covariance estimation along RR-averaged
constant-stepsize LSA trajectories is not proved here and is left for future
work. That future analysis should combine the OBM and lugsail theory above
with finite-sample spectral-density corrections in the spirit of Ng and
Perron (1996) and with recent batch-means methodology for stochastic-gradient
inference such as Singh, Shukla, and Vats (2025).

== Main comparison

The numerical values in this subsection are taken from the completed report
`reports/2026-04-23_main_comparison.md`. The table is a compact summary of
the main methods; the report also records additional methods, coverage
percentiles, problem diagnostics, and raw output locations.

The expected behavior is as follows. A single large constant stepsize should
produce short intervals but can leave substantial steady-state bias. A single
small constant stepsize reduces this bias only partially at a fixed horizon.
The Richardson--Romberg combination is designed to cancel the leading
stepsize bias while keeping the long-run variance target essentially the
same. Therefore the desirable empirical signature is lower L2 error and
near-nominal coverage without a major increase in interval width.

The main comparison confirms this pattern. The entries are medians over the
100 generated problems.

#table(
  columns: (1.7fr, 0.8fr, 0.8fr, 0.8fr),
  inset: 4pt,
  [*Method*], [*L2*], [*CI width*], [*Coverage*],
  [$alpha=0.2$ constant], [$26.67$], [$8.36$], [$0.5%$],
  [$alpha=0.02$ constant], [$13.93$], [$8.27$], [$40.5%$],
  [RR constant stepsizes], [$4.52$], [$8.22$], [$94.0%$],
  [Diminishing $0.2 slash sqrt(k)$], [$6.80$], [$10.34$], [$90.0%$],
  [PR + OBM], [$5.35$], [$8.04$], [$92.0%$],
  [RR + OBM], [$4.52$], [$8.23$], [$95.0%$],
  [RR + OBM-RR], [$4.52$], [$8.22$], [$95.0%$],
)

The RR combination reduces the median L2 error to $4.52 dot 10^(-3)$ and
brings the median scalar coverage close to the nominal 95% level. Its
interval widths are comparable to the single-stepsize intervals, so the
coverage gain is not obtained by simply making intervals wider.

The single constant-stepsize branches undercover because the centers of the
intervals are biased. This is most visible for $alpha=0.2$, but it remains
substantial for $alpha=0.02$. Changing the variance estimator cannot repair
this failure: OBM, MSB, or OBM-RR can adjust the estimated uncertainty around
the center, but they do not move the biased center itself.

In this long-horizon run, OBM-RR/lugsail is essentially neutral. At
$T=10^6$, the OBM window bias is already small enough that the lugsail
correction does not visibly improve coverage. The remaining differences in
CI width between RR+OBM and RR+OBM-RR are within the scale of Monte Carlo
variation in this comparison.

== Theory-aligned RR stepsize sweep

The main comparison above uses the wide practical pair
$alpha in {0.2, 0.02}$. A separate sweep, recorded in
`reports/2026-05-26_theory_rr_alpha_sweep.md`, repeats the comparison for
the adjacent two-level pairs $(2 alpha, alpha)$ with
$alpha in {0.02, 0.05, 0.10}$. This is the stepsize geometry closest to the
first-order Richardson--Romberg expansion used in the theory.

The setup is the same finite-state LSA experiment as in the main comparison:
100 problems, 100 trajectories per problem, $T=10^6$, $d=5$, 10 Markov
states, and a random scalar projection direction per problem. The same
problem seeds are used across the three rows. The table reports medians over
problems; L2 errors and CI widths are again in units of $10^(-3)$.

#table(
  columns: (1.15fr, 1.65fr, 0.8fr, 0.8fr, 0.8fr),
  inset: 4pt,
  [*RR pair*], [*Single-alpha coverage*], [*RR L2*], [*RR width*], [*RR coverage*],
  [$(0.04, 0.02)$], [$0.04$: $92.0%$; $0.02$: $94.0%$],
  [$2.97$], [$5.36$], [$94.0%$],
  [$(0.10, 0.05)$], [$0.10$: $86.0%$; $0.05$: $91.5%$],
  [$2.97$], [$5.37$], [$94.0%$],
  [$(0.20, 0.10)$], [$0.20$: $67.5%$; $0.10$: $86.0%$],
  [$2.97$], [$5.38$], [$94.0%$],
)

The sweep shows that the RR gain is not an artifact of one particular
wide-pair tuning. As the larger single-alpha branch moves from $0.04$ to
$0.20$, its median coverage deteriorates from $92.0%$ to $67.5%$. After
Richardson--Romberg extrapolation, however, the median L2 error, CI width,
and coverage are essentially unchanged across the three adjacent pairs.

The OBM and OBM-RR versions of the RR intervals give the same qualitative
message. In this sweep, RR+OBM and RR+OBM-RR have median coverage between
$94.0%$ and $95.0%$, with width changes below about one percent. Thus the
new run reinforces the separation between the two corrections: RR in
$alpha$ removes the point-estimator bias, while OBM or lugsail in $b$ only
modifies the estimated long-run variance.

== Lugsail bias--variance diagnostic

A separate lugsail bias--variance experiment isolates the covariance
estimator itself. The numerical values in this subsection are taken from
`reports/2026-04-23_lugsail_bias_variance.md`. The experiment compares OBM
and OBM-RR over a grid of block sizes and uses the analytic value of
$sigma^2(u)$ as ground truth.

In the short-horizon regime, lugsail helps because the raw OBM estimator has
visible negative Bartlett-window bias. For PR iterates, OBM-RR with
$lambda=2$ reduces the leading window bias and shifts the useful block-size
range downward. At $T=10^5$, the best OBM point has relative bias about
$-2.5%$ and MSE $0.72$, while OBM-RR with $lambda=2$ has relative bias about
$-0.7%$ and MSE $0.34$. Across $T in {20000, 50000, 100000}$, the observed
MSE reduction is roughly 40--55%.

For constant-stepsize and RR iterate paths, the improvement is more limited
because part of the error is SA transient or point-estimator bias, not
Bartlett-window bias. Lugsail can correct the leading window bias of the
variance estimator, but it cannot correct this separate stochastic
approximation bias component.

These experiments support the following interpretation. Step-size RR is the
primary tool for improving the center of the confidence interval. OBM is the
baseline practical estimator of $Sigma_infinity$. Lugsail/OBM-RR is useful
when the long-run-variance estimator has visible negative window bias, mainly
in shorter-horizon or more persistent regimes.

== Limitations and planned experimental extensions

The current experiments support the qualitative message of the theory, but
they do not yet cover all diagnostics needed for a complete empirical study.
The following extensions are planned to separate theorem validation,
variance-estimation accuracy, and robustness to problem conditioning.

#table(
  columns: (1.35fr, 2.75fr),
  inset: 4pt,
  [*Extension*], [*Purpose*],
  [Coverage as a function of $T$],
  [Repeating RR+OBM and RR+OBM-RR for
   $T in {2 dot 10^4, 5 dot 10^4, 10^5, 3 dot 10^5, 10^6}$ would show the
   horizon where lugsail improves coverage and the horizon where ordinary OBM
   is already sufficient.],
  [Oracle-variance comparison],
  [Intervals based on the analytic $sigma^2(u)$ should be compared with
   OBM, MSB, and OBM-RR intervals. This would separate normal-approximation
   error and point-estimator bias from variance-estimation error.],
  [Burn-in and initialization sweep],
  [Varying $n_0$, $theta_0$, and the initial law of $Z_0$ would test the
   deterministic-start transfer and the practical size of the
   $(alpha a)^(-1) log^2 n$ burn-in window.],
  [Block-size sweep for coverage],
  [For each main estimator, a sweep over $b = floor(T^eta)$ should report
   coverage, width, variance-estimator bias, and the frequency of negative or
   clamped lugsail estimates.],
  [Mixing and conditioning stress test],
  [Varying the Markov-chain mixing rate, the spectral gap of $overline(A)$,
   and the noise amplitude would check whether the empirical behavior follows
   the theorem's dependence on $t_"mix"$, $a$, and bounded-noise constants.],
  [Matrix covariance diagnostics],
  [The present coverage results are scalar random-direction diagnostics.
   Additional runs should check several directions and the full estimated
   covariance matrix, including positive-semidefiniteness of lugsail
   corrections.],
)

The theory-aligned stepsize sweep has now been completed for three adjacent
pairs. The most important remaining diagnostics are therefore the horizon
sweep and the oracle-variance comparison: the first would show when lugsail
matters for coverage, and the second would separate variance-estimation error
from point-estimator bias and normal-approximation error.

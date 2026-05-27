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
  [RR + OBM-LW],
  [RR point estimator with OBM long-run variance estimation using a lugsail
   window.],
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
the lugsail-window OBM estimator, abbreviated OBM-LW, is
$
hat(sigma)_(op("OBM-LW"))^2(b, lambda)
  = frac(lambda, lambda - 1) hat(sigma)_(op("OBM"))^2(lambda b)
    - frac(1, lambda - 1) hat(sigma)_(op("OBM"))^2(b),
quad lambda > 1.
$ <eq:obm-lw-estimator>
For the Bartlett kernel, $lambda=2$ cancels the leading $1 slash b$ bias term.
This correction is conceptually separate from Richardson--Romberg
extrapolation in the stepsize. RR in $alpha$ targets the stochastic
approximation bias of the point estimator
$overline(theta)^(("RR", alpha))$; the lugsail window in $b$ targets the
window bias of the long-run variance estimator. Thus stepsize extrapolation
and lugsail-window variance estimation operate on different quantities and
are not substitutes for each other.

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

The table below gives a compact summary of the completed main comparison
run. The entries are medians over the 100 generated problems and focus on
the methods most directly relevant to the thesis message.

The expected behavior is as follows. A single large constant stepsize should
produce short intervals but can leave substantial steady-state bias. A single
small constant stepsize reduces this bias only partially at a fixed horizon.
The Richardson--Romberg combination is designed to cancel the leading
stepsize bias while keeping the long-run variance target essentially the
same. Therefore the desirable empirical signature is lower L2 error and
near-nominal coverage without a major increase in interval width.

The main comparison confirms this pattern.

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
  [RR + OBM-LW], [$4.52$], [$8.22$], [$95.0%$],
)

#figure(
  image("../figures/experiments/main_methods_comparison.svg", width: 100%),
  caption: [Main comparison at $T=10^6$. The left panel shows median
    Euclidean error and the right panel shows median scalar coverage across
    the same 100 generated problems.],
) <fig:main-methods-comparison>

The RR combination reduces the median L2 error to $4.52 dot 10^(-3)$ and
brings the median scalar coverage close to the nominal 95% level. Its
interval widths are comparable to the single-stepsize intervals, so the
coverage gain is not obtained by simply making intervals wider.

The single constant-stepsize branches undercover because the centers of the
intervals are biased. This is most visible for $alpha=0.2$, but it remains
substantial for $alpha=0.02$. Changing the variance estimator cannot repair
this failure: OBM, MSB, or OBM-LW can adjust the estimated uncertainty around
the center, but they do not move the biased center itself.

In this long-horizon run, OBM-LW is essentially neutral. At
$T=10^6$, the OBM window bias is already small enough that the lugsail
correction does not visibly improve coverage. The remaining differences in
CI width between RR+OBM and RR+OBM-LW are within the scale of Monte Carlo
variation in this comparison.

== Theory-aligned RR stepsize sweep

The main comparison above uses the wide practical pair
$alpha in {0.2, 0.02}$. To check that the effect is not specific to this
wide pair, a separate sweep repeats the comparison for the adjacent
two-level pairs $(2 alpha, alpha)$ with
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

The OBM and OBM-LW versions of the RR intervals give the same qualitative
message. In this sweep, RR+OBM and RR+OBM-LW have median coverage between
$94.0%$ and $95.0%$, with width changes below about one percent. Thus this
experiment reinforces the separation between the two corrections: RR in
$alpha$ removes the point-estimator bias, while OBM and OBM-LW in $b$ only
modify the estimated long-run variance.

== Oracle-variance diagnostic

The theory-aligned sweep still uses estimated long-run variances. To separate
variance-estimation error from point-estimator bias and normal-approximation
error, we repeat the largest adjacent-pair experiment, $(0.20, 0.10)$, with
an oracle interval based on the analytic finite-state value
$sigma^2(u)=u^top Sigma_infinity u$. The interval center is the same
RR-averaged estimator in every row.

#table(
  columns: (1.7fr, 0.8fr, 0.8fr, 0.8fr, 0.8fr),
  inset: 4pt,
  [*Method*], [*L2*], [*CI width*], [*Coverage*], [*Width/oracle*],
  [RR + batch means], [$2.97$], [$5.38$], [$94.0%$], [$0.990$],
  [RR + oracle variance], [$2.97$], [$5.43$], [$95.0%$], [$1.000$],
  [RR + OBM], [$2.97$], [$5.42$], [$95.0%$], [$0.998$],
  [RR + OBM-LW], [$2.97$], [$5.39$], [$95.0%$], [$0.993$],
  [RR + MSB], [$2.97$], [$5.36$], [$95.0%$], [$0.987$],
)

The oracle interval is nearly indistinguishable from the practical
variance-estimator intervals. OBM is only about $0.2%$ narrower than the
oracle interval at the median, and OBM-LW is about $0.7%$ narrower. The
median coverage remains $95%$ for the oracle, OBM, OBM-LW, and MSB rows.
Thus, at $T=10^6$, long-run-variance estimation is not the bottleneck for
the RR confidence intervals. The remaining coverage error is already at the
scale of Monte Carlo variation across the 100 generated problems.

== Coverage over trajectory length

We next repeat the oracle-variance comparison over
$T in {2 dot 10^4, 5 dot 10^4, 10^5, 3 dot 10^5, 10^6}$, using the same
RR pair $(0.20,0.10)$ and the same problem seeds across horizons. The goal
is to identify whether undercoverage at shorter horizons comes from the RR
center and normal approximation, or from estimating the long-run variance.

#table(
  columns: (0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  inset: 4pt,
  [*$T$*], [*Oracle cov.*], [*OBM cov.*], [*OBM-LW cov.*],
  [*OBM/oracle width*], [*OBM-LW/oracle width*],
  [$2 dot 10^4$], [$95.5%$], [$94.0%$], [$93.0%$], [$0.942$], [$0.944$],
  [$5 dot 10^4$], [$95.0%$], [$94.0%$], [$93.0%$], [$0.967$], [$0.958$],
  [$10^5$], [$95.0%$], [$94.0%$], [$94.0%$], [$0.978$], [$0.972$],
  [$3 dot 10^5$], [$95.0%$], [$94.0%$], [$94.0%$], [$0.989$], [$0.989$],
  [$10^6$], [$95.0%$], [$95.0%$], [$95.0%$], [$0.998$], [$0.993$],
)

The oracle coverage is already close to the nominal level at all horizons.
This suggests that, in this problem class, the RR center and the normal
approximation are not the main cause of the short-horizon coverage loss.
The gap is instead in the practical variance estimators: for example, at
$T=2 dot 10^4$ the OBM interval is about $5.8%$ narrower than the oracle
interval, and its median coverage is $94.0%$ instead of $95.5%$. This gap
shrinks monotonically in the width ratio as $T$ grows.

With the default block-size rule $b=floor(T^0.6)$, OBM-LW does not
improve coverage over OBM in this sweep. It is neutral at the largest
horizon and slightly lower at the smaller horizons. Therefore lugsail should
not be presented as an automatic coverage fix. Rather, together with the
bias--variance diagnostic below, the evidence says that lugsail is a
variance-estimator bias-reduction device whose practical benefit depends on
the block-size regime.

== Block-size sweep for coverage

To test this block-size dependence directly, we repeat the RR coverage
experiment for
$b = floor(T^eta)$, $eta in {0.3,0.4,0.5,0.6,0.7,0.8}$, and
$T in {2 dot 10^4, 10^5, 10^6}$. The RR center is fixed at the adjacent pair
$(0.20,0.10)$, so the differences across rows are caused by the
long-run variance estimator. The figure plots the most informative slice of
this grid.

#figure(
  image("../figures/experiments/blocksize_lugsail_diagnostics.svg", width: 100%),
  caption: [Block-size sensitivity of OBM and OBM-LW at $T=10^5$.
    The left panel shows median scalar coverage, and the right panel shows
    median relative bias of the corresponding long-run variance estimator.],
) <fig:blocksize-lugsail-diagnostics>

This sweep resolves the apparent tension between the default-rule coverage
sweep and the lugsail bias--variance diagnostic. Lugsail improves coverage
when OBM is still dominated by negative Bartlett-window bias, as shown in
@fig:blocksize-lugsail-diagnostics. For example,
at $T=2 dot 10^4$ and $eta=0.5$, OBM-LW changes the median relative bias
from $-0.222$ to $-0.022$, the width/oracle ratio from $0.878$ to $0.979$,
and median coverage from $91.5%$ to $95.0%$. At $T=10^5$, OBM-LW with
$eta=0.4$ or $eta=0.5$ reaches $95.0%$ median coverage, while OBM at the
same block sizes remains too narrow.

At the production rule $eta=0.6$, OBM is already close to the oracle width
in this problem class. The lugsail correction is therefore neutral rather
than beneficial. For very large blocks, however, OBM-LW becomes unstable:
at $eta=0.8$ the signed lugsail estimate is negative in $14.77%$ of
trajectory-level estimates at $T=2 dot 10^4$, $6.71%$ at $T=10^5$, and
$1.74%$ at $T=10^6$. Thus lugsail should be reported together with
negative/clamped-estimate diagnostics, not only with the final clamped CI
width.

== Conditioning and noise stress test

The conditioning/noise stress test checks whether the block-size conclusion
survives moderate changes in the problem generator. The run keeps the same
RR pair $(0.20,0.10)$ and the same random finite-state Markov-chain
generator, but varies the eigenvalue range of
$-overline(A)$ and the target norm of the state-dependent matrix noise. Thus
this is a conditioning and matrix-noise stress test, not yet a mixing-time
stress test.

The main conclusion is short: the oracle coverage is $95%$ in every moderate
conditioning/noise scenario, which indicates that the RR center and the normal
approximation remain adequate for these stress levels. Weaker mean
contraction increases the scale of the problem: at $T=10^6$, the median L2
error is $6.04 dot 10^(-3)$ in the weak-mean scenario, compared with
$2.97 dot 10^(-3)$ at baseline. Nevertheless, OBM-LW at $eta=0.5$ stays near
the oracle width and has median coverage $95%$.

The smaller block-size row $eta=0.4$ makes the OBM window bias more visible.
At $T=10^6$, OBM coverage is $91%$ in the weak-mean scenario and $92%$ in
the weak-plus-noise scenario, while OBM-LW restores $95%$ coverage in both
cases. The corresponding raw variance-estimator bias changes from $-0.252$
to $-0.004$ in the weak-mean scenario and from $-0.208$ to approximately zero
in the weak-plus-noise scenario. This supports the same interpretation as the
block-size sweep: lugsail helps when OBM is too narrow because of negative
Bartlett-window bias, but the useful block-size range must still be checked.

== Mixing-rate stress test

The mixing-rate stress test changes the Markov-chain dependence directly.
Starting from the same dense random transition matrix $P_0$, the experiment
uses the lazy mixture
$P_rho = rho I + (1-rho) P_0$. This keeps the stationary distribution fixed
but decreases the spectral gap. The LSA problem generator and the RR pair
$(0.20,0.10)$ are otherwise unchanged.

#figure(
  image("../figures/experiments/mixing_stress_diagnostics.svg", width: 70%),
  caption: [Slow-mixing stress test at $T=10^6$. The figure compares oracle
    variance intervals with OBM-LW intervals at $eta=0.5$ as the spectral gap
    decreases.],
) <fig:mixing-stress-diagnostics>

@fig:mixing-stress-diagnostics reports the OBM-LW row with $eta=0.5$, where
variance estimation is already close to oracle in the non-slow-mixing
experiments. For moderate slowdown, $rho=0.5$, the oracle and OBM-LW rows
remain near nominal. The intervals become wider because the long-run variance
is larger, but the qualitative behavior is unchanged.

For slower chains, the failure mode changes. At $rho=0.8$ and $T=10^6$,
OBM-LW has essentially oracle width, but both oracle and OBM-LW coverage are
only $76.5%$. At $rho=0.95$, the oracle row itself collapses: coverage is
$77.0%$ at $T=10^5$ and $6.5%$ at $T=10^6$. Increasing $T$ reduces the
oracle CI width from $107.45 dot 10^(-3)$ to $33.82 dot 10^(-3)$, while the
median L2 error only decreases from $124.79 dot 10^(-3)$ to
$111.76 dot 10^(-3)$. Thus the problem is no longer variance estimation.
The center or finite-sample normal approximation is failing relative to the
$T^(-1/2)$ interval width. This experiment shows why the dependence on
$t_"mix"$ in the theory is practically important.

== Lugsail bias--variance diagnostic

A separate lugsail bias--variance experiment isolates the covariance
estimator itself. It compares OBM and OBM-LW over a grid of block sizes and
uses the analytic value of $sigma^2(u)$ as ground truth.

In the short-horizon regime, lugsail helps because the raw OBM estimator has
visible negative Bartlett-window bias. For PR iterates, OBM-LW with
$lambda=2$ reduces the leading window bias and shifts the useful block-size
range downward. At $T=10^5$, the best OBM point has relative bias about
$-2.5%$ and MSE $0.72$, while OBM-LW with $lambda=2$ has relative bias about
$-0.7%$ and MSE $0.34$. Across $T in {20000, 50000, 100000}$, the observed
MSE reduction is roughly 40--55%.

For constant-stepsize and RR iterate paths, the improvement is more limited
because part of the error is SA transient or point-estimator bias, not
Bartlett-window bias. Lugsail can correct the leading window bias of the
variance estimator, but it cannot correct this separate stochastic
approximation bias component.

These experiments support the following interpretation. Step-size RR is the
primary tool for improving the center of the confidence interval. OBM is the
baseline practical estimator of $Sigma_infinity$. OBM-LW is useful
when the long-run-variance estimator has visible negative window bias, mainly
in shorter-horizon or more persistent regimes.

== Limitations and future diagnostics

The current experiments support the qualitative message of the theory, but
they do not yet cover all diagnostics needed for a complete empirical study.
The following diagnostics would make the empirical study more complete by
separating theorem validation, variance-estimation accuracy, and robustness
to problem conditioning.

#table(
  columns: (1.35fr, 2.75fr),
  inset: 4pt,
  [*Extension*], [*Purpose*],
  [Burn-in and initialization sweep],
  [Varying $n_0$, $theta_0$, and the initial law of $Z_0$ would test the
   deterministic-start transfer and the practical size of the
   $(alpha a)^(-1) log^2 n$ burn-in window.],
  [Mixing-aware tuning],
  [The mixing-rate stress test shows that strong slow mixing breaks oracle
   coverage. Additional runs should vary horizon, burn-in, and stepsize as
   functions of the spectral gap or $t_"mix"$.],
  [Matrix covariance diagnostics],
  [The present coverage results are scalar random-direction diagnostics.
   Additional runs should check several directions and the full estimated
   covariance matrix, including positive-semidefiniteness of lugsail
   corrections.],
)

The present experiment set already includes a theory-aligned stepsize sweep
for three adjacent pairs, an oracle $T$-sweep, a block-size coverage sweep,
and conditioning/noise and mixing-rate stress tests for the largest adjacent
pair. The most important remaining diagnostics are therefore mixing-aware
tuning and matrix-valued covariance checks: the current evidence is still
scalar-directional.

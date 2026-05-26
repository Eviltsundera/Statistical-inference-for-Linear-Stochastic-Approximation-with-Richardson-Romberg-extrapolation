#import "../defs.typ": *

== Contribution map and notation guide

The proof combines existing non-asymptotic tools with RR-specific algebra and
the deterministic-start transfer developed here. The main proof blocks are:

#table(
  columns: (1.25fr, 2.3fr, 2.2fr),
  inset: 4pt,
  [*Block*], [*Role in the proof*], [*Status*],
  [RR weight algebra],
  [Controls the deterministic kernels $cal(Q)_l^("RR")$ and
   $Q_(l;n_0,n)^("RR")$ and their variation.],
  [Derived in @sec:zeroth_order_rr and @sec:pr_weights.],
  [Poisson and martingale reduction],
  [Turns the Markovian depth-zero weighted sum into a martingale plus a
   bounded Poisson remainder.],
  [Adapted from the Samsonov et al. framework and reproved for RR weights.],
  [Stationary misadjustment],
  [Controls the non-martingale RR remainder
   $J^((1)) + J^((2)) + H^((2))$ under the stationary augmented chain.],
  [Uses Levin et al. inputs; the stationary RR assembly is carried out here.],
  [Burn-in transfer],
  [Transfers the stationary theorem to finite starts by controlling the
   deterministic transient, random initial product, and startup discrepancy.],
  [Developed in @sec:burn_in_transfer.],
)

The theorem-level outputs should be read in the following order:

#table(
  columns: (1.25fr, 1.2fr, 2.65fr),
  inset: 4pt,
  [*Output*], [*Label*], [*Object controlled*],
  [Stationary augmented-chain assembly],
  [@thm:RR-BE],
  [$S_(n, "stat")^("RR")(u)$ with the finite-window variance
   $sigma_n^("RR")(u)$.],
  [Stationary balanced triangular-array corollaries],
  [@cor:RR-BE-working and @cor:RR-BE-sigma],
  [$S_(n, "stat")^("RR")(u)$ at $alpha_n = c n^(-1\/2)$, with either
   finite-window or asymptotic normalization.],
  [Deterministic-start finite-window transfer],
  [@thm:burn-RR-BE-master],
  [$Xi_(n,n_0)^("bRR")(u)$ after burn-in, still normalized by
   $sigma_(n,n_0)^("bRR")(u)$.],
  [Deterministic-start balanced theorem],
  [@thm:burn-final-balanced and @cor:burn-sqrt-n-transfer],
  [$Xi_(n,n_0)^("asy,RR")(u)$ and then the final $sqrt(n)$ statistic
   $Xi_(n,n_0)^("n,RR")(u)$.],
)

The following notation is used throughout the later chapters:

#table(
  columns: (1.2fr, 3.3fr),
  inset: 4pt,
  [*Notation*], [*Meaning*],
  [$alpha, 2 alpha$], [The two constant step sizes used by the two-level RR estimator.],
  [$n_0$ and $m = n - n_0$], [Burn-in length and effective averaging window.],
  [$(Z_k)_(k >= 0)$ and $xi$],
  [Base Markov chain and initial law $xi = cal(L)(Z_0)$. The recursion uses
   observations $Z_k$, $k >= 1$; stationary covariance formulas may use $Z_0$
   by stationarity.],
  [$cal(F)_k$],
  [Natural filtration $sigma(Z_0, dots, Z_k)$. Local displays that begin at
   $Z_1$ use the same filtration with the harmless extra variable $Z_0$.],
  [$(Z_k^("stat"))_(k in ZZ)$],
  [Two-sided stationary copy of the base chain. Under the stationary
   augmented-chain convention the superscript is usually omitted.],
  [$B_w = I - w overline(A)$], [Deterministic linearized contraction at step size $w$.],
  [$cal(Q)_l^("RR")$], [Full-window RR PR weight in the stationary $n_0 = 0$ chapter.],
  [$Q_(l;n_0,n)^("RR")$ or $Q_l^("bRR")$], [Burned-in RR PR weight for the window $k = n_0, dots, n - 1$.],
  [$S_(n, "stat")^("RR")(u)$], [Stationary augmented-chain scalar statistic.],
  [$T_(n,n_0)^("RR")(u)$], [Finite-start burned-in scalar statistic with $sqrt(m)$ normalization.],
  [$Xi_(n,n_0)^("bRR")(u)$], [Finite-window normalized burned-in statistic.],
  [$sigma(u)^2 = u^top Sigma_infinity u$], [Asymptotic scalar variance target.],
  [$"polylog"(n)$],
  [A generic fixed power of $log n$, possibly changing from line to line.
   It never hides polynomial dependence on $n$ or any dependence on
   $alpha$, $p$, or $q$ that is rate-relevant in the surrounding display.],
)

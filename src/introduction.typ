#import "defs.typ": *

// === 1. Introduction ===

Stochastic approximation (SA) algorithms are a cornerstone of modern computational statistics, optimization, and reinforcement learning.
Introduced by Robbins and Monro (1951), these iterative procedures provide a principled way to find roots of equations or optimize objectives when only noisy observations are available. A particularly important subclass is the _linear stochastic approximation_ (LSA) algorithm, which arises naturally in temporal-difference (TD) learning, policy evaluation, and stochastic gradient descent for linear models.

In this work, we study the LSA recursion with a _constant step size_ $alpha > 0$:
$ theta_k^((alpha)) = theta_(k-1)^((alpha)) - alpha {A(Z_k) theta_(k-1)^((alpha)) - b(Z_k)}, quad k >= 1, $ <eq:lsa>
where ${Z_k}_(k in bb(N))$ is a time-homogeneous Markov chain on a measurable space $(sans(Z), cal(Z))$ with transition kernel $sans(Q)$ and unique invariant distribution $pi$.
The mappings $A : sans(Z) -> bb(R)^(d times d)$ and $b : sans(Z) -> bb(R)^d$ are measurable functions satisfying $overline(A) := integral_(sans(Z)) A(z) d pi(z)$ and $overline(b) := integral_(sans(Z)) b(z) d pi(z)$.
We use the stability convention that $-overline(A)$ is Hurwitz, equivalently all eigenvalues of $overline(A)$ have strictly positive real parts. Then the target parameter $theta^* = overline(A)^(-1) overline(b)$ is uniquely defined.

The _Polyak--Ruppert averaging_ procedure (Polyak, 1990; Ruppert, 1988) provides an effective variance reduction technique.
Given a burn-in period $n_0 >= 0$, the averaged iterate is defined as
$ overline(theta)_n^((alpha)) = frac(1, n - n_0) sum_(k=n_0)^(n-1) theta_k^((alpha)). $ <eq:pr-average>

== Bias of constant step-size iterates

The use of a constant step size $alpha > 0$ offers several practical advantages: it enables geometrically fast forgetting of the initial condition (Dieuleveut, Durmus, and Bach, 2020) and simplifies hyperparameter tuning compared to diminishing step-size schedules.
However, unlike the classical regime $alpha_k -> 0$ with $sum alpha_k = infinity$ and $sum alpha_k^2 < infinity$, a constant step size produces iterates that converge only _in distribution_ to a stationary measure $Pi_alpha$, rather than almost surely to $theta^*$.
The stationary expectation $bb(E)[theta_infinity^((alpha))]$ is generally _biased_ with respect to $theta^*$, and this bias cannot be eliminated by Polyak--Ruppert averaging alone.

As shown in Levin, Naumov, and Samsonov (2025), the stationary bias has a
leading linear term in $alpha$:
$ lim_(n -> infinity) bb(E)[theta_n^((alpha))] = theta^* + alpha Delta + O(alpha^(3\/2)), $ <eq:bias-expansion>
where $Delta = overline(A)^(-1) sum_(k=1)^infinity bb(E)[{sans(Q)^k tilde(A)(Z_infinity)} epsilon.alt(Z_infinity)]$ depends on the correlation structure of the Markov chain, and $tilde(A)(z) = A(z) - overline(A)$ is the centered matrix-valued function.
Under stronger expansion assumptions, the power-series approach of Huo, Chen,
and Xie (2024) gives higher-order bias expansions in integer powers of
$alpha$; in the Levin decomposition, the first misadjustment bias component
itself has an $O(alpha^2)$ remainder after the leading $alpha Delta$ term.

== Richardson--Romberg extrapolation

To eliminate the leading $O(alpha)$ bias term, we employ the _Richardson--Romberg_ (RR) _extrapolation_ procedure.
Two LSA sequences are run _on the same Markov chain trajectory_ ${Z_k}$ with step sizes $alpha$ and $2 alpha$, and the RR iterate is formed as
$ overline(theta)_n^((alpha, "RR")) = 2 overline(theta)_n^((alpha)) - overline(theta)_n^((2 alpha)). $ <eq:rr-iterate>
Since both sequences share the same noise realization, the leading bias term $alpha Delta$ cancels, leaving a residual bias of order $O(alpha^(3\/2))$ or higher (Levin et al., 2025).

More generally, one can consider the multi-level extrapolation with $M$ step sizes $cal(A) = {alpha_1, dots, alpha_M}$ and coefficients ${h_m}$ determined by the Vandermonde system (Huo et al., 2024):
$ sum_(m=1)^M h_m = 1, quad sum_(m=1)^M h_m alpha_m^l = 0, quad l = 1, dots, M-1, $
which cancels successive powers in settings where such a power-series
expansion is available.

== Problem statement and goals

The high-order moment bounds for the PR-averaged RR iterate
$overline(theta)_n^((alpha, "RR"))$ established in Levin et al. (2025) show
that the leading error term scales as
$sqrt("Tr" Sigma_epsilon.alt^(("M"))) dot n^(-1\/2)$, where
$Sigma_epsilon.alt^(("M"))$ is the Markovian noise covariance defined below in
the key quantities subsection. This is the usual parametric $n^(-1\/2)$
benchmark for averaged LSA; this thesis does not prove a separate
Hájek--Le Cam or minimax lower bound.
Berry--Esseen type bounds and bootstrap inference procedures for the
_standard_ Polyak--Ruppert average $overline(theta)_n$ (without extrapolation)
under Markovian noise have been obtained in Samsonov, Sheshukova, Moulines,
and Naumov (2025).

These results do not directly give the distributional approximation needed for
the PR-averaged Richardson--Romberg statistic under Markovian noise. Two
distinctions are essential here. First, the stationary theorem is proved for an
augmented-chain scalar statistic, where the martingale noise, Poisson boundary
term, and Richardson--Romberg misadjustment are controlled separately. Second,
the deterministic-start estimator requires an additional burn-in transfer,
because the deterministic transient and startup remainders are absent from the
stationary augmented-chain theorem.

The main goal of this work is therefore to build a non-asymptotic
distributional approximation for scalar projections of the PR-averaged
Richardson--Romberg statistic. The stationary result should be read as an
$n_0 = 0$ augmented-chain theorem, not as a fixed-$alpha$ central limit theorem
centered exactly at $theta^*$. Its final deterministic-start consequence is obtained
at the balanced triangular-array scale $alpha_n = c n^(-1\/2)$, where the
residual RR terms are absorbed into explicit remainders and the covariance
target is $Sigma_infinity$.

Concretely, we establish:

+ A stationary full-window augmented-chain Berry--Esseen assembly for scalar
  RR statistics, including the martingale approximation, predictable-variance
  comparison, Poisson remainder, and stationary RR misadjustment.
+ A balanced triangular-array specialization, in particular for
  $alpha_n = c n^(-1\/2)$, which identifies $Sigma_infinity$ as the covariance
  target and gives the resulting stationary $n_0 = 0$ CLT interpretation.
+ A deterministic-start transfer theorem under mixing-scale burn-in conditions
  with logarithmic factors, yielding the corresponding balanced-scale bound for
  the main burned-in statistic.

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

== Setting and assumptions <sec:assumptions>

We now formalize the setting and state the assumptions that will be used throughout this work.

Let ${Z_k}_(k in bb(N))$ be a Markov chain on a complete separable metric space $(sans(Z), cal(Z))$ with transition kernel $sans(Q)$.

#let assumption-counter = counter("assumption")

#let assumption(name, body) = {
  assumption-counter.step()
  block(width: 100%, spacing: 0.8em)[
    *Assumption #context assumption-counter.display() (#name).* #body
  ]
}

#assumption("Uniform geometric ergodicity")[
  The kernel $sans(Q)$ admits a unique invariant distribution $pi$ and is _uniformly geometrically ergodic_: there exists $t_"mix" in bb(N)^*$ such that for all $k in bb(N)^*$,
  $ Delta(sans(Q)^k) := sup_(z, z' in sans(Z)) frac(1, 2) ||sans(Q)^k (z, dot) - sans(Q)^k (z', dot)||_"TV" <= (1\/4)^(floor(k \/ t_"mix")). $
  Equivalently, there exist constants $zeta > 0$ and $rho in (0, 1)$ such that $sup_z ||sans(Q)^k (z, dot) - pi||_"TV" <= zeta rho^k$ for all $k >= 1$.
]

#assumption("Hurwitz condition and boundedness")[
  The matrix $-overline(A)$ is Hurwitz; equivalently, all eigenvalues of $overline(A)$ have strictly positive real parts. Moreover,
  $ C_A := max( sup_(z in sans(Z)) ||A(z)|| , sup_(z in sans(Z)) ||tilde(A)(z)|| ) < infinity, $
  where $tilde(A)(z) := A(z) - overline(A)$.
]

#assumption("Noise regularity")[
  The noise function $epsilon.alt(z) = tilde(A)(z) theta^* - tilde(b)(z)$, where $tilde(b)(z) = b(z) - overline(b)$, satisfies
  $ ||epsilon.alt||_infinity := sup_(z in sans(Z)) ||epsilon.alt(z)|| < +infinity. $
]

By construction, $pi(tilde(A)) = 0$, $pi(tilde(b)) = 0$, and hence
$pi(epsilon.alt) = 0$.

Under Assumptions 1--3, the error $theta_k^((alpha)) - theta^*$ satisfies the recursion
$ theta_k^((alpha)) - theta^* = (I - alpha A(Z_k))(theta_(k-1)^((alpha)) - theta^*) - alpha epsilon.alt(Z_k). $ <eq:error-recursion>

== Key quantities

The _Markovian noise covariance matrix_ captures both the marginal variance and the temporal correlations of the noise:
$ Sigma_epsilon.alt^(("M")) = bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_0)^top] + sum_(ell=1)^infinity lr((bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_ell)^top] + bb(E)_pi [epsilon.alt(Z_ell) epsilon.alt(Z_0)^top])). $ <eq:noise-cov>
The series is absolutely convergent under Assumptions 1 and 3. Indeed, for
centered bounded $epsilon.alt$,
$||sans(Q)^ell epsilon.alt||_infinity <=
2 ||epsilon.alt||_infinity (1 slash 4)^(floor(ell slash t_"mix"))$, and hence
$
||bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_ell)^top]||
  <= ||epsilon.alt||_infinity ||sans(Q)^ell epsilon.alt||_infinity
  <= 2 ||epsilon.alt||_infinity^2 (1 slash 4)^(floor(ell slash t_"mix")).
$
The same bound applies to the transposed covariance term, so the covariance
series converges absolutely in operator norm.
This matrix is the limiting covariance in the Markov chain CLT for the partial sums $n^(-1\/2) sum_(t=0)^(n-1) epsilon.alt(Z_t)$ (cf. Douc et al., 2018, Theorem 21.2.10).

The _asymptotically optimal covariance matrix_ is given by
$ Sigma_infinity = overline(A)^(-1) Sigma_epsilon.alt^(("M")) (overline(A)^(-1))^top. $ <eq:asymp-cov>
This is the covariance target attained by the averaged linearized recursion. We call it optimal in the usual averaged-SA sense; a full Hájek--Le Cam optimality statement would require an additional local-asymptotic experiment argument, which is not part of this thesis.

The _Lyapunov equation_ plays a central role in the contraction analysis. For any $P = P^top succ 0$, there exists a unique $Q = Q^top succ 0$ satisfying $overline(A)^top Q + Q overline(A) = P$. Defining $a = lambda_"min" (P) \/ (2 ||Q||)$ and $kappa_Q = lambda_"max" (Q) \/ lambda_"min" (Q)$, the key contraction property holds: for all $alpha in [0, alpha_infinity]$,
$ ||I - alpha overline(A)||_Q^2 <= 1 - alpha a. $ <eq:contraction>

Since the iterates $theta_k^((alpha))$ alone are generally not Markovian (due to the Markovian noise), we consider the _joint process_ $(theta_k^((alpha)), Z_(k+1))$ with kernel
$ overline(sans(P))_alpha f(theta, z) = integral_(sans(Z)) sans(Q)(z, d z') f(F_z (theta), z'), $
where $F_z (theta) = (I - alpha A(z)) theta + alpha b(z)$. Thus the current
second coordinate $z$ is the observation used to update $theta$, and the next
coordinate $z'$ is carried forward to the following step. Under Assumptions
1--3, this joint chain admits a unique invariant distribution $Pi_alpha$ for
sufficiently small $alpha > 0$ (Levin et al., 2025).

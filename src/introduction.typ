#import "defs.typ": *

// === 1. Introduction ===

Stochastic approximation (SA) algorithms are a cornerstone of modern computational statistics, optimization, and reinforcement learning.
Introduced by Robbins and Monro (1951), these iterative procedures provide a principled way to find roots of equations or optimize objectives when only noisy observations are available. A particularly important subclass is the _linear stochastic approximation_ (LSA) algorithm, which arises naturally in temporal-difference (TD) learning, policy evaluation, and stochastic gradient descent for linear models.

In this work, we study the LSA recursion with a _constant step size_ $alpha > 0$:
$ theta_k^((alpha)) = theta_(k-1)^((alpha)) - alpha {A(Z_k) theta_(k-1)^((alpha)) - b(Z_k)}, quad k >= 1, $ <eq:lsa>
where ${Z_k}_(k in bb(N))$ is a time-homogeneous Markov chain on a measurable space $(sans(Z), cal(Z))$ with transition kernel $sans(Q)$ and unique invariant distribution $pi$.
The mappings $A : sans(Z) -> bb(R)^(d times d)$ and $b : sans(Z) -> bb(R)^d$ are measurable functions satisfying $overline(A) := integral_(sans(Z)) A(z) d pi(z)$ and $overline(b) := integral_(sans(Z)) b(z) d pi(z)$.
Assuming that $-overline(A)$ is a _Hurwitz matrix_ (all eigenvalues have strictly negative real parts), the target parameter $theta^* = overline(A)^(-1) overline(b)$ is uniquely defined.

The _Polyak--Ruppert averaging_ procedure (Polyak, 1990; Ruppert, 1988) provides an effective variance reduction technique.
Given a burn-in period $n_0 >= 0$, the averaged iterate is defined as
$ overline(theta)_n^((alpha)) = frac(1, n - n_0) sum_(k=n_0)^(n-1) theta_k^((alpha)). $ <eq:pr-average>

== Bias of constant step-size iterates

The use of a constant step size $alpha > 0$ offers several practical advantages: it enables geometrically fast forgetting of the initial condition (Dieuleveut, Durmus, and Bach, 2020) and simplifies hyperparameter tuning compared to diminishing step-size schedules.
However, unlike the classical regime $alpha_k -> 0$ with $sum alpha_k = infinity$ and $sum alpha_k^2 < infinity$, a constant step size produces iterates that converge only _in distribution_ to a stationary measure $Pi_alpha$, rather than almost surely to $theta^*$.
The stationary expectation $bb(E)[theta_infinity^((alpha))]$ is generally _biased_ with respect to $theta^*$, and this bias cannot be eliminated by Polyak--Ruppert averaging alone.

As shown in Levin, Naumov, and Samsonov (2025) and Huo, Chen, and Xie (2023), the asymptotic bias admits a power-series expansion in $alpha$:
$ lim_(n -> infinity) bb(E)[theta_n^((alpha))] = theta^* + alpha Delta + O(alpha^2), $ <eq:bias-expansion>
where $Delta = overline(A)^(-1) sum_(k=1)^infinity bb(E)[{sans(Q)^k tilde(A)(Z_infinity)} epsilon.alt(Z_infinity)]$ depends on the correlation structure of the Markov chain, and $tilde(A)(z) = A(z) - overline(A)$ is the centered matrix-valued function.

== Richardson--Romberg extrapolation

To eliminate the leading $O(alpha)$ bias term, we employ the _Richardson--Romberg_ (RR) _extrapolation_ procedure.
Two LSA sequences are run _on the same Markov chain trajectory_ ${Z_k}$ with step sizes $alpha$ and $2 alpha$, and the RR iterate is formed as
$ overline(theta)_n^((alpha, "RR")) = 2 overline(theta)_n^((alpha)) - overline(theta)_n^((2 alpha)). $ <eq:rr-iterate>
Since both sequences share the same noise realization, the leading bias term $alpha Delta$ cancels, leaving a residual bias of order $O(alpha^(3\/2))$ or higher (Levin et al., 2025).

More generally, one can consider the multi-level extrapolation with $M$ step sizes $cal(A) = {alpha_1, dots, alpha_M}$ and coefficients ${h_m}$ determined by the Vandermonde system (Huo et al., 2023):
$ sum_(m=1)^M h_m = 1, quad sum_(m=1)^M h_m alpha_m^l = 0, quad l = 1, dots, M-1, $
which reduces the bias from $O(alpha)$ to $O(alpha^M)$.

== Problem statement and goals

The high-order moment bounds for the RR iterate $overline(theta)_n^((alpha, "RR"))$ established in Levin et al. (2025) show that the leading error term scales as $sqrt("Tr" Sigma_epsilon.alt^(("M"))) dot n^(-1\/2)$, matching the minimax optimal rate.
Berry--Esseen type bounds and bootstrap inference procedures for the _standard_ Polyak--Ruppert average $overline(theta)_n$ (without extrapolation) under Markovian noise have been obtained in Samsonov, Sheshukova, Moulines, and Naumov (2025).

However, the _distributional approximation_ --- specifically, the central limit theorem and Berry--Esseen bounds --- for the _averaged Richardson--Romberg iterates_ $overline(theta)_n^((alpha, "RR"))$ remains an open problem.
The main goal of this work is to fill this gap by establishing:

+ A _central limit theorem_ for $sqrt(n)(overline(theta)_n^((alpha, "RR")) - theta^*)$, identifying the limiting covariance matrix.
+ _Non-asymptotic Berry--Esseen bounds_ for the rate of convergence to the Gaussian limit.
+ A practical _inference procedure_ (confidence intervals) based on the RR-extrapolated iterates.

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
  The matrix $-overline(A)$ is Hurwitz, i.e., all eigenvalues of $overline(A)$ have strictly positive real parts. Moreover,
  $ C_A := sup_(z in sans(Z)) ||A(z)|| or sup_(z in sans(Z)) ||tilde(A)(z)|| < infinity, $
  where $tilde(A)(z) = A(z) - overline(A)$.
]

#assumption("Noise regularity")[
  The noise function $epsilon.alt(z) = tilde(A)(z) theta^* - tilde(b)(z)$, where $tilde(b)(z) = b(z) - overline(b)$, satisfies
  $ ||epsilon.alt||_infinity := sup_(z in sans(Z)) ||epsilon.alt(z)|| < +infinity. $
]

Under Assumptions 1--3, the error $theta_k^((alpha)) - theta^*$ satisfies the recursion
$ theta_k^((alpha)) - theta^* = (I - alpha A(Z_k))(theta_(k-1)^((alpha)) - theta^*) - alpha epsilon.alt(Z_k). $ <eq:error-recursion>

== Key quantities

The _Markovian noise covariance matrix_ captures both the marginal variance and the temporal correlations of the noise:
$ Sigma_epsilon.alt^(("M")) = bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_0)^top] + 2 sum_(ell=1)^infinity bb(E)_pi [epsilon.alt(Z_0) epsilon.alt(Z_ell)^top]. $ <eq:noise-cov>
This matrix is the limiting covariance in the Markov chain CLT for the partial sums $n^(-1\/2) sum_(t=0)^(n-1) epsilon.alt(Z_t)$ (cf. Douc et al., 2018, Theorem 21.2.10).

The _asymptotically optimal covariance matrix_ is given by
$ Sigma_infinity = overline(A)^(-1) Sigma_epsilon.alt^(("M")) (overline(A)^(-1))^top. $ <eq:asymp-cov>
This covariance is optimal in the Hájek--Le Cam sense and represents the best achievable asymptotic variance for estimating $theta^*$ via averaged LSA.

The _Lyapunov equation_ plays a central role in the contraction analysis. For any $P = P^top succ 0$, there exists a unique $Q = Q^top succ 0$ satisfying $overline(A)^top Q + Q overline(A) = P$. Defining $a = lambda_"min" (P) \/ (2 ||Q||)$ and $kappa_Q = lambda_"max" (Q) \/ lambda_"min" (Q)$, the key contraction property holds: for all $alpha in [0, alpha_infinity]$,
$ ||I - alpha overline(A)||_Q^2 <= 1 - alpha a. $ <eq:contraction>

Since the iterates $theta_k^((alpha))$ alone are generally not Markovian (due to the Markovian noise), we consider the _joint process_ $(theta_k^((alpha)), Z_(k+1))$ with kernel
$ overline(sans(P))_alpha f(theta, z) = integral_(sans(Z)) sans(Q)(z, d z') f(F_(z') (theta), z'), $
where $F_z (theta) = (I - alpha A(z)) theta + alpha b(z)$. Under Assumptions 1--3, this joint chain admits a unique invariant distribution $Pi_alpha$ for sufficiently small $alpha > 0$ (Levin et al., 2025).

#import "../defs.typ": *

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

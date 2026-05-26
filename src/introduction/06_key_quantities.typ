#import "../defs.typ": *

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

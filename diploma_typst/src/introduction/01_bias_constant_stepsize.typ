#import "../defs.typ": *

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

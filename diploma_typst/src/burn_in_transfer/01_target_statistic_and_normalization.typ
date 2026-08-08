#import "../defs.typ": *

== Target Statistic and Normalization

Let $xi = cal(L)(Z_0)$ be the initial law of the base Markov chain, let
$0 <= n_0 < n$, and set $m := n - n_0$. The burned-in PR average is
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
on the same Markov trajectory.
// This coupling is part of the RR statistic; if the two levels use different
// paths, the deterministic-weight decomposition below is a different object.

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
// The stationary augmented-chain statistic from the previous chapter is used
// only as a comparison object after the startup transfer.

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
$Xi_(n,n_0)^("asy,RR")(u)$.
// A final corollary converts the $sqrt(m)$ statistic to the final $sqrt(n)$
// statistic when the burn-in is at the mixing scale, of order $alpha^(-1)$
// times logarithmic factors.

#import "../defs.typ": *

== Burned-in Deterministic Weights

With the $sqrt(m)$ normalization, the depth-zero linearized term has the
weights
$
Q_(l; n_0, n)^((alpha))
  := alpha sum_(k = max(n_0, l))^(n - 1) B_alpha^(k - l),
quad
1 <= l <= n - 1,
$ <eq:burn-Q-alpha>
and $Q_(l; n_0, n)^((alpha)) = 0$ for $l >= n$. The RR weight is
$
Q_(l; n_0, n)^("RR")
  := 2 Q_(l; n_0, n)^((alpha))
     - Q_(l; n_0, n)^((2 alpha)).
$ <eq:burn-Q-RR>
The leading burned-in sum is
$
W_(n,n_0)^("RR")
  := -frac(1, sqrt(m)) sum_(l = 1)^(n - 1)
      Q_(l; n_0, n)^("RR") epsilon.alt(Z_l).
$ <eq:burn-W-RR>

// For $l >= n_0$ these weights are full-window weights with horizon $n-l$. For
// $l < n_0$, the lower summation limit is $n_0$ rather than $l$, so the weight
// comparison, Poisson decomposition, and variance proxy must be restated for
// $Q_(l; n_0, n)^("RR")$.

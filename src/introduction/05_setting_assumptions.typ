#import "../defs.typ": *

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

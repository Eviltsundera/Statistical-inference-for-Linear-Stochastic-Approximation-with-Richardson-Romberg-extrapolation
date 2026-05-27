#import "../defs.typ": *

== Stationary Small-Step Convention <sec:stationary-small-step>

We collect the stationary small-step restrictions used in this chapter. The
individual ceilings are stated in @sec:external-inputs.
// The cited papers sometimes use the plus-form SA convention; throughout this
// thesis we use
// $theta_(k+1) = theta_k - w(A(Z_(k+1)) theta_k - b(Z_(k+1)))$, so their
// stability assumptions are read after this sign conversion.

For stationary depth-two arguments define
$
alpha_*(q,t_"mix")
  := min (
    alpha_("L,P2")(q,t_"mix"),
    alpha_("L,C6")(q,t_"mix"),
    alpha_("L,P5")(q,t_"mix"),
    alpha_("L,P8")(q,t_"mix"),
    alpha_("L,P9")(q,t_"mix"),
    alpha_("L,inv")(q,t_"mix")
  ).
$ <eq:levin-stationary-threshold>
// The six ceilings correspond to @lem:levin-prop-2, @lem:levin-cor-6,
// @lem:levin-prop-5-component, @lem:levin-prop-8, @lem:levin-prop-9, and
// @lem:levin-invariant-depth-two-law.

For the shifted-to-unshifted first-order transfer we also use the local inverse
ceiling
$
alpha_("inv") := frac(1, 2 || overline(A) ||).
$ <eq:alpha-inv>
If $w <= alpha_("inv")$, the Neumann series yields
$|| (I - w overline(A))^(-1) || <= 2$.

Define the stationary admissibility threshold
$
alpha_("stat")(q)
  := min (
    alpha_infinity,
    alpha_("inv"),
    alpha_*(q, t_"mix")
  ).
$ <eq:alpha-stationary-threshold>
Throughout this chapter, the small-step condition is
$2 alpha <= alpha_("stat")(q)$.
// The first condition makes both RR step sizes admissible for the Lyapunov,
// inverse, and Levin stationary depth-two inputs.

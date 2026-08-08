#import "../defs.typ": *


This chapter transfers the stationary $n_0 = 0$ Berry--Esseen bound to the
deterministic-start Richardson--Romberg average after burn-in.
// The previous chapter proves the stationary bound for
// $S_(n, "stat")^("RR")(u)$. Burn-in changes the deterministic PR kernels, so
// the result is not obtained by simply inserting $n_0 > 0$ into the stationary
// theorem.

// Throughout this chapter, constants with a `"burn"` subscript may depend on fixed
// problem parameters and the external/local startup constants of
// @sec:external-inputs.
// Such constants may absorb fixed powers of $a^(-1)$. They never hide
// dependence on $n$, $n_0$, $m$, $alpha$, $p$, or $q$ unless these variables
// appear as arguments of the constant or named quantity. Powers of $t_"mix"$,
// $p$, $q$, $d$, and rate-relevant powers of $a^(-1)$ are displayed explicitly
// in the surrounding bound.
// Unless explicitly stated otherwise, all burn-in estimates are uniform over the
// initial law $xi = cal(L)(Z_0)$ of the base Markov chain. Constants may depend on
// the deterministic initial point $theta_0$ only when that dependence is displayed,
// but they do not depend on $xi$.

For deterministic starts we strengthen the stationary condition by adding the
startup/product-stability restrictions:
$
alpha_("st")(p)
  := min (
    alpha_("prod")(p),
    alpha_("prod")(2p),
    frac(1, 2 a)
  ),
quad
alpha_("burn")(p,q)
  := min (
    alpha_("stat")(q),
    alpha_("st")(p)
  ).
$ <eq:alpha-admissibility-thresholds>
Throughout this chapter, the burn-in small-step condition is
$2 alpha <= alpha_("burn")(p,q)$.
// The local random-time product and full-state startup estimates do not introduce
// additional named step-size ceilings: @lem:burn-random-time-product follows from
// @lem:burn-product-stability and $w a <= 1$, while @lem:burn-full-startup uses
// the Levin depth-two ceilings collected in $alpha_("stat")(q)$ together with
// the product-stability ceiling above.

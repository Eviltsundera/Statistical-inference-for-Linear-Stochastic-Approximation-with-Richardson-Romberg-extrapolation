#import "../defs.typ": *


The previous chapter proves a stationary $n_0 = 0$ Berry--Esseen bound for
$S_(n, "stat")^("RR")(u)$. This chapter transfers that bound to the
deterministic-start Richardson--Romberg average after burn-in. Burn-in changes
the deterministic PR kernels, so the result is not obtained by simply inserting
$n_0 > 0$ into the stationary theorem.

Throughout this chapter, constants with a `"burn"` subscript may depend on
fixed problem parameters and the external/local startup constants of
@sec:imported-inputs, and may absorb
fixed powers of $a^(-1)$. They never hide dependence on $n$, $n_0$, $m$,
$alpha$, $p$, or $q$ unless these variables appear as arguments of the
constant or named quantity. Powers of $t_"mix"$, $p$, $q$, $d$, and any
rate-relevant powers of $a^(-1)$ are displayed explicitly in the surrounding
bound.
Unless explicitly stated otherwise, all burn-in estimates are uniform over the
initial law $xi = cal(L)(Z_0)$ of the base Markov chain. Constants may depend on
the deterministic initial point $theta_0$ only when that dependence is displayed,
but they do not depend on $xi$.

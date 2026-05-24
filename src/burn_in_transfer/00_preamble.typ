#import "../defs.typ": *


The previous chapter proves a stationary $n_0 = 0$ Berry--Esseen bound for
$S_(n, "stat")^("RR")(u)$. This chapter transfers that bound to the
deterministic-start Richardson--Romberg average after burn-in. Burn-in changes
the deterministic PR kernels, so the result is not obtained by simply inserting
$n_0 > 0$ into the stationary theorem.

Throughout this chapter, constants with a `"burn"` subscript are independent
of $n$, $n_0$, $m$, $alpha$, $p$, and $q$ unless these variables appear as
arguments. Dependencies on fixed problem constants and imported Levin/startup
constants are absorbed into named constants; powers of $a$, $t_"mix"$, $p$,
$q$, and $d$ are kept explicit when they affect the final rate.


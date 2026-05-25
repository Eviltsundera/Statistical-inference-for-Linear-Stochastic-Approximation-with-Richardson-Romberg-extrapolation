#import "../defs.typ": *

== Admissibility Thresholds <sec:admissibility-thresholds>

The exact external working forms are collected in @sec:imported-inputs. This
section records only the admissibility ceilings used
by the stationary and burned-in theorems. The cited papers sometimes use the
plus-form SA convention; throughout this thesis we use
$theta_(k+1) = theta_k - w(A(Z_(k+1)) theta_k - b(Z_(k+1)))$, so their
stability assumptions are read after this sign conversion.

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
Here the six ceilings are the positive restrictions needed for
@lem:levin-prop-2, @lem:levin-cor-6, @lem:levin-prop-5-component,
@lem:levin-prop-8, @lem:levin-prop-9, and
@lem:levin-invariant-depth-two-law. Thus Input C is
exactly the minimum of the Levin stationary depth-two ceilings whose working
forms are stated in @sec:external-direct-inputs.

For deterministic-start burn-in arguments define
$
alpha_("st")(p)
  := min (
    alpha_("prod")(2p),
    alpha_("rand-prod")(2p),
    alpha_("full-start")(2p)
  ).
$ <eq:startup-local-threshold>
The first ceiling belongs to the deterministic product-stability working form
@lem:burn-product-stability. The second and third belong to the local
extensions @lem:burn-random-time-product and @lem:burn-full-startup. These two
random-coupling and full-state startup statements are proved in the thesis and
are not direct citations from Levin et al. Thus Input D is exactly the local
threshold @eq:startup-local-threshold plus the local statements named above,
not an additional external theorem. This separation is recorded in
@sec:external-local-extensions.

For the shifted-to-unshifted first-order transfer we also use the local inverse
ceiling
$
alpha_("inv") := frac(1, 2 || overline(A) ||).
$ <eq:alpha-inv>
If $w <= alpha_("inv")$, the Neumann series yields
$|| (I - w overline(A))^(-1) || <= 2$.

We collect the small-step ceilings used in the stationary and deterministic
burn-in results into two admissibility thresholds:
$
alpha_("stat")(q)
  := min (
    alpha_infinity,
    alpha_("inv"),
    alpha_*(q, t_"mix")
  ),
quad
alpha_("burn")(p,q)
  := min (
    alpha_("stat")(q),
    frac(1, 2 a),
    alpha_("st")(p)
  ).
$ <eq:alpha-admissibility-thresholds>
Thus $2 alpha <= alpha_("stat")(q)$ makes both RR step sizes admissible for
the Lyapunov, inverse, and Levin stationary depth-two inputs. The stronger
condition $2 alpha <= alpha_("burn")(p,q)$ additionally gives
$alpha a <= 1 slash 4$ and the startup/product-stability ceilings used in the
burn-in transfer. In the full-state startup proof the $H^((2))$ comparison
uses Levin moment bounds at order $2p$; the burned-in theorems therefore
choose $q$ so that $p <= q slash 4$.

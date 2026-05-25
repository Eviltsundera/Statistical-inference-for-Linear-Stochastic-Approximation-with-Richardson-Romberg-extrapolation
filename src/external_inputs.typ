#import "defs.typ": *

== Direct Imported Working Forms <sec:external-direct-inputs>

This appendix records the external statements in the exact working forms used
in the thesis. Items explicitly marked as direct citations are imported from
the cited papers after the sign convention conversion described in
@sec:admissibility-thresholds. Items marked as local extensions are proved in
the main text and should not be read as separate external theorems.

#lemma[
  *(Direct citation: Levin et al. (2025, Lemma 11) -- Markov concentration.)*
  Under *UGE 1*, there is a universal constant $C_("MC")$ such that, for any
  bounded measurable functions $g_i$ with $pi(g_i) = 0$ and
  $||g_i||_infinity <= c_i$, every initial distribution $xi$, and every
  $p >= 2$,
  $
  || sum_(i=1)^N g_i(Z_i) ||_(L_p(xi))
    <= C_("MC") sqrt(p thin t_"mix" thin sum_(i=1)^N c_i^2).
  $ <eq:imported-markov-conc>
  This is the scalar time-inhomogeneous consequence used in the
  predictable-variation estimates and in the preliminary last-iterate bounds.
] <lem:imported-markov-concentration>

#lemma[
  *(Direct citation: Samsonov et al. (2025, Lemma 21) -- Bolthausen--Fan
  martingale Berry--Esseen.)*
  For scalar martingale differences $X_l$ with $|X_l| <= kappa$, partial sum
  $S_N = sum_l X_l$, predictable variation
  $V_N^2 = sum_l bb(E)[X_l^2 | cal(F)_(l - 1)]$, and deterministic scale
  $s_N^2 > 0$, for every $p >= 1$,
  $
  d_K(S_N slash s_N, cal(N)(0,1))
    &<= L_B(kappa) frac((2 N + 1) log(2 N + 1), s_N^3) \
    &quad + C_1 sqrt(p) s_N^(- 2 p slash (2 p + 1))
         (bb(E)|V_N^2 - s_N^2|^p)^(1 slash (2 p + 1)) \
    &quad + C_2 p s_N^(- 2 p slash (2 p + 1))
         kappa^(2 p slash (2 p + 1)).
  $ <eq:imported-bolthausen-fan>
  Here $L_B(kappa) < infinity$ and $C_1, C_2$ are the constants in that
  martingale theorem.
] <lem:imported-bolthausen-fan>

#lemma[
  *(Direct citation: Levin et al. (2025, Proposition 2) -- stationary bias of
  $J^((1))$.)*
  Under stationarity for the augmented chain
  $(Z_(t + 1), J_t^((0, w)), J_t^((1, w)))$ and for
  $w <= alpha_("L,P2")(q,t_"mix")$,
  $
  bb(E)_pi [J_infinity^((1, w))] = w thin Delta + R(w),
  quad
  || R(w) ||
    <= 12 thin || overline(A)^(-1) || thin C_A^2 thin t_"mix"^2 thin w^2
       thin || epsilon.alt ||_infinity,
  $
  with
  $
  Delta := overline(A)^(-1) sum_(k >= 1)
    bb(E)_pi [(sans(Q)^k tilde(A))(Z_0) thin epsilon.alt(Z_0)].
  $
] <lem:levin-prop-2>

#lemma[
  *(Direct citation: Levin et al. (2025, Corollary 6) -- centered bilinear
  $L_p$ bound.)*
  Define
  $
  overline(psi)_w (j, z)
    := tilde(A)(z) j
      - bb(E)_(Pi_(J^((0)), w)) [tilde(A)(Z_1) thin J_0^((0, w))].
  $
  For $w <= alpha_("L,C6")(q,t_"mix")$, every initial distribution, every
  $r >= 1$, and every $p >= 2$,
  $
  lr(|| sum_(t = 0)^(r - 1) overline(psi)_w (J_t^((0, w)), Z_(t + 1)) ||)_(L_p)
    <= c_(W, 1) thin p^(3 slash 2) sqrt(w r)
       + c_(W, 2) thin p^3 thin w^(-1 slash 2)
         thin log^(1 slash 2)(1 slash (w a)),
  $
  with $c_(W, 1), c_(W, 2)$ depending only on
  $C_A, kappa_Q, t_"mix", || epsilon.alt ||_infinity$. The logarithmic factor
  is the one displayed in Levin et al. (2025, Corollary 6, Eq. (67)); it is
  absorbed into $"polylog"(n)$ in the working-scale corollaries.
] <lem:levin-cor-6>

#lemma[
  *(Direct citation: Levin et al. (2025, Proposition 8) -- depth-two
  $J^((2))$ moment bound.)*
  For every $q >= 2$ and every $p$ satisfying $2 <= p <= q slash 2$, if
  $w <= alpha_("L,P8")(q,t_"mix")$, then
  $
  || J_n^((2, w)) ||_(L_p)
    <= D_J thin t_"mix"^(5 slash 2) thin p^(7 slash 2) thin w^(3 slash 2)
       thin log^(3 slash 2)(1 slash (w a)),
  $
  uniformly in $n$, with $D_J$ depending only on
  $C_A, kappa_Q, || overline(A)^(-1) ||, || epsilon.alt ||_infinity$.
] <lem:levin-prop-8>

#lemma[
  *(Direct citation: Levin et al. (2025, Proposition 9) -- one-trajectory
  $H^((2))$ moment bound.)*
  For every $q >= 2$ and every $p$ satisfying $2 <= p <= q slash 2$, if
  $w <= alpha_("L,P9")(q,t_"mix")$, then
  $
  || H_n^((2, w)) ||_(L_p)
    <= D_H thin d^(1 slash q) thin t_"mix"^(5 slash 2) thin p^(7 slash 2)
       thin w^(3 slash 2) thin log^(3 slash 2)(1 slash (w a)),
  $
  uniformly in $n$, with $D_H$ depending only on
  $C_A, kappa_Q, || overline(A)^(-1) ||, || epsilon.alt ||_infinity$.
  The proof uses the representation
  $
  H_k^((2,w))
    = - w sum_(l = 1)^k Gamma_(l + 1:k)^((w))
          tilde(A)(Z_l) J_(l - 1)^((2,w)).
  $ <eq:levin-H2-representation>
] <lem:levin-prop-9>

#lemma[
  *(Direct citation/extraction: Levin et al. (2025, Appendix B.2,
  Proposition 5) -- depth-two startup contraction for the $J$ coordinates.)*
  For
  $
  Y_k^((w)) := (Z_(k + 1), J_k^((0,w)), J_k^((1,w)), J_k^((2,w)))
  $
  and $y = (z, j_0, j_1, j_2)$,
  $y' = (z', j'_0, j'_1, j'_2)$, define the depth-two augmented-chain cost
  from Levin et al. (2025, Appendix B.2, Eq. (49)):
  $
  c_(J,2)^((w))(y,y')
    &:= ||j_0 - j'_0|| + ||j_1 - j'_1|| + ||j_2 - j'_2|| \
    &quad + lr((||j_0|| + ||j'_0|| + ||j_1|| + ||j'_1||
          + ||j_2|| + ||j'_2|| + sqrt(w a) ||epsilon.alt||_infinity))
          thin 1_(z != z').
  $ <eq:levin-depth-two-cost>
  Under the Proposition-5 step-size restriction
  $w <= alpha_("L,P5")(p,t_"mix")$, two copies of $Y_k^((w))$ started from
  deterministic states $y,y'$ can be coupled so that, for
  $ell = 0,1,2$,
  $
  || J_k^((ell,w))(y) - J_k^((ell,w))(y') ||_(L_p)
    <= C_W thin p^(7 slash 2) thin t_"mix"^(5 slash 2)
       thin log^(3 slash 2)(1 slash (w a))
       thin exp(-w a k slash (12 p))
       thin c_(J,2)^((w))(y,y').
  $ <eq:levin-depth-two-component-contraction>
  The displayed componentwise estimate is the coordinate projection of the
  Wasserstein contraction, with constants as in Levin et al. (2025, Eq. (55)).
] <lem:levin-prop-5-component>

#lemma[
  *(Direct citation: Levin et al. (2025, Corollary 4) -- invariant depth-two
  $J$ law.)*
  Under the same stationary depth-two ceilings as above, the Markov chain
  $Y_k^((w))$ admits a unique invariant law $Pi_(J,2,w)$. Its stationary
  version has finite moments at the orders used in
  @lem:levin-prop-8, and the contraction in
  @lem:levin-prop-5-component gives convergence to this law from every
  deterministic initial augmented state.
] <lem:levin-invariant-depth-two-law>

== Local Extensions Used in This Thesis <sec:external-local-extensions>

The following statements are local extensions or assemblies, not direct
citations from Levin et al. or Samsonov et al.:

- @lem:finite-past-full-augmented-state constructs the stationary full
  depth-two augmented state including $H^((2))$ by a finite-past limit. It uses
  @lem:levin-invariant-depth-two-law, @lem:levin-prop-5-component, and
  @lem:levin-prop-9.
- @lem:burn-random-time-product proves conditional product stability at a
  random coupling time from the deterministic product-stability working form
  @lem:burn-product-stability and the coupling-time tail.
- @lem:burn-full-startup proves the full-state startup contraction for
  $J^((1)) + J^((2)) + H^((2))$ by combining the $J$ contraction
  @lem:levin-prop-5-component with the local random-time product estimate.

Whenever the main proof invokes these three statements, the invoked result is
the local lemma named above, not an unquoted external proposition.

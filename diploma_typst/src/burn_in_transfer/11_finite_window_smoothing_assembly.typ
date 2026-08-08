#import "../defs.typ": *

== Finite-Window Smoothing Assembly

The finite-start scalar statistic decomposes as
$
T_(n,n_0)^("RR")(u)
  = -frac(u^top M_(n,n_0)^("bRR"), sqrt(m))
    + cal(R)_(n,n_0, op("fin"))^("bRR")(u),
$ <eq:burn-full-decomp>
where the composite non-Gaussian remainder is
$
cal(R)_(n,n_0, op("fin"))^("bRR")(u)
  := D_(op("tr"), n, n_0)^("RR")(u)
     + cal(I)_(n,n_0)^("init,RR")(u)
     + u^top D_(2,n,n_0)^("bRR")
     + u^top R_(n,n_0, op("fin"))^("mis,RR").
$ <eq:burn-composite-remainder>
// The terms are controlled by @lem:burn-deterministic-transient,
// @lem:burn-random-initial-product, @lem:burn-poisson-decomp, and
// @thm:burn-misadjustment, respectively.

// For reference, the finite-window assembly uses the following components:

// #table(
//   columns: (1.45fr, 2.4fr, 1.8fr),
//   inset: 4pt,
//   [*Component*], [*Input bound*], [*Role in smoothing*],
//   [$D_(op("tr"), n, n_0)^("RR")(u)$],
//   [@eq:burn-RR-transient-bound],
//   [Deterministic transient from $theta_0 - theta^*$.],
//   [$cal(I)_(n,n_0)^("init,RR")(u)$],
//   [@eq:burn-random-init-bound],
//   [Random initial-product discrepancy.],
//   [$u^top D_(2,n,n_0)^("bRR")$],
//   [@eq:burn-D2-bound],
//   [Poisson Abel boundary remainder.],
//   [$u^top R_(n,n_0, op("fin"))^("mis,RR")$],
//   [@eq:burn-mis-bound],
//   [Depth-two RR misadjustment plus startup transfer.],
//   [$u^top M_(n,n_0)^("bRR")$],
//   [@thm:burn-M-BE and @cor:burn-bracket-asymp],
//   [Leading martingale Berry--Esseen term.],
// )

Let $cal(B)_("mis")(m,n_0,p,q,alpha)$ denote the right-hand side of
@eq:burn-mis-bound.

#lemma[
  *(Burned-in composite remainder bound.)*
  Assume the hypotheses of @lem:burn-deterministic-transient,
  @lem:burn-random-initial-product, @lem:burn-poisson-decomp, and
  @thm:burn-misadjustment. Then, for every $p >= 2$ and every $q >= 2$
  satisfying $p <= q slash 4$ and $2 alpha <= alpha_("burn")(p,q)$,
  $
  || cal(R)_(n,n_0, op("fin"))^("bRR")(u) ||_(L_p)
    <= || u || thin lr((
      cal(B)_("start")(m,n_0,p,alpha)
      + frac(C_("burn,D2"), sqrt(m))
      + cal(B)_("mis")(m,n_0,p,q,alpha)
    )),
  $ <eq:burn-R-bound>
  where
  $
  cal(B)_("start")(m,n_0,p,alpha)
    &:= frac(|| theta_0 - theta^* ||, alpha a sqrt(m)) thin lr([
      5 sqrt(kappa_Q) thin (1 - alpha a)^(n_0 slash 2)
  + C_("init,RR") thin p thin exp(-c_("init") alpha a n_0 slash p)
    ]), \
  C_("burn,D2")
    &:= 3 thin t_"mix" thin || epsilon.alt ||_infinity
        thin lr((C_("burn,Q") + frac(C_("burn,V"), a^2))).
  $
] <lem:burn-R-bound>

_Proof._ Apply the triangle inequality in @eq:burn-composite-remainder. The
deterministic part is @eq:burn-RR-transient-bound. The random initial-product
term is @eq:burn-random-init-bound. The Poisson boundary term is bounded by
$
|| u^top D_(2,n,n_0)^("bRR") ||_(L_p)
  <= ||u|| thin ||D_(2,n,n_0)^("bRR")||_infinity
  <= frac(C_("burn,D2") ||u||, sqrt(m))
$
using @eq:burn-D2-bound. The last term is exactly @eq:burn-mis-bound. $square$

Dividing @eq:burn-full-decomp by the finite-window normalization gives
$
Xi_(n,n_0)^("bRR")(u) = X_(n,n_0)^("bRR") + Y_(n,n_0)^("bRR"),
$ <eq:burn-XY-split>
where
$
X_(n,n_0)^("bRR")
  := -frac(u^top M_(n,n_0)^("bRR"),
           sqrt(m) thin sigma_(n,n_0)^("bRR")(u)),
quad
Y_(n,n_0)^("bRR")
  := frac(cal(R)_(n,n_0, op("fin"))^("bRR")(u),
          sigma_(n,n_0)^("bRR")(u)).
$

#theorem[
  *(Finite-window deterministic-start burned-in PR-averaged RR Berry--Esseen bound.)*
  Assume the hypotheses of @thm:burn-M-BE, @lem:burn-random-initial-product,
  and @thm:burn-misadjustment. Let
  $m := n - n_0$ and
  $p = max(2, ceil(log n))$ and
  $q = max(4 p, ceil(log(e thin d)), 2)$. Then $p >= 2$, $q >= 2$, and
  $p <= q slash 4$. If $(n,n_0,p,q,alpha,u)$ is in the admissible burn-in
  regime @eq:admissible-burn-regime, then
  $
  d_K lr((Xi_(n,n_0)^("bRR")(u), cal(N)(0, 1)))
    &<= frac(C_("bK,1")(u) thin log^(3 slash 4) n, m^(1 slash 4))
     + frac(C_("bK,2")(u) thin log n, sqrt(m))
    + frac(e thin
        || cal(R)_(n,n_0, op("fin"))^("bRR")(u) ||_(L_p),
        sqrt(2 pi) thin sigma_(n,n_0)^("bRR")(u))
     + frac(e, n),
  $ <eq:burn-RR-BE-master>
  with the composite remainder bounded by @lem:burn-R-bound. The bound is
  uniform over $xi = cal(L)(Z_0)$.
] <thm:burn-RR-BE-master>

_Proof._ Apply the smoothing inequality @eq:smoothing-Lp to the split
@eq:burn-XY-split. The martingale Berry--Esseen term is @thm:burn-M-BE; the
minus sign in $X_(n,n_0)^("bRR")$ is handled by applying that theorem to the
signed increments $-u^top Delta M_l^("bRR")$. The bounded-increment constant
and predictable quadratic variation are unchanged. Since
$p = max(2, ceil(log n))$, the smoothing tail $e^(-p)$ is at most
$e slash n$. The $L_p$ norm of the perturbation is
$
||Y_(n,n_0)^("bRR")||_(L_p)
  = frac(||cal(R)_(n,n_0, op("fin"))^("bRR")(u)||_(L_p),
         sigma_(n,n_0)^("bRR")(u)),
$
which gives @eq:burn-RR-BE-master. $square$

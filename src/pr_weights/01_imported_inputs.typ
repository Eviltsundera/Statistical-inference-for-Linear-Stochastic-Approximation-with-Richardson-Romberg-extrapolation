#import "../defs.typ": *

== Imported Inputs and Admissibility Thresholds <sec:imported-inputs>

The proof below is a self-contained assembly of RR weights, Poisson
decomposition, smoothing, and burn-in transfer. It uses the following external
non-asymptotic inputs in the displayed forms.

*Input A: Markov concentration for centered inhomogeneous sums.* Under *UGE 1*,
there is a universal constant $C_("MC")$ such that, for any bounded measurable
functions $g_i$ with $pi(g_i) = 0$ and $||g_i||_infinity <= c_i$, every initial
distribution $xi$, and every $p >= 2$,
$
|| sum_(i=1)^N g_i(Z_i) ||_(L_p(xi))
  <= C_("MC") sqrt(p thin t_"mix" thin sum_(i=1)^N c_i^2).
$ <eq:imported-markov-conc>
This is the scalar time-inhomogeneous concentration consequence of Levin et
al. (2025, Lemma 11) used in the predictable-variation estimates and in the
preliminary last-iterate bounds.

*Input B: Bolthausen--Fan martingale Berry--Esseen.* For scalar martingale
differences $X_l$ with $|X_l| <= kappa$, partial sum $S_N = sum_l X_l$,
predictable variation $V_N^2 = sum_l bb(E)[X_l^2 | cal(F)_(l - 1)]$, and
deterministic scale $s_N^2 > 0$, Samsonov et al. (2025, Lemma 21) gives, for
every $p >= 1$,
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

*Input C: Levin stationary depth-two inputs.* There exists a step-size ceiling
$alpha_*(q,t_"mix") > 0$ such that, for $q >= 2$, $2 <= p <= q slash 2$, and
$w <= alpha_*(q,t_"mix")$, the stationary bias, centered bilinear, and
depth-two moment estimates of Levin et al. (2025) hold in the working forms
stated later as @lem:levin-prop-2, @lem:levin-cor-6, and
@lem:levin-prop-89. Their constants may depend on the fixed problem
parameters displayed in those lemmas, but not on $n$, $p$, or $w$ beyond the
explicit factors. The cited papers sometimes use the plus-form SA convention;
throughout this thesis we use
$theta_(k+1) = theta_k - w(A(Z_(k+1)) theta_k - b(Z_(k+1)))$, so their
stability assumptions are read after this sign conversion.

*Input D: startup and random-product stability.* There exists a positive
threshold $alpha_("st")(p)$ such that the product-stability and full-state
startup contractions used in the burn-in chapter hold for $2 alpha <=
alpha_("st")(p)$; their working forms are @lem:burn-product-stability and
@lem:burn-full-startup. This input is not used in the stationary theorem, only
in the deterministic-start transfer.

For the shifted-to-unshifted first-order transfer we also use the local inverse
ceiling
$
alpha_("inv") := frac(1, 2 || overline(A) ||).
$ <eq:alpha-inv>
If $w <= alpha_("inv")$, the Neumann series yields
$|| (I - w overline(A))^(-1) || <= 2$.


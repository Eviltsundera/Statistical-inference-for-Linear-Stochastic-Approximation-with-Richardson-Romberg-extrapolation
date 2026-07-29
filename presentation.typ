// Defense presentation
#set page(paper: "presentation-16-9", margin: (x: 18mm, y: 12mm))
#set text(font: "New Computer Modern", size: 15pt, lang: "en")
#set par(justify: false, leading: 0.55em)

#let blue = rgb("#24476f")
#let teal = rgb("#2f7f79")
#let green = rgb("#3d7a4f")
#let red = rgb("#a94a4a")
#let gray = rgb("#5f6670")
#let pale_blue = rgb("#eef4f8")
#let pale_green = rgb("#edf7f0")
#let pale_red = rgb("#f8eeee")
#let line_gray = rgb("#d7dce2")

#let slide_title(body) = {
  text(size: 22pt, weight: "bold", fill: blue)[#body]
  v(0.22em)
  line(length: 100%, stroke: 0.8pt + line_gray)
  v(0.5em)
}

#let accent(body) = text(fill: teal, weight: "bold")[#body]
#let small(body) = text(size: 12pt, fill: gray)[#body]

// Colored bold column header (no box).
#let head(body, color: blue) = text(fill: color, weight: "bold")[#body]

// Single highlighted equation panel: filled, no border.
#let eqbox(body) = align(center, block(
  inset: (x: 14pt, y: 9pt),
  radius: 3pt,
  fill: pale_blue,
)[#body])

// Key-message panel: filled with a left accent bar, no full border.
#let takeaway(body) = block(
  width: 100%,
  inset: (x: 12pt, y: 9pt),
  radius: 3pt,
  fill: pale_green,
  stroke: (left: 3pt + green),
)[#body]

// Light inline highlight for a short result line.
#let note_box(body) = align(center, block(
  inset: (x: 11pt, y: 6pt),
  radius: 3pt,
  fill: pale_green,
)[#body])

#let footer() = {
  place(bottom + right, dx: -0.25cm, dy: 0.25cm,
    text(size: 9pt, fill: rgb("#8b929a"))[
      #context counter(page).display()
    ]
  )
}

// ============================================================
// 1. Title
// ============================================================

#v(1fr)
#align(center)[
  #text(size: 30pt, weight: "bold", fill: blue)[
    Statistical Inference for Linear \
    Stochastic Approximation with \
    Richardson--Romberg Extrapolation
  ]
  #v(1.3em)
  #text(size: 18pt)[Danil Gainanov]
  #v(0.5em)
  #text(size: 13pt, fill: gray)[Supervisor: Ilya Levin]
  #v(0.4em)
  #text(size: 12pt, fill: gray)[Master's thesis defense, 2026]
]
#v(1fr)
#footer()

// ============================================================
// 2. Motivation and plan
// ============================================================
#pagebreak()

#slide_title[Motivation: What the Recursion Solves]

Linear stochastic approximation (LSA) drives #accent[TD-learning and policy
evaluation] in reinforcement learning. From a #accent[dependent data stream]
$Z_1, Z_2, dots$ (a Markov chain), the constant-step recursion
$theta_k^((alpha)) = theta_(k-1)^((alpha)) - alpha (A(Z_k) thin theta_(k-1)^((alpha)) - b(Z_k))$
solves the mean linear system

#eqbox[
  $ overline(A) thin theta^* = overline(b),
    quad overline(A) = bb(E)_pi [A(Z)], quad overline(b) = bb(E)_pi [b(Z)],
    quad theta^* = overline(A)^(-1) overline(b). $
]

#grid(columns: (1.05fr, 1fr), column-gutter: 16pt,
  [
    #head[Canonical instance: TD(0).] \
    Features $phi(s)$, $V(s) approx phi(s)^top theta$, transition
    $Z_k = (s_k, r_k, s_(k+1))$:
    $ A(Z_k) = phi(s_k)(phi(s_k) - gamma phi(s_(k+1)))^top,
      quad b(Z_k) = r_k thin phi(s_k); $
    $theta^*$ is the TD fixed point.
  ],
  [
    #head[Why inference is hard.]
    - Constant step does not converge to $theta^*$: a steady-state
      #accent[bias of order $alpha$].
    - A biased center invalidates a CI; no variance estimate repairs a wrong
      center.
  ],
)

#takeaway[
  Goal: a valid *confidence interval* for $u^top theta^*$ --- RR corrects the
  center, long-run variance sets the width.
]

#small[
  *Plan:* (1) setup & target; (2) Richardson--Romberg & the main Berry--Esseen
  bound; (3) proof architecture; (4) confidence intervals & experiments.
]

#footer()

// ============================================================
// 3. Setup and assumptions
// ============================================================
#pagebreak()

#slide_title[Setup and Assumptions]

Centering the recursion at $theta^*$ gives the #accent[error recursion]

#eqbox[
  $ theta_k^((alpha)) - theta^*
    = (I - alpha A(Z_k))(theta_(k-1)^((alpha)) - theta^*)
      - alpha thin epsilon(Z_k),
    quad
    epsilon(z) = tilde(A)(z) theta^* - tilde(b)(z), $
]

with $tilde(A) = A - overline(A)$, $tilde(b) = b - overline(b)$, so the noise is
centered: $pi(epsilon) = 0$.

#v(0.4em)

#grid(columns: (1.1fr, 1fr), column-gutter: 16pt,
  [
    #head[Assumptions]
    - *(A1)* Uniform geometric ergodicity, mixing time $t_"mix"$.
    - *(A2)* $-overline(A)$ Hurwitz; $A$, $tilde(A)$ bounded.
    - *(A3)* Bounded noise $||epsilon||_infinity < infinity$.
  ],
  [
    #head[Polyak--Ruppert average] \
    Burn-in $n_0$, window $m = n - n_0$:
    $ overline(theta)_(n,n_0)^((alpha))
      = frac(1, m) sum_(k=n_0)^(n-1) theta_k^((alpha)). $
  ],
)

#v(0.4em)

#small[
  Markovian noise means the randomness in $A(Z_k)$, $b(Z_k)$ is serially
  dependent; the constant step $alpha$ does not vanish, so the iterates have a
  steady-state bias.
]

#footer()

// ============================================================
// 4. Statistical target and error sources
// ============================================================
#pagebreak()

#slide_title[Statistical Target and Two Error Sources]

For confidence intervals we need a normal approximation for scalar projections
$sqrt(n) thin u^top (overline(theta)_n - theta^*) approx cal(N)(0, u^top Sigma_infinity u)$.
Two distinct obstacles:

#v(0.5em)

#grid(columns: (1fr, 1fr), column-gutter: 18pt,
  [
    #head[Centering error]
    $ bb(E) thin overline(theta)_n^((alpha)) - theta^*
       = alpha Delta + "higher order". $
    A variance estimator cannot correct a biased interval center.
  ],
  [
    #head[Dependent fluctuation]
    $ Sigma_epsilon^("M")
      = bb(E)[epsilon_0 epsilon_0^top]
        + 2 sum_(l >= 1) bb(E)[epsilon_0 epsilon_l^top]. $
    The normal variance must include serial correlations.
  ],
)

#v(0.6em)

#takeaway[
  Two separate tasks: *correct the center* of the estimator, and *approximate
  the dependent-noise distribution* around that center. RR handles the first.
]

#footer()

// ============================================================
// 5. Richardson--Romberg
// ============================================================
#pagebreak()

#slide_title[Richardson--Romberg in the Step Size]

Run two PR-averaged LSA trajectories on the #accent[same Markov chain path],
with steps $alpha$ and $2 alpha$, and extrapolate:
$ overline(theta)_n^("RR") = 2 overline(theta)_n^((alpha)) - overline(theta)_n^((2 alpha)). $

#v(0.3em)

Both branches carry the same leading bias slope $Delta$ (Levin et al., 2025),
so the linear term cancels:

#eqbox[
  $ overline(theta)_n^((alpha)) approx theta^* + alpha Delta,
    quad
    overline(theta)_n^((2 alpha)) approx theta^* + 2 alpha Delta
    quad arrow.r.double quad
    overline(theta)_n^("RR")
      approx theta^* + (2 alpha - 2 alpha) Delta
      = theta^* + O(alpha^(3 slash 2)). $
]

#v(0.5em)

#takeaway[
  RR moves the *center* of the confidence interval; it does not estimate the
  covariance. The shared path keeps the two branches' noise highly correlated,
  so extrapolation cancels bias instead of adding simulation error.
]

#footer()

// ============================================================
// 6. Inference target
// ============================================================
#pagebreak()

#slide_title[What Distribution Should We Approximate?]

For a direction $u$ and window $m = n - n_0$, the practical statistic is

#eqbox[
  $ sqrt(m) thin u^top (overline(theta)_(n,n_0)^("RR") - theta^*)
    quad arrow.r.double quad
    cal(N)(0, sigma^2(u)),
    quad sigma^2(u) = u^top Sigma_infinity u. $
]

#v(0.5em)

Because the data are dependent, the target is the #accent[long-run] covariance,
not a one-step variance:

$ Sigma_infinity = overline(A)^(-1) Sigma_epsilon^("M") overline(A)^(-top),
  quad
  Sigma_epsilon^("M") = bb(E)_pi [epsilon_0 epsilon_0^top]
    + sum_(j >= 1) ( bb(E)_pi [epsilon_0 epsilon_j^top]
                     + bb(E)_pi [epsilon_j epsilon_0^top] ). $

#v(0.5em)

#takeaway[
  The inference goal is a Gaussian approximation of the scalar RR statistic with
  the Markovian long-run variance $sigma^2(u) = u^top Sigma_infinity u$.
]

#footer()

// ============================================================
// 7. Main result
// ============================================================
#pagebreak()

#slide_title[Main Theoretical Result]

Under (A1)--(A3), at the #accent[balanced scale]
$alpha = c thin n^(-1 slash 2)$ with burn-in
$n_0 asymp (alpha a)^(-1) log^2 n$ --- where $c > 0$ is the step-size constant
and $a$ the Lyapunov contraction rate --- the burned-in RR average satisfies a
Berry--Esseen bound:

#eqbox[
  $ d_K (
      sqrt(m) thin u^top (overline(theta)_(n,n_0)^("RR") - theta^*) slash sigma(u),
      thin cal(N)(0,1)
    )
    <= C(u, c, theta_0) thin "polylog"(n) thin n^(-1 slash 4),
    quad sigma^2(u) = u^top Sigma_infinity u. $
]

#v(0.5em)

#takeaway[
  The theorem turns the RR averaged estimator into a valid
  normal-approximation object for scalar confidence intervals.
]

#v(0.3em)

#small[
  The constant $C(u, c, theta_0)$ depends only on the direction $u$, the
  step-size constant $c$ in $alpha = c thin n^(-1 slash 2)$, and the start
  $theta_0$. The $n^(-1 slash 4)$ rate is the martingale Berry--Esseen rate;
  polylog factors absorb the mixing time and the burn-in.
]

#footer()

// ============================================================
// 8a. Theorem I (1/3): decomposition and weights
// ============================================================
#pagebreak()

#slide_title[Theorem I (1/3): Decomposition and RR Weights]

In the stationary augmented chain ($n_0 = 0$), unfolding the error recursion and
PR-averaging splits the scaled error into a noise sum plus deterministic
remainders:

#eqbox[
  $ sqrt(n) thin (overline(theta)_n^("RR") - theta^*) = W^("RR") + D^("RR"),
    quad
    W^("RR") = -frac(1, sqrt(n)) sum_(l=1)^(n-1) cal(Q)_l^("RR") thin epsilon(Z_l). $
]

The deterministic *RR weight* is
$cal(Q)_l^("RR") = 2 Q_l^((alpha)) - Q_l^((2 alpha))$, with
$Q_l^((alpha)) = overline(A)^(-1)(I - B_alpha^(n-l))$ and
$B_alpha = I - alpha overline(A)$ (write $k = n - l$):

#grid(columns: (1fr, 1fr), column-gutter: 16pt,
  [
    #head[Closeness to $overline(A)^(-1)$]
    $ cal(Q)_l^("RR") - overline(A)^(-1)
        = -overline(A)^(-1)(2 B_alpha^k - B_(2 alpha)^k), $
    $ sum_(l) || cal(Q)_l^("RR") - overline(A)^(-1) ||^2 <= C_1 slash (alpha a). $
  ],
  [
    #head[Bounded variation]
    $ sum_(l) || cal(Q)_(l+1)^("RR") - cal(Q)_l^("RR") || <= C_2 slash a^2. $
    Controls the Abel / Poisson boundary remainder.
  ],
)

#v(0.3em)

#takeaway[
  All stochastic content of the leading term sits in $epsilon(Z_l)$; the kernel
  is deterministic and asymptotically $overline(A)^(-1)$.
]

#footer()

// ============================================================
// 8b. Theorem I (2/3): Poisson martingale and variance
// ============================================================
#pagebreak()

#slide_title[Theorem I (2/3): Poisson Martingale and Variance]

$W^("RR")$ is a dependent sum. Solve the *Poisson equation*
$hat(epsilon) - sans(Q) hat(epsilon) = epsilon$ and split each
$epsilon(Z_l)$ into a martingale increment plus a telescoping term; Abel
summation gives

#eqbox[
  $ W^("RR") = -frac(1, sqrt(n)) M_n^("RR") + D_(2,n)^("RR"),
    quad
    Delta M_l^("RR")
      = cal(Q)_l^("RR") thin (hat(epsilon)(Z_l) - sans(Q) hat(epsilon)(Z_(l-1))), $
]

where the Abel remainder is $|| D_(2,n)^("RR") ||_(L_p) = O(t_"mix" thin n^(-1 slash 2))$.

#grid(columns: (1fr, 1fr), column-gutter: 16pt,
  [
    #head[Variance matches the target]
    $ Sigma_n^("RR") = frac(1, n) sum_(l=2)^(n-1)
        cal(Q)_l^("RR") Sigma_epsilon^("M") (cal(Q)_l^("RR"))^top, $
    $ || Sigma_n^("RR") - Sigma_infinity || <= C_3 slash (n thin alpha a). $
  ],
  [
    #head[Martingale Berry--Esseen]
    $ d_K (
        frac(u^top M_n^("RR"), sqrt(n) thin sigma_n^("RR")(u)),
        cal(N)(0,1)
      )
      <= frac(C log^(3 slash 4) n, n^(1 slash 4))
       + frac(C log n, sqrt(n)). $
  ],
)

#v(0.3em)

#takeaway[
  The dependent fluctuation reduces to a bounded-increment martingale whose
  variance is $Sigma_infinity$ up to $O(1 slash (n thin alpha a))$.
]

#footer()

// ============================================================
// 8c. Theorem I (3/3): misadjustment and assembly
// ============================================================
#pagebreak()

#slide_title[Theorem I (3/3): Misadjustment and Assembly]

The remaining non-martingale piece is the RR *misadjustment*
$ R_n^("mis,RR") = frac(1, sqrt(n)) sum_(k=0)^(n-1)
    (2 R_k^((alpha)) - R_k^((2 alpha))). $
A depth-two refinement of the perturbation recursions (Levin et al., 2025)
controls it at the same order as the martingale rate:

#eqbox[
  $ || R_n^("mis,RR") ||_(L_p) <= C thin "polylog"(n) thin n^(-1 slash 4)
    quad "at" quad alpha = c thin n^(-1 slash 2). $
]

#v(0.4em)

A Bobkov--Götze *smoothing inequality* then merges the martingale Berry--Esseen
term and this remainder into a single Kolmogorov bound,
$d_K (X_n + Y_n, dot) <= d_K (X_n, dot) + e thin ||Y_n||_(L_p) slash sqrt(2 pi) + e^(-p)$.

#v(0.4em)

#takeaway[
  Stationary balanced-scale bound:
  $d_K (S_(n,"stat")^("RR")(u) slash sigma_n^("RR")(u), cal(N)(0,1))
   <= C(u) thin "polylog"(n) thin n^(-1 slash 4)$,
  with covariance target $Sigma_infinity$.
]

#footer()

// ============================================================
// 9. Burn-in theorem
// ============================================================
#pagebreak()

#slide_title[Theorem II: Burn-in Transfer to Deterministic Starts]

The real algorithm starts from a fixed $theta_0$ and a non-stationary $Z_0$. We
transfer the stationary bound to the burned-in statistic
$ T_(n,n_0)^("RR")(u)
   = sqrt(m) thin u^top (overline(theta)_(n,n_0)^("RR") - theta^*),
  quad m = n - n_0. $

#v(0.3em)

#grid(columns: (1.05fr, 1fr), column-gutter: 16pt,
  [
    #head[Four extra sources, all controlled]
    - deterministic transient $B_alpha^k (theta_0 - theta^*)$;
    - random initial products $Gamma_(1:k)^((alpha)) - B_alpha^k$;
    - augmented-chain startup discrepancy;
    - finite-window variance normalization.
  ],
  [
    #head[Balanced burn-in result] \
    With $n_0 asymp (alpha a)^(-1) log^2 n$ and $alpha = c thin n^(-1 slash 2)$:
    #eqbox[
      $ d_K ( Xi_(n,n_0)^("RR")(u), cal(N)(0,1) )
        <= C(u,c,theta_0) thin "polylog"(n) thin n^(-1 slash 4). $
    ]
  ],
)

#v(0.4em)

#takeaway[
  This is the theorem behind the estimator used in experiments: discard the
  burn-in, average the remaining RR iterates, and use the same covariance target
  $Sigma_infinity$.
]

#footer()

// ============================================================
// 10. Practical intervals
// ============================================================
#pagebreak()

#slide_title[From Theory to Confidence Intervals]

In practice $Sigma_infinity$ is unknown, so the long-run variance of the
projected trajectory $Y_t = u^top theta_t$ is estimated from one dependent run.

#v(0.45em)

#grid(columns: (1fr, 1fr), column-gutter: 16pt,
  [
    #head[OBM] (overlapping batch means, block $b$)
    $ hat(sigma)_("OBM")^2(b)
      = frac(b, T-b+1) sum_(s=0)^(T-b)
        (overline(Y)_(s,b) - overline(Y)_T)^2. $
    Window bias $approx c_1(u) slash b$, often negative.
  ],
  [
    #head[OBM-LW / lugsail] ($lambda > 1$)
    $ hat(sigma)_("LW")^2
      = frac(lambda, lambda-1) hat(sigma)_("OBM")^2(lambda b)
        - frac(1, lambda-1) hat(sigma)_("OBM")^2(b). $
    $lambda = 2$ cancels the leading $1 slash b$ bias.
  ],
)

#v(0.6em)

#takeaway[
  *RR changes the center; OBM and lugsail change the width.* The thesis proves
  the RR distributional approximation; non-asymptotic OBM/lugsail theory along
  RR trajectories is left as future work.
]

#footer()

// ============================================================
// 11. Experiments setup
// ============================================================
#pagebreak()

#slide_title[Numerical Experiments]

#table(
  columns: (1.15fr, 2.2fr),
  inset: 7pt,
  stroke: 0.45pt + line_gray,
  fill: (x, y) => if y == 0 { pale_blue },
  [*Quantity*], [*Main setup*],
  [Problem class], [Finite-state Markovian LSA],
  [Dimension and states], [$d=5$, 10 Markov states],
  [Monte Carlo design], [100 problems, 100 trajectories per problem],
  [Trajectory length], [$T = 10^6$],
  [RR stepsizes], [$alpha in {0.2, 0.02}$ in the main comparison],
  [Inference], [95% scalar intervals in a random projection direction],
  [Variance estimators], [Batch means, OBM, OBM-LW, MSB, and oracle diagnostics],
)

#v(0.4em)

#takeaway[
  The experiments are designed to separate the point-estimator bias from the
  long-run variance estimation error.
]

#footer()

// ============================================================
// 12. Main comparison
// ============================================================
#pagebreak()

#slide_title[Main Comparison: RR Fixes the Center]

#grid(columns: (1.0fr, 1.15fr), column-gutter: 14pt,
  [
    #table(
      columns: (1.55fr, 0.65fr, 0.65fr),
      inset: 5pt,
      align: (left, center, center),
      stroke: 0.4pt + line_gray,
      fill: (x, y) => if y == 0 { pale_blue } else if y == 3 or y >= 6 { pale_green },
      [*Method*], [*L2*], [*Cov.*],
      [$alpha=0.2$ const.], [$26.67$], [$0.5%$],
      [$alpha=0.02$ const.], [$13.93$], [$40.5%$],
      [RR const.], [$4.52$], [$94.0%$],
      [Diminishing], [$6.80$], [$90.0%$],
      [PR + OBM], [$5.35$], [$92.0%$],
      [RR + OBM], [$4.52$], [$95.0%$],
      [RR + OBM-LW], [$4.52$], [$95.0%$],
    )

    #small[L2 is reported in units of $10^(-3)$; medians over 100 problems.]
  ],
  [
    #image("figures/experiments/main_methods_comparison.svg", width: 100%)
  ],
)

#v(0.2em)

#takeaway[
  The coverage gain is not achieved by widening intervals: RR mainly improves
  the interval center by reducing step-size bias.
]

#footer()

// ============================================================
// 13. Stepsize and oracle diagnostics
// ============================================================
#pagebreak()

#slide_title[Diagnostics: Stepsize and Variance Estimation]

#table(
  columns: (1.15fr, 1.75fr, 0.8fr, 0.8fr),
  inset: 6pt,
  align: (left, left, center, center),
  stroke: 0.45pt + line_gray,
  fill: (x, y) => if y == 0 { pale_blue } else { if y == 1 or y == 2 or y == 3 { pale_green } },
  [*RR pair*], [*Single-step coverage*], [*RR L2*], [*RR cov.*],
  [$(0.04,0.02)$], [$0.04$: $92.0%$; $0.02$: $94.0%$], [$2.97$], [$94.0%$],
  [$(0.10,0.05)$], [$0.10$: $86.0%$; $0.05$: $91.5%$], [$2.97$], [$94.0%$],
  [$(0.20,0.10)$], [$0.20$: $67.5%$; $0.10$: $86.0%$], [$2.97$], [$94.0%$],
)

#v(0.5em)

For the largest adjacent pair, oracle and practical variance intervals are
almost identical:

#note_box[
  RR + oracle variance: 95.0%; RR + OBM: 95.0%; RR + MSB: 95.0% coverage.
]

#v(0.4em)

#small[
  Across $T in {2 dot 10^4, dots, 10^6}$ the oracle interval stays near $95%$
  coverage, so the RR center and the normal approximation are not the
  bottleneck. The only short-horizon gap is variance estimation: at
  $T = 2 dot 10^4$ the data-driven interval is about $5.8%$ too narrow, and this
  gap closes as $T$ grows.
]

#footer()

// ============================================================
// 14. OBM and lugsail
// ============================================================
#pagebreak()

#slide_title[Variance Estimation: When Does Lugsail Help?]

#grid(columns: (1.0fr, 1.2fr), column-gutter: 14pt,
  [
    The remaining short-horizon gap is in *variance estimation*, not the center.

    #v(0.4em)

    *OBM can be too narrow* when the Bartlett-window bias is negative.

    #v(0.4em)

    *Lugsail helps at small blocks / short horizons:* at $T = 2 dot 10^4$,
    $eta = 0.5$ it lifts median coverage $91.5% -> 95.0%$.

    #v(0.4em)

    *But not automatic:* neutral at the production rule $eta = 0.6$, and the
    signed estimate can turn negative for very large blocks.
  ],
  [
    #image("figures/experiments/blocksize_lugsail_diagnostics.svg", width: 100%)
  ],
)

#v(0.2em)

#takeaway[
  Step-size RR and lugsail OBM solve different problems: center bias versus
  variance-estimator window bias.
]

#footer()

// ============================================================
// 15. Contributions
// ============================================================
#pagebreak()

#slide_title[Contributions and Limitations]

#grid(columns: (1fr, 1fr), column-gutter: 16pt,
  [
    #head[Main contributions]
    - Formulated RR-averaged constant-step LSA inference under Markovian noise.
    - Proved a scalar Berry--Esseen bound at the balanced scale.
    - Transferred the result from stationary starts to burned-in deterministic starts.
    - Verified the bias-correction mechanism in finite-state experiments.
  ],
  [
    #head[Limitations and future work]
    - Full multivariate confidence regions.
    - Non-asymptotic OBM and lugsail theory along RR trajectories.
    - Sharper dependence on mixing time and stability constants.
    - More diagnostics for slow mixing and matrix covariance estimates.
  ],
)

#v(0.5em)

#takeaway[
  The central message: RR makes constant-step LSA statistically usable by
  correcting the center, while standard long-run variance tools handle the
  interval width.
]

#footer()

// ============================================================
// 16. Thank you
// ============================================================
#pagebreak()

#v(2fr)
#align(center)[
  #text(size: 34pt, weight: "bold", fill: blue)[Thank you!]
  #v(0.9em)
  #text(size: 20pt, fill: gray)[Questions?]
]
#v(2fr)
#footer()

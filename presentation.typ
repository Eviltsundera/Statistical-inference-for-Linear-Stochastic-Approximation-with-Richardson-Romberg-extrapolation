// Presentation: Statistical Inference for LSA with RR Extrapolation
#set page(paper: "presentation-16-9", margin: (x: 24mm, y: 18mm))
#set text(font: "New Computer Modern", size: 16pt, lang: "en")
#set par(justify: true, leading: 0.5em)

#let slide-title(body) = {
  text(size: 22pt, weight: "bold", fill: eastern)[#body]
  v(0.3em)
  line(length: 100%, stroke: 0.8pt + eastern)
  v(0.3em)
}

#let accent(body) = text(fill: eastern, weight: "bold")[#body]

#let footer() = {
  place(bottom + right, dy: 0.5cm, dx: -0.5cm,
    text(size: 10pt, fill: rgb("#888"),
      context counter(page).display()
    )
  )
}

// ============================================================
// SLIDE 1: Title
// ============================================================

#v(2fr)
#align(center)[
  #text(size: 26pt, weight: "bold")[
    Statistical Inference for Linear \
    Stochastic Approximation with \
    Richardson--Romberg Extrapolation
  ]
  #v(1.5em)
  #text(size: 18pt)[Danil Gainanov]
  #v(0.8em)
  #text(size: 14pt, fill: rgb("#555"))[
    Supervisor: Ilya Levin
  ]
  #v(0.5em)
  #text(size: 13pt, fill: rgb("#888"))[2025]
]
#v(2fr)
#footer()

// ============================================================
// SLIDE 2: Problem setup
// ============================================================
#pagebreak()

#slide-title[Linear Stochastic Approximation (LSA)]

Recursion with a #accent[constant step size] $alpha > 0$:
$ theta_k^((alpha)) = theta_(k-1)^((alpha)) - alpha [A(Z_k) theta_(k-1)^((alpha)) - b(Z_k)], quad k >= 1, $
where ${Z_k}$ is a time-homogeneous Markov chain with invariant distribution $pi$.

*Goal:* estimate $theta^* = overline(A)^(-1) overline(b)$, $quad overline(A) = integral A(z) d pi(z)$.

*Polyak--Ruppert averaging:* $quad overline(theta)_n^((alpha)) = frac(1, n - n_0) sum_(k=n_0)^(n-1) theta_k^((alpha))$

*Problem:* constant step size introduces #accent[bias]:
$ lim_(n -> infinity) bb(E)[theta_n^((alpha))] = theta^* + alpha Delta + O(alpha^2) $

#footer()

// ============================================================
// SLIDE 3: Richardson--Romberg extrapolation
// ============================================================
#pagebreak()

#slide-title[Richardson--Romberg Extrapolation]

*Idea:* run two LSA sequences _on the same trajectory_ ${Z_k}$ with step sizes $alpha$ and $2alpha$:
$ overline(theta)_n^((alpha, "RR")) = 2 overline(theta)_n^((alpha)) - overline(theta)_n^((2 alpha)) $

The leading bias term $alpha Delta$ #accent[cancels] $arrow.r.double$ residual bias $O(alpha^2)$.

*General case* ($M$ step sizes, Vandermonde weights):
$ sum_(m=1)^M h_m = 1, quad sum_(m=1)^M h_m alpha_m^l = 0, quad l = 1, dots, M-1 quad arrow.r.double quad "bias" O(alpha^M). $

#rect(inset: 10pt, width: 100%, stroke: 1pt + eastern, radius: 4pt)[
  *Known:* $L_p$ error bounds for RR iterates (Levin et al., 2025). \
  *Open question:* CLT and Berry--Esseen for $overline(theta)_n^((alpha, "RR"))$.
]

#footer()

// ============================================================
// SLIDE 4: Goals
// ============================================================
#pagebreak()

#slide-title[Goals of This Work]

#set text(size: 15pt)

#rect(inset: 10pt, width: 100%, stroke: 1.2pt + eastern, radius: 4pt)[
  Establish for the averaged RR iterates $overline(theta)_n^((alpha, "RR"))$:

  + *CLT:* $quad sqrt(n)(overline(theta)_n^((alpha, "RR")) - theta^*) arrow.r^(d) cal(N)(0, Sigma_infinity)$
    with $Sigma_infinity = overline(A)^(-1) Sigma_epsilon.alt^(("M")) (overline(A)^(-1))^top$.

  2. *Berry--Esseen bound:* for all convex $A in cal(A)$,
    $ sup_(A in cal(A)) |bb(P)(sqrt(n) Sigma_infinity^(-1\/2)(overline(theta)_n^("RR") - theta^*) in A) - bb(P)(cal(N)(0, I_d) in A)| lt.tilde Delta_n (alpha, d, t_"mix") $
    with rate $Delta_n -> 0$ as $n -> infinity$, $alpha -> 0$.

  3. *Practical inference procedures* (OBM, MSB bootstrap confidence intervals).
]

#set text(size: 16pt)

*Key references:*
- Huo, Chen, Xie (2023) --- constant step + batch-mean CI
- Samsonov, Sheshukova, Moulines, Naumov (2025) --- CLT + bootstrap for PR average
- Levin, Naumov, Samsonov (2025) --- $L_p$ bounds for RR

#footer()

// ============================================================
// SLIDE 5: Assumptions
// ============================================================
#pagebreak()

#slide-title[Assumptions]

#set text(size: 15pt)

*A1 (Uniform geometric ergodicity).* $exists t_"mix" in bb(N)^*$:
$ Delta(sans(Q)^k) := sup_(z, z') frac(1, 2) ||sans(Q)^k (z, dot) - sans(Q)^k (z', dot)||_"TV" <= (1\/4)^(floor(k \/ t_"mix")) $

*A2 (Hurwitz condition and boundedness).* $-overline(A)$ is Hurwitz, $C_A = sup_z ||A(z)|| < infinity$.

*A3 (Noise regularity).* The noise $epsilon.alt(z) = tilde(A)(z) theta^* - tilde(b)(z)$ is bounded:
$||epsilon.alt||_infinity = sup_z ||epsilon.alt(z)|| < infinity$.

#set text(size: 16pt)

*Key quantities:*
- Markovian noise covariance: $Sigma_epsilon.alt^(("M")) = bb(E)_pi [epsilon.alt_0 epsilon.alt_0^top] + 2 sum_(ell=1)^infinity bb(E)_pi [epsilon.alt_0 epsilon.alt_ell^top]$
- Optimal covariance: $Sigma_infinity = overline(A)^(-1) Sigma_epsilon.alt^(("M")) (overline(A)^(-1))^top$
- Contraction: $||I - alpha overline(A)||_Q^2 <= 1 - alpha a$ for $alpha <= alpha_infinity$

#footer()

// ============================================================
// SLIDE 6: Error decomposition
// ============================================================
#pagebreak()

#slide-title[Error Decomposition of the RR Iterate]

#set text(size: 15pt)

We analyze the #accent[last iterate] $theta_n^(("RR"))$ (not the average) --- the analysis is simpler and captures the key structure.

$ theta_n^(("RR")) - theta^* = underbrace([2 Gamma_(1:n)^((alpha)) - Gamma_(1:n)^((2alpha))](theta_0 - theta^*), -> 0) + underbrace(tilde(J)_n^((0, alpha)), O(sqrt(alpha))) + underbrace(2 J_n^((1, alpha)) - J_n^((1, 2alpha)), "2nd order") + dots $

*Leading term:* $quad J_n^((0,alpha)) = -alpha sum_(j=1)^n (I - alpha overline(A))^(n-j) epsilon.alt(Z_j)$

#rect(inset: 8pt, width: 100%, stroke: 1pt + rgb("#cc4444"), radius: 4pt)[
  *Key observation:* RR cancels the bias $bb(E)[theta_n^((alpha))] - theta^* = O(alpha)$, but the #accent[fluctuations] $theta_n^((alpha)) - bb(E)[theta_n^((alpha))]$ are $O(1)$ regardless of RR.
]

#set text(size: 16pt)

*Task:* show $tilde(J)_n^((0,alpha))$ and $J_n^((1,alpha))$ are #accent[negligible] after averaging.

#footer()

// ============================================================
// SLIDE 7: Bound on J~_n^(0)
// ============================================================
#pagebreak()

#slide-title[Result 1: Bound on $tilde(J)_n^((0,alpha))$]

#set text(size: 15pt)

*Why isolate $tilde(J)_n^((0,alpha))$?* It is a #accent[linear functional] of the Markov chain $sum c_j epsilon.alt(Z_j)$, hence it will be the #accent[leading term] in the Berry--Esseen expansion (CLT rate is determined by this term).

Representation: $tilde(J)_n^((0,alpha)) = -2alpha^2 overline(A) sum_(j=1)^n H_j^((n)) epsilon.alt(Z_j)$, where
$ H_j^((n)) = sum_(i=1)^(n-j) (I - alpha overline(A))^(i-1) (I - 2alpha overline(A))^(n-j-i) $

*Norm bound:* $quad ||H_j^((n))|| <= overline(C)_A (1-alpha a)^((n-j-1)/2) dot frac(2, alpha a)$

*Sub-Gaussian concentration* (Markov chain inequality):
$ bb(P)(||tilde(J)_n^((0,alpha))|| >= t) <= 2 exp(-t^2 \/ (2 u_n^2)), quad u_n^2 lt.tilde hat(C)_A^2 alpha $

#set text(size: 16pt)

#rect(inset: 10pt, width: 100%, stroke: 1pt + eastern, radius: 4pt)[
  *Result:* $quad bb(E)^(1\/2) ||tilde(J)_n^((0,alpha))||^2 = O(sqrt(alpha))$ --- negligible upon $1\/sqrt(n)$ averaging.
]

#footer()

// ============================================================
// SLIDE 8: Bound on J^(1)
// ============================================================
#pagebreak()

#slide-title[Result 2: Bound on $J_n^((1,alpha))$]

#set text(size: 14pt)

$J_n^((1,alpha))$ contains products $tilde(A)(Z_j) dot J^((0))$. Define
$ S_n = sum_(t=1)^(n-1) (I - alpha overline(A))^(n-t) tilde(A)(Z_(t+1)) J_t^((0,alpha)) $
Decomposition: $quad S_n - bb(E) S_n = -alpha(U_1^((w)) + U_2^((w)) + U_3^((w)))$

#grid(columns: (1fr, 1fr), column-gutter: 16pt,
  [
    *Martingale part $U_2^((w))$:*
    $ ||U_2^((w))||_(L_p) lt.tilde frac(C p^(3\/2) t_"mix"^(1\/2) ||epsilon.alt||_infinity, alpha a) $
    _(Burkholder inequality)_
  ],
  [
    *Residual $U_3^((w))$:*
    $ ||U_3^((w))||_(L_p) lt.tilde frac(C p^(1\/2) t_"mix" ||epsilon.alt||_infinity, sqrt(alpha a)) $
    _(Markov chain concentration)_
  ]
)

#v(0.3em)

*Combined:*
$ (bb(E) ||S_n - bb(E) S_n||^p)^(1\/p) lt.tilde alpha (p^(3\/2) t_"mix"^(1\/2) frac(1, alpha a) + p^(1\/2) t_"mix" frac(1, sqrt(alpha a))) $

#footer()

// ============================================================
// SLIDE 9: OBM and MSB procedures
// ============================================================
#pagebreak()

#slide-title[Variance Estimation: OBM and MSB]

#set text(size: 14.5pt)

Given iterate projections $theta_1, dots, theta_T$ and overall mean $overline(theta)_T$, block size $b_n$:

*Overlapping Batch Mean (OBM):*

Block averages: $Y_t = frac(1, b_n) sum_(k=t)^(t+b_n-1) theta_k$, $quad t = 1, dots, T - b_n + 1$

$ hat(sigma)_"OBM"^2 = frac(b_n, T - b_n + 1) sum_(t=1)^(T-b_n+1) (Y_t - overline(theta)_T)^2 $

*Multiplier Subsample Bootstrap (MSB):*

Bootstrap statistic: $S_b = frac(sqrt(b_n), sqrt(T - b_n + 1)) sum_(t=1)^(T-b_n+1) w_(b,t) (Y_t - overline(theta)_T)$, $quad w_(b,t) tilde.op cal(N)(0,1)$

CI from empirical quantiles of ${S_b}_(b=1)^B$: $quad overline(theta)_T plus.minus hat(q)_(1-alpha\/2) \/ sqrt(T)$

#set text(size: 16pt)

Both methods are #accent[nonparametric]: no knowledge of $Sigma_epsilon.alt^(("M"))$ or mixing time required.

#footer()

// ============================================================
// SLIDE 10: Experiments — design
// ============================================================
#pagebreak()

#slide-title[Numerical Experiments: Design]

Comparison of #accent[9 strategies] for confidence interval construction:

#set text(size: 14pt)

#table(
  columns: (0.35fr, 1fr),
  inset: 7pt,
  stroke: 0.5pt,
  fill: (x, y) => if y == 0 { rgb("#dde4f0") },
  [*Source*], [*Methods*],
  [Huo et al.], [
    1--2. Constant step $alpha in {0.2, 0.02}$ + batch-mean CI \
    3. RR ($alpha + 2alpha$) + batch-mean CI \
    4--5. Diminishing step $alpha_0\/sqrt(k)$ + CLTZ20 batch-mean CI
  ],
  [Samsonov et al.], [
    6--7. Diminishing step $c_0\/(t+k_0)^gamma$ + PR averaging + OBM / MSB bootstrap CI
  ],
  [#accent[New]], [
    #accent[8--9. RR (constant step) + OBM / MSB bootstrap CI]
  ],
)

#set text(size: 16pt)

#footer()

// ============================================================
// SLIDE 11: Experiment parameters
// ============================================================
#pagebreak()

#slide-title[Experiment Parameters]

*Problem setup:* $quad d = 5$, $quad |cal(X)| = 10$ states, $quad T = 10^4$

#v(0.5em)

#table(
  columns: (1fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt,
  fill: (x, y) => if y == 0 { rgb("#dde4f0") },
  [*Huo et al. (batch-mean CI)*], [*Samsonov et al. (OBM / MSB CI)*],
  [
    Batch count: $K = ceil(T^(0.3)) = 20$ \
    Burn-in: $n_0 = min(1000, T\/10) = 1000$ \
    RR step sizes: $alpha_1 = 0.2$, $alpha_2 = 0.02$
  ],
  [
    PR step size: $alpha_t = c_0 \/ (t + k_0)^gamma$ \
    $c_0 = 1.0$, $k_0 = 10$, $gamma = 2\/3$ \
    $arrow.r.double alpha_0 = 1\/10^(2\/3) approx 0.22$
  ],
)

#v(0.3em)

#table(
  columns: (1fr,),
  inset: 8pt,
  stroke: 0.5pt,
  fill: (x, y) => if y == 0 { rgb("#dde4f0") },
  [*OBM / MSB parameters (shared by methods 6--9)*],
  [
    Block size: $b_n = ceil(T^(0.6)) = 251$ #h(2em)
    Bootstrap replications (MSB): $B = 500$ #h(2em)
    CI level: $95%$
  ],
)

#footer()

// ============================================================
// SLIDE 12: Results
// ============================================================
#pagebreak()

#slide-title[Experimental Results]

#set text(size: 13pt)

#table(
  columns: (1.4fr, 0.7fr, 0.9fr, 0.6fr, 0.5fr),
  inset: 6pt,
  align: (left, center, center, center, center),
  stroke: 0.5pt,
  fill: (x, y) => if y == 0 { rgb("#dde4f0") } else if y >= 8 { rgb("#e0f5e0") },
  [*Method*], [*L2 $times 10^(-3)$*], [*CI width $times 10^(-3)$*], [*Cov %*], [*Div*],
  [$alpha = 0.2$ (const)],            [8.75],  [0.90],  [0],   [500],
  [$alpha = 0.02$ (const)],           [1.01],  [0.82],  [72],  [98],
  [RR (batch-mean)],                  [0.50],  [0.79],  [93],  [500],
  [$0.2\/sqrt(k)$ (dim)],            [0.69],  [1.13],  [94],  [0],
  [$0.02\/sqrt(k)$ (dim)],           [1.16],  [1.35],  [84],  [0],
  [PR + OBM],                         [0.48],  [0.61],  [86],  [0],
  [PR + MSB],                         [0.48],  [0.60],  [86],  [0],
  [#accent[RR + OBM]],               [#accent[0.50]],[#accent[0.81]],[#accent[94]],[#accent[576]],
  [#accent[RR + MSB]],               [#accent[0.50]],[#accent[0.79]],[#accent[94]],[#accent[576]],
)

_Median over problems. Div = diverged trajectories (out of total)._

#set text(size: 16pt)

RR + OBM/MSB: #accent[best coverage (94%) + lowest L2 among constant-step methods]

#footer()

// ============================================================
// SLIDE 11: Discussion
// ============================================================
#pagebreak()

#slide-title[Discussion]

+ *RR eliminates bias:* L2 error $0.50 times 10^(-3)$ --- #accent[2$times$ better] than the best single constant step ($alpha = 0.02$: $1.01 times 10^(-3)$), despite using aggressive $alpha = 0.2$.

#v(0.2em)

2. *RR + OBM/MSB* achieve #accent[best coverage (94%)] among all 9 methods, matching the best diminishing-step method ($0.2\/sqrt(k)$: 94%) but with lower L2 error.

#v(0.2em)

3. *PR + OBM/MSB* have the lowest absolute L2 ($0.48 times 10^(-3)$) and narrowest CIs, but coverage 86% --- below nominal 95%.

#v(0.2em)

4. *Constant step $alpha = 0.2$ alone* diverges (500 div): 0% coverage. RR rescues it by cancelling bias.

#v(0.2em)

5. *Divergence* in RR methods (576) comes from the $alpha = 0.2$ component --- mitigated by NaN-filtering, but motivates studying RR stability.

#footer()

// ============================================================
// SLIDE 12: Status and plan
// ============================================================
#pagebreak()

#slide-title[Current Status and Plan]

*Done:*
- Problem formulation, assumptions
- Bounds on residual terms $tilde(J)_n^((0,alpha))$ and $J_n^((1,alpha))$ in the RR error decomposition
- Martingale + residual decomposition with $L_p$ bounds
- Numerical experiments: 9 methods, confirming RR advantages

#v(0.5em)

*Remaining:*
- Complete CLT proof for $sqrt(n)(overline(theta)_n^((alpha, "RR")) - theta^*)$
- Derive explicit Berry--Esseen bounds
- Justify consistency of OBM/MSB variance estimators for RR iterates
- RR for OBM by block size $b_n$: since $bb(E)[hat(sigma)_"OBM"^2] = sigma^2(u) + c_1\/b_n + o(1\/b_n)$, one can apply RR with two block sizes to cancel the $1\/b_n$ bias
- Extended experiments: dependence on $T$, $d$, $t_"mix"$

#footer()

// ============================================================
// SLIDE 13: Thank you
// ============================================================
#pagebreak()

#v(3fr)
#align(center)[
  #text(size: 32pt, weight: "bold")[Thank you!]
  #v(1em)
  #text(size: 20pt, fill: rgb("#555"))[Questions?]
]
#v(3fr)
#footer()

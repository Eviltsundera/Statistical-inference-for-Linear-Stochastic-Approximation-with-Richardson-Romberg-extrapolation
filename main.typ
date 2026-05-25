// Document settings
#set document(
  title: "Statistical Inference for Linear Stochastic Approximation with Richardson-Romberg Extrapolation",
)
#set page(paper: "us-letter", margin: (x: 1in, y: 1in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)", supplement: [Eq.])
#set par(justify: true, first-line-indent: 1em)

#import "src/defs.typ": *

// Title block
#align(center)[
  #v(0.5in)
  #text(size: 16pt, weight: "bold")[
    Statistical Inference for Linear Stochastic Approximation \
    with Richardson-Romberg Extrapolation
  ]
  #v(1.5em)
  #text(size: 12pt)[--- ---]
  #v(2em)
]

#align(center)[
  *Abstract* \
  This thesis studies statistical inference for constant-stepsize linear
  stochastic approximation with Markovian noise. The main object is the
  Polyak--Ruppert averaged Richardson--Romberg estimator formed from two
  coupled step sizes, $alpha$ and $2 alpha$. We prove a stationary
  augmented-chain Berry--Esseen assembly for scalar RR statistics, identify
  the asymptotic covariance target $Sigma_infinity$, and then transfer the
  result to deterministic starts under mixing-scale burn-in conditions with
  logarithmic factors. At the balanced scale $alpha = c n^(-1\/2)$, the final
  burned-in statistic has a non-asymptotic normal approximation with
  $n^(-1\/4)$ polynomial rate up to logarithmic factors.
]

#v(2em)

= Introduction <sec:introduction>

#include "src/introduction.typ"

#pagebreak()

= Zeroth-Order Richardson--Romberg Difference <sec:zeroth_order_rr>

#include "src/zeroth_order_rr.typ"

#pagebreak()

= Last Iterate Analysis <sec:last_iterate>

#include "src/last_iterate.typ"

#pagebreak()

= Richardson--Romberg PR Weight Bounds <sec:pr_weights>

#include "src/pr_weights.typ"

#pagebreak()

= Burn-in Transfer for Deterministic Starts <sec:burn_in_transfer>

#include "src/burn_in_transfer.typ"

#pagebreak()

= Appendix: External Inputs and Local Extensions <sec:imported-inputs>

#include "src/external_inputs.typ"

#pagebreak()

= References <sec:references>

- Bobkov, S. G. and Goetze, F. (1999). Exponential integrability and
  transportation cost related to logarithmic Sobolev inequalities. _Journal of
  Functional Analysis_, 163(1), 1--28.
- Bolthausen, E. (1982). Exact convergence rates in some martingale central
  limit theorems. _Annals of Probability_, 10(3), 672--688.
- Dieuleveut, A., Durmus, A., and Bach, F. (2020). Bridging the gap between
  constant step-size stochastic gradient descent and Markov chains. _Annals of
  Statistics_, 48(3), 1348--1382.
- Douc, R., Moulines, E., Priouret, P., and Soulier, P. (2018). _Markov
  Chains_. Springer.
- Fan, X. (2019). Exact rates of convergence in some martingale central limit
  theorems. _Journal of Mathematical Analysis and Applications_, 469(2),
  1028--1044. https://doi.org/10.1016/j.jmaa.2018.09.049.
- Huo, D., Chen, Y., and Xie, Q. (2024). Effectiveness of constant stepsize in
  Markovian LSA and statistical inference. _Proceedings of the AAAI Conference
  on Artificial Intelligence_, 38(18), 20447--20455.
  https://doi.org/10.1609/aaai.v38i18.30028.
- Levin, I., Naumov, A., and Samsonov, S. (2025). High-order error bounds for
  Markovian LSA with Richardson--Romberg extrapolation. Extended version,
  arXiv:2508.05570. Conference version in _Proceedings of the AAAI Conference
  on Artificial Intelligence_, 40(43), 36696--36704, 2026.
  https://doi.org/10.1609/aaai.v40i43.40994.
- Polyak, B. T. (1990). New stochastic approximation type procedures.
  _Automation and Remote Control_, 51(7), 937--946.
- Robbins, H. and Monro, S. (1951). A stochastic approximation method. _Annals
  of Mathematical Statistics_, 22(3), 400--407.
- Ruppert, D. (1988). Efficient estimations from a slowly convergent
  Robbins--Monro process. Technical Report 781, Cornell University Operations
  Research and Industrial Engineering.
- Samsonov, S., Sheshukova, M., Moulines, E., and Naumov, A. (2025).
  Statistical inference for linear stochastic approximation with Markovian
  noise. _Advances in Neural Information Processing Systems_, 38.
  arXiv:2505.19102.

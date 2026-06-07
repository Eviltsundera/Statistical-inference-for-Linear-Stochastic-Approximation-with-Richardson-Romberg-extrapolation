// Document settings
#set document(
  title: "Statistical Inference for Linear Stochastic Approximation with Richardson-Romberg Extrapolation",
)
#let chapter-numbering(..nums) = context numbering("1.1", counter(heading).get().first(), ..nums)
#let chapter-equation-numbering(..nums) = context numbering("(1.1)", counter(heading).get().first(), ..nums)

#set page(
  paper: "a4",
  margin: (left: 2cm, right: 2cm, top: 2cm, bottom: 2cm),
  footer: context {
    if counter(page).get().first() > 2 {
      align(center, counter(page).display("1"))
    }
  },
)
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set heading(numbering: "1.1")
#set math.equation(numbering: chapter-equation-numbering, supplement: [Eq.])
#set figure(numbering: chapter-numbering)
#set figure.caption(separator: [. ])
#set par(justify: true, first-line-indent: 1cm, leading: 0.35em)

#import "src/defs.typ": *
#import "src/abstracts.typ": english-abstract

#show figure.where(kind: image): set figure(supplement: [Figure])
#show figure.where(kind: image): set block(width: 100%)
#show figure.where(kind: image): set align(center)
#show figure.caption.where(kind: image): set align(center)

#show figure.where(kind: table): set figure(supplement: [Table])
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: table): set block(width: 100%)
#show figure.where(kind: table): set align(center)
#show figure.caption.where(kind: table): set align(right)

#show ref.where(form: "normal"): it => context {
  let el = it.element
  if el == none {
    return it
  }

  let loc = el.location()
  let chapter = counter(heading).at(loc).first()
  if el.func() == math.equation {
    let object = counter(math.equation).at(loc).first()
    link(loc, [Eq. #numbering("(1.1)", chapter, object)])
  } else if el.func() == figure {
    let kind = el.kind
    let object = counter(figure.where(kind: kind)).at(loc).first()
    let supplement = if el.supplement == none { [] } else { el.supplement + [ ] }
    link(loc, [#supplement#numbering("1.1", chapter, object)])
  } else {
    it
  }
}

#show heading.where(level: 1): it => {
  counter(math.equation).update(0)
  counter(figure.where(kind: image)).update(0)
  counter(figure.where(kind: table)).update(0)
  counter(figure.where(kind: "theorem")).update(0)
  counter(figure.where(kind: "lemma")).update(0)
  counter(figure.where(kind: "remark")).update(0)
  counter(figure.where(kind: "corollary")).update(0)
  counter(footnote).update(0)
  it
}

// Title block
#let title-page-line(body) = block(width: 100%)[
  #align(center)[
    #text(weight: "bold")[#body]
  ]
]

#let signature-block(role, name) = align(right)[
  #block(width: 4.6cm)[
    #align(center)[#role]
    #v(1em)
    #align(center)[#name]
  ]
]

#[
  #set par(justify: false, first-line-indent: 0pt, leading: 0.25em)
  #set text(size: 12pt)

  #align(center)[
    #text(weight: "bold")[
      FEDERAL STATE AUTONOMOUS EDUCATIONAL INSTITUTION FOR \
      HIGHER PROFESSIONAL EDUCATION NATIONAL RESEARCH \
      UNIVERSITY \
      «HIGHER SCHOOL OF ECONOMICS»
    ]

    #v(0.5em)

    #text(style: "italic", weight: "bold")[Faculty of Computer Science]
  ]

  #v(2em)

  #title-page-line[Gainanov Danil]

  #v(0.8em)

  #title-page-line[Статистический вывод для линейной стохастической аппроксимации с экстраполяцией Ричардсона-Ромберга]

  #v(0.8em)

  #title-page-line[Statistical Inference for Linear Stochastic Approximation with Richardson-Romberg Extrapolation]

  #v(1em)

  #align(center)[
    Qualification paper -- Master of Science Dissertation \
    Field of study 01.04.02 «Applied Mathematics and Informatics» \
    Program: Modern computer sciences
  ]

  #v(3.2em)

  #signature-block[Supervisor][Levin Ilya]

  #v(1.8em)

  #signature-block[Student][Gainanov Danil]

  #v(1fr)

  #align(center)[Moscow, 2026]
]

#pagebreak()

#align(center)[#text(size: 14pt, weight: "bold")[Abstract]]

#english-abstract

#pagebreak()

#outline(title: [Contents])

#pagebreak()

= Introduction <sec:introduction>

#include "src/introduction.typ"

== Chapter Summary and Results

The chapter fixes the statistical problem, the Markovian LSA model, the
constant-stepsize convention, and the Richardson--Romberg estimator used in
the rest of the thesis. It also states the main assumptions and explains why
the target is a non-asymptotic distributional approximation for scalar
projections of the PR-averaged RR statistic.

#pagebreak()

= Zeroth-Order Richardson--Romberg Difference <sec:zeroth_order_rr>

#include "src/zeroth_order_rr.typ"

== Chapter Summary and Results

The chapter isolates the deterministic-product zeroth-order RR cancellation
mechanism. The main output is a terminal-iterate decomposition, a geometric
kernel bound for $H_j^((n))$, and a scalar $L^p$ estimate showing the extra
$sqrt(alpha)$-scale decay of the leading RR difference.

#pagebreak()

= Last Iterate Analysis <sec:last_iterate>

#include "src/last_iterate.typ"

== Chapter Summary and Results

The chapter proves a centered bound for the shifted first-order perturbation
and records how it applies to PR-averaged RR misadjustment terms. It also
identifies the limitation of a single-$alpha$ estimate, motivating the
separate RR weight analysis developed in the following chapter.

#pagebreak()

= Richardson--Romberg PR Weight Bounds <sec:pr_weights>

#include "src/pr_weights.typ"

== Chapter Summary and Results

The chapter derives the closed forms, pointwise estimates, energy bounds, and
total-variation controls for the PR-averaged RR weights. These bounds feed
into the Poisson martingale approximation, predictable-variation comparison,
depth-two misadjustment control, and the stationary Berry--Esseen assembly.

#pagebreak()

= Burn-in Transfer for Deterministic Starts <sec:burn_in_transfer>

#include "src/burn_in_transfer.typ"

== Chapter Summary and Results

The chapter transfers the stationary augmented-chain result to deterministic
initial conditions. It controls the burned-in deterministic weights,
transient terms, random initial products, startup discrepancies, and the
finite-window normalization, culminating in the balanced burn-in
Berry--Esseen bound.

#pagebreak()

= Numerical Experiments <sec:experiments>

#include "src/experiments.typ"

== Chapter Summary and Results

The experiments support the theoretical interpretation: step-size
Richardson--Romberg extrapolation primarily improves the interval center by
reducing bias, while OBM and lugsail OBM affect the estimated long-run
variance. The diagnostics also show that the benefit of lugsail correction is
block-size and dependence-regime dependent.

#pagebreak()

= Conclusion <sec:conclusion>

#include "src/conclusion.typ"

#pagebreak()

= References <sec:references>

+ Bobkov, S. G. and Goetze, F. (1999). Exponential integrability and
  transportation cost related to logarithmic Sobolev inequalities. _Journal of
  Functional Analysis_, 163(1), 1--28.
+ Dieuleveut, A., Durmus, A., and Bach, F. (2020). Bridging the gap between
  constant step-size stochastic gradient descent and Markov chains. _Annals of
  Statistics_, 48(3), 1348--1382.
+ Douc, R., Moulines, E., Priouret, P., and Soulier, P. (2018). _Markov
  Chains_. Springer.
+ Durmus, A., Moulines, E., Naumov, A., and Samsonov, S. (2025). Finite-time
  high-probability bounds for Polyak--Ruppert averaged iterates of linear
  stochastic approximation. _Mathematics of Operations Research_, 50(2),
  935--964.
+ Durmus, A., Moulines, E., Naumov, A., Samsonov, S., Scaman, K., and Wai, H.-T.
  (2021). Tight high probability bounds for linear stochastic approximation
  with fixed stepsize. In _Advances in Neural Information Processing Systems_,
  34, 30063--30074.
+ Flegal, J. M. and Jones, G. L. (2010). Batch means and spectral variance
  estimators in Markov chain Monte Carlo. _Annals of Statistics_, 38(2),
  1034--1070. https://doi.org/10.1214/09-AOS735.
+ Huo, D., Chen, Y., and Xie, Q. (2024). Effectiveness of constant stepsize in
  Markovian LSA and statistical inference. _Proceedings of the AAAI Conference
  on Artificial Intelligence_, 38(18), 20447--20455.
  https://doi.org/10.1609/aaai.v38i18.30028.
+ Levin, I., Naumov, A., and Samsonov, S. (2025). High-order error bounds for
  Markovian LSA with Richardson--Romberg extrapolation. Extended version,
  arXiv:2508.05570. Conference version in _Proceedings of the AAAI Conference
  on Artificial Intelligence_, 40(43), 36696--36704, 2026.
  https://doi.org/10.1609/aaai.v40i43.40994.
+ Liu, Y., Vats, D., and Flegal, J. M. (2022). Batch size selection for
  variance estimators in MCMC. _Statistics and Computing_, 32, 28.
  https://doi.org/10.1007/s11222-022-10080-1.
+ Ng, S. and Perron, P. (1996). The exact error in estimating the spectral
  density at the origin. _Journal of Time Series Analysis_, 17(4), 379--408.
+ Polyak, B. T. (1990). New stochastic approximation type procedures.
  _Automation and Remote Control_, 51(7), 937--946.
+ Robbins, H. and Monro, S. (1951). A stochastic approximation method. _Annals
  of Mathematical Statistics_, 22(3), 400--407.
+ Ruppert, D. (1988). Efficient estimations from a slowly convergent
  Robbins--Monro process. Technical Report 781, Cornell University Operations
  Research and Industrial Engineering.
+ Samsonov, S., Sheshukova, M., Moulines, E., and Naumov, A. (2025).
  Statistical inference for linear stochastic approximation with Markovian
  noise. _Advances in Neural Information Processing Systems_, 38.
  arXiv:2505.19102.
+ Singh, R., Shukla, A., and Vats, D. (2025). On the utility of equal batch
  sizes for inference in stochastic gradient descent. _Journal of Machine
  Learning Research_, 26(31), 1--49.
+ Vats, D. and Flegal, J. M. (2022). Lugsail lag windows for estimating
  time-average covariance matrices. _Biometrika_, 109(3), 735--750.
  https://doi.org/10.1093/biomet/asab049.

#pagebreak()

#include "src/appendix.typ"

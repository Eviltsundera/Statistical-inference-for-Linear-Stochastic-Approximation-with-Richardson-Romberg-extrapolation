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
  Your abstract.
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

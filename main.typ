// Document settings
#set document(
  title: "Statistical Inference for Linear Stochastic Approximation with Richardson-Romberg Extrapolation",
)
#set page(paper: "us-letter", margin: (x: 1in, y: 1in))
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set heading(numbering: "1.")
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

= Constant Asymptotic <appendix:constants>

#include "src/constant_asymp.typ"

#pagebreak()

= New Lemma <appendix:new_lemma>

#include "src/new_lemma.typ"

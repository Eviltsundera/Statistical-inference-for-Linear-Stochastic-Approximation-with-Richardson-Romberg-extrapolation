// Theorem environments
#let theorem-counter = counter("theorem")
#let lemma-counter = counter("lemma")
#let remark-counter = counter("remark")
#let corollary-counter = counter("corollary")

#let theorem(body) = {
  theorem-counter.step()
  block(width: 100%, spacing: 0.8em)[
    *Theorem #context theorem-counter.display().* #body
  ]
}

#let lemma(body) = {
  lemma-counter.step()
  block(width: 100%, spacing: 0.8em)[
    *Lemma #context lemma-counter.display().* #body
  ]
}

#let remark(body) = {
  remark-counter.step()
  block(width: 100%, spacing: 0.8em)[
    *Remark #context remark-counter.display().* #body
  ]
}

#let corollary(body) = {
  corollary-counter.step()
  block(width: 100%, spacing: 0.8em)[
    *Corollary #context corollary-counter.display().* #body
  ]
}

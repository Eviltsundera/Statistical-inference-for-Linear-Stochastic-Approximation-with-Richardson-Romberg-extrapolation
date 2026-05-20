// Theorem environments. These are implemented as captionless figures so that
// labels after theorem-like blocks can be referenced with @lem:..., @thm:....
#let theorem-like(name, kind, supplement, body) = figure(
  block(width: 100%, spacing: 0.8em)[
    *#name #context counter(figure.where(kind: kind)).display().* #body
  ],
  kind: kind,
  supplement: supplement,
  numbering: "1.",
  caption: none,
)

#let theorem(body) = theorem-like([Theorem], "theorem", [Theorem], body)
#let lemma(body) = theorem-like([Lemma], "lemma", [Lemma], body)
#let remark(body) = theorem-like([Remark], "remark", [Remark], body)
#let corollary(body) = theorem-like([Corollary], "corollary", [Corollary], body)

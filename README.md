# Statistical Inference for Linear Stochastic Approximation with Richardson-Romberg Extrapolation

Master's thesis on statistical inference methods for Linear Stochastic Approximation (LSA) using Richardson-Romberg extrapolation.

## Prerequisites

- [Rust](https://rustup.rs/) toolchain
- [Typst](https://typst.app/) — install via Cargo:

```bash
cargo install --locked typst-cli
```

## Building

```bash
typst compile --root . diploma_typst/main.typ diploma_typst/main.pdf
```

This produces `diploma_typst/main.pdf`.

For live preview with auto-recompilation on file changes:

```bash
typst watch --root . diploma_typst/main.typ diploma_typst/main.pdf
```

Build the slide deck:

```bash
typst compile --root . diploma_typst/presentation.typ diploma_typst/presentation.pdf
```

## Project Structure

```
diploma_typst/
  main.typ               — main document (entry point)
  main.pdf               — compiled thesis
  presentation.typ       — slide deck source
  presentation.pdf       — compiled slide deck
  src/
    defs.typ              — shared definitions (theorem environments)
    introduction.typ      — LSA setting, assumptions, RR motivation, goals
    zeroth_order_rr.typ   — zeroth-order RR difference $\tilde J^{(0,\alpha)}$
    last_iterate.typ      — first-order/last-iteration centered bound
    pr_weights.typ        — Richardson--Romberg PR weight bounds
papers/                  — reference papers
summaries/               — paper summaries
code/                    — numerical experiments (see code/README.md)
```

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
typst compile main.typ
```

This produces `main.pdf` in the project root.

For live preview with auto-recompilation on file changes:

```bash
typst watch main.typ
```

## Project Structure

```
main.typ                 — main document (entry point)
src/
  defs.typ               — shared definitions (theorem environments)
  constant_asymp.typ     — constant asymptotic analysis
  new_lemma.typ          — new lemma and martingale decomposition
```

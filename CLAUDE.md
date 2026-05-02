# CLAUDE.md

Repository for a Master's thesis on statistical inference (CLT / Berry–Esseen) for Richardson–Romberg averaged iterates in Linear Stochastic Approximation (LSA) with Markovian noise.

The repo has two largely independent halves:

1. **Thesis** — Typst sources at the root (`main.typ`, `src/*.typ`).
2. **Numerical experiments** — Python package under `code/`, managed with `uv`.

Papers, summaries, research notes, and experiment reports sit in their own top-level directories (`papers/`, `summaries/`, `research/`, `reports/`).

## Typst thesis

Entry point: `main.typ` (includes `src/introduction.typ`, `src/constant_asymp.typ`, `src/new_lemma.typ`). Shared theorem/lemma environments live in `src/defs.typ`.

Build:

```bash
typst compile main.typ          # → main.pdf
typst watch main.typ            # live reload
```

`presentation.typ` / `presentation.pdf` is a slide deck and builds the same way.

Existing Typst sources mix English prose with Russian inline notes (e.g. `// ПРОВЕРИТЬ ЕЩЕ РАЗ`). Preserve author's language when editing — don't silently translate comments.

## Python experiments (`code/`)

Package `lsa_inference` with four modules:

- `markov_chain.py` — random irreducible transition matrix `P`, stationary `π`, batched chain simulation.
- `lsa_problem.py` — generates state-dependent `A(x)`, `b(x)` with Hurwitz mean `Ā`; computes `θ* = -Ā⁻¹ b̄`; diagnostics (`problem_diagnostics`) flag `‖A‖>1` and `ρ(I+αA)≥1`.
- `lsa_engine.py` — vectorized LSA iterations over all trajectories at once:
  - `run_lsa_const` / `run_lsa_const_full` — constant step (Huo 2023).
  - `run_lsa_diminishing` — `α/√k` with CLTZ20 non-uniform batching.
  - `run_lsa_polyak_ruppert` — diminishing step + PR averaging (Samsonov 2025).
  - `run_rr_full` — Richardson–Romberg: same trajectories, multiple α's, Lagrange weights via `rr_coefficients`.
  - Divergence guard: any trajectory whose iterate exceeds `1e6` is clamped to `NaN` (`_DIVERG_THRESH`).
- `inference.py` — CI construction: `batch_mean_ci` (Huo), `obm_ci` (overlapping batch mean, Samsonov), `obm_rr_ci` (lugsail-style RR on variance estimator, cancels O(1/b) bias), `msb_ci` (multiplier subsample bootstrap).

### Runners

All runners are at `code/`'s top level and use `multiprocessing` with one worker per LSA problem.

- `run_comparison.py` — main 16-method bench (Huo batch-mean, PR/RR × OBM/MSB/OBM-RR).
- `run_bn_sweep.py` — sweeps OBM block size `b_n` to pick the right T-exponent.
- `run_lugsail_bias_variance.py` — bias/variance of OBM vs OBM-RR (lugsail) vs ground truth `σ²_∞ = uᵀ Ā⁻¹ Γ_ε Ā⁻ᵀ u`.

Run everything through `uv` from `code/`:

```bash
cd code
uv sync
uv run python run_comparison.py                                      # smoke test
uv run python run_comparison.py --n-problems 100 --n-traj 100 --T 1000000
```

CSV outputs land in `code/` (top level) for `run_comparison.py` / `run_bn_sweep.py`, and in `code/results/` for `run_lugsail_bias_variance.py`.

### Non-obvious choices to respect

- **OBM block size**: `b_n ~ T^0.6` (per Samsonov et al. 2025 Table 2). `T^0.75` oversmooths and under-covers — see comment in `run_comparison.py:298`.
- **PR schedule**: `c0=200, k0=20000, γ=0.65` matches the Samsonov paper notebook exactly — large `k0` keeps iterates quasi-stationary on the OBM block scale.
- **RR pair**: default `α ∈ {0.2, 0.02}`. At `α=0.2` a significant fraction of randomly generated problems have `ρ(I+0.2A)≳1` — last-iterate instability shows up as low coverage of the individual α=0.2 method, *not* as NaN divergence.
- **OBM-RR (lugsail)**: `σ²_RR = λ/(λ-1)·σ²(λb) - 1/(λ-1)·σ²(b)`, clamped to ≥0. `λ=2` cancels the leading 1/b bias for the Bartlett kernel (Vats & Flegal 2022). See `inference.py:obm_rr_ci`.
- **Memory**: `run_comparison.py` processes trajectories in chunks of 100 (`_CHUNK_SIZE`) to bound per-worker RAM. `_obm_variance_from_proj` streams in chunks too — don't materialize `(n_traj, n_blocks)` for T≳10⁶.

## Supporting directories

- `papers/` — source PDFs of every paper the thesis builds on (Huo 2023, Samsonov 2025, Levin 2025, Vats & Flegal 2022, Flegal & Jones 2010, Ng & Perron 1996, Liu/Vats/Flegal 2022, Singh/Shukla/Vats 2025).
- `summaries/` — one markdown summary per paper, plus `RR_for_OBM_variance_estimator.md` (derivation of the OBM-RR construction).
- `research/` — longer-form research notes (currently `obm_rr_markov_lsa_report.md`).
- `conversations/` — standalone Markdown notes answering math/theory questions. Use Markdown-preview-friendly LaTeX: inline formulas as `$...$`, display formulas as `$$...$$`, and avoid code blocks for formulas unless showing literal source syntax.
- `reports/` — dated experiment reports (`YYYY-MM-DD_<name>.md`) summarizing finished runs. Not gitignored — these are the canonical record of what has been tried.
- `code/docs/` — reproduction specs for Huo 2023 and Samsonov 2025 + a Russian-language `code_explanation.md`.
- `tmp/` — scratch / reference notebooks (gitignored). `tmp/MarkovLSAConstantStepSize/` is an older clone of the Huo-style experiments kept as a reference.

## Conventions

- `uv` for Python env (don't use pip/conda directly in `code/`).
- Reports in `reports/` are dated ISO (`YYYY-MM-DD_<slug>.md`) and link back to the script and raw CSVs that produced them.
- Math notes in `conversations/` should render cleanly in Markdown preview: use `$...$` and `$$...$$` for formulas.
- Code comments and docstrings mix English and Russian — keep the language the surrounding text uses.
- Don't commit `tmp/`, `__pycache__/`, or generated `main.pdf` changes unless the user asks.

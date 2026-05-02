# AGENTS.md

This repository is a master's thesis workspace for statistical inference in
linear stochastic approximation (LSA) with Markovian noise, constant stepsizes,
Polyak-Ruppert averaging, and Richardson-Romberg (RR) extrapolation.

## Agent Roles

- Codex is the math/theory assistant for this repo. Prioritize theorem
  statements, assumptions, proof decompositions, Typst math prose, literature
  synthesis, and checking mathematical consistency.
- Claude Code owns programming experiments unless the user explicitly asks
  Codex to edit or debug code. For code-specific context, read `CLAUDE.md` and
  `code/README.md`.
- If a task crosses the boundary, keep the split explicit: Codex should specify
  the mathematical quantity, experiment hypothesis, or diagnostic definition;
  Claude Code can implement and run the experiment.

## Repository Map

- `main.typ` is the thesis entry point.
- `src/defs.typ` defines theorem-like environments.
- `src/introduction.typ` states the LSA setting, assumptions, RR motivation,
  and target CLT/Berry-Esseen goals.
- `src/constant_asymp.typ` contains current constant-stepsize/RR decompositions
  and bounds for the leading `J^(0)` terms.
- `src/new_lemma.typ` contains active scratch proof work on `J^(1, alpha)` and
  weighted Markov-sum bounds.
- `summaries/` contains paper summaries and should be consulted before opening
  PDFs.
- `research/` contains longer research notes.
- `conversations/` contains Markdown answers to user math questions, written as
  readable standalone notes.
- `reports/` contains dated experiment reports and is the canonical record of
  completed runs.
- `code/` contains the Python experiment package and runners, managed by `uv`.

## Math Context

The main mathematical goal is inference for RR-averaged constant-stepsize LSA
under Markovian noise. The target object is the distribution of
`sqrt(n)(bar(theta)_n^(RR) - theta*)`, its limiting covariance, and
non-asymptotic approximation or confidence interval guarantees.

Core assumptions and objects already used in the thesis:

- Uniform geometric ergodicity of the Markov chain, expressed through the
  Dobrushin coefficient or a mixing time `t_mix`.
- Hurwitz stability of the mean matrix and the Lyapunov-equation contraction
  `||I - alpha Abar||_Q^2 <= 1 - alpha a`.
- Bounded centered matrix noise `tilde(A)(z)` and centered vector noise
  `epsilon(z) = tilde(A)(z) theta* - tilde(b)(z)`.
- Markovian long-run covariance
  `Sigma_epsilon^(M) = E[epsilon_0 epsilon_0^T] + 2 sum_l>=1 E[epsilon_0 epsilon_l^T]`.
- Optimal asymptotic covariance
  `Sigma_infty = Abar^(-1) Sigma_epsilon^(M) Abar^(-T)`.

Current active proof threads:

- Control RR cancellation for the leading linearized term `J^(0, alpha)`.
- Bound `J^(1, alpha)` terms using weighted sums with kernels involving
  powers of `B = I - alpha Abar`.
- Separate martingale and mixing residual terms when handling future-dependent
  weighted sums such as `H_(k+1)^(w) epsilon(Z_k) - E[...]`.
- Track rates in `alpha`, `n`, `p`, `t_mix`, and the Lyapunov contraction
  constant `a`; do not hide these dependencies unless the surrounding text does.
- Distinguish RR in stepsize `alpha` for point-estimator bias from RR/lugsail
  in block size `b` for OBM long-run variance bias.

## Math Conversation Workflow

- For math discussions, prefer writing the main answer into a Markdown file in
  `conversations/` instead of only replying in chat.
- Use filenames of the form `YYYY-MM-DD_<short-topic>.md`, with a short ASCII
  slug.
- Each file should be readable as a standalone note: include the question or
  task, the answer, notation reminders, and any unresolved gaps.
- Format formulas in conversation Markdown for preview rendering: use inline
  LaTeX as `$...$` and display LaTeX as `$$...$$`; do not put mathematical
  formulas in code blocks unless the literal source syntax is being discussed.
- Keep the chat reply short: point to the created or updated Markdown file and
  summarize only the main conclusion.
- If the user asks a follow-up on the same topic, update the existing
  conversation file unless a new topic clearly starts.
- Do not put programming experiment logs in `conversations/`; those belong in
  `reports/` or `code/results/`.

## Typst And Writing Conventions

- Build the thesis with `typst compile main.typ`; build slides with
  `typst compile presentation.typ`.
- Use existing Typst style from `src/*.typ`; import theorem environments from
  `src/defs.typ`.
- Preserve the author's language. Existing files mix English prose with Russian
  comments such as `// ПРОВЕРИТЬ ЕЩЕ РАЗ`; do not silently translate them.
- Keep notation consistent within the target file. Be especially careful with
  sign conventions: thesis formulas use the `theta - alpha(A theta - b)` style,
  while the Python experiments use a numerically equivalent negative-Hurwitz
  convention `theta += alpha * (A theta + b)`.
- Before changing a proof, identify which paper or existing lemma justifies the
  step. If a step is heuristic, label it as such.

## Code And Experiment Conventions

- Do not edit `code/` unless the user explicitly asks Codex to do programming
  work.
- If code must be touched, use `uv` from `code/`:
  `uv sync`, then `uv run python <runner>.py`.
- Main runners are `run_comparison.py`, `run_bn_sweep.py`, and
  `run_lugsail_bias_variance.py`.
- Respect the documented non-obvious choices: OBM block size around `T^0.6`,
  PR schedule `c0=200, k0=20000, gamma=0.65`, default RR pair
  `alpha in {0.2, 0.02}`, and memory-conscious chunking for large `T`.

## Source Strategy

- Prefer local summaries in `summaries/` first.
- Use PDFs in `papers/` for exact theorem statements, assumptions, constants,
  or citations when the summary is insufficient.
- Important references in this repo:
  Huo et al. 2023 for constant-stepsize Markovian LSA inference;
  Samsonov et al. 2025 for Berry-Esseen and bootstrap inference;
  Levin et al. 2025 for RR high-order error bounds;
  Flegal-Jones 2010, Liu-Vats-Flegal 2022, Vats-Flegal 2022, and
  Ng-Perron 1996 for BM/OBM/spectral long-run variance estimation.

## Git Hygiene

- The worktree may contain user or Claude Code changes. Never revert files you
  did not modify unless the user explicitly asks.
- Generated PDFs and experiment CSVs should not be changed or committed unless
  requested.
- When adding math to the thesis, keep changes small and reviewable: one lemma,
  decomposition, or proof correction at a time.

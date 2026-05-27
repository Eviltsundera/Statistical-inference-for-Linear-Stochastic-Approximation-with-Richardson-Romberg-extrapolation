# Where the Levin Small-Step Thresholds Are Defined

## Question

In the stationary threshold

$$
\alpha_*(q,t_{\mathrm{mix}})
  := \min\{\alpha_{\mathrm{L,P2}}(q,t_{\mathrm{mix}}),
           \alpha_{\mathrm{L,C6}}(q,t_{\mathrm{mix}}),
           \alpha_{\mathrm{L,P5}}(q,t_{\mathrm{mix}}),
           \alpha_{\mathrm{L,P8}}(q,t_{\mathrm{mix}}),
           \alpha_{\mathrm{L,P9}}(q,t_{\mathrm{mix}}),
           \alpha_{\mathrm{L,inv}}(q,t_{\mathrm{mix}})\},
$$

where are the individual thresholds defined?

## Answer

Before the May 27 edit, the individual quantities
$\alpha_{\mathrm{L,P2}}$, $\alpha_{\mathrm{L,C6}}$,
$\alpha_{\mathrm{L,P5}}$, $\alpha_{\mathrm{L,P8}}$,
$\alpha_{\mathrm{L,P9}}$, and $\alpha_{\mathrm{L,inv}}$ are not given by
separate displayed formula definitions. They are used as local names for the
step-size ceilings required by the imported Levin et al. results.

The combined threshold is introduced in
`src/pr_weights/01_imported_inputs.typ`, equation
`<eq:levin-stationary-threshold>`. The nearby comment says that the six ceilings
correspond respectively to:

- Levin Proposition 2, recorded locally as `<lem:levin-prop-2>`;
- Levin Corollary 6, recorded locally as `<lem:levin-cor-6>`;
- Levin Proposition 5, recorded locally as `<lem:levin-prop-5-component>`;
- Levin Proposition 8, recorded locally as `<lem:levin-prop-8>`;
- Levin Proposition 9, recorded locally as `<lem:levin-prop-9>`;
- Levin Corollary 4 / invariant depth-two law, recorded locally as
  `<lem:levin-invariant-depth-two-law>`.

Those working forms are in `src/appendix/external_inputs.typ`. They state
conditions of the form $w \le \alpha_{\mathrm{L,\cdot}}(\cdot)$, but do not
define the $\alpha_{\mathrm{L,\cdot}}$ functions explicitly.

## Mapping in the Current Text

The current local mapping is:

$$
\alpha_{\mathrm{L,P2}}(q,t_{\mathrm{mix}})
\quad\leftrightarrow\quad
\text{Levin et al. Proposition 2, stationary bias of } J^{(1)}.
$$

$$
\alpha_{\mathrm{L,C6}}(q,t_{\mathrm{mix}})
\quad\leftrightarrow\quad
\text{Levin et al. Corollary 6, centered bilinear } L_p \text{ bound.}
$$

$$
\alpha_{\mathrm{L,P5}}(p,t_{\mathrm{mix}})
\quad\leftrightarrow\quad
\text{Levin et al. Appendix B.2, Proposition 5, depth-two startup contraction.}
$$

$$
\alpha_{\mathrm{L,P8}}(q,t_{\mathrm{mix}})
\quad\leftrightarrow\quad
\text{Levin et al. Proposition 8, } J^{(2)} \text{ moment bound.}
$$

$$
\alpha_{\mathrm{L,P9}}(q,t_{\mathrm{mix}})
\quad\leftrightarrow\quad
\text{Levin et al. Proposition 9, } H^{(2)} \text{ moment bound.}
$$

$$
\alpha_{\mathrm{L,inv}}(q,t_{\mathrm{mix}})
\quad\leftrightarrow\quad
\text{Levin et al. Corollary 4 / invariant depth-two law.}
$$

The last name is potentially misleading: the thesis also defines a separate
local inverse ceiling

$$
\alpha_{\mathrm{inv}} := \frac{1}{2\|\bar A\|},
$$

which is unrelated to the placeholder $\alpha_{\mathrm{L,inv}}$ except by
name. To avoid confusion, $\alpha_{\mathrm{L,inv}}$ should probably be renamed
to something like $\alpha_{\mathrm{L,C4}}$ or
$\alpha_{\mathrm{L,statJ2}}$.

## Recommendation

Either:

1. keep these as abstract imported ceilings and add one sentence saying that
   they denote the step-size restrictions in the cited Levin statements, not
   explicit constants re-derived in the thesis; or
2. replace them by the actual Levin thresholds when possible.

From the Levin paper, the common explicit thresholds are expressed through
their $\alpha_\infty$, $\alpha_{p,\infty}$, and
$\alpha_{q,\infty}t_{\mathrm{mix}}^{-1}$ style constants. However, since the
current thesis only needs an admissible positive ceiling, the abstract-minimum
presentation is acceptable if it is stated clearly.

## Update

The temporary Section-8 remark was removed. The working convention remains in
`src/pr_weights/01_imported_inputs.typ`, where the combined
`alpha_*` threshold lists the six imported Levin ceilings and the adjacent
comment maps them to the corresponding local imported lemmas.

## Follow-up: Where `alpha_infinity` Is Defined

The explicit definition currently appears in
`src/zeroth_order_rr/02_h_kernel_norm_bound.typ`, inside Lemma
`<lem:lyapunov-contraction-local>`:

$$
\alpha_\infty
  := \min\left\{
    \frac{\lambda_{\min}(P)}
         {2\kappa_Q\|\bar A\|_Q^2},
    \frac{\|Q\|}{\lambda_{\min}(P)}
  \right\}.
$$

Here $Q$ solves

$$
\bar A^\top Q + Q\bar A = P,
$$

and

$$
a := \frac{\lambda_{\min}(P)}{2\|Q\|},
\qquad
\kappa_Q := \frac{\lambda_{\max}(Q)}{\lambda_{\min}(Q)}.
$$

The same symbol is used globally later, for example in
`src/appendix/key_quantities.typ`, but that appendix currently states only the
contraction property for all $\alpha\in[0,\alpha_\infty]$ and does not repeat
the displayed definition.

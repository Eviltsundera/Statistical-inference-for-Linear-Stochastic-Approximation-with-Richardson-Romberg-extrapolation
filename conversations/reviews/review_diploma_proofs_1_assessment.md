# Assessment of `review_diploma_proofs_1.md`

Date: 2026-05-18.

## Question

Do I agree with the proof-review notes in
`conversations/reviews/review_diploma_proofs_1.md`?

## Short answer

Mostly yes. The review correctly identifies the main proof risks. Some
Section 4--5 issues have already been improved in the current text, but the
early-chapter algebra/concentration issues and the burned-in startup transfer
for $H^{(2,w)}$ remain real.

I do not see a fatal conceptual error in the RR weight / Poisson / martingale
Berry--Esseen architecture. The problems are local proof gaps and
self-containedness issues.

## Points I agree with and still consider active

1. **Lost factor $1/a$ in Section 2.4.**  
   The current bound for $H_j^{(n)}$ has the factor $2/(\alpha a)$, but the
   next display for $\|g_j\|_\infty$ drops the resulting $1/a$. If dependence
   on $a$ is kept explicit, the summed proxy should scale like
   $\alpha/a^3$, not $\alpha/a$.

2. **Endpoint $j=n$ in $H_j^{(n)}$.**  
   For $j=n$ the defining sum is empty, while the displayed bound uses the
   exponent $(n-j-1)/2=-1/2$. This is formal but easy to fix by treating
   $1\le j\le n-1$ and setting $H_n^{(n)}=0$.

3. **Vector-valued Markov concentration.**  
   The stated concentration lemma is vector-valued with a dimension-free
   prefactor. Unless the cited Durmus/Levin result is genuinely Hilbert-valued
   in this form, this is underjustified. The safest thesis route is to state
   the lemma for scalar projections and use only fixed $u^\top(\cdot)$ bounds
   in the Berry--Esseen proof, or else add a real vector-valued argument with
   an explicit dimension dependence.

4. **Section 3.2 should state centered fluctuation plus bias.**  
   The local conclusion
   $\|D_1^{\mathrm{mis,RR}}\|_{L_p}=O(\sqrt n\,\alpha)$ is not what was proved
   from the centered last-iterate estimate. It should be written as a centered
   $L_p$ bound plus the stationary RR bias
   $O(\sqrt n\,\alpha^2)$.

5. **Startup transfer for $H^{(2,w)}$ is the weakest burn-in bridge.**  
   The current burn-in chapter now cites Levin's depth-two contraction and the
   Appendix D.1 representation of $H^{(2,w)}$, but the proof is still terse.
   The missing detail is not just contraction of $J^{(2,w)}$; one also has to
   control the difference between the random products along the coupled
   finite-start and stationary trajectories. If Levin contains exactly this
   full-state startup estimate, it should be quoted as such. Otherwise this
   should be stated as an imported technical assumption or proved separately.

6. **Final burn-in theorem assumptions should be more self-contained.**  
   The current theorem is better than the version reviewed because it has
   $\alpha_{\mathrm{adm}}(p,q)$ and explicit finite-$n$ conditions. Still, a
   final reader should not have to infer that Assumptions 1--3, bounded
   $\widetilde A$, and all Levin admissibility restrictions are active.

## Points that are already mostly fixed in the current text

1. **Stationary augmented-chain convention.**  
   This review point was correct, but the current stationary chapter now
   explicitly marks the Section 4 results as stationary augmented-chain
   statements and separates them from the burn-in theorem.

2. **Section 3.1 dependence on $a$ in the $U_R$ assembly.**  
   The current `last_iterate.typ` statement uses
   $\sqrt{\alpha/a}$ for the $U_R$ contribution after multiplying by
   $\alpha$, so this particular complaint no longer appears in the same form.
   The projection-to-vector issue remains separate.

3. **Final theorem assumption grouping.**  
   The current burn-in chapter already groups the elementary step-size
   restrictions through $\alpha_{\mathrm{adm}}(p,q)$ and a large-$n$ paragraph.
   I would still polish the theorem statement, but the main redundancy has
   been reduced.

## Points I agree are low priority

1. The matrix conditional covariance should be notationally separated from
   the vector noise $\epsilon$ to reduce reading risk in the variance/bracket
   section.
2. The convention on which constants absorb dependence on $a,t_{\rm mix},C_A$
   should be made uniform.

## Recommended order

1. Fix Section 2.4 algebra and the endpoint $j=n$.
2. Replace the vector concentration lemma by a scalar projection lemma, or add
   a real vector-valued proof.
3. Rewrite the Section 3.2 misadjustment statement as centered fluctuation
   plus bias.
4. Strengthen the burn-in startup contraction for $H^{(2,w)}$: exact imported
   proposition if available, otherwise explicit technical assumption.
5. Make the final burn-in theorem assumptions self-contained.

## Work Plan

Use this as the tracking checklist for the proof-cleanup pass.

- [x] **Fix Section 2.3--2.4 algebra in `src/zeroth_order_rr.typ`.**
  - Treat the endpoint $j=n$ separately: the sum defining $H_n^{(n)}$ is empty.
  - Restore the missing factor $1/a$ in $\|g_j\|_\infty$.
  - Recompute $\sum_j \|g_j\|_\infty^2$, $u_n^2$, and the displayed
    $L_p$ bound with explicit powers of $a$.
  - Rename the local constant currently competing with the assumption constant
    $C_A$ if needed.
  - Done 2026-05-18: the bound now sums only the nonzero terms
    $j\le n-1$, gives $\sum_j\|g_j\|_\infty^2\lesssim \alpha/a^3$,
    and sets $\widehat C_A=32\widetilde C_A\|\epsilon\|_\infty
    \sqrt{t_{\rm mix}/a^3}$.

- [x] **Make the Markov concentration input scalar or fully justify the vector
  form.**
  - Preferred route: restate the lemma for scalar functions
    $g_i:\mathsf Z\to\mathbb R$.
  - Rewrite the immediate application as a fixed projection
    $u^\top \widetilde J_n^{(0,\alpha)}$.
  - Check whether later uses in `src/pr_weights.typ` and
    `src/burn_in_transfer.typ` already use scalar quantities; if yes, point
    this out and avoid unnecessary vector claims.
  - Done 2026-05-18: Section 2 now states the concentration lemma for
    scalar time-dependent functions and applies it to
    $u^\top\widetilde J_n^{(0,\alpha)}$. The resulting bound is
    $\|u^\top\widetilde J_n^{(0,\alpha)}\|_{L_p}\lesssim
    \sqrt p\,\|u\|\,\widehat C_A\sqrt\alpha$. Later stationary and burn-in
    uses are scalar bracket/projection bounds, so no dimension-free vector
    concentration is claimed.

- [x] **Patch Section 3.2 in `src/last_iterate.typ`.**
  - Replace the statement
    $\|D_1^{\mathrm{mis,RR}}\|_{L_p}=O(\sqrt n\,\alpha)$ by a centered bound.
  - Add the separate stationary bias estimate
    $\|\mathbb E D_1^{\mathrm{mis,RR}}\|\lesssim \sqrt n\,\alpha^2$.
  - Keep the conclusion that the depth-one route is insufficient at
    $\alpha\asymp n^{-1/2}$.
  - Done 2026-05-18: Section 3.2 now defines
    $D_{1,\mathrm c}^{\mathrm{mis,RR}}=D_1^{\mathrm{mis,RR}}-\mathbb E
    D_1^{\mathrm{mis,RR}}$, proves the centered bound
    $\|D_{1,\mathrm c}^{\mathrm{mis,RR}}\|_{L_p}\lesssim
    \sqrt n\,\alpha\,\Phi(p,\alpha)$, and records the separate stationary
    bias bound $\|\mathbb E D_1^{\mathrm{mis,RR}}\|\lesssim
    \sqrt n\,\alpha^2$.

- [x] **Strengthen the burn-in startup bridge for $H^{(2,w)}$ in
  `src/burn_in_transfer.typ`.**
  - First check the Levin source or local summary for an exact proposition
    controlling the full augmented state including $H^{(2,w)}$.
  - If it exists, quote it explicitly and replace the current compressed proof
    paragraph.
  - If it does not, state this as a named imported technical assumption, with
    a short explanation that componentwise contraction for
    $J^{(0)},J^{(1)},J^{(2)}$ alone is not enough.
  - Done 2026-05-18: Levin Proposition 5 was checked against the local PDF
    text; it contracts only the state $(Z,J^{(0)},J^{(1)},J^{(2)})$, while
    Proposition 9 gives a one-trajectory moment bound for $H^{(2)}$. This was
    first isolated as a technical assumption and then replaced by the lemma in
    the next checklist item.

- [x] **Prove the full-state startup contraction previously stated as
  `Technical Assumption`.**
  - Extend the Levin Proposition-5 coupling from
    $(Z,J^{(0)},J^{(1)},J^{(2)})$ to the enlarged state
    $(Z,J^{(0)},J^{(1)},J^{(2)},H^{(2)})$.
  - Start from the representation
    $H_k^{(2,w)}=-w\sum_{\ell=1}^k
    \Gamma_{\ell+1:k}^{(w)}\widetilde A(Z_\ell)J_{\ell-1}^{(2,w)}$.
  - Split on the base-chain coupling time: before coupling, use the coupling
    tail and one-trajectory moment bounds; after coupling, the products are
    common and the only source term is
    $J_{\ell-1}^{(2,w)}-\widetilde J_{\ell-1}^{(2,w)}$.
  - Reuse Levin's random-product estimates from Proposition 9 and the
    Proposition-5 contraction/moment envelopes to recover
    $\exp(-cwa k/p)$ decay with an acceptable polynomial factor.
  - After proving it, replace or downgrade `Technical Assumption` in
    `src/burn_in_transfer.typ` to a lemma/proposition.
  - Done 2026-05-18: `src/burn_in_transfer.typ` now has
    `@lem:burn-full-startup`, proving the startup contraction by extending the
    Levin Proposition-5 exact-coupling argument to $H^{(2,w)}$. The proof uses
    the representation of $H^{(2,w)}$, splits on the base-chain coupling time,
    controls the bad event by Levin Proposition 9, and controls the post-coupling
    convolution by the random-product estimate plus the $J^{(2,w)}$ contraction.
    The old `Technical Assumption` environment and `@ass:burn-full-startup`
    references were removed.

- [x] **Make final burn-in theorem assumptions self-contained.**
  - State that Assumptions 1--3 are in force.
  - Keep $\alpha_{\mathrm{adm}}(p,q)$ as the compact holder for Levin and
    small-step admissibility restrictions.
  - Make clear which conditions are finite-$n$ and which become automatic for
    all sufficiently large $n$ under $\alpha=c n^{-1/2}$.
  - Done 2026-05-18: the final balanced burn-in theorem now explicitly
    assumes Assumptions 1--3, the Lyapunov contraction, the Levin depth-two
    stationary moment/misadjustment inputs, and $\sigma^2(u)>0$. The definition
    of $\alpha_{\mathrm{adm}}(p,q)$ now states which threshold comes from the
    Levin depth-two bounds and which comes from the proved startup-contraction
    lemma `@lem:burn-full-startup`.

- [x] **Notation cleanup after the mathematical fixes.**
  - Separate notation for vector noise $\epsilon$ and matrix conditional
    covariance in the bracket section.
  - Make the convention on constants consistent, especially dependence on
    $a$, $t_{\rm mix}$, $C_A$, and $\kappa_Q$.
  - Done 2026-05-18: the conditional covariance in the stationary and burn-in
    bracket sections is now denoted by $\mathcal V_\epsilon(z)$ rather than an
    overlined $\epsilon$, and both chapters state a short convention for which
    dependencies are absorbed into named constants and which remain explicit.

- [x] **Verification.**
  - Run `typst compile main.typ`.
  - Query the affected labels if labels are added or changed.
  - Skim the rendered Section 2.4 and burn-in startup pages for layout.
  - Done 2026-05-18: final verification pass completed. `typst compile
    main.typ` passes; queried `@lem:burn-full-startup`,
    `@eq:burn-startup-transfer`, `@thm:burn-final-balanced`,
    `@thm:misadjustment`, `@thm:burn-misadjustment`, `@eq:bar-eps-def`,
    `@eq:burn-bar-eps-def`, and `@eq:burn-final-alpha-adm`. The built PDF text
    has no `Lemma ,`, `equation )`, `Eq. ,`, unresolved `??`, or
    `Technical Assumption` occurrences. Rendered pages 6--7, 23, 39--40, and
    45 were skimmed for layout.

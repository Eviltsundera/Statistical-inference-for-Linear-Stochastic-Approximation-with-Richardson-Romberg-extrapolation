# Assessment of `review_diploma_proofs_2.md`

Date: 2026-05-20  
Source review: `conversations/reviews/review_diploma_proofs_2.md`  
Current source snapshot: after the finite-start/burn-in and full-state startup-contraction edits.

## Short conclusion

I agree with the main critical direction of the review: the remaining weak
points are not the PR-weight identities or the final burn-in assembly, but
rather the rigor of early concentration lemmas and the exact presentation of
the last-iterate versus PR-average notation.

Some remarks are stale after the latest edits. In particular, the burned-in
Poisson remainder now has the correct linear dependence on
$\|\varepsilon\|_\infty / \sqrt m$, the final theorem no longer assumes an
unnamed `Technical Assumption`, and several finite-start/burn-in normalization
issues have already been separated.

The review should therefore be read as:

1. keep the concentration and notation objections as real action items;
2. downgrade several algebraic and burn-in-transfer comments to verification
   or polish tasks;
3. do not treat the final theorem as fully presentation-ready until the
   concentration inputs are restated with exact citations.

## Item-by-item assessment

### 1.1 Last iterate versus PR average

Verdict: agree.

The introduction uses $\theta_k^{(\alpha)}$ for the last iterate and
$\overline\theta_n^{(\alpha)}$ for the PR average, which is correct. The earlier
Section 2 notation was easy to confuse with the thesis-facing averaged RR
estimator `overline(theta)_n^((alpha, "RR"))`, because it used
`theta_n^(("RR", alpha))` for a last-iterate decomposition.

This has been fixed as a notation/presentation issue: Chapter 2 now uses a
chapter-local last-iterate RR object and Chapter 4 still rederives the
PR-weight representation for the averaged statistic.

### 1.2 Vector bound versus scalar projection in the shifted first-order lemma

Verdict: agree; this is the strongest remaining technical objection.

The lemma in `src/last_iterate.typ` states a vector-valued bound
`||S_n - E S_n||_(L_p)`, but Step 2 proves the difficult future term by first
projecting onto a deterministic direction $u$. There is no final argument that
converts fixed scalar projections into a Euclidean-norm estimate, and such a
conversion would introduce an explicit dimension factor in fixed dimension.

Best fix: make the lemma scalar:

$$
\|u^\top(S_n-\mathbb E S_n)\|_{L_p}
  \le C\|u\|\|\varepsilon\|_\infty(\cdots).
$$

This matches the downstream Berry--Esseen use, which is scalar-projected in
direction $u$.

### 1.3 Conditional concentration for the future-centered term `U_M`

Verdict: agree, but the current text is better than the version described in
the review.

The present proof now subtracts the conditional expectation
$\mathbb E[H_{k+1}^{(w)}u\mid\mathcal F_k]$ and separately handles the
ordinary Markov additive term. That fixes the most naive mistake of applying a
$\pi$-centered inequality directly to a future chain started from $Z_k$.

However, the line "applying the Markov concentration lemma to the future chain
conditionally on $\mathcal F_k$" is still too compressed. The proof needs an
exact cited inequality around the conditional mean, or a self-contained lemma
for future-centered bilinear Markov sums. The later invocation of Samsonov et
al. Proposition 9 also needs its hypotheses and rate matched explicitly to the
kernel used here.

So I agree with the review that this remains a real rigor gap.

### 1.4 Markov concentration for arbitrary initial distribution

Verdict: agree that it must be checked/cited precisely.

The scalar concentration lemma in Section 2 is stated for arbitrary initial
distribution and $\pi(g_i)=0$. Such a statement may be true under the specific
uniform-ergodicity/McDiarmid inequality being imported, because the variance
proxy is large enough to absorb the initial bias. But the current proof says
only "follows the lines of Durmus et al.", which is not sufficient for a
thesis proof.

Fix: quote the exact lemma used, including whether it is concentration around
zero, around the initial-law mean, or around the stationary mean. If the source
only gives concentration around the mean, add the initial-bias correction or
restrict this particular lemma to stationary start.

### 1.5 Full-state startup contraction / Lemma 21

Verdict: partially agree; the review is partly stale.

The latest source no longer leaves this as a bare Technical Assumption. It now
states and proves `lem:burn-full-startup`, including the coupling time, the
Levin depth-two component contraction, the representation of $H^{(2,w)}$, the
bad pre-coupling event, the post-coupling initial term, and the convolution
term.

This has now been strengthened as a Levin-extension style technical
proposition. The proof explicitly says which parts are imported from Levin
Appendix B.2 and Appendix D.1, states the product-stability estimate used in
the $H^{(2)}$ comparison, and expands the bad-event, post-coupling initial
term, and convolution bounds. It is not a fully self-contained reproof of
Levin Proposition 9, but the scope of the imported estimates is now explicit.

### 2.1 Power of `a` in `hat(C)_A`

Verdict: mostly stale / not an error in the current source.

The current definition is

$$
\widehat C_A
  = 32\,\widetilde C_A\,\|\varepsilon\|_\infty
    \sqrt{t_{\rm mix}/a^3},
$$

which is exactly an $a^{-3/2}$ dependence. The preceding squared variance
proxy has $a^{-3}$, as it should. I do not see the alleged $a^{-3}$ mistake in
the current constant.

### 2.2 Power of `a` in the `U_R` assembly

Verdict: stale; current text has the better power.

The current bound after multiplying by $\alpha$ is written with
$\sqrt{\alpha/a}$, not $\sqrt\alpha/a$. This matches the review's requested
correction.

### 2.3 Reuse of `C_A`

Verdict: largely already fixed in Section 2, but keep as a style check.

The old confusing local reuse has been replaced by `overline(C)_A := kappa_Q`,
while `C_A` remains the Assumption-2 sup-norm constant. The derived
`tilde(C)_A := C_A overline(C)_A` is clear enough.

No major mathematical issue remains here.

### 2.4 Burned-in Poisson remainder typo/order

Verdict: stale; current source is correct.

The current bound for `D_(2,n,n_0)^("bRR")` is linear in
$\|\varepsilon\|_\infty$ and has order $1/\sqrt m$. I do not see a remaining
`\sqrt{\|\varepsilon\|_\infty}/m` issue.

### 2.5 Hurwitz wording

Verdict: agree as a wording fix.

The assumption section is mathematically clear:

$$
-\overline A \text{ is Hurwitz}
\quad\text{i.e.}\quad
\operatorname{Re}\lambda(\overline A)>0.
$$

The introduction now states the convention explicitly: $-\overline A$ is
Hurwitz, equivalently all eigenvalues of $\overline A$ have strictly positive
real parts. Assumption 2 uses the same wording.

### 3.1 Depth-one route is too weak

Verdict: agree, but already handled as exposition.

The current text says that the depth-one centered-fluctuation bound is
$O(1)$ at $\alpha\asymp n^{-1/2}$ and therefore does not yield the desired
Berry--Esseen remainder. That is the right role for this section: motivation,
not final proof.

### 3.2 Stationary bias cancellation for `J^(1)`

Verdict: agree, low-to-medium priority.

The statement is plausible, but it is terse. If this section remains in the
thesis, it should specify that the expectation is under the step-size-dependent
stationary augmented law, and that the same leading coefficient $\Delta$
appears at $\alpha$ and $2\alpha$.

### 4.1 RR weight identities

Verdict: agree with the review; these look correct.

The deterministic PR/RR weight identities are one of the more solid parts of
the current proof.

### 4.2 Burned-in weight normalization

Verdict: mostly resolved.

The burn-in chapter now consistently distinguishes the $\sqrt m$ finite-window
statistic, the asymptotic normalization by $\sigma(u)$, and the final
$\sqrt n$ corollary. This no longer looks like a blocker.

### 4.3 Poisson martingale approximation and boundary terms

Verdict: mostly okay, but worth a final proof pass.

The stationary and burned-in Poisson decompositions now both keep the left
boundary and explain why the right boundary vanishes. The predictable-variance
comparison is also separated.

I would still do a final index audit, because an off-by-one error in the
martingale range $l=2,\ldots,n-1$ or the weight $Q_1$ would propagate into the
variance proxy. But conceptually this block is not the main weakness.

### 5.1 Finite-start theorem depends on startup transfer

Verdict: partly stale.

The theorem is no longer conditional on an unnamed technical assumption. It
uses the stated `lem:burn-full-startup`. The remaining issue is whether that
lemma's proof is detailed enough. So this item should be merged with 1.5
rather than treated as an additional missing assumption.

### 5.2 Burn-in scale

Verdict: agree.

The scale
$n_0\asymp(\alpha a)^{-1}\log^2 n$ with
$\alpha=c n^{-1/2}$ gives $n_0=o(n)$ and is consistent with $m\ge n/2$ for
large $n$.

### 6.1 Novelty / "open problem" wording

Verdict: agree partially.

The introduction now uses the narrower wording: existing results do not
directly give the distributional approximation needed for the PR-averaged RR
statistic under Markovian noise, especially after deterministic-start burn-in
transfer. It no longer reads as a claim to be the first CLT for constant-step
LSA in general.

### 6.2 Explicit hypotheses in final theorems

Verdict: mostly resolved, but polish remains.

The final burned-in theorem explicitly assumes Assumptions 1--3, the Lyapunov
contraction, Levin depth-two inputs, and $\sigma^2(u)>0$. The theorem also
states $m=n-n_0$, the step-size admissibility condition, and the burn-in
conditions.

Remaining polish: state "fix $u\in\mathbb R^d$" directly in the theorem
statement, and keep scalar-projected wording visible in theorem names and
normalization definitions.

## Priority list I would use

1. Make the shifted first-order perturbation lemma scalar, or add an explicit
   dimension factor if a vector statement is really needed.
2. Replace the conditional future-chain concentration paragraph by an exact
   imported proposition or a stated lemma for future-centered bilinear Markov
   sums.
3. Quote the exact Markov concentration inequality used for arbitrary initial
   distribution, or restrict it and add a bias correction.
4. Rename the Section 2 last-iterate RR object so it cannot be confused with
   the PR-averaged RR estimator.
5. Expand `lem:burn-full-startup` only if we want the final theorem to be
   fully self-contained rather than "Levin-extension" style.
6. Polish the introduction: Hurwitz wording and novelty claim.

## Current checklist

- [x] Scalarize or dimension-correct the shifted first-order lemma.
  Done in `src/last_iterate.typ`: the lemma and its PR-misadjustment
  application now bound fixed scalar projections $u^\top(\cdot)$, and the
  future-kernel projection is written through $H^\top u$.
- [x] Make the future-centered `U_M` concentration input exact.
  Done in `src/last_iterate.typ`: Step 2 now uses a stated imported
  future-centered bilinear estimate with conditional centering
  $\mathbb E[\cdot\mid\mathcal F_k]$, and the concrete kernel
  $\beta_{k,l}\lesssim(1-\alpha a)^{(n-k)/2}$ is substituted directly.
- [x] Verify/cite Markov concentration for arbitrary initial distribution.
  Done in `src/zeroth_order_rr.typ`: Section 2.2 now identifies the lemma as
  the scalar specialization of Levin et al. (2025, Lemma 11), which is stated
  for arbitrary initial law and gives a tail bound around zero under
  $\pi(g_i)=0$. The bracket-concentration uses in `src/pr_weights.typ` and
  `src/burn_in_transfer.typ` now point back to this exact imported input.
- [x] Rename last-iterate RR notation in Section 2.
  Done in `src/zeroth_order_rr.typ`: the ambiguous
  $\theta_n^{(\mathrm{RR},\alpha)}$ object is now the chapter-local
  $\theta_{n,\mathrm{last}}^{(\mathrm{RR},\alpha)}$, explicitly separated from
  the PR-averaged RR estimator, and the zeroth-order bound is written for
  $\widetilde J_{n,\mathrm{last}}^{(0,\alpha)}$.
- [x] Decide whether to expand `lem:burn-full-startup` further.
  Done in `src/burn_in_transfer.typ`: the lemma is kept as a
  Levin-extension technical proposition, but the proof now explicitly imports
  the product-stability estimate, spells out the bad-event Holder step, and
  shows the geometric convolution used for the $H^{(2)}$ coordinate.
- [x] Clean Hurwitz wording in the introduction.
  Done in `src/introduction.typ`: the opening stability sentence and
  Assumption 2 now state the sign convention through the explicit equivalence
  $-\overline A$ Hurwitz iff $\operatorname{Re}\lambda(\overline A)>0$.
- [x] Narrow the novelty claim to PR-averaged RR plus burn-in transfer.
  Done in `src/introduction.typ`: the problem statement now frames the
  contribution as CLT/Berry--Esseen assembly for the PR-averaged RR statistic
  plus deterministic-start logarithmic burn-in transfer, rather than a broad
  claim about distributional approximation for constant-step LSA.
- [x] Burned-in Poisson remainder has correct $1/\sqrt m$ and
  $\|\varepsilon\|_\infty$ order.
- [x] Final theorem no longer depends on an unnamed Technical Assumption.
- [x] Burn-in $\sqrt m$ and $\sqrt n$ normalizations are separated.

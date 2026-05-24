# Assessment of `review_diploma_proofs_5.md`

Question: study the fifth proof review and decide which comments are real issues in the current thesis text.

Short answer: I agree with the review on the main mathematical and presentation risks. Status update: the action checklist at the end of this note has now been completed in the current source.

## Main verdict

The review is not saying that the whole proof strategy is wrong. The stationary augmented-chain route plus burned-in deterministic-start transfer is coherent. The weak points identified by the review were:

1. Some introductory and theorem-facing wording still sounds stronger than what is proved.
2. The burn-in is repeatedly called "logarithmic", although at the balanced scale $\alpha=c n^{-1/2}$ and $p\simeq \log n$ the current conditions require
   $$
   n_0 \gtrsim (\alpha a)^{-1}\log^2 n
   \asymp n^{1/2}\log^2 n .
   $$
3. Lemma 24, the full-state startup contraction including $H^{(2)}$, is the main load-bearing technical extension. The current proof is much better than the old assumption-style text, but it is still written as a Levin-extension proof sketch rather than a completely self-contained contraction proof.
4. The sign argument in the Kolmogorov distance should be removed. Apply the martingale Berry--Esseen theorem directly to the signed martingale increments.
5. Notation/style issues could distract a reader: `C_A`-like constants, source-file references such as `last_iterate.typ`, missing bibliography, and the placeholder abstract.

## Item-by-item assessment

### 1.1 Fixed-alpha CLT wording

Agree. In `src/introduction.typ` the goals still say:

> A stationary full-window central limit theorem for $\sqrt n(\bar\theta_n^{(\alpha,\mathrm{RR})}-\theta^*)$.

This is too strong if read as fixed-$\alpha$ and centered exactly at $\theta^*$. The stationary chapter proves a statement for the stationary augmented-chain assembly, and the clean thesis-facing interpretation is triangular-array/balanced-scale, where residual RR bias is absorbed into the non-asymptotic remainder.

Recommended fix: rewrite the goal as a stationary augmented-chain Berry--Esseen/CLT assembly, then state that at $\alpha_n=c n^{-1/2}$ it identifies $\Sigma_\infty$ as the covariance target.

### 1.2 Stationary theorem vs deterministic-start theorem

Agree, mostly as an introduction/organization issue. Chapters 4--5 now separate
$S_{n,\mathrm{stat}}^{\mathrm{RR}}(u)$ and the burned-in deterministic-start statistic, and Chapter 5 explicitly adds deterministic transient, random initial-product discrepancy, and startup discrepancy. The introduction still compresses this into "CLT for the PR-averaged RR statistic", which invites the wrong reading.

Recommended fix: in the goal list, say "stationary augmented-chain RR assembly" for the stationary result and "deterministic-start transfer for the burned-in statistic" for the finite-start result.

### 1.3 Lemma 24 / full-state startup contraction

Agree with the review's risk assessment. The current `src/burn_in_transfer.typ` proof now decomposes $H^{(2)}$ after coupling and bounds bad-event, initial-product, and convolution terms. That is the right structure. However, a critical reader can still object that several steps are imported in a compressed way:

- applying product stability conditionally at the random coupling time $T$;
- defining the invariant augmented chain including $H^{(2)}$, not only the Levin $J$-coordinates;
- justifying that the one-trajectory Levin Proposition 9 moment bounds give the exact initial-cost terms needed after coupling;
- tracking the constants in the convolution bound uniformly in $n,n_0,m,\alpha$.

So I would not demote it back to a bare assumption, but I would expand the proof and explicitly state a short "conditional random-product stability" sublemma, plus a finite-past construction of the stationary $H^{(2)}$ coordinate.

### 1.4 "Logarithmic burn-in"

Agree. This is currently misleading in several places. The text can still say that the conditions are logarithmic in the contraction exponent, but not that the burn-in itself is logarithmic in $n$ at the balanced scale.

Preferred wording:

- "mixing-scale burn-in with logarithmic factors";
- "at $\alpha=c n^{-1/2}$, the required burn-in is $O(n^{1/2}\operatorname{polylog} n)$";
- for the final corollary, "logarithmic-square mixing-scale burn-in".

### 1.5 Sign in Kolmogorov distance

Agree. The equality $d_K(X,N)=d_K(-X,N)$ is not a safe general statement when $X$ may have atoms or discontinuities. It is also unnecessary. The martingale theorem can be applied to increments $-\Delta M_l$; boundedness and predictable quadratic variation are unchanged.

Recommended fix: replace the symmetry paragraph in Chapter 4 by the signed-increment argument.

### 2.1 `C_A` collision

Mostly agree. The current source already uses `overline(C)_A := kappa_Q`, not literally `C_A := kappa_Q`, but the notation is still too close to the assumption constant `C_A`. The product `tilde(C)_A := C_A overline(C)_A` is readable to us but poor for a thesis reader.

Recommended fix: rename the local norm-equivalence constant to something like $K_Q$ or $C_Q^{\mathrm{eq}}$.

### 2.2 Matrix power identity

Agree. The identity is correct here because $B_\alpha$ and $B_{2\alpha}$ are polynomials in the same matrix $\bar A$, hence commute. The text should explicitly say this in the zeroth-order chapter and PR-weight chapter.

### 2.3 Powers of $a$ in Lemma 5

Already fixed in the current source. `src/last_iterate.typ` now states the second term as
$$
p^{1/2}t_{\mathrm{mix}}^{3/2}\sqrt{\alpha/a},
$$
so the review's older concern about $\sqrt\alpha/a$ no longer applies.

### 2.4 `Phi + 1 <= C Phi`

Agree. The proof previously used
$$
\|J_n^{(1,w)}\|_{L_p}\le Cw(\Phi(p,w)+1)\le Cw\Phi(p,w).
$$
This silently assumes $\Phi(p,w)$ is bounded below. Safer fix: define
$\Phi_+(p,w)=1+\Phi(p,w)$ and propagate it only in the boundary term. The final balanced rate is unchanged.

Status: fixed in the current source by using $\Phi_+$ in the stationary and
burned-in misadjustment bounds.

### 2.5 Lower variance condition

Mostly already handled. Chapter 4 and Chapter 5 explicitly assume $\sigma^2(u)>0$ and impose finite-$n$ lower variance conditions. The introduction should still say that scalar results are for directions with $\sigma(u)>0$, unless a global positive-definiteness assumption is added.

### 2.6 Burned-in bracket concentration uses $n$ with $m\ge n/2$

Agree, but this is mostly already built into the final theorem. The theorem statements should keep $m\ge n/2$ visible, because it is what lets the $n$-based concentration bound be converted to an $m$-normalized statistic.

### 2.7 Decomposition table before final burn-in theorem

Agree as a clarity improvement. It would make the final theorem easier to audit by separating:

- deterministic transient;
- random initial-product discrepancy;
- augmented-chain startup discrepancy;
- Poisson martingale Berry--Esseen term;
- variance comparison.

Status: fixed in the current source by adding a finite-window assembly table in
the burn-in chapter and a contribution/notation guide in the introduction.

## Other comments

The review was also correct about these presentation issues, now fixed in the
current source:

- the abstract placeholder has been replaced;
- a bibliography section has been added;
- source-file references such as `last_iterate.typ` were replaced by
  section/lemma references;
- the introduction explicitly notes $\pi(\varepsilon)=0$ after the definitions;
- a contribution map and notation table were added.

Some PDF-layout artifacts mentioned in the review should be checked in the rendered PDF. They may be Typst rendering/hyphenation artifacts rather than source mistakes, but the broken bracket issue should be searched for directly in the PDF or via rendered screenshots.

## Recommended fix order

- [x] Rephrase introduction goals: stationary augmented-chain assembly; balanced triangular-array interpretation; deterministic-start burned-in transfer.
- [x] Replace "logarithmic burn-in" by "mixing-scale burn-in with logarithmic factors" and state the balanced-scale order $n_0=O(n^{1/2}\operatorname{polylog}n)$.
- [x] Replace the Kolmogorov sign-symmetry argument by signed martingale increments.
- [x] Expand Lemma 24: conditional product-stability at random $T$, stationary finite-past construction for $H^{(2)}$, and term-by-term constants.
- [x] Rename local `C_A`-style constants in the zeroth-order chapter.
- [x] Add the commutativity comment for $B_\alpha$ and $B_{2\alpha}$.
- [x] Replace $\Phi$ by $\Phi_+=1+\Phi$ where the proof uses $\Phi+1$.
- [x] Add the contribution map and notation table.
- [x] Remove internal file references and replace them by section/lemma references.
- [x] Add bibliography and, when ready, replace the abstract placeholder.

## Bottom line

I agree with the review enough to treat it as actionable. The first fixes should be wording and local correctness fixes, because they are cheap and reduce reader confusion. The only mathematically heavy item is Lemma 24; it is the one that most directly supports the deterministic-start burned-in theorem, so it should be strengthened before the final proof pass.

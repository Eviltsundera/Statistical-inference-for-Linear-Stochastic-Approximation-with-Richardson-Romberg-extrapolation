# Assessment of `review_diploma_proofs_6.md`

Question: does the review identify real remaining problems in the current
thesis sources?

Short answer: I agree with most of the review's mathematical direction, but
not with the implication that all listed issues are still open. Several P0/P1
items were already fixed in the current Typst sources. The remaining important
work is mostly exposition: make imported inputs more self-contained, remove a
few misleading phrases, and smooth notation.

Update 2026-05-24: the remaining exposition fixes listed below have now been
applied in the Typst sources. The note is kept as an audit trail.

## Fix status

- [x] Removed the misleading "logarithmic burn-in window" wording.
- [x] Replaced "Chapters Section" / "Chapter Section" wording in the
  contribution map.
- [x] Replaced "thesis-facing" by academic wording.
- [x] Clarified full-window versus burned-in RR-weight notation in the
  introduction table.
- [x] Added `Imported Inputs and Admissibility Thresholds` in
  `src/pr_weights.typ`.
- [x] Added the stationary indexing convention for
  $(J_{k-1}^{(0,w)}, Z_k)$.
- [x] Added the fixed-$w$ / triangular-array clarification after the
  stationary-limit lemma.
- [x] Added an explicit $m \ge n/2$ conversion sentence in the final burn-in
  theorem proof.
- [x] Added a comment explaining why the burned-in bracket concentration keeps
  a $\sqrt{pn}$ term.
- [x] Polished the manual references section into a more consistent format.

## Already fixed in current sources

- Lemma 28 normalization interval.
  The review is mathematically correct: the upper endpoint must be
  $\sqrt{3/2}$, not $\sqrt{3}/2$. Current
  `src/burn_in_transfer.typ` already has
  $r_{n,n_0}(u) \le \sqrt{3/2}$.

- Section 3, Lemma 5, power of $a$.
  The review is correct: the $U_R$ contribution after multiplication by
  $\alpha$ is $\sqrt{\alpha/a}$, not $\sqrt{\alpha}/a$, unless an extra
  normalization of $a$ is imposed. Current `src/last_iterate.typ` already uses
  $\sqrt{\alpha/a}$.

- Sign-symmetry issue in smoothing.
  The current text handles the minus sign by applying the martingale
  Berry--Esseen theorem to signed increments, so the old Kolmogorov
  sign-symmetry shortcut is no longer present in the relevant assembly proofs.

- Burned-in variance normalization.
  The final normalization transfer already uses the corrected compact interval
  and a generic universal Gaussian-comparison constant.

## I agree: still worth fixing

- Imported inputs are still not self-contained enough.
  Section 4.9 already has a "Cited inputs" block, but the final theorem still
  relies on objects such as $\alpha_*(q,t_{\rm mix})$, $\alpha_{\rm st}(p)$,
  Levin depth-two bounds, startup contraction, Markov concentration, and
  Bolthausen--Fan smoothing without a single consolidated assumptions block.
  This is the main remaining structural issue.

- The phrase "burn-in window is logarithmic" is still present in
  `src/burn_in_transfer.typ`. It should be replaced by "mixing-scale burn-in,
  of order $\alpha^{-1}$ times logarithmic factors" or by the balanced-scale
  statement $n_0 = O(\sqrt n \log^2 n)$.

- The introduction contribution map still says "Derived in Chapters Section 2
  and Section 4" and "Developed in Chapter Section 5". These are simple wording
  mistakes.

- The notation guide should distinguish the full-window RR weight from the
  burned-in weight more explicitly: $Q_l^{\rm RR}$ or
  $\mathcal Q_l^{\rm RR}$ in the stationary chapter versus
  $Q_{l;n_0,n}^{\rm RR}$ / $Q_l^{\rm bRR}$ in the burn-in chapter.

- Lemma 14 / the centered telescoping identity should explicitly state the
  stationarity indexing convention: under the stationary augmented-chain law,
  $(J_{k-1}^{(0,w)}, Z_k)$ has the same law as
  $(J_0^{(0,w)}, Z_1)$, so the centered $\bar\psi_w$ is centered for every
  summand.

- The final burned-in theorem proof already uses $m \ge n/2$ in several
  places, but an explicit line saying that all polynomial prefactors in $n$
  can be converted to the $m$ scale would make the proof easier to audit.

- The bibliography is still too skeletal for a final diploma. It is enough as
  a working references section, but not yet a polished bibliography.

- The phrase "thesis-facing" is still present and should be replaced by
  "final", "main", or "deterministic-start" in academic prose.

## Partially agree / not a proof blocker

- Stationary-limit transfer uniformity as $\alpha \to 0$.
  The review points at a real readability risk, but I do not view it as a
  mathematical contradiction in the current formulation. The lemma is proved
  for each fixed admissible $w$, and the resulting bound displays the
  $w$-dependence through $\Phi_+(p,w)$. The triangular-array substitution
  $w=\alpha_n$ is legitimate if the constants in the zero-start bound are
  independent of $w$ on the admissible range. Still, adding one explicit
  sentence after the lemma would remove the ambiguity.

- The stationary theorem is already separated from the deterministic-start
  estimator in the text. Renaming the theorem to include "comparison
  statistic" would help, but this is now an exposition improvement rather than
  a serious mathematical issue.

- The concentration over indices $2,\dots,n-1$ in the burned-in bracket is not
  wrong: the current text explains that the resulting $\sqrt{pn}$ term is
  acceptable under $m \ge n/2$. A short reminder in the corollary/proof is
  still useful.

- The exact Gaussian-comparison constant $C_\Phi$ is not important. The text
  can safely replace exact-looking constants by a generic universal constant,
  but the bound's validity does not depend on the displayed sharp value.

- Section-prefixed theorem numbering would improve navigation, but changing
  numbering is a formatting decision, not a mathematical fix.

## Recommended fix order

1. Fix the remaining misleading wording:
   "logarithmic burn-in", "Chapters Section", and "thesis-facing".
2. Add a compact "Imported inputs and admissibility thresholds" block that
   states exactly what is imported from Levin et al. and Samsonov et al.
3. Clarify stationary indexing in the $J^{(1)}$ telescoping step.
4. Add one explicit $m \ge n/2$ conversion sentence in the final burned-in
   theorem proof.
5. Clean up RR-weight notation in the introduction table.
6. Polish references into a consistent bibliographic format.

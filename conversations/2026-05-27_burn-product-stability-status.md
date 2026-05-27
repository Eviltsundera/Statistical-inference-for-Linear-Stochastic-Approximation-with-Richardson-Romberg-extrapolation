# Status of `lem:burn-product-stability`

Question: is the lemma
`Technical input: deterministic and conditional product stability`
proved in the thesis text?

Short answer after the 2026-05-27 edit: the deterministic-vector product bound
is imported, and the conditional/adapted-vector form is now proved locally from
that imported bound.

The source statement is in
`src/burn_in_transfer/04_deterministic_transient.typ`, label
`<lem:burn-product-stability>`. The lemma now has three layers:

- the deterministic-vector estimate
  `eq:burn-product-stability-deterministic`, imported from the random-product
  stability machinery of Durmus et al. and the form used in Levin et al.;
- the conditional/adapted-vector estimate
  `eq:burn-product-stability-conditional`, proved by conditioning on the past
  and applying the deterministic-vector bound pointwise;
- the unconditional `L_p` consequences used later.

The appendix also classifies it this way. In
`src/appendix/external_inputs.typ`, the burn-in transfer section says that
`@lem:burn-product-stability` records the deterministic-vector imported
estimate and proves the conditional/adapted-vector version. Then it
distinguishes this input from local extensions/assemblies such as:

- `@lem:finite-past-full-augmented-state`;
- `@lem:burn-random-time-product`;
- `@lem:burn-full-startup`.

So the logical status is now:

$$
\text{deterministic product stability} \quad = \quad \text{imported input},
$$

and

$$
\text{conditional product stability} \quad = \quad
\text{local conditioning consequence}.
$$

The later random-time and full-startup lemmas are local consequences using this
conditional form.

The formerly missing load-bearing part was the conditional version:

$$
\mathbb E^{1/p}
\left[
  \|\Gamma_{s+1:k}^{(w)} W_s\|^p \mid \mathcal G_s
\right]
\le
C_{\rm prod}
e^{-c_{\rm prod}wa(k-s)/p}
\|W_s\|.
$$

That is not just a cosmetic extension: it is used later at coupling times. The
current source now closes this gap by explicitly proving the conditional form
from the deterministic-vector import.

## Likely Import Source

The closest import source is not Huo et al. (2023). Huo cites the same
stability literature, but does not appear to state the product estimate in the
form needed here.

The relevant chain is:

1. Durmus--Moulines--Naumov--Samsonov--Wai (2021), *On the stability of random
   matrix product with Markovian noise: Application to linear stochastic
   approximation and TD learning*. This is the original random-matrix-product
   stability source cited by the later papers.

2. Durmus--Moulines--Naumov--Samsonov (2025), *Finite-time high-probability
   bounds for Polyak--Ruppert averaged iterates of linear stochastic
   approximation*. Levin et al. cite this as the source of the product bound;
   in their notation it is used as “Durmus et al. (2025, Proposition 7)”.

3. Levin et al. (2025), Appendix D.1, Proposition 9. Inside the proof of the
   $H_n^{(2,\alpha)}$ bound, Levin et al. use:

   $$
   \mathbb E_\xi^{1/(2p)}
   \left[
     \|\Gamma_{\ell+1:n}^{(\alpha)}\widetilde A(Z_\ell)\|^{2p}
   \right]
   \le
   2\kappa_Q C_A e^2 d^{1/q}
   e^{-\alpha a(n-\ell)/12}.
   $$

   This is the closest displayed formula to the product-stability estimate used
   in the thesis.

4. Samsonov et al. (2025), Appendix E, Proposition 10, also states a product
   stability result:

   $$
   \mathbb E_\xi^{1/p}
   \left[
     \|\Gamma_{j:n}^{(\alpha)}\|^p
   \right]
   \le
   C_\Gamma d^{1/\log n}
   \exp\left\{
     -\frac a{12}\sum_{k=j}^n \alpha_k
   \right\}.
   $$

   It is proved by a block decomposition and matrix-product concentration, and
   is based on the same Durmus et al. line of results.

So the thesis lemma can be justified as a *working extraction/extension* of
the Durmus--Levin/Samsonov product-stability machinery:

$$
\mathbb E^{1/p}
\left[
  \|\Gamma_{s+1:k}^{(w)}v\|^p \mid Z_s
\right]
\lesssim
e^{-cwa(k-s)}
\|v\|.
$$

The thesis currently states the weaker version with
$e^{-cwa(k-s)/p}$, which is harmless for later convolution estimates but less
sharp than the displayed product bounds in Levin/Samsonov.

The exact conditional/adapted-vector statement

$$
\mathbb E^{1/p}
\left[
  \|\Gamma_{s+1:k}^{(w)}W_s\|^p \mid \mathcal G_s
\right]
\le
C_{\rm prod}e^{-c_{\rm prod}wa(k-s)/p}\|W_s\|
$$

is not literally quoted in Levin or Samsonov. The source now makes the import
rigorous by adding the short derivation:

- the deterministic-vector bound is uniform over the initial base-chain state;
- conditionally on $\mathcal G_s$, the future product law depends on the past
  only through the current base state $Z_s$;
- apply the deterministic-vector estimate pointwise to the realised vector
  $W_s$;
- then take the outer $L_p$ norm to get the unconditional consequence.

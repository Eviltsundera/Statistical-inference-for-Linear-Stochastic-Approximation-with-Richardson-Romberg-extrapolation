# OBM Bias, Richardson–Romberg, and Markov LSA / Constant-Step SA

## Executive summary

This note rewrites the earlier report in plain Markdown and uses **ASCII-style formulas** instead of LaTeX-only display math, so it remains readable even in Markdown viewers that do not render math.

The main conclusion is the following:

- **RR in the step size `alpha`** is aimed at the **bias of the point estimator** (the center of the confidence interval).
- **RR in the batch size `b`** is aimed at the **bias of the long-run variance / covariance estimator** (the standard error part).
- For OBM / Bartlett-type estimators, the literature strongly supports a leading bias term of order `1/b` and a variance term of order `b/n`.
- A more refined expansion of the form

```text
E[sigma_hat^2_{n,b}(u)]
  = sigma^2(u) + c1(u)/b + c2(u)/b^2 + c3(u) * b/n + o(1/b^2 + b/n)
```

is a **very plausible working template**, but in the papers I reviewed the rigorous results are usually stated in a slightly more general way: a first-order window-bias term plus lower-order remainder terms, together with an asymptotic MSE expansion involving `1/b^2` and `b/n`.

So, for your Markov LSA / constant-step SA problem, the cleanest interpretation is:

1. use **RR in `alpha`** to reduce point-estimator bias from `O(alpha)` to `O(alpha^2)`;
2. use **OBM / BM / weighted BM / lugsail** ideas to estimate the long-run covariance;
3. if a `1/b` expansion is available, then **RR in `b`** (or, equivalently, a lugsail-style bias correction) removes the leading truncation bias in the variance estimator.

---

## 1. Basic setup

Suppose `Y_t` is a weakly dependent process with mean `theta`, and let

```text
Ybar_n = (1/n) * sum_{t=1}^n Y_t.
```

A standard central limit theorem is

```text
sqrt(n) * (Ybar_n - theta)  =>  N(0, Sigma),
```

where `Sigma` is the long-run covariance (also called the time-average covariance matrix):

```text
Sigma = sum_{k=-infinity}^{infinity} Cov(Y_0, Y_k).
```

For Markov chain Monte Carlo, time series, and steady-state simulation, a large literature studies estimators of `Sigma` based on lag windows, spectral variance estimators, and batch means. Flegal and Jones (2010) is a foundational reference for BM and OBM consistency in MCMC, and the same variance-estimation logic is what later gets imported into stochastic approximation and SGD inference.  

---

## 2. The OBM estimator in the notation relevant here

Let `theta_1, ..., theta_n` be a trajectory of iterates. For batch size `b`, define overlapping batch averages

```text
theta_bar_{b,t} = (1/b) * sum_{k=t}^{t+b-1} theta_k,
t = 1, ..., n-b+1,
```

and the full average

```text
theta_bar_n = (1/n) * sum_{k=1}^n theta_k.
```

For a direction `u`, the scalar OBM estimator is

```text
sigma_hat^2_{n,b}(u)
  = (b / (n-b+1)) * sum_{t=1}^{n-b+1} [ u^T (theta_bar_{b,t} - theta_bar_n) ]^2.
```

The matrix version is

```text
Sigma_hat_{n,b}
  = (b / (n-b+1)) * sum_{t=1}^{n-b+1}
      (theta_bar_{b,t} - theta_bar_n)(theta_bar_{b,t} - theta_bar_n)^T.
```

The standard asymptotic regime is

```text
b = b_n -> infinity,
b_n / n -> 0.
```

This is the same regime used in classical BM / OBM theory for Markov chains and simulation output analysis.

---

## 3. What the classical MCMC / time-series literature proves

### 3.1 Flegal and Jones (2010): consistency and the `n^(1/3)` picture

Flegal and Jones (2010) study batch means, overlapping batch means, and spectral variance estimators in MCMC. The paper proves:

- strong consistency for batch means and spectral methods under suitable conditions;
- mean-square consistency for BM and OBM;
- an “optimal batch size up to proportionality” point, which is the classical source of the `b ~ n^(1/3)` rule for Bartlett / OBM-type estimators.

Conceptually, this comes from the bias-variance balance

```text
bias ~ 1/b,
variance ~ b/n,
MSE ~ 1/b^2 + b/n.
```

Balancing `1/b^2` against `b/n` yields `b ~ n^(1/3)`.

This paper is the right first citation if you want to justify why OBM has a leading truncation-bias term and why `b/n` naturally appears in the variability term.

### 3.2 Vats, Flegal, and Jones (2018): multivariate spectral estimators

Vats, Flegal, and Jones (2018) move to multivariate spectral variance estimators and prove strong consistency for a broad class of multivariate lag-window estimators of the time-average covariance matrix. The importance of this paper here is not that it gives the exact OBM expansion you want, but that it puts the multivariate covariance-estimation theory on solid ground.

It is useful when your target is not only a scalar variance `sigma^2(u)`, but a full covariance matrix for Wald regions or simultaneous inference.

### 3.3 Vats, Flegal, and Jones (2019): multivariate batch means

Vats, Flegal, and Jones (2019) is especially relevant if you want a direct multivariate batch-means statement. One of the central contributions is strong consistency of the **multivariate batch means estimator**. The paper is framed around stopping rules and multivariate effective sample size, but for your purposes the key message is:

- the BM estimator of the covariance matrix in the Markov chain CLT is itself a rigorous object;
- it can be used to calibrate multivariate confidence regions.

This is one of the cleanest references when you want to say “batch means is not just a heuristic; it is a consistent covariance estimator for the multivariate CLT.”

### 3.4 Liu, Vats, and Flegal (2022): asymptotic MSE for a general class of multivariate batch means estimators

This is one of the closest papers to the expansion you asked about.

Their main contribution is to study **batch size selection** for a **general class of multivariate batch means variance estimators** in MCMC, and they derive the **asymptotic MSE** for that class. The paper explicitly analyzes the MSE tradeoff that leads to optimal batch-size formulas.

The point that matters most for your question is:

```text
MSE = (bias)^2 + variance,
and for Bartlett / BM / OBM-type procedures the two dominant scales are
1/b^2  and  b/n.
```

This is exactly the structure behind the heuristic expansion

```text
E[sigma_hat^2_{n,b}(u)]
  = sigma^2(u) + c1(u)/b + c2(u)/b^2 + c3(u) * b/n + ...
```

Even when the paper is not written in this exact scalar-direction notation, it supports the same asymptotic logic:

- the **first-order bias** is of window / truncation type;
- its square contributes the `1/b^2` term in MSE;
- the stochastic fluctuation contributes the `b/n` term.

This is probably the single most relevant paper if you want a rigorous paper to cite for the phrase “the asymptotic MSE has the structure `const/b^2 + const * b/n`.”

### 3.5 Vats and Flegal (2022): lugsail lag windows

This paper is crucial for the bias-correction story.

Its central message is that standard lag-window estimators of the time-average covariance matrix usually have **negative first-order bias** under positive correlation. The authors introduce **lugsail lag windows** precisely to offset this bias. They study both spectral variance estimators and weighted batch means estimators.

At a high level, the paper says:

- the usual lag-window / truncation estimator has a first-order bias term of order `1/b^q`;
- for Bartlett-type procedures, `q = 1`;
- by modifying the window into a lugsail version, one can force the leading bias coefficient to be zero (or even positive).

This is exactly the same philosophy as Richardson–Romberg in `b`.

If your baseline estimator has

```text
E[Sigma_hat_b] = Sigma + C1/b + C2/b^2 + lower-order terms,
```

then a lugsail or RR correction is designed to kill the `C1/b` term.

So, if someone asks whether there is literature showing that the leading `1/b` bias can be removed without changing the fundamental estimator family, this is the paper to cite.

### 3.6 Ng and Perron (1996): exact finite-sample error at frequency zero

Ng and Perron (1996) is a very important supporting reference from time series / econometrics.

They derive **exact finite-sample bias and variance** formulas for a broad class of spectral density estimators at frequency zero. Their main lesson is that finite-sample behavior can differ substantially from what asymptotic theory suggests, and that the error depends on whether the mean is known or estimated.

This is directly relevant because the finite-sample correction induced by estimating the mean is of the same flavor as the `b/n` term that appears in BM / OBM asymptotics.

So this paper is extremely valuable for explaining **where** a finite-sample term like `c3 * b/n` comes from:

- estimated mean versus known mean,
- edge effects,
- finite bandwidth,
- and, more generally, the difference between asymptotic and exact finite-sample error.

### 3.7 Singh, Shukla, and Vats (2025): direct bridge to SGD / Markovian inference

This is the most directly relevant modern paper for the stochastic-approximation / SGD side.

They study inference for averaged SGD, emphasize the Markovian nature of the SGD process, and show that a **batch-means estimator of the asymptotic covariance matrix** can be constructed using **equal batch sizes**. A key point in the paper is that their batching method allows **bias correction of the variance estimate** without extra memory.

So, if you want a direct bridge from the MCMC / simulation literature to Markovian stochastic approximation or SGD, this paper is exactly that bridge.

It does not by itself prove your exact scalar OBM expansion in the notation `sigma_hat^2_{n,b}(u)`, but it confirms that:

- batch-means covariance estimation is the right object for inference in Markovian SGD / SA;
- bias correction of that variance estimator is practically useful;
- equal-batch constructions can be memory efficient.

---

## 4. The most useful synthesis for your question

The literature most strongly supports the following hierarchy.

### 4.1 First-order template

For OBM / Bartlett-type estimators:

```text
E[Sigma_hat_{n,b}] = Sigma + C1/b + lower-order terms.
```

or in a scalar projection:

```text
E[sigma_hat^2_{n,b}(u)] = sigma^2(u) + c1(u)/b + lower-order terms.
```

This is the level at which the lugsail paper is naturally phrased: the leading bias is of order `1/b` for Bartlett-type constructions.

### 4.2 MSE template

The asymptotic MSE behaves like

```text
MSE(sigma_hat^2_{n,b}(u))
  = const / b^2 + const * b/n + lower-order terms.
```

This is the level at which Flegal–Jones and especially Liu–Vats–Flegal are naturally interpreted.

### 4.3 Refined expansion

A more refined expansion of the form

```text
E[sigma_hat^2_{n,b}(u)]
  = sigma^2(u) + c1(u)/b + c2(u)/b^2 + c3(u) * b/n + o(1/b^2 + b/n)
```

is a very reasonable **working asymptotic expansion** for OBM / Bartlett-type procedures, but I would state it carefully:

- it is well aligned with the existing BM / OBM / spectral / lag-window theory;
- the literature I reviewed most often proves the first-order bias term and the MSE structure rather than this exact scalar expansion in one theorem.

So, in a paper or note, I would present it as:

> a natural refined asymptotic template motivated by existing MCMC and spectral-density theory,

rather than claiming that exactly this formula is already established for Markov LSA in the same notation.

---

## 5. What RR in batch size `b` does

Suppose the refined expansion holds:

```text
E[Sigma_hat_b]
  = Sigma + C1/b + C2/b^2 + C3 * b/n + o(1/b^2 + b/n).
```

Now define the RR correction in `b` by

```text
Sigma_hat_b^RR = 2 * Sigma_hat_{2b} - Sigma_hat_b.
```

Then

```text
E[Sigma_hat_b^RR]
  = Sigma - C2/(2 b^2) + 3 C3 * b/n + o(1/b^2 + b/n).
```

So RR in `b` removes the `1/b` term, but it does **not** remove the `b/n` finite-sample term.

That is the cleanest summary of the bias-reduction mechanism.

This is also why the lugsail idea is so close in spirit to Richardson–Romberg:

- both are trying to remove the leading truncation / window bias;
- neither is a universal cure for every source of finite-sample error.

---

## 6. What RR in step size `alpha` does

Now switch to constant-step stochastic approximation or SGD.

If the point estimator has an expansion

```text
E[theta_hat^(alpha)]
  = theta_star + d1 * alpha + d2 * alpha^2 + ...
```

then the Richardson–Romberg combination

```text
theta_hat^RR = 2 * theta_hat^(alpha) - theta_hat^(2 alpha)
```

satisfies

```text
E[theta_hat^RR] = theta_star + O(alpha^2).
```

So:

- **RR in `alpha`** fixes the **center bias**;
- **RR in `b`** fixes the **variance-estimator bias**.

This separation is conceptually very important. These are different bias sources and they should not be conflated.

---

## 7. What this means for Markov LSA

For Markov LSA / constant-step SA, the natural conclusion is:

### 7.1 What is well supported

The following strategy is well supported by the combined MCMC + SGD literature:

1. Construct the point estimator using averaging (for example, Polyak–Ruppert or averaged constant-step iterates).
2. Estimate the asymptotic covariance of the averaged estimator using BM / OBM / weighted BM.
3. If needed, use a bias-corrected covariance estimator (lugsail or RR-in-`b` style).
4. Use the resulting covariance estimate in a standard Wald / CLT confidence interval.

### 7.2 What is still a transfer rather than a theorem

The exact scalar-direction expansion

```text
E[sigma_hat^2_{n,b}(u)]
  = sigma^2(u) + c1(u)/b + c2(u)/b^2 + c3(u) * b/n + o(1/b^2 + b/n)
```

for the specific Markov LSA recursion you have in mind is best viewed as a **transfer principle / conjectural template** unless you prove it directly in your SA setting.

The MCMC and spectral-density papers make this template very plausible, but they do not automatically give you a theorem for every stochastic approximation recursion.

---

## 8. Confidence intervals

For a scalar projection `u^T theta_star`, the standard CI after RR in `alpha` is

```text
u^T theta_hat^RR  +/-  z_{1-delta/2} * sqrt( u^T Sigma_hat u / n ).
```

Here `Sigma_hat` can be:

- OBM / BM computed on the RR path;
- a lugsail-corrected covariance estimator;
- or an RR-in-`b` corrected estimator.

For a multivariate confidence region:

```text
{ vartheta :
    n * (theta_hat^RR - vartheta)^T * (Sigma_hat)^(-1) * (theta_hat^RR - vartheta)
    <= chi^2_{d,1-delta}
}.
```

A practical caveat:

- if you form `Sigma_hat_b^RR = 2*Sigma_hat_{2b} - Sigma_hat_b` directly, it may fail to be positive semidefinite in finite samples;
- in that case, a PSD projection may be needed before inversion.

This is one reason practitioners often like lugsail-style corrections or compute the batch estimator directly on the bias-corrected path.

---

## 9. Bottom line

Here is the sharpest possible bottom line.

### If your question is:

> Is there literature supporting a bias decomposition for OBM / batch means of the form  
> `leading 1/b bias + finite-sample b/n term`, and hence supporting RR bias reduction in `b`?

then the answer is **yes**.

### If your question is:

> Is there already a theorem in the exact Markov LSA notation  
> `E[sigma_hat^2_{n,b}(u)] = sigma^2(u) + c1(u)/b + c2(u)/b^2 + c3(u)b/n + o(...)`?

then the answer is **not exactly in the papers I reviewed**.

What the literature gives very clearly is:

- first-order truncation / window bias;
- `1/b^2 + b/n` asymptotic MSE structure;
- direct bias-correction mechanisms (lugsail / linear combinations);
- and modern transfer of these ideas to SGD / Markovian inference.

That is enough to justify the following practical statement:

> For Markov LSA or constant-step SA, RR in `alpha` should be used to reduce point-estimator bias, while RR or lugsail in `b` should be used to reduce long-run variance estimator bias.

---

## 10. References and links

Below I list the papers mentioned above, with article pages and PDF links where available.

### 1. Flegal, J. M. and Jones, G. L. (2010)
**Batch means and spectral variance estimators in Markov chain Monte Carlo**

- Article / DOI: [10.1214/09-AOS735](https://doi.org/10.1214/09-AOS735)
- Article page: [Experts@Minnesota entry](https://experts.umn.edu/en/publications/batch-means-and-spectral-variance-estimators-in-markov-chain-mont/)
- Free PDF (arXiv preprint): [arXiv PDF](https://arxiv.org/pdf/0811.1729.pdf)

### 2. Vats, D., Flegal, J. M., and Jones, G. L. (2018)
**Strong consistency of multivariate spectral variance estimators in Markov chain Monte Carlo**

- Article / DOI: [10.3150/16-BEJ914](https://doi.org/10.3150/16-BEJ914)
- Article page: [Experts@Minnesota entry](https://experts.umn.edu/en/publications/strong-consistency-of-multivariate-spectral-variance-estimators-i/)

### 3. Vats, D., Flegal, J. M., and Jones, G. L. (2019)
**Multivariate output analysis for Markov chain Monte Carlo**

- Article / DOI: [10.1093/biomet/asz002](https://doi.org/10.1093/biomet/asz002)
- Article page: [Biometrika page](https://academic.oup.com/biomet/article/106/2/321/5426969)
- Free PDF (preprint): [arXiv PDF](https://arxiv.org/pdf/1512.07713.pdf)

### 4. Liu, Y., Vats, D., and Flegal, J. M. (2022)
**Batch Size Selection for Variance Estimators in MCMC**

- Article / DOI: [10.1007/s11009-020-09841-7](https://doi.org/10.1007/s11009-020-09841-7)
- Article page: [Ideas / RePEc entry](https://ideas.repec.org/a/spr/metcap/v24y2022i1d10.1007_s11009-020-09841-7.html)
- Free PDF (preprint): [arXiv PDF](https://arxiv.org/pdf/1804.05975.pdf)

### 5. Vats, D. and Flegal, J. M. (2022)
**Lugsail lag windows for estimating time-average covariance matrices**

- Article / DOI: [10.1093/biomet/asab049](https://doi.org/10.1093/biomet/asab049)
- Article page: [Biometrika page](https://academic.oup.com/biomet/article/109/3/735/6395353)
- PDF from publisher: [OUP PDF](https://academic.oup.com/biomet/article-pdf/109/3/735/45512212/asab049.pdf)
- Free PDF (preprint): [arXiv PDF](https://arxiv.org/pdf/1809.04541.pdf)

### 6. Ng, S. and Perron, P. (1996)
**The Exact Error in Estimating the Spectral Density at the Origin**

- Article page / abstract: [EconPapers entry](https://econpapers.repec.org/RePEc:bla:jtsera:v:17:y:1996:i:4:p:379-408)
- Direct PDF: [Columbia-hosted PDF](https://www.columbia.edu/~sn2294/pub/jts-96.pdf)

### 7. Singh, R., Shukla, A., and Vats, D. (2025)
**On the Utility of Equal Batch Sizes for Inference in Stochastic Gradient Descent**

- Article page: [JMLR article page](https://www.jmlr.org/papers/v26/24-0094.html)
- Direct PDF: [JMLR PDF](https://www.jmlr.org/papers/volume26/24-0094/24-0094.pdf)

### 8. Optional additional preprint connection
**Weighted batch means estimators in Markov chain Monte Carlo** (Liu and Flegal, preprint)

- Summary / entry: [Emergent Mind entry](https://www.emergentmind.com/papers/1805.08283)
- Likely preprint PDF location: [arXiv PDF](https://arxiv.org/pdf/1805.08283.pdf)

---

## 11. Suggested citation strategy for your own write-up

If you write this up for a paper, talk, or note, a clean citation strategy is:

- cite **Flegal and Jones (2010)** for BM / OBM consistency and the classical bias-variance tradeoff;
- cite **Liu, Vats, and Flegal (2022)** for asymptotic MSE and batch-size selection;
- cite **Vats and Flegal (2022)** for first-order bias correction via lugsail windows;
- cite **Ng and Perron (1996)** for exact finite-sample insight on spectral-density estimation at frequency zero;
- cite **Singh, Shukla, and Vats (2025)** for the SGD / ASGD inference bridge.

That gives a very coherent literature chain from classical MCMC variance estimation to modern stochastic-approximation inference.

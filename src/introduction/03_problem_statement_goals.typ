#import "../defs.typ": *

== Problem statement and goals

The high-order moment bounds for the PR-averaged RR iterate
$overline(theta)_n^((alpha, "RR"))$ established in Levin et al. (2025) show
that the leading error term scales as
$sqrt("Tr" Sigma_epsilon.alt^(("M"))) dot n^(-1\/2)$, where
$Sigma_epsilon.alt^(("M"))$ is the Markovian noise covariance defined below in
the key quantities subsection. This is the usual parametric $n^(-1\/2)$
benchmark for averaged LSA; this thesis does not prove a separate
Hájek--Le Cam or minimax lower bound.
Berry--Esseen type bounds and bootstrap inference procedures for the
_standard_ Polyak--Ruppert average $overline(theta)_n$ (without extrapolation)
under Markovian noise have been obtained in Samsonov, Sheshukova, Moulines,
and Naumov (2025).

These results do not directly give the distributional approximation needed for
the PR-averaged Richardson--Romberg statistic under Markovian noise. Two
distinctions are essential here. First, the stationary theorem is proved for an
augmented-chain scalar statistic, where the martingale noise, Poisson boundary
term, and Richardson--Romberg misadjustment are controlled separately. Second,
the deterministic-start estimator requires an additional burn-in transfer,
because the deterministic transient and startup remainders are absent from the
stationary augmented-chain theorem.

The main goal of this work is therefore to build a non-asymptotic
distributional approximation for scalar projections of the PR-averaged
Richardson--Romberg statistic. The stationary result should be read as an
$n_0 = 0$ augmented-chain theorem, not as a fixed-$alpha$ central limit theorem
centered exactly at $theta^*$. Its final deterministic-start consequence is obtained
at the balanced triangular-array scale $alpha_n = c n^(-1\/2)$, where the
residual RR terms are absorbed into explicit remainders and the covariance
target is $Sigma_infinity$.

Concretely, we establish:

+ A stationary full-window augmented-chain Berry--Esseen assembly for scalar
  RR statistics, including the martingale approximation, predictable-variance
  comparison, Poisson remainder, and stationary RR misadjustment.
+ A balanced triangular-array specialization, in particular for
  $alpha_n = c n^(-1\/2)$, which identifies $Sigma_infinity$ as the covariance
  target and gives the resulting stationary $n_0 = 0$ CLT interpretation.
+ A deterministic-start transfer theorem under mixing-scale burn-in conditions
  with logarithmic factors, yielding the corresponding balanced-scale bound for
  the main burned-in statistic.

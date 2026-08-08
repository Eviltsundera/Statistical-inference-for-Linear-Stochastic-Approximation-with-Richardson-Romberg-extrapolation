#import "../defs.typ": *

== Problem statement and goals

The high-order moment bounds for the PR-averaged RR iterate
$overline(theta)_n^((alpha, "RR"))$ established in Levin et al. (2025) show
that the leading error term scales as
$sqrt("Tr" Sigma_epsilon.alt^(("M"))) dot n^(-1\/2)$, where
$Sigma_epsilon.alt^(("M"))$ is the Markovian noise covariance recorded in
@sec:key-quantities. This is the usual parametric $n^(-1\/2)$
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
  comparison, Poisson remainder, and stationary RR misadjustment. The
  controlled comparison statistic is
  $
  S_(n, "stat")^("RR")(u)
    = - frac(u^top M_n^("RR"), sqrt(n))
      + u^top cal(R)_(n, "stat")^("RR"),
  quad
  cal(R)_(n, "stat")^("RR")
    = D_(2,n)^("RR") + R_n^("mis,RR").
  $
  Its finite-window variance proxy is
  $
  sigma_n^(2, "RR")(u)
    = u^top lr((
        frac(1, n) sum_(l=2)^(n-1)
          cal(Q)_l^("RR") Sigma_epsilon.alt^(("M"))
          (cal(Q)_l^("RR"))^top
      )) u,
  $
  and the theorem has the schematic form
  $
  d_K lr((
    frac(S_(n, "stat")^("RR")(u), sigma_n^("RR")(u)),
    cal(N)(0,1)
  ))
    <= frac("polylog"(n), n^(1\/4))
       + frac(||u^top cal(R)_(n, "stat")^("RR")||_(L_p),
              sigma_n^("RR")(u)).
  $
  In the balanced triangular-array specialization $alpha_n = c n^(-1\/2)$,
  the same stationary theorem identifies
  $
  Sigma_infinity
    = overline(A)^(-1) Sigma_epsilon.alt^(("M")) overline(A)^(-top),
  quad
  sigma^2(u) = u^top Sigma_infinity u,
  $
  as the covariance target and gives, for every fixed non-degenerate direction
  $u$, the stationary $n_0 = 0$ CLT interpretation
  $
  d_K lr((
    frac(S_(n, "stat")^("RR")(u), sigma(u)),
    cal(N)(0,1)
  ))
    <= C(u,c) frac("polylog"(n), n^(1\/4)).
  $
+ A deterministic-start transfer theorem under mixing-scale burn-in conditions
  with logarithmic factors, yielding the corresponding balanced-scale bound for
  the main burned-in statistic. Writing $m = n - n_0$ and
  $
  overline(theta)_(n,n_0)^(("RR", alpha))
    = 2 overline(theta)_(n,n_0)^((alpha))
      - overline(theta)_(n,n_0)^((2 alpha)),
  quad
  T_(n,n_0)^("RR")(u)
    = sqrt(m) thin u^top lr((
        overline(theta)_(n,n_0)^(("RR", alpha)) - theta^*
      )),
  $
  the transfer gives
  $
  d_K lr((
    frac(T_(n,n_0)^("RR")(u), sigma(u)),
    cal(N)(0,1)
  ))
    <= C(u,c,theta_0) frac("polylog"(n), n^(1\/4))
  $
  at $alpha = c n^(-1\/2)$ when
  $
  n_0 asymp (alpha a)^(-1) log^2 n.
  $
  Under the same burn-in window, the final $sqrt(n)$ statistic satisfies the
  same polynomial rate up to logarithmic factors.

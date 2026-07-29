# Уточнение гипотезы для MSE OBM / OBM-LW

## Вопрос

После эксперимента по fixed-\(\eta\) MSE для OBM и OBM-LW оказалось, что
простая теория хорошо предсказывает slope на правой ветке, но не предсказывает
эмпирически лучший \(\eta\). Можно ли уточнить теоретическую гипотезу?

## Короткий ответ

Да. Текущий результат лучше интерпретировать не как опровержение классической
асимптотики, а как признак выраженного pre-asymptotic режима. Двухчленная модель

$$
\mathrm{MSE}_{OBM}(T,b) \approx C_1 b^{-2} + C_2 b/T
$$

и lugsail-модель

$$
\mathrm{MSE}_{LW}(T,b) \approx C_1 b^{-4} + C_2 b/T
$$

описывают только asymptotic window, где \(b\to\infty\), \(b/T\to 0\), и
truncation bias уже действительно ведет себя как \(1/b\) или \(1/b^2\).
В наших finite-\(T\) экспериментах малые \(b=T^\eta\) остаются слишком малыми:
например при \(\eta=0.2\) и \(T\le 10^6\) блоки имеют размер порядка
\(6\)--\(16\). Это еще не зона, где Bartlett-window bias обязан выглядеть как
чистый \(c_1/b\).

Поэтому рабочую гипотезу стоит заменить на **многорежимную MSE-модель**.

## Уточненная гипотеза

Для OBM разумнее писать

$$
\mathrm{MSE}_{OBM}(T,b)
\approx
\left[
  B_{sat}(b,T)
  + \frac{c_1}{b}
  + \frac{c_2}{b^2}
  + c_3\frac{b}{T}
  + R_{SA}(T)
\right]^2
+ v_1\frac{b}{T}.
$$

Для OBM-LW с параметром \(\lambda\):

$$
\mathrm{MSE}_{LW,\lambda}(T,b)
\approx
\left[
  B_{sat,\lambda}(b,T)
  + d_{2,\lambda}\frac{1}{b^2}
  + d_{3,\lambda}\frac{b}{T}
  + R_{SA}(T)
\right]^2
+ v_\lambda\frac{b}{T}.
$$

Здесь:

- \(c_1/b\) -- классический Bartlett truncation bias;
- \(d_{2,\lambda}/b^2\) -- residual truncation bias после lugsail cancellation;
- \(c_3 b/T\), \(d_{3,\lambda} b/T\) -- centering / unknown-mean finite-sample
  correction;
- \(v_\lambda b/T\) -- stochastic variance estimator fluctuation;
- \(R_{SA}(T)\) -- residual bias/noise from finite-time SA trajectory and PR
  transient;
- \(B_{sat}(b,T)\) -- pre-asymptotic saturation/floor term for small \(b\).

The essential new term is \(B_{sat}\). It is not meant as a final theorem
statement; it is a phenomenological term saying: before \(b\) exceeds the
correlation / relaxation scale of the iterate sequence, the window estimator
does not yet behave like its formal \(1/b\) expansion.

## Why this explains the experiment

The fixed-\(\eta\) experiment found:

| Method | best empirical \(\eta\) | empirical rate | theory rate at that \(\eta\) |
|---|---:|---:|---:|
| OBM | 0.600 | 0.3949 | 0.4000 |
| OBM-LW \(\lambda=2\) | 0.450 | 0.5489 | 0.5500 |
| OBM-LW \(\lambda=3\) | 0.425 | 0.5666 | 0.5750 |
| OBM-LW \(\lambda=4\) | 0.425 | 0.5684 | 0.5750 |

These points lie on the **right branch** of the simple asymptotic formulas:

$$
r_{OBM}(\eta)=\min(2\eta,1-\eta),
$$

and

$$
r_{LW}(\eta)=\min(4\eta,1-\eta).
$$

On the right branch, the dominating term is \(b/T=T^{-(1-\eta)}\). The
observed rates match \(1-\eta\) almost exactly. So the variance branch is
working.

What fails is the left branch. For small \(\eta\), the theory predicts that the
bias term \(b^{-2}\) or \(b^{-4}\) should decay as \(T^{-2\eta}\) or
\(T^{-4\eta}\). Empirically, in the accessible range \(T\le 10^6\), those small
blocks are too short and the MSE does not follow that clean decay. That is why
the empirical optimum is pushed to larger \(\eta\): the experiment chooses
large enough blocks to exit the saturation regime, then lands on the
variance-dominated right branch.

## More precise regime hypothesis

A useful way to state the refined hypothesis is to introduce a transition scale
\(b_{0}(T)\), or approximately a constant \(b_0\), such that:

1. **Small-block regime:** \(b \lesssim b_0\).

   The estimator underestimates long-run variance severely. Bias is saturated,
   not proportional to \(1/b\). MSE may decay slowly or even appear to grow
   over moderate \(T\) when \(b=T^\eta\) remains tiny.

2. **Asymptotic truncation regime:** \(b_0 \ll b \ll T\).

   The classical expansions become visible:

   $$
   \mathrm{Bias}_{OBM}(b,T) \approx c_1/b + c_3 b/T,
   $$

   $$
   \mathrm{Bias}_{LW}(b,T) \approx d_2/b^2 + d_3 b/T.
   $$

3. **Large-block / finite-sample regime:** \(b/T\) is not very small.

   The centering correction and variance \(b/T\) dominate. In this regime,
   MSE decays like \(T^{-(1-\eta)}\), which is exactly what the fixed-\(\eta\)
   experiment saw near the empirical best points.

Under this hypothesis, the classical optimum \(\eta=1/3\) for OBM and
\(\eta=1/5\) for OBM-LW is a true asymptotic prediction only after the
small-block threshold becomes negligible relative to \(T^\eta\). Our current
\(T\)-range has not reached that threshold for the smaller theoretical
\(\eta\)'s.

## Experimentally testable refinements

The next experiments should separate the terms, not only fit the total MSE.

1. **Bias-rate plot.**

   For each fixed \(\eta\), fit

   $$
   |\mathbb E\hat\sigma^2-\sigma^2| \approx C T^{-q(\eta)}.
   $$

   Compare \(q(\eta)\) against \(\eta\) for OBM and \(2\eta\) for OBM-LW.
   If the left branch is not visible, \(q(\eta)\) will be far below the
   predicted line for small \(\eta\).

2. **Variance-rate plot.**

   Fit

   $$
   \mathrm{Var}(\hat\sigma^2) \approx C T^{-s(\eta)}.
   $$

   The prediction is \(s(\eta)=1-\eta\). This is likely already working.

3. **Threshold diagnostic.**

   Plot MSE or bias against the actual block size \(b\), pooling across \(T\),
   and mark the region where the bias curve first becomes approximately a
   straight line in log-log coordinates. That identifies the empirical
   \(b_0\).

4. **Model comparison.**

   Fit several nested models:

   $$
   M_1: \quad \mathrm{MSE}=A b^{-2p}+V b/T,
   $$

   $$
   M_2: \quad \mathrm{MSE}=(A b^{-p}+C b/T)^2+V b/T,
   $$

   $$
   M_3: \quad \mathrm{MSE}=(A(b+b_0)^{-p}+C b/T+D T^{-\rho})^2+V b/T.
   $$

   Here \(p=1\) for OBM and \(p=2\) for OBM-LW. If \(M_3\) gives a much better
   log-MSE fit with stable \(b_0\), that supports the pre-asymptotic threshold
   interpretation.

## Thesis-facing wording

A cautious thesis statement would be:

> Classical OBM/Bartlett theory predicts \(b^{-2}+b/T\) MSE for the standard
> estimator and \(b^{-4}+b/T\) after lugsail cancellation of the first-order
> bias. Our experiments confirm the \(b/T\) variance-dominated branch very
> clearly. However, for the available \(T\)-range, the small-block truncation
> branch is not yet in its asymptotic regime; finite-window saturation and
> centering/transient effects shift the empirical optimum to larger block
> exponents.

This preserves the literature-backed asymptotic theory while accurately
describing the finite-sample behavior observed in the experiments.

## Unresolved gaps

- Need to verify whether \(B_{sat}\) is mostly an OBM small-window effect, a PR
  finite-time effect, or a Markov mixing/autocorrelation-length effect.
- Need separate bias and variance rate fits for the latest 2026-06-09 run.
- Need to check whether increasing \(T\) beyond \(10^6\), even with fewer
  trajectories, begins to reveal the left branch for OBM-LW near \(\eta=0.2\).

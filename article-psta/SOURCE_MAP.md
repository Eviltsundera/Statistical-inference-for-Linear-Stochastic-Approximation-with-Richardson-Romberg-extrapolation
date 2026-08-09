# Карта источников статьи

Центральный результат статьи — неасимптотическая Gaussian/Berry–Esseen
аппроксимация для RR-оценки с PR-усреднением после разогрева. Доказательство
отделяет ведущий пуассоновский мартингал от нелинейного Lp-остатка.

## Теория

- Модель и условия: `diploma_typst/src/introduction/05_setting_assumptions.typ`.
- Разложение смещения и точная атрибуция Levin et al.:
  `diploma_typst/src/introduction/01_bias_constant_stepsize.typ` и
  `diploma_typst/src/appendix/external_inputs.typ`.
- PR/RR-веса после разогрева и их оценки:
  `diploma_typst/src/burn_in_transfer/02_burned_in_deterministic_weights.typ`
  и `03_closed_forms_and_weight_bounds.typ`.
- Сравнение детерминированной ковариационной прокси:
  `diploma_typst/src/burn_in_transfer/05_variance_proxy.typ`.
- Пуассоновское разложение и мартингальная аппроксимация:
  `diploma_typst/src/burn_in_transfer/06_poisson_martingale_approximation.typ`
  и `10_martingale_berry_esseen.typ`.
- Master smoothing assembly и balanced-scale theorem:
  `diploma_typst/src/burn_in_transfer/11_finite_window_smoothing_assembly.typ`
  и `12_balanced_burn_in_berry_esseen.typ`.
- Прямые внешние moment inputs:
  `diploma_typst/src/appendix/external_inputs.typ`.

В `rr_gaussian.tex` атрибуция разделена следующим образом:

- стационарные оценки отдельных компонент берутся из Proposition 2,
  Corollary 6, Propositions 8–9 и Lemma 8 расширенной версии Levin et al.;
  Lemma 8 используется для граничных значений `J^(1)` через конечнопрошлый
  предел и лемму Фату;
- концентрация случайной предсказуемой скобки выводится локально из
  неоднородного марковского moment inequality, Lemma 11 Levin et al.;
- инвариантный закон `J`-координат берётся из Corollary 4 Levin et al., а
  стационарная `H^(2)`-координата строится локально конечнопрошлым пределом;
  перенос на произвольный старт доказывается на едином блочном sticky-coupling,
  используя покомпонентный расчёт Proposition 5 Levin et al. и условную
  устойчивость случайных произведений из Proposition 7 Durmus et al.;
- распределительный шаг использует общую мартингальную Berry–Esseen-оценку,
  Lemma 21 Samsonov et al.;
- Gaussian approximation последней итерации в статью не включена.

## Эксперименты

- Генератор цепей и LSA-задач:
  `code/lsa_inference/markov_chain.py` и
  `code/lsa_inference/lsa_problem.py`.
- CDF/KS-диагностика на balanced scale:
  `code/run_rr_cdf_experiment.py`,
  `reports/2026-06-02_rr_cdf_dense_theory_proxy.md` и
  `code/results/cdf/`.
- Рисунок главного эксперимента строится скриптом
  `article-psta/figures/plot_rr_main_experiment.py` и сохраняется как
  `article-psta/figures/rr_main_experiment.pdf`.
- Sweep соседних пар `(2 alpha, alpha)`:
  `reports/2026-05-26_theory_rr_alpha_sweep.md` и
  `code/results/theory_rr_sweep/`.
- Сравнение oracle и OBM:
  `reports/2026-05-26_oracle_variance_rr.md`,
  `reports/2026-05-26_rr_coverage_T_sweep.md` и
  `code/results/oracle_variance/`.
- Неизменяемый публичный срез сохранённых данных:
  <https://github.com/Eviltsundera/Statistical-inference-for-Linear-Stochastic-Approximation-with-Richardson-Romberg-extrapolation/tree/1d7e4d7914c1b0f4c0c5c840418e250e21ae75b5/code/results>.

Сюжет lugsail/OBM-RR не включён: он относится к другой экстраполяции — по
размеру блока оценки дисперсии, а не по шагу точечной оценки.

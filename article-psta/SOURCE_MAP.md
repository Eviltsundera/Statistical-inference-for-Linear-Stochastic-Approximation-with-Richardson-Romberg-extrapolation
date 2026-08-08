# Карта источников статьи

Статья сознательно использует один компактный теоретический сюжет: RR по шагу
уменьшает смещение, а при общей траектории сохраняет ковариационную цель
ведущего линейного члена.

## Теория

- Модель и условия: `diploma_typst/src/introduction/05_setting_assumptions.typ`.
- Разложение стационарного смещения и точная атрибуция Levin et al.:
  `diploma_typst/src/introduction/01_bias_constant_stepsize.typ` и
  `diploma_typst/src/appendix/external_inputs.typ`.
- PR- и RR-веса:
  `diploma_typst/src/pr_weights/02_error_decomposition_and_rr_weight.typ` и
  `diploma_typst/src/pr_weights/03_closed_form_identities.typ`.
- Поточечная оценка весов:
  `diploma_typst/src/pr_weights/04_pointwise_bounds.typ`.
- Суммированные оценки:
  `diploma_typst/src/pr_weights/05_summed_bounds.typ`.
- Сравнение ковариационной прокси с `Sigma_infinity`:
  `diploma_typst/src/pr_weights/06_variance_comparison.typ`.

Полная теорема Берри–Эссеена, depth-two misadjustment и перенос с разогрева в
статью не включены. Они требуют существенно более длинного доказательства;
кроме того, внутренний аудит диплома отмечает как наиболее нагруженное место
full-state startup contraction для остатка `H^(2)`.

## Эксперименты

- CDF/KS-диагностика на balanced scale:
  `reports/2026-06-02_rr_cdf_dense_theory_proxy.md`,
  `code/results/cdf/rr_cdf_dense_M10000_T20k_1M_summary.csv` и
  `code/results/cdf/rr_cdf_dense_M10000_T20k_1M_z.csv`.
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

Сюжет lugsail/OBM-RR не включён: он относится к другой экстраполяции — по
размеру блока оценки дисперсии, а не по шагу точечной оценки.


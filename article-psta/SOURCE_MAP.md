# Карта источников статьи

Центральный результат статьи — неасимптотическая Gaussian/Berry–Esseen
аппроксимация для RR-оценки с PR-усреднением после разогрева. Доказательство
отделяет ведущий пуассоновский мартингал от нелинейного Lp-остатка.

## Теория

- Модель и условия: `diploma_typst/src/introduction/05_setting_assumptions.typ`.
- Разложение смещения и точная атрибуция Levin et al.:
  `diploma_typst/src/introduction/01_bias_constant_stepsize.typ` и
  `diploma_typst/src/appendix/external_inputs.typ`.
- Разогрев $n_0=n/2$, разложение RR-среднего и $L_p$-оценка остатка:
  Theorem 2, equations (25)--(27), точная оценка (87) и Appendix D.1
  расширенной версии Levin et al.
- Пуассоновское разложение ведущей аддитивной суммы и мартингальная
  Berry--Esseen-сборка адаптированы из
  `diploma_typst/src/burn_in_transfer/06_poisson_martingale_approximation.typ`
  и `10_martingale_berry_esseen.typ`.

В `rr_gaussian.tex` импортируемая $L_p$-оценка отделена от локального
распределительного шага. Концентрация случайной предсказуемой скобки опирается
на Lemma 11 Levin et al., а Berry--Esseen-переход --- на Lemma 21 Samsonov et
al. Перенос с мартингальной части на полный RR-объект использует inequality
(11) Samsonov et al. Gaussian approximation последней итерации в статью не
включена.

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

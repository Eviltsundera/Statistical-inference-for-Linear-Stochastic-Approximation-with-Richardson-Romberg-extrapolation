# Assessment of `review_diploma_proofs_3.md`

Дата: 2026-05-21

## Краткий вывод

С главным замечанием ревью я согласен: в текущем доказательстве Lemma 22 есть реальный разрыв в convolution estimate для координаты $H^{(2,w)}$. Это блокирует безусловное использование Lemma 22 в downstream transfer-цепочке до правки. В текущем тексте product-stability estimate имеет экспоненту $e^{-cwa(k-s)/p}$, но в convolution term дальше используется $e^{-cwa(k-\ell)}$. Поэтому нужно либо доказать более сильную product-stability estimate без деления на $p$, либо честно увеличить startup constant.

При этом несколько замечаний ревью уже не соответствуют текущим исходникам: степень $a$ в Section 2.4 уже записана как $a^{-3/2}$, вклад $U_R$ уже имеет вид $\sqrt{\alpha/a}$, gamma-window уже объяснен, верхнее ограничение на burn-in уже вынесено в $sqrt(n)$-corollary, а локальный конфликт обозначения $C_A$ уже снят через $\overline C_A$.

## Проверка замечаний

| Приоритет | Замечание ревью | Мой статус |
|---|---|---|
| P0 | Lemma 22: convolution для $H^{(2)}$ теряет `/p` | Согласен. Это главный фикс. |
| P1 | Section 2.4: неверная степень $a$ в $\widehat C_A$ | Не согласен для текущих исходников: сейчас стоит $\sqrt{t_{\rm mix}/a^3}$, то есть $a^{-3/2}$. |
| P1 | Lemma 5: вклад $U_R$ должен быть $\sqrt{\alpha/a}$ | Уже исправлено в текущих исходниках. |
| P1 | Local inverse bound $\|(I-w\bar A)^{-1}\|\le 2$ не следует из Lyapunov contraction | Согласен. Нужно добавить small-step ceiling, например $2\alpha\|\bar A\|\le 1/2$, или сформулировать отдельное admissibility condition. |
| P2 | Lemma 13: stationary finite-past construction не uniform in $w$ | Согласен как стилистико-техническое уточнение. В lemma уже стоит `Fix w`, но proof лучше явно сказать, что domination is for each fixed admissible $w$. |
| P2 | Corollary 5: gamma-window требует пояснения остальных terms | В текущем тексте это уже пояснено. |
| P2 | Corollary 12: upper bound на burn-in нужно вынести | В текущем тексте upper bound уже есть в $sqrt(n)$-corollary. Можно только сделать statement визуально более симметричным. |
| P3 | Переиспользование $C_A$ | Уже исправлено: используется $\overline C_A := \kappa_Q$, а $C_A$ остается sup-norm constant. |
| P3 | Опечатка `E⟦X|^p]` | В текущих исходниках не обнаружена. |
| P3 | Сломанная верстка `Phi(84) p, alpha)` | В текущих исходниках не обнаружена. |

## Деталь по Lemma 22

Текущий текст:

$$
\|\Gamma_{s+1:k}^{(w)}V_s\|_{L_p}
\le C e^{-cwa(k-s)/p}\|V_s\|_{L_{2p}}.
$$

Тогда convolution term должен давать

$$
w\sum_{\ell=1}^k
e^{-cwa(k-\ell)/p}e^{-cwa\ell/p}
= wk e^{-cwak/p}.
$$

После потери константы в экспоненте:

$$
wk e^{-cwak/p}
\le C\frac{p}{a}e^{-c'wak/p}.
$$

Значит текущая строка с $e^{-cwa(k-\ell)}$ в convolution proof не следует из заявленной product-stability estimate. Минимальная правка: заменить convolution calculation и увеличить $A_{\rm st}(p,q,w)$ на дополнительный множитель порядка $p/a$ (или хотя бы на $p$, если $a^{-1}$ намеренно поглощается в problem constants). После этого нужно обновить downstream текст, где используется $A_{\rm st}$.

## Влияние на финальную скорость

Ожидаемо итоговая balanced скорость не ломается. При $\alpha=c n^{-1/2}$, $m\asymp n$, $p\asymp\log n$ дополнительный множитель $p/a$ в $A_{\rm st}$ только увеличивает polylog constant в startup transfer:

$$
\frac{p A_{\rm st}(p,q,\alpha)}{\alpha a\sqrt m}n^{-\beta}.
$$

Такой term остается lower order при выбранном logarithmic burn-in. Но это нужно явно провести после исправления Lemma 22, иначе Theorem 7 / final burned-in theorem остается условным.

## Рекомендуемый чеклист исправлений

- [x] Исправить convolution proof в Lemma 22: использовать экспоненту `/p` и вывести фактор $p/a$.
- [x] Увеличить $A_{\rm st}(p,q,w)$ и все downstream occurrences: accumulated startup transfer, burned-in misadjustment, final balanced theorem.
- [x] Проверить, что lower burn-in condition остается $n_0 \gtrsim (\alpha a)^{-1}\log^2 n$ при $p\asymp\log n$, а изменяется только polylog factor.
- [x] Добавить small-step ceiling для локального inverse bound, например $2\alpha\|\bar A\|\le 1/2$, и включить его в admissible stepsize clauses.
- [x] В Lemma 13 добавить фразу, что finite-past domination is for each fixed admissible $w>0$, not uniform in $w$.
- [x] Опционально сделать burn-in lower/upper window в final corollary визуально парным:
  $n_0 \gtrsim (\alpha a)^{-1}\log^2 n$ и $n_0 \lesssim (\alpha a)^{-1}\log^2 n$.

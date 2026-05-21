# Assessment of `review_diploma_proofs_4.md`

Дата: 2026-05-21

## Краткий вывод

С главным замечанием ревью я согласен. В текущем finite-start / burn-in
разложении действительно отсутствует stochastic initial-product discrepancy

$$
(\Gamma_{1:k}^{(w)} - B_w^k)(\theta_0-\theta^*).
$$

Текущая burn-in глава отдельно контролирует deterministic transient
$B_w^k(\theta_0-\theta^*)$, Poisson boundary term и depth-two noise-driven
misadjustment, но не контролирует разность между random initial product и
deterministic initial product. Поэтому burned-in theorem в текущем виде нельзя
считать полностью доказанной для общего deterministic start
$\theta_0\neq\theta^*$.

При этом часть остальных замечаний уже закрыта в текущих исходниках:
$\widehat C_A$ сейчас имеет правильную степень $a^{-3/2}$, Lemma 2 уже
сформулирована как scalar specialization of Levin Lemma 11 with arbitrary
initial law and tail around zero, а конфликт $C_A$ уже снят через
$\overline C_A$.

## Проверка замечаний

| Приоритет | Замечание ревью | Мой статус |
|---|---|---|
| P0 | Потерян stochastic initial-product discrepancy | Согласен. Это реальный разрыв в finite-start theorem. |
| P1 | Неверная степень $a$ в $\widehat C_A$ | Не согласен для текущих исходников: сейчас стоит $\sqrt{t_{\rm mix}/a^3}$, то есть $a^{-3/2}$. Можно только переписать как $\sqrt{t_{\rm mix}}/a^{3/2}$ для читаемости. |
| P1 | Нужно унифицировать admissibility для $2\alpha$ | Частично согласен. Основные theorem statements уже содержат $2\alpha$, но Section 3.2 / last-iterate exposition стоит локально подтянуть. |
| P2 | Lemma 2 concentration around zero for arbitrary initial law | В текущем тексте уже сделан вариант 1 из ревью: точная ссылка на Levin Lemma 11. Проверка PDF подтверждает, что lemma stated for arbitrary initial probability and tail around zero. |
| P2 | Section 3.2 оформить как failed/exploratory route | Согласен стилистически. Математически это уже сказано в конце раздела, но лучше вынести в title/первый абзац. |
| P2 | Усилить stationary vs deterministic-start conventions | Согласен, особенно после добавления missing initial-product term. |
| P3 | Конфликт $C_A$ | Уже исправлено: используется $\overline C_A := \kappa_Q$, а $C_A$ остается sup-norm constant из Assumption 2. |
| P3 | Placeholder abstract | Согласен: `main.typ` всё еще содержит `Your abstract.` |
| P3 | Typo `E⟦X|^p]` и старые glitches | В текущих исходниках не обнаружены. |

## Почему пункт 1 реален

В Section 4.1 stationary chapter текущий текст сначала пишет exact identity

$$
\theta_k^{(\alpha)}-\theta^*
=-\alpha\sum_{\ell=1}^k
\Gamma_{\ell+1:k}^{(\alpha)}\epsilon(Z_\ell)
+\Gamma_{1:k}^{(\alpha)}(\theta_0-\theta^*).
$$

После замены random products на deterministic products в тексте стоит

$$
\theta_k^{(\alpha)}-\theta^*
=J_k^{(0,\alpha)}
+B_\alpha^k(\theta_0-\theta^*)
+R_k^{(\alpha)},
$$

где $R_k^{(\alpha)}$ затем отождествляется с $J_k^{(1,\alpha)}+H_k^{(1,\alpha)}$
или depth-two version $J_k^{(1,\alpha)}+J_k^{(2,\alpha)}+H_k^{(2,\alpha)}$.
Но эти $J/H$ processes в текущих определениях initialized at zero and driven by
the noise component $J^{(0)}$. Они не содержат product error from the initial
condition:

$$
R_{k,\mathrm{init}}^{(w)}
:=(\Gamma_{1:k}^{(w)}-B_w^k)(\theta_0-\theta^*).
$$

В burn-in chapter это проявляется в composite remainder:

$$
\mathcal R_{n,n_0}^{\mathrm{bRR}}
=D_{\mathrm{tr}}^{\mathrm{RR}}
+u^\top D_2^{\mathrm{bRR}}
+u^\top R_{n,n_0,\mathrm{fin}}^{\mathrm{mis,RR}}.
$$

Здесь $D_{\mathrm{tr}}^{\mathrm{RR}}$ контролирует только deterministic part
$B_w^k(\theta_0-\theta^*)$, а $R_{n,n_0,\mathrm{fin}}^{\mathrm{mis,RR}}$
контролирует only noise-driven finite-start perturbation variables. Значит
нужно добавить четвертый term.

## Какой фикс нужен

Нужно ввести, например,

$$
\mathcal I_{n,n_0}^{\mathrm{rand,RR}}(u)
:=\frac{1}{\sqrt m}\sum_{k=n_0}^{n-1}
u^\top\left[
2(\Gamma_{1:k}^{(\alpha)}-B_\alpha^k)
-(\Gamma_{1:k}^{(2\alpha)}-B_{2\alpha}^k)
\right](\theta_0-\theta^*).
$$

После этого composite remainder должен стать

$$
\mathcal R_{n,n_0}^{\mathrm{bRR}}
=D_{\mathrm{tr}}^{\mathrm{RR}}
+\mathcal I_{n,n_0}^{\mathrm{rand,RR}}
+u^\top D_2^{\mathrm{bRR}}
+u^\top R_{n,n_0,\mathrm{fin}}^{\mathrm{mis,RR}}.
$$

Доказательство можно сделать через уже импортируемую random-product stability
estimate. Для deterministic $\theta_0$ ожидаемый bound:

$$
\|\mathcal I_{n,n_0}^{\mathrm{rand,RR}}(u)\|_{L_p}
\le
C\|u\|\|\theta_0-\theta^*\|
\frac{p}{\alpha a\sqrt m}
\exp\{-c\alpha a n_0/p\}.
$$

Достаточно даже грубо оценить

$$
\|(\Gamma_{1:k}^{(w)}-B_w^k)(\theta_0-\theta^*)\|_{L_p}
\le
\|\Gamma_{1:k}^{(w)}(\theta_0-\theta^*)\|_{L_p}
+\|B_w^k(\theta_0-\theta^*)\|,
$$

а затем применить product stability to the first term and Lyapunov contraction
to the second. При $p\asymp\log n$, $\alpha=c n^{-1/2}$, $m\asymp n$ и
$n_0\gtrsim(\alpha a)^{-1}\log^2 n$ этот term становится lower order.

## Остальные правки

1. Переписать $\widehat C_A$ как
   $32\widetilde C_A\|\epsilon\|_\infty\sqrt{t_{\rm mix}}/a^{3/2}$, чтобы
   исключить неправильное чтение $\sqrt{t_{\rm mix}}/a^3$.
2. В Section 3.2 / `last_iterate.typ` явно добавить, что применение RR bound
   requires $2\alpha\le\alpha_\infty$ and $2\alpha\le\alpha_{\rm inv}$.
3. Переименовать section `Application to the PR-averaged RR Misadjustment` в
   что-то вроде `A depth-one bound and why it is insufficient`, и первым
   абзацем сказать, что this subsection is not used in the final
   Berry--Esseen assembly.
4. После добавления initial-product term пройти theorem statements:
   deterministic $\theta_0$ or random $\theta_0$, same Markov trajectory for
   $\alpha$ and $2\alpha$, same initial condition for both RR trajectories.
5. Заменить placeholder abstract before final submission.

## Рекомендуемый чеклист исправлений

- [x] Add $R_{k,\mathrm{init}}^{(w)}$ to the finite-start deterministic-product decomposition.
- [x] Add $\mathcal I_{n,n_0}^{\mathrm{rand,RR}}(u)$ to the burned-in RR composite remainder.
- [x] Prove the accumulated $L_p$ bound for $\mathcal I_{n,n_0}^{\mathrm{rand,RR}}(u)$ using random-product stability plus deterministic Lyapunov contraction.
- [x] Propagate this new term through `@lem:burn-R-bound`, `@thm:burn-RR-BE-master`, and the final balanced burned-in theorem.
- [x] Clarify the finite-start theorem assumptions on $\theta_0$, $Z_0$, and the shared RR trajectory.
- [x] Polish local admissibility clauses for $2\alpha$ in the depth-one/last-iterate exposition.
- [x] Rename the depth-one misadjustment subsection as an exploratory/insufficient route.
- [x] Rewrite $\widehat C_A$ with explicit $a^{-3/2}$ formatting.
- [ ] Replace the placeholder abstract.

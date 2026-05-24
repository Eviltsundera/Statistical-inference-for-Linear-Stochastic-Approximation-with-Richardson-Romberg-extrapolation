# Ревью текущей версии диплома: комментарии к доказательствам

**Файл:** `main.pdf`  
**Фокус ревью:** только текущие материалы, прежде всего математическая корректность доказательств, согласованность разложений и условий.  
**Статус работы:** черновик; часть разделов ещё не завершена.

---

## Краткий вывод

Общая стратегия работы выглядит разумной: Markovian LSA с constant stepsize, PR averaging, Poisson/martingale approximation, Richardson–Romberg extrapolation и перенос stationary/full-window результата на deterministic-start burned-in statistic. Направление согласуется с современной литературой по Berry–Esseen bounds для Markovian LSA, RR bias reduction и high-order bounds.

Однако в текущей версии есть **одна критическая проблема**, которую нужно исправить до того, как finite-start/burned-in теорема будет считаться доказанной: при переходе от exact random-product representation к deterministic-product decomposition теряется stochastic transient от начального условия. Кроме этого, есть несколько локальных ошибок в константах, admissibility conditions, формулировках imported inequalities и нотации.

---

## Опорный внешний контекст

Эти комментарии сверялись с текущей линией работ:

1. Samsonov, Sheshukova, Moulines, Naumov, *Statistical inference for Linear Stochastic Approximation with Markovian Noise* — Berry–Esseen bounds порядка \(O(n^{-1/4})\) для PR-averaged LSA с Markovian noise и bootstrap inference.
2. Levin, Naumov, Samsonov, *High-Order Error Bounds for Markovian LSA with Richardson-Romberg Extrapolation* — RR bias cancellation и high-order moment bounds с leading covariance \(\Sigma_\infty\).
3. Huo, Chen, Xie, *Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference* — CLT/inference procedure для constant-stepsize Markovian LSA и использование RR для bias reduction.

---

## Таблица замечаний

| № | Severity | Место | Суть |
|---|---|---|---|
| 1 | **Critical** | Section 4.1, burned-in theorem | Потерян random initial-product discrepancy \((\Gamma_{1:k}^{(\alpha)}-B_\alpha^k)(\theta_0-\theta^*)\). |
| 2 | Major | Section 2.4, Eq. (40) | Неверная степень \(a\) в \(\widehat C_A\): из предыдущей строки следует \(a^{-3/2}\), а не \(a^{-3}\), если это равенство. |
| 3 | Major | Леммы/теоремы с RR | Нужно унифицировать условия admissibility для \(2\alpha\), особенно contraction и inverse-transfer. |
| 4 | Medium/Major | Lemma 2 | Сильная concentration around zero для arbitrary initial law требует точной ссылки или самостоятельного доказательства. |
| 5 | Medium | Section 3.2 | Depth-one route даёт недостаточный bound; лучше явно оформить как failed/exploratory route. |
| 6 | Medium | Stationary vs deterministic-start conventions | Нужно жёстко разделить stationary theorem, finite-start theorem и условия на \(\theta_0\), \(Z_0\), augmented chain. |
| 7 | Minor/Medium | Нотация | Конфликт обозначения \(C_A\), typos, placeholder abstract, мелкие LaTeX glitches. |

---

# 1. Критическая проблема: потерян stochastic initial transient

## Где возникает

В Section 4.1 используется exact random-product representation:

\[
\theta_k^{(\alpha)}-\theta^*
=
-\alpha\sum_{l=1}^k \Gamma_{l+1:k}^{(\alpha)}\epsilon(Z_l)
+
\Gamma_{1:k}^{(\alpha)}(\theta_0-\theta^*).
\]

Затем случайные произведения заменяются deterministic products:

\[
\Gamma_{l+1:k}^{(\alpha)} \rightsquigarrow B_\alpha^{k-l},
\qquad
\Gamma_{1:k}^{(\alpha)} \rightsquigarrow B_\alpha^k.
\]

В результате пишется decomposition вида

\[
\theta_k^{(\alpha)}-\theta^*
=
J_k^{(0,\alpha)}
+B_\alpha^k(\theta_0-\theta^*)
+R_k^{(\alpha)},
\]

а далее \(R_k^{(\alpha)}\) отождествляется с perturbation remainder типа

\[
R_k^{(\alpha)} := J_k^{(1,\alpha)} + H_k^{(1,\alpha)}.
\]

## В чём ошибка

Для \(\theta_0\neq \theta^*\) это отождествление неполное. Из exact representation при замене random product на deterministic product появляется дополнительный член

\[
R_{k,\mathrm{init}}^{(\alpha)}
:=
\left(\Gamma_{1:k}^{(\alpha)}-B_\alpha^k\right)(\theta_0-\theta^*).
\]

Он не входит в стандартные noise-driven perturbation terms \(J^{(1)}+H^{(1)}\), потому что эти члены строятся из \(J^{(0)}\), то есть из шумовой части, а не из initial condition.

Иными словами, корректная finite-start decomposition должна иметь вид

\[
\theta_k^{(\alpha)}-\theta^*
=
J_k^{(0,\alpha)}
+B_\alpha^k(\theta_0-\theta^*)
+R_{k,\mathrm{noise}}^{(\alpha)}
+R_{k,\mathrm{init}}^{(\alpha)},
\]

где

\[
R_{k,\mathrm{noise}}^{(\alpha)}
=
J_k^{(1,\alpha)}+H_k^{(1,\alpha)}
\]

на depth-one уровне, и

\[
R_{k,\mathrm{noise}}^{(\alpha)}
=
J_k^{(1,\alpha)}+J_k^{(2,\alpha)}+H_k^{(2,\alpha)}
\]

на depth-two уровне.

## Почему это критично

В stationary full-window theorem этот член можно обойти, если initialization действительно stationary/centered in the augmented chain или если явно ставится \(\theta_0=\theta^*\) в соответствующей convention. Но в deterministic-start burned-in theorem он не исчезает автоматически.

В burned-in RR statistic должен появиться дополнительный remainder:

\[
\mathcal R^{bRR}_{\mathrm{init,rand}}
=
\frac{1}{\sqrt m}
\sum_{k=n_0}^{n-1}
 u^\top
\left[
2\left(\Gamma_{1:k}^{(\alpha)}-B_\alpha^k\right)
-
\left(\Gamma_{1:k}^{(2\alpha)}-B_{2\alpha}^k\right)
\right](\theta_0-\theta^*),
\]

где \(m=n-n_0\), если statistic нормируется через \(\sqrt m\).

Сейчас final burned-in composite remainder содержит deterministic transient, Poisson boundary и misadjustment, но не содержит этот stochastic initial-product discrepancy. Поэтому finite-start theorem в текущем виде не полностью доказана.

## Минимальная правка

В Section 4.1 нужно заменить определение remainder на раздельное:

\[
R_k^{(\alpha)}
=
R_{k,\mathrm{noise}}^{(\alpha)}
+
R_{k,\mathrm{init}}^{(\alpha)}.
\]

Далее во всех PR/RR decompositions нужно отдельно вести

\[
D_{\mathrm{init,rand}}^{(\alpha)}
=
\frac{1}{\sqrt n}\sum_{k=0}^{n-1}
\left(\Gamma_{1:k}^{(\alpha)}-B_\alpha^k\right)(\theta_0-\theta^*)
\]

для full-window normalization, и аналогичный burned-in член для \(k=n_0,\ldots,n-1\).

Для RR:

\[
D_{\mathrm{init,rand}}^{RR}
=
2D_{\mathrm{init,rand}}^{(\alpha)}
-D_{\mathrm{init,rand}}^{(2\alpha)}.
\]

## Что нужно доказать

Нужна lemma вида:

\[
\left\|
\frac{1}{\sqrt m}
\sum_{k=n_0}^{n-1}
\left(\Gamma_{1:k}^{(w)}-B_w^k\right)(\theta_0-\theta^*)
\right\|_{L_p}
\le
\text{small term}(w,n,n_0,p)\|\theta_0-\theta^*\|
\]

для \(w\in\{\alpha,2\alpha\}\).

Ожидаемый вид bound после logarithmic burn-in, вероятно:

\[
\lesssim
\frac{\|\theta_0-\theta^*\|}{\sqrt m\,\alpha a}
\exp(-c\alpha a n_0/p),
\]

или близкий вариант, в зависимости от используемой random-product stability lemma.

Важно: этот bound нельзя просто заменить deterministic transient estimate, потому что здесь стоит difference между random and deterministic matrix products.

## Альтернативная правка

Если не хочется добавлять эту lemma, можно сузить формулировку finite-start theorem:

- либо поставить \(\theta_0=\theta^*\), тогда \(R_{k,\mathrm{init}}^{(\alpha)}=0\);
- либо в stationary theorem работать только с augmented stationary initialization;
- либо честно оставить finite-start theorem как conjectural extension до добавления bound на \(R_{k,\mathrm{init}}\).

Для диплома лучше первый путь не выбирать, если цель — именно deterministic-start inference. Лучше добавить missing term и оценить его.

---

# 2. Локальная алгебраическая ошибка в \(\widehat C_A\)

## Где

Section 2.4, переход от оценки суммы квадратов к variance proxy.

В тексте получается

\[
\sum_j \|g_j^u\|_\infty^2
\le
\frac{16\alpha\|u\|^2\widetilde C_A^2\|\epsilon\|_\infty^2}{a^3}.
\]

Lemma 2 даёт

\[
v_n^2
=64t_{\rm mix}\sum_j\|g_j^u\|_\infty^2.
\]

Следовательно,

\[
v_n^2
\le
\|u\|^2
\frac{1024\alpha\widetilde C_A^2\|\epsilon\|_\infty^2t_{\rm mix}}{a^3}.
\]

Если писать

\[
v_n^2\le \|u\|^2\widehat C_A^2\alpha,
\]

то должно быть

\[
\widehat C_A
=
\frac{32\widetilde C_A\|\epsilon\|_\infty\sqrt{t_{\rm mix}}}{a^{3/2}}.
\]

В текущей версии стоит знаменатель \(a^3\), что даёт \(\widehat C_A^2\sim a^{-6}\), а из предыдущей строки следует \(a^{-3}\).

## Как исправить

Вариант 1 — точное исправление:

\[
\boxed{
\widehat C_A
=
32\widetilde C_A\|\epsilon\|_\infty\sqrt{t_{\rm mix}}\,a^{-3/2}
}
\]

Вариант 2 — если намеренно хочется грубее:

написать, что после enlarging constants и при дополнительном условии вроде \(a\le 1\) можно взять более грубую константу с \(a^{-3}\). Тогда текст должен быть не “defining gives”, а “after enlarging the constant, one may take”.

## Влияние

Rate по \(\alpha\) и \(n\) не ломается. Но если в работе отслеживаются степени \(a\), текущая формула неверна как равенство/непосредственное следствие.

---

# 3. Условия admissibility для \(2\alpha\)

## Проблема

В RR-доказательствах одновременно используются шаги \(\alpha\) и \(2\alpha\). Поэтому все contraction, inverse-transfer и closed-form bounds должны быть валидны для обоих шагов.

В некоторых местах это явно написано:

\[
\alpha,2\alpha\in(0,\alpha_\infty].
\]

Но в ранних леммах местами формулируется только

\[
0<\alpha\le \alpha_\infty,
\]

хотя внутри доказательства используется \(B_{2\alpha}=I-2\alpha A\) и contraction for \(I-2\alpha A\).

## Минимальная правка

Везде, где фигурирует \(2\alpha\), формулировать assumptions как:

\[
2\alpha\le \alpha_\infty.
\]

Если используется inverse transfer

\[
J^{(1,w)}_k=(I-wA)^{-1}T^{(1,w)}_k,
\]

то нужно также:

\[
2\alpha\le \alpha_{\rm inv}.
\]

То есть лучше писать одним блоком:

\[
0<2\alpha\le \alpha_\infty\wedge \alpha_{\rm inv}.
\]

## Где особенно проверить

- last-iterate RR estimates;
- Section 3.2 при переносе от \(T^{(1,w)}\) к \(J^{(1,w)}\), \(w\in\{\alpha,2\alpha\}\);
- Section 4.2 closed-form identities;
- all RR Berry–Esseen/burn-in theorems.

---

# 4. Lemma 2: concentration around zero для arbitrary initial law

## Что сейчас написано

Lemma 2 утверждает tail bound для time-inhomogeneous centered functions \(g_i\) under UGE:

\[
\mathbb P_\xi\left(\left|\sum_{i=1}^n g_i(Z_i)\right|\ge t\right)
\le
2\exp\left(-\frac{t^2}{2v_n^2}\right)
\]

для arbitrary initial distribution \(\xi\), при условии \(\pi(g_i)=0\), причём bound around zero, not around expectation.

## Почему это уязвимо

Для nonstationary initial law обычно есть initial-bias term:

\[
\mathbb E_\xi\sum_i g_i(Z_i)
\neq 0.
\]

Поэтому concentration around zero для arbitrary start — сильная форма. Она может быть верной, если именно такая lemma импортируется из Levin et al. через coupling/Dobrushin construction, но это нужно сделать максимально явно.

## Что исправить

Нужно одно из двух:

1. Привести точную формулировку imported lemma с теми же assumptions и подчеркнуть, что она даёт bound around zero for arbitrary initial law.
2. Добавить короткое доказательство или хотя бы proof sketch:
   - как строится coupling/block decomposition;
   - почему centering \(\pi(g_i)=0\) достаточно;
   - где исчезает initial-bias term.

## Почему это важно

Эта lemma используется не только в раннем last-iterate bound, но и концептуально похожие concentration statements появляются в burned-in quadratic variation / variance-comparison arguments. Если читатель усомнится в Lemma 2, downstream bounds тоже станут уязвимыми.

---

# 5. Section 3.2 лучше оформить как “failed route”

## Что происходит

Section 3.2 применяет depth-one estimate к PR-averaged RR misadjustment:

\[
D^{\mathrm{mis,RR}}_1
=
\frac{\sqrt n}{n-n_0}
\sum_{k=n_0}^{n-1}
\left(2J_k^{(1,\alpha)}-J_k^{(1,2\alpha)}\right).
\]

Получается bound вида

\[
\|u^\top D^{\mathrm{mis,RR}}_{1,c}\|_{L_p}
\le
C\|u\|\sqrt n\,\alpha\Phi(p,\alpha).
\]

При \(\alpha\asymp n^{-1/2}\) это даёт \(O(1)\), то есть не становится subleading Berry–Esseen remainder нужного порядка.

Текст сам признаёт, что depth-one route does not yield a useful Berry–Esseen remainder.

## Рекомендация

Переименовать раздел в один из вариантов:

- `A depth-one bound and why it is insufficient`;
- `Exploratory depth-one control of the misadjustment`;
- `Why the first-order perturbation bound is not enough`.

В начале раздела явно написать:

> This subsection is not used in the final Berry–Esseen assembly. It explains why a depth-two decomposition is needed.

Так читатель не будет думать, что слабый bound входит в финальную теорему.

---

# 6. Stationary vs deterministic-start conventions

## Что сделано хорошо

В работе уже есть попытка отделить stationary full-window theorem от deterministic-start burned-in theorem. Это правильно, потому что:

- stationary full-window analysis удобно проводить через augmented chain;
- deterministic-start inference требует burn-in и отдельных transfer estimates;
- zero-start/full-window result нельзя автоматически перенести на deterministic-start одной terminal \(\rho^n\)-оценкой.

## Что нужно усилить

После исправления missing initial-product discrepancy нужно ещё раз пройти все theorem statements и явно указать:

1. В stationary theorem:
   - стартуется ли augmented chain из \(\Pi_\alpha\)?
   - или только \(Z_0\sim\pi\), а \(\theta_0\) fixed?
   - если \(\theta_0=\theta^*\), это нужно написать.

2. В finite-start theorem:
   - \(\theta_0\) deterministic или random?
   - какие моменты требуются от \(\theta_0-\theta^*\)?
   - где именно используется burn-in \(n_0\)?

3. В RR theorem:
   - две chains с шагами \(\alpha\) и \(2\alpha\) должны использовать same Markov trajectory;
   - initial condition для обеих trajectories должно быть одинаково или это нужно оговорить;
   - если разные initializations допустимы, transient terms меняются.

---

# 7. Что выглядит корректно

Ниже — части, которые по текущему чтению не вызывают существенных возражений.

## 7.1 Last-iterate RR zeroth-order algebra

Разложение

\[
\widetilde J^{(0,\alpha)}_{n,\mathrm{last}}
=2J_n^{(0,\alpha)}-J_n^{(0,2\alpha)}
\]

и identity

\[
B_\alpha^m-B_{2\alpha}^m
=\alpha A\sum_{i=1}^m B_\alpha^{i-1}B_{2\alpha}^{m-i}
\]

выглядят правильными. Коммутация допустима, потому что \(B_\alpha\) и \(B_{2\alpha}\) являются полиномами от одной матрицы \(A\).

Знак и фактор \(-2\alpha^2 A\) также согласованы с определением

\[
J_n^{(0,\alpha)}=-\alpha\sum_j B_\alpha^{n-j}\epsilon(Z_j).
\]

## 7.2 Poisson / Abel summation part

В stationary chapter Poisson-equation / Abel-summation mechanism выглядит согласованно. Особенно важно, что right boundary disappears because

\[
Q^{RR}_{n-1}=2\alpha I-2\alpha I=0.
\]

Это хороший structural cancellation для RR weights.

## 7.3 Variance target

Целевая covariance

\[
\Sigma_\infty=A^{-1}\Sigma_\epsilon^{(M)}A^{-\top}
\]

согласована с Markov-chain CLT covariance for partial sums of \(\epsilon(Z_t)\) и standard averaged-LSA covariance target.

## 7.4 Burn-in structure

Сама структура burned-in section разумна:

- burned-in weights;
- \(\sqrt m\)-normalization;
- finite-window normalization;
- comparison with \(\sqrt n\)-normalization;
- final transfer from stationary to deterministic-start.

Но она станет полной только после добавления stochastic initial-product discrepancy из пункта 1.

---

# 8. Мелкие, но важные правки по нотации и тексту

## 8.1 Конфликт обозначений \(C_A\)

В Assumption 2 \(C_A\) — это sup-norm constant для \(A(z)\) и \(\widetilde A(z)\):

\[
C_A:=\max\left(\sup_z\|A(z)\|,\sup_z\|\widetilde A(z)\|\right).
\]

В Section 2.3 затем локально задаётся

\[
C_A:=\kappa_Q.
\]

Это опасно: далее появляется выражение типа

\[
\widetilde C_A=C_A C_A,
\]

которое формально выглядит бессмысленно.

Лучше переименовать:

- \(C_A^{\rm sup}\) — bound on random matrices;
- \(C_Q\) или \(C_{\rm Lyap}\) — Lyapunov/norm-equivalence constant;
- \(\widetilde C_A=C_A^{\rm sup}C_Q\).

## 8.2 Placeholder abstract

В начале стоит `Your abstract.` Это нужно убрать до любой отправки научруку/рецензенту.

## 8.3 Typo в Lemma 3

В Lemma 3 виден LaTeX/glitch:

\[
\mathbb E⟦X|^p]
\]

Нужно заменить на стандартное:

\[
\mathbb E[|X|^p]
\]

или

\[
\|X\|_{L_p}\le 2^{1/p}\sqrt p\,\sigma.
\]

## 8.4 Ссылки на equations

Некоторые места ссылаются на “Eq.” или “Chapter 4” без полного локального statement assumptions. Для диплома это допустимо в черновике, но перед финальной версией лучше сделать theorem/lemma statements самодостаточными.

## 8.5 Нотация \(n\), \(m\), \(n_0\)

В burned-in part нужно жёстко держать:

\[
m=n-n_0.
\]

И всегда явно указывать, где normalization \(\sqrt n\), а где \(\sqrt m\). Иначе легко получить неверный factor \(\sqrt{n/m}\).

---

# 9. Приоритетный checklist исправлений

## Must fix до защиты / отправки

1. Добавить stochastic initial-product discrepancy:
   \[
   (\Gamma_{1:k}^{(\alpha)}-B_\alpha^k)(\theta_0-\theta^*).
   \]
2. Добавить его RR/burned-in version.
3. Доказать bound на этот член после burn-in или сузить theorem assumptions.
4. Исправить \(\widehat C_A\) и степень \(a\).
5. Везде заменить conditions на \(2\alpha\le\alpha_\infty\) и, где нужно, \(2\alpha\le\alpha_{\rm inv}\).

## Should fix

6. Уточнить Lemma 2: exact imported statement или proof sketch.
7. Переименовать Section 3.2 как exploratory/failed route.
8. Развести stationary and deterministic-start theorem assumptions.
9. Убрать конфликт \(C_A\).

## Cosmetic / final polish

10. Написать abstract.
11. Исправить typos и LaTeX glitches.
12. Проверить consistency обозначений \(\Sigma_\epsilon^{(M)}\), \(\Sigma_\infty\), \(Q_l\), \(\mathcal Q_l^{RR}\).
13. Сделать statements лемм самодостаточными.

---

# 10. Предлагаемый патч для Section 4.1

Ниже — компактный текст, который можно адаптировать прямо в диплом.

```tex
The replacement of the random products by deterministic products produces two distinct remainders. We write
\[
\theta_k^{(\alpha)}-\theta^*
=J_k^{(0,\alpha)}+B_\alpha^k(\theta_0-\theta^*)
+R_{k,\mathrm{noise}}^{(\alpha)}
+R_{k,\mathrm{init}}^{(\alpha)},
\]
where
\[
R_{k,\mathrm{init}}^{(\alpha)}
:=\left(\Gamma_{1:k}^{(\alpha)}-B_\alpha^k\right)(\theta_0-\theta^*)
\]
accounts for the stochastic perturbation of the initial-condition product, while
\[
R_{k,\mathrm{noise}}^{(\alpha)}
:=-\alpha\sum_{l=1}^k
\left(\Gamma_{l+1:k}^{(\alpha)}-B_\alpha^{k-l}\right)\epsilon(Z_l)
\]
accounts for the stochastic perturbation of the noise-driven component.
At depth one,
\[
R_{k,\mathrm{noise}}^{(\alpha)}=J_k^{(1,\alpha)}+H_k^{(1,\alpha)},
\]
whereas at depth two,
\[
R_{k,\mathrm{noise}}^{(\alpha)}
=J_k^{(1,\alpha)}+J_k^{(2,\alpha)}+H_k^{(2,\alpha)}.
\]
```

И затем для burned-in RR:

```tex
The finite-start burned-in RR remainder contains the additional initial-product fluctuation
\[
\mathcal R^{bRR}_{\mathrm{init,rand}}
=\frac{1}{\sqrt m}\sum_{k=n_0}^{n-1}u^\top
\left\{2\left(\Gamma_{1:k}^{(\alpha)}-B_\alpha^k\right)
-\left(\Gamma_{1:k}^{(2\alpha)}-B_{2\alpha}^k\right)\right\}(\theta_0-\theta^*).
\]
```

---

# 11. Итоговая оценка

После исправления missing random initial transient схема доказательства, скорее всего, станет жизнеспособной:

- stationary RR Berry–Esseen part выглядит концептуально согласованной;
- RR weight algebra и Abel/Poisson cancellation выглядят корректно;
- depth-two misadjustment route — правильное направление;
- logarithmic burn-in transfer — естественный способ перейти к deterministic start.

Но без пункта 1 finite-start theorem в текущей форме нельзя считать полностью доказанной.

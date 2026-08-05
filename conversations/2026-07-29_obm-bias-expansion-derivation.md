# Как расписать $\mathbb{E}[\hat\sigma^2(b)] = \sigma_\infty^2 + \frac{c_1}{b} + \frac{c_2}{b^2} + c_3\frac{b}{T} + o(\cdot)$ для OBM: полный вывод

**Дата:** 2026-07-29
**Вопрос:** как строго расписать разложение смещения OBM-оценщика, которое до сих пор фигурировало как «working template» (`research/obm_rr_markov_lsa_report.md` §4.3, `summaries/RR_for_OBM_variance_estimator.md` §3, `conversations/2026-07-29_obm-lw-rate-min-formula-assumptions.md` шаг 4).
**Правки 2026-08-05** (полная адверсариальная проверка + arXiv:2505.08456): (i) исправлен знак члена $\frac{b}{T}\rho(T)$ в тождестве §2 — было $-\frac{b}{T}(\frac{\Gamma_1}{T}+\rho(T))$, верно $-\frac{b}{T}(\frac{\Gamma_1}{T}-\rho(T))$; прежние численные тесты ошибку не ловили, потому что во всех них $\rho(T)$ нулевой или экспоненциально мал; (ii) оценка $\varepsilon_1$ переписана через $\tilde\Gamma_1=\sum|k|\,|\gamma(k)|$ — со знакопеременными $\gamma$ прежняя форма неверна (есть контрпример); (iii) уточнены константа на границе $k^{-3}$ в §4 и порядок дрейфового члена в §8(a); исправлено предсказание о «пересечении нуля» в §6; (iv) добавлен §7 — явный MSE (bias² + Var) с точными константами.

**Главный итог:** для стационарного процесса разложение не нужно постулировать — оно выводится как **точное тождество** (без единого $o(\cdot)$) из подсчёта пар в квадратичной форме. При этом:

- $c_1 = -\Gamma_1 = -\sum_k |k|\,\gamma(k)$;
- $c_3 = -\sigma_\infty^2$ **точно** — для нормировки $\frac{b}{T-b+1}$, которая используется у Самсонова и в `code/lsa_inference/inference.py`;
- члена $c_2/b^2$ в честном стационарном геометрически перемешивающем случае **нет**: на его месте стоит хвост $\rho(b)$, который экспоненциально мал по $b/\ell$; $c_2/b^2$ — это верхняя огибающая для более общих (полиномиально перемешивающих / нестационарных) случаев.

Тождество проверено численно до машинной точности (AR(1) вплоть до $\phi=0.99$, MA(2) со знакопеременными $\gamma$, IID, тяжёлый хвост $k^{-2.2}$, включая $b\sim T/2$ и вырожденный $b=T$): скрипт `tmp/check_obm_identity.py`. Независимо: сборка §3 проверена покоэффициентно в точной арифметике (по каждому $\gamma(k)$, включая лаги $k>T$) для шести пар $(T,b)$, а матожидание — через точный след гауссовской квадратичной формы $\frac{b}{N}\operatorname{tr}(A\Sigma A^\top)$ при $T\le 8000$ (расхождение $\le 5\cdot10^{-13}$).

---

## 1. Оценщик и обозначения

Скалярная последовательность $Y_1,\dots,Y_T$ (у нас $Y_t = u^\top\theta_t$), блоки $B_j = \{j+1,\dots,j+b\}$, $j = 0,\dots,N-1$, $N = T-b+1$; блочные средние $\bar Y_j(b) = b^{-1}\sum_{t\in B_j} Y_t$ и глобальное $\bar Y_T = T^{-1}\sum_t Y_t$. Оценщик — в точности как в `_obm_variance_from_proj`:

$$\hat\sigma^2(b) \;=\; \frac{b}{N}\sum_{j=0}^{N-1}\bigl(\bar Y_j(b) - \bar Y_T\bigr)^2 .$$

Предположения для тождества: **стационарность** ($\gamma(k) = \mathrm{Cov}(Y_t, Y_{t+k})$ не зависит от $t$) и суммируемость $\sum_k |k|\,|\gamma(k)| < \infty$. Обозначения:

$$\sigma_\infty^2 = \sum_{k\in\mathbb{Z}}\gamma(k), \qquad
\Gamma_1 = \sum_{k\in\mathbb{Z}} |k|\,\gamma(k), \qquad
\tilde\Gamma_1 = \sum_{k\in\mathbb{Z}} |k|\,|\gamma(k)|, \qquad
\Gamma_2 = \sum_{k\in\mathbb{Z}} k^2\,|\gamma(k)|,$$

— в самом тождестве участвует **знаковая** $\Gamma_1$; абсолютные $\tilde\Gamma_1, \Gamma_2$ нужны только в оценках остатков (при $\gamma\ge0$ различие исчезает: $\tilde\Gamma_1=\Gamma_1$),

и ключевой объект — **хвостовой функционал окна Бартлетта**

$$\rho(m) \;=\; \frac{1}{m}\sum_{|k|\ge m}\bigl(|k| - m\bigr)\,\gamma(k).$$

## 2. Точное тождество

Для любых $1 \le b \le T$ и любого стационарного процесса с $\Gamma_1<\infty$:

$$\boxed{\;
\mathbb{E}\bigl[\hat\sigma^2(b)\bigr]
= \sigma_\infty^2
\;-\; \frac{\Gamma_1}{b}
\;-\; \frac{b}{T}\,\sigma_\infty^2
\;+\; \rho(b)
\;-\; \frac{b}{T}\Bigl(\frac{\Gamma_1}{T} - \rho(T)\Bigr)
\;+\; \frac{2\varepsilon_1}{NT}
\;}$$

где $\varepsilon_1 = \sum_{t=1}^{T} w_t\, r_t$ — краевая константа (определения $w_t, r_t$ в шаге 3 ниже), с оценкой $|\varepsilon_1| \le \tfrac12(\tilde\Gamma_1 + \Gamma_2)$; при $\gamma\ge0$ дополнительно $\varepsilon_1 \ge 0$. Со знаковой $\Gamma_1$ оценка неверна: для MA(1) с $\gamma(0)=\tfrac54$, $\gamma(1)=-\tfrac12$ получается $|\varepsilon_1|=1$ при $\tfrac12(\Gamma_1+\Gamma_2)=0$; абсолютная версия на этом же примере достигается с равенством.

Сопоставление с шаблоном $\sigma_\infty^2 + \frac{c_1}{b} + \frac{c_2}{b^2} + c_3\frac{b}{T} + o(\frac{1}{b^2}+\frac{b}{T})$:

| член шаблона | что это на самом деле | величина |
|---|---|---|
| $c_1/b$ | $-\Gamma_1/b$ | главное смещение усечения; $\frac{\Gamma_1}{\sigma_\infty^2}\asymp \ell$ (корреляционная длина) |
| $c_3\,b/T$ | $-\frac{b}{T}\sigma_\infty^2$ | смещение центрирования; **коэффициент известен точно**, $c_3=-\sigma_\infty^2$ |
| $c_2/b^2$ | $\rho(b)$ | при $\Gamma_2<\infty$: $\lvert\rho(b)\rvert \le \frac{1}{b^2}\sum_{\lvert k\rvert\ge b}k^2\lvert\gamma(k)\rvert = o(1/b^2)$; при геометрическом затухании — $O\!\bigl(\tfrac{\Gamma_1}{b}e^{-b/\ell}\bigr)$ |
| $o(\cdot)$ | $-\frac{b}{T}\bigl(\frac{\Gamma_1}{T}-\rho(T)\bigr) + \frac{2\varepsilon_1}{NT}$ | явные $O\!\bigl(\frac{b\Gamma_1}{T^2}\bigr)$ и $O\!\bigl(\frac{\tilde\Gamma_1+\Gamma_2}{T^2}\bigr)$ |

Обе главные поправки **отрицательны** при положительной корреляции: OBM недобирает и из-за усечения окна, и из-за центрирования выборочным средним — согласуется с систематическим недопокрытием в экспериментах.

Проверка на IID ($\gamma(k)=\sigma^2\delta_{k0}$): все $\Gamma$-члены нулевые, и $\mathbb{E}\hat\sigma^2(b) = \sigma^2\bigl(1-\tfrac{b}{T}\bigr)$ — точно (это же видно прямым счётом).

Вырожденная проверка $b=T$: оценщик тождественно нулевой ($N=1$, единственный блок совпадает со всей выборкой), и тождество даёт ровно $0$ — при $b=T$ все $w_t = 1$ и $\varepsilon_1 = \Gamma_1 - T\rho(T)$, после подстановки всё сокращается. Версия со старым знаком $\rho(T)$ этот тест не проходит (даёт $4\rho(T)\ne0$) — на нём ошибка и была поймана.

## 3. Вывод

### Шаг 1: разложение квадратичной формы

$\bar Y_j - \bar Y_T = (\bar Y_j - \mu) - (\bar Y_T - \mu)$, поэтому

$$\mathbb{E}\sum_j(\bar Y_j - \bar Y_T)^2
= \underbrace{\sum_j \operatorname{Var}(\bar Y_j)}_{A}
\;-\; 2\underbrace{\sum_j \operatorname{Cov}(\bar Y_T, \bar Y_j)}_{B}
\;+\; \underbrace{N\operatorname{Var}(\bar Y_T)}_{C}.$$

Каждый из трёх членов считается точно.

### Шаг 2: блочный член $A$ — точное окно Бартлетта

Внутри блока длины $b$ упорядоченная пара точек с лагом $k$ ($|k|<b$) встречается ровно $b-|k|$ раз, и **каждый блок полный** (стартовые позиции $j\le T-b$), поэтому без всяких краевых поправок

$$\frac{b}{N}\,A \;=\; b\operatorname{Var}(\bar Y_0(b)) \;=\; \frac1b\sum_{|k|<b}(b-|k|)\,\gamma(k) \;=\; \sum_{|k|<b}\Bigl(1-\frac{|k|}{b}\Bigr)\gamma(k).$$

Это усиливает тождество «OBM $=$ Бартлетт-SV $+\,O_p(b/T)$» из предыдущей заметки: *в среднем* блочная часть OBM — это **точно** Бартлетт (в отличие от SV-оценщика через $\hat R(k)$, где сидят множители $\frac{T-k}{T}$); вся разница между OBM и «чистым окном» — в членах центрирования из шага 3.

Дальше — арифметика с хвостами:

$$\sum_{|k|<b}\Bigl(1-\frac{|k|}{b}\Bigr)\gamma(k)
= \sum_k \gamma(k) - \frac1b\sum_k |k|\gamma(k) - \sum_{|k|\ge b}\Bigl(1-\frac{|k|}{b}\Bigr)\gamma(k)
= \sigma_\infty^2 - \frac{\Gamma_1}{b} + \rho(b),$$

поскольку $-\bigl(1-\frac{|k|}{b}\bigr) = \frac{|k|-b}{b}$ при $|k|\ge b$. Отсюда $c_1 = -\Gamma_1$, а «второй член разложения» — это ровно $\rho(b)$, не более и не менее.

### Шаг 3: члены центрирования $B$ и $C$

**Кросс-член.** $\sum_j \bar Y_j = \frac1b\sum_{t=1}^T w_t Y_t$, где $w_t = \#\{j: t\in B_j\} = \min(t,\,b,\,T-t+1,\,N)$ — «шатёр» с $\sum_t w_t = Nb$. Далее,

$$\operatorname{Cov}(\bar Y_T, Y_t) = \frac1T\sum_{s=1}^T\gamma(t-s) = \frac1T\bigl(\sigma_\infty^2 - r_t\bigr),
\qquad r_t = \sum_{u\ge t}\gamma(u) + \sum_{u\ge T-t+1}\gamma(u)$$

($r_t$ — недостающая корреляционная масса, заметная только на расстоянии $\lesssim\ell$ от краёв). Поэтому

$$-\frac{2b}{N}\,B = -\frac{2}{NT}\sum_t w_t(\sigma_\infty^2 - r_t) = -\frac{2b}{T}\sigma_\infty^2 + \frac{2\varepsilon_1}{NT},
\qquad \varepsilon_1 = \sum_t w_t r_t.$$

Оценка $\varepsilon_1$: $w_t \le \min(t, T-t+1)$, и $\sum_{t\ge1} t\sum_{u\ge t}|\gamma(u)| = \sum_u \frac{u(u+1)}{2}|\gamma(u)|$, откуда (два края) $|\varepsilon_1| \le \sum_{u\ge1}u(u+1)|\gamma(u)| = \frac12(\tilde\Gamma_1+\Gamma_2)$ — константа, не растущая ни с $b$, ни с $T$. Аргумент идёт через $|\gamma|$, поэтому абсолютная $\tilde\Gamma_1$ здесь неустранима.

**Глобальный член.** Дисперсия среднего — тот же Бартлетт на масштабе $T$:

$$\frac{b}{N}\,C = b\operatorname{Var}(\bar Y_T) = \frac{b}{T}\sum_{|k|<T}\Bigl(1-\frac{|k|}{T}\Bigr)\gamma(k)
= \frac{b}{T}\Bigl(\sigma_\infty^2 - \frac{\Gamma_1}{T} + \rho(T)\Bigr).$$

Внутренняя самоподобность приятна: $\rho$ появляется дважды — на масштабе окна $b$ и на масштабе выборки $T$.

**Сборка.** $-\frac{2b}{T}\sigma_\infty^2 + \frac{b}{T}\sigma_\infty^2 = -\frac{b}{T}\sigma_\infty^2$ — вот откуда $c_3 = -\sigma_\infty^2$; остальные слагаемые дают формулу из §2 без остатка. $\blacksquare$

## 4. Честный статус $c_2/b^2$

Из тождества видно, что «$c_2/b^2$» — это $\rho(b)$, и его порядок зависит от скорости затухания $\gamma$:

- **геометрическое перемешивание**, $|\gamma(k)|\le \gamma_0 e^{-|k|/\ell}$: $\rho(b) = O\!\bigl(\gamma_0\frac{\ell^2}{b}e^{-b/\ell}\bigr)$ — экспоненциально мал; для AR(1) с $\gamma(k)=\gamma_0\phi^{|k|}$ вообще в замкнутой форме
  $$\rho(b) = \frac{\Gamma_1\,\phi^b}{b}, \qquad\text{т.е.}\qquad
  \mathbb{E}\hat\sigma^2(b) = \sigma_\infty^2\Bigl(1-\frac{b}{T}\Bigr) - \frac{\Gamma_1}{b}\bigl(1-\phi^b\bigr) + O\!\Bigl(\frac{b\,\Gamma_1}{T^2}\Bigr)$$
  (остаток $O(b\Gamma_1/T^2)$ поглощает $b$-независимый вклад $2\varepsilon_1/(NT) = O(\ell\,\Gamma_1/T^2)$ только при $b \gtrsim \ell$; в рабочем режиме $b\gg\ell$ это выполнено);
- **память второго порядка** $\Gamma_2<\infty$: $|\rho(b)| \le \frac{1}{b^2}\sum_{|k|\ge b}k^2|\gamma(k)| = o(1/b^2)$ (используется $|k|-b \le k^2/b$);
- настоящий $\Theta(1/b^2)$ возникает только на границе $\gamma(k) = C|k|^{-3}$ (двусторонняя): тогда $\rho(b) \to \frac{C}{b^2}$ (проверено численно: $b^2\rho(b)\to C$ с точностью $10^{-7}$; ранее здесь стояло $\frac{C}{2b^2}$ — потеря множителя 2 от отрицательных лагов); при $\gamma(k)\asymp C|k|^{-r}$, $r>2$, вообще $\rho(b) \sim \frac{2C}{(r-1)(r-2)}\,b^{1-r}$.

Итого: писать $c_2/b^2$ в шаблоне корректно как **верхнюю огибающую** при условии $\Gamma_2<\infty$ (что и оговорено в заметке про $r(\eta)$), но для наших геометрически эргодичных цепей при фиксированном $\alpha$ практически значимый «второй член» — не $c_2/b^2$, а центрирование $-\sigma_\infty^2\,b/T$; а при убывающем шаге — рост $\ell(T)$, см. §8.

## 5. Масштабы коэффициентов и связь с LSA

Линеаризация LSA даёт для проекции AR(1)-структуру $\phi = 1 - c\alpha$, $c \asymp |\mathrm{Re}\,\lambda(\bar A)|$, и

$$\frac{\Gamma_1}{\sigma_\infty^2} = \frac{2\phi}{1-\phi^2} \;\approx\; \frac{1}{c\alpha} \;=\; \ell,$$

т.е. $|c_1|/\sigma_\infty^2 \asymp \ell$, $|c_3|/\sigma_\infty^2 = 1$. Относительные смещения: усечение $\asymp \ell/b$, центрирование $= b/T$. Они сравниваются при $b \asymp \sqrt{\ell T}$, но MSE-оптимум сидит раньше, при $b^{*} \asymp (\Gamma_1^2 T/\sigma_\infty^4)^{1/3} \asymp \ell^{2/3}T^{1/3}$, где усечение доминирует над центрированием в $(2/3)^{2/3}(T/\ell)^{1/3}$ раз — поэтому lugsail бьёт по правильному члену (точные константы $b^{*}$ и $\mathrm{MSE}^{*}$ — в §7). Для RR-итератов с меньшим $\alpha$ (у нас $\alpha=0.02$ против $0.2$) $\ell$ в 10 раз больше — усечённое смещение в 10 раз тяжелее, что и объясняет, почему lugsail важнее именно в RR-ветке.

## 6. Следствие для lugsail (OBM-LW)

Подстановка тождества в $\hat\sigma^2_{LW}(b) = \frac{\lambda}{\lambda-1}\hat\sigma^2(\lambda b) - \frac{1}{\lambda-1}\hat\sigma^2(b)$:

$$\mathbb{E}\bigl[\hat\sigma^2_{LW}(b)\bigr]
= \sigma_\infty^2
\;-\; (\lambda+1)\,\frac{b}{T}\,\sigma_\infty^2
\;+\; \frac{\lambda\,\rho(\lambda b) - \rho(b)}{\lambda-1}
\;+\; O\!\Bigl(\frac{b\,\Gamma_1}{T^2}\Bigr).$$

- Член $-\Gamma_1/b$ уничтожается тождественно (для этого и строилось).
- Кернельный остаток при геометрическом затухании $\approx -\frac{\rho(b)}{\lambda-1} < 0$ (хвост на масштабе $\lambda b$ пренебрежим рядом с хвостом на $b$) — экспоненциально мал.
- **Главный остаток — центрирование, усиленное в $\lambda+1$ раз**: при $\lambda=2$ это $-3\,\frac{b}{T}\,\sigma_\infty^2$.
- Точный вид $O(\cdot)$-хвоста: $-(\lambda+1)\frac{b}{T}\bigl(\frac{\Gamma_1}{T}-\rho(T)\bigr)$ плюс $b$-независимая $\varepsilon_1$-комбинация порядка $O\bigl(\frac{\tilde\Gamma_1+\Gamma_2}{T^2}\bigr)$; запись $O\bigl(\frac{b\,\Gamma_1}{T^2}\bigr)$ корректна при $b\gtrsim\ell$ (константа $\propto\lambda+1$).

Проверяемое предсказание для `run_lugsail_bias_variance.py`: в квазистационарном режиме ($b \gg \ell$, фиксированный $\alpha$) смещение OBM при больших $b$ линейно по $b$ с наклоном $-\sigma_\infty^2/T$, а у OBM-RR — с наклоном $-3\sigma_\infty^2/T$. При положительной корреляции **обе кривые остаются строго ниже нуля и подходят к нему снизу** — пересечения нуля нет (в первой версии заметки утверждалось обратное для OBM; опровергнуто точным тождеством: все члены смещения $\le0$). Кривая OBM немонотонна («горб»): ближайший подход к нулю $\approx -2\sigma_\infty^2\sqrt{\ell/T}$ достигается при $b\approx\sqrt{\ell T}$ (AR(1), $\phi=0.9$, $T=10^5$: максимум $-0.370$ при $b\approx970$, предсказание $-0.3699$ при $b=973$); LW подходит к нулю на порядок ближе и при много меньших $b$, затем уходит вниз с наклоном $-3\sigma_\infty^2/T$. Две оговорки для самого прогона: (i) раннер усредняет OBM-RR **после** клампа $\max(\cdot,0)$ (`run_lugsail_bias_variance.py`, `rr_clamped`), поэтому при больших $b$ измеренный наклон будет систематически менее отрицательным, чем $-3\sigma_\infty^2/T$, тогда как OBM не клампится; (ii) ветки const/RR работают на $T_{\mathrm{post}} = T - \mathrm{burn\_in}$ при номинальном $T$ в логах — их истинный наклон $-\sigma_\infty^2/T_{\mathrm{post}}$ (на ~11% круче при $T=10^4$); чисто предсказание тестирует только PR-ветка (без burn-in).

**Замечание о нормировке.** Нормировка Флегала–Джонса $\frac{Tb}{(T-b)(T-b+1)}$ отличается от нашей множителем $\frac{T}{T-b}$ и в точности убивает член $-\frac{b}{T}\sigma_\infty^2$ (в IID-случае оценщик становится строго несмещённым; остаток — $-\Gamma_1/T$). В коде это однострочная поправка `* T/(T-b)` в `_obm_variance_from_proj`, и для lugsail выигрыш утраивается. Но это смена конвенции оценщика — менять её среди готовых прогонов не стоит без отдельного решения (сравнимость с сериями результатов и с бумагой Самсонова, где нормировка $\frac{b}{T-b+1}$).

## 7. Явный MSE: $\mathrm{bias}^2 + \operatorname{Var}$ через $\sigma_\infty^2$ и $\Gamma_1$ (дополнение 2026-08-05)

### 7.1. Закон дисперсии и его lugsail-инфляция

Для OBM (Flegal–Jones 2010, Thm 4: $(T/b)\operatorname{Var} \to c\,\sigma^4$ с $c=4/3$ для OBM против $c=2$ для непересекающихся BM):

$$\operatorname{Var}\bigl[\hat\sigma^2(b)\bigr] = \frac{4}{3}\,\frac{b}{T}\,\sigma_\infty^4\,\bigl(1+o(1)\bigr)$$

(условия: стационарность + суммируемость четвёртых кумулянтов, $b\to\infty$, $b/T\to0$; конечно-выборочная поправка отрицательна, по точному гауссовскому расчёту $\approx \frac43 - 1.1\,\frac{b}{T}$, т.е. на уровне процента при $b/T\le0.02$). Для lugsail-комбинации эквивалентное лаг-окно — «плоская вершина»: $W(x)=1$ при $|x|\le1$ и $W(x)=\frac{\lambda-|x|}{\lambda-1}$ при $1\le|x|\le\lambda$ (в единицах $x=k/b$), откуда через $\operatorname{Var} \approx \frac{b}{T}\sigma_\infty^4\cdot 2\!\int W^2$:

$$\operatorname{Var}\bigl[\hat\sigma^2_{LW}(b)\bigr] = \frac{4(\lambda+2)}{3}\,\frac{b}{T}\,\sigma_\infty^4\,\bigl(1+o(1)\bigr),$$

при $\lambda=2$ это $\frac{16}{3}$ — ровно **×4** к OBM при том же $b$ (эквивалентно ×2 к OBM с окном $2b$; совпадает с $\int k_L^2 = \frac43$ против $\frac23$ у Бартлетта в Vats–Flegal 2022, Appendix A). Составные части: $\operatorname{Cov}\bigl(\hat\sigma^2(b),\hat\sigma^2(\lambda b)\bigr) \approx \frac{2(3\lambda-1)}{3\lambda}\,\frac{b}{T}\,\sigma_\infty^4$ (при $\lambda=2$: $\frac53$), корреляция $\frac{3\lambda-1}{2\lambda^{3/2}}$ ($\approx0.88$ при $\lambda=2$); тождество $\operatorname{Var}(aX-cY)=a^2\operatorname{Var}X+c^2\operatorname{Var}Y-2ac\operatorname{Cov}$ с этими константами замыкается на $\frac{4(\lambda+2)}{3}$ точно ($\lambda^3-3\lambda+2=(\lambda-1)^2(\lambda+2)$). MC-проверка (гауссовский AR(1), $\phi=0.5$, $T=2\cdot10^4$, 4000 реплик, оба окна на одних траекториях): $4/3$, $16/3$ ($\lambda=2$), $20/3$ ($\lambda=3$), ковариация и корреляция — всё в пределах $\sim$1 MC-ошибки ($\approx2\%$); эмпирические отношения $\operatorname{Var}_{LW}/\operatorname{Var}_{OBM} = 4.01$–$4.04$.

### 7.2. MSE OBM: явное $\mathbb{E}\bigl[(\hat\sigma^2(b)-\sigma_\infty^2)^2\bigr]$

По определению $\mathrm{MSE}(b) = \mathbb{E}\bigl[(\hat\sigma^2(b)-\sigma_\infty^2)^2\bigr] = \mathrm{Bias}(b)^2 + \operatorname{Var}\bigl[\hat\sigma^2(b)\bigr]$, и смещение известно **точно** из §2:

$$\mathrm{Bias}(b) \;=\; -\frac{\Gamma_1}{b} \;-\; \frac{b}{T}\,\sigma_\infty^2 \;+\; \rho(b) \;+\; \delta(b),
\qquad
\delta(b) = -\frac{b}{T}\Bigl(\frac{\Gamma_1}{T}-\rho(T)\Bigr) + \frac{2\varepsilon_1}{NT},
\quad |\delta| \lesssim \frac{b\,\Gamma_1}{T^2}+\frac{\tilde\Gamma_1+\Gamma_2}{T^2},$$

так что единственная асимптотика во всей формуле — закон дисперсии из §7.1. Раскрывая квадрат смещения и подставляя $\operatorname{Var} = \frac43\frac{b}{T}\sigma_\infty^4(1+o(1))$:

$$\boxed{\;
\mathbb{E}\bigl[(\hat\sigma^2(b)-\sigma_\infty^2)^2\bigr]
= \underbrace{\frac{\Gamma_1^2}{b^2}}_{\text{усечение}^2}
+ \underbrace{\frac43\,\frac{b}{T}\,\sigma_\infty^4}_{\text{дисперсия}}
+ \underbrace{\frac{2\,\Gamma_1\sigma_\infty^2}{T}}_{\text{перекрёстный}}
+ \underbrace{\frac{b^2}{T^2}\,\sigma_\infty^4}_{\text{центрирование}^2}
\underbrace{\,-\,2\rho(b)\Bigl(\frac{\Gamma_1}{b}+\frac{b}{T}\sigma_\infty^2\Bigr)+\rho(b)^2}_{\text{эксп. малые при геом. перемешивании}}
+\; \mathcal{R}(b)
\;}$$

с явно оценённым остатком (первое слагаемое — $o(1)$ из закона дисперсии, второе — вклады $\delta$: $-2\delta\cdot(\mathrm{Bias}-\delta)$ и $\delta^2$):

$$|\mathcal{R}(b)| \;\lesssim\; \frac{b}{T}\,\sigma_\infty^4\cdot o(1)
\;+\; \Bigl(\frac{\Gamma_1}{b}+\frac{b}{T}\sigma_\infty^2\Bigr)\cdot\Bigl(\frac{b\,\Gamma_1}{T^2}+\frac{\tilde\Gamma_1+\Gamma_2}{T^2}\Bigr).$$

При $b=T^\eta$ иерархия членов: $T^{-2\eta}$ (усечение²), $T^{\eta-1}$ (дисперсия), $T^{-1}$ (перекрёстный), $T^{2\eta-2}$ (центрирование²) — главные всегда первые два при любом $\eta\in(0,1)$: перекрёстный не зависит от $b$ и его $T^{-1}$ мажорируется $\max(T^{-2\eta},T^{\eta-1})$, а центрирование² подавлено множителем $b/T$ относительно дисперсии. Оптимум и его цена:

$$b^{*} = \Bigl(\frac{3\,\Gamma_1^2\,T}{2\,\sigma_\infty^4}\Bigr)^{1/3} = \Bigl(\tfrac32\Bigr)^{1/3}\ell^{2/3}\,T^{1/3},
\qquad
\mathrm{MSE}^{*} = 3\Bigl(\tfrac23\Bigr)^{2/3}\sigma_\infty^4\Bigl(\frac{\ell}{T}\Bigr)^{2/3}
\approx 2.29\,\sigma_\infty^4\Bigl(\frac{\ell}{T}\Bigr)^{2/3},$$

где $\ell = \Gamma_1/\sigma_\infty^2$; на оптимуме дисперсия ровно вдвое больше $\mathrm{bias}^2$ (условие первого порядка). Численно (AR(1), $\phi=0.9$, $T=10^6$): argmin полной кривой (точное тождество + $\frac43\frac{b}{T}\sigma^4$) — $b=512$ против $b^{*}=512.5$; полный MSE превышает двухчленный $\mathrm{MSE}^{*}$ на 1.9%, и это в точности перекрёстный член плюс $(b^{*}\sigma_\infty^2/T)^2$.

### 7.3. MSE lugsail: явное $\mathbb{E}\bigl[(\hat\sigma^2_{LW}(b)-\sigma_\infty^2)^2\bigr]$

Точное смещение (комбинация тождества §2 при окнах $b$ и $\lambda b$; член $\Gamma_1/b$ сокращается тождественно):

$$\mathrm{Bias}_{LW}(b) = -(\lambda+1)\frac{b}{T}\sigma_\infty^2 + \frac{\lambda\rho(\lambda b)-\rho(b)}{\lambda-1} + \delta_{LW}(b),
\qquad
\delta_{LW} = -(\lambda+1)\frac{b}{T}\Bigl(\frac{\Gamma_1}{T}-\rho(T)\Bigr) + \frac{2}{\lambda-1}\Bigl[\frac{\lambda\,\varepsilon_1(\lambda b)}{N_{\lambda b}T}-\frac{\varepsilon_1(b)}{N_b T}\Bigr],$$

где $\varepsilon_1(m), N_m$ — величины из §2 при окне $m$, $|\delta_{LW}| \lesssim \frac{b\Gamma_1}{T^2} + \frac{\tilde\Gamma_1+\Gamma_2}{T^2}$. Отсюда, с законом дисперсии §7.1:

$$\boxed{\;
\mathbb{E}\bigl[(\hat\sigma^2_{LW}(b)-\sigma_\infty^2)^2\bigr]
= \underbrace{(\lambda+1)^2\frac{b^2}{T^2}\,\sigma_\infty^4}_{\text{центрирование}^2}
+ \underbrace{\frac{4(\lambda+2)}{3}\,\frac{b}{T}\,\sigma_\infty^4}_{\text{дисперсия}}
+ \underbrace{\frac{2(\lambda+1)\,b\,\sigma_\infty^2}{T}\cdot\frac{\rho(b)-\lambda\rho(\lambda b)}{\lambda-1}
+ \Bigl(\frac{\lambda\rho(\lambda b)-\rho(b)}{\lambda-1}\Bigr)^2}_{\text{эксп. малые при геом. перемешивании}}
+\; \mathcal{R}_{LW}(b)
\;}$$

с тем же типом остатка: $|\mathcal{R}_{LW}| \lesssim \frac{b}{T}\sigma_\infty^4\cdot o(1) + \bigl(\frac{b}{T}\sigma_\infty^2 + \frac{|\rho(b)|}{\lambda-1}\bigr)\cdot\bigl(\frac{b\Gamma_1}{T^2}+\frac{\tilde\Gamma_1+\Gamma_2}{T^2}\bigr)$.

Оба полиномиальных члена **растут** по $b$, поэтому оптимум определяется балансом экспоненциального $\rho(b)^2$-хвоста с дисперсией: при геометрическом перемешивании $b^{*}_{LW}$ решает $b \approx \frac{\ell}{2}\log\bigl(\mathrm{const}\cdot\tfrac{T}{b^2}\bigr)$, т.е. живёт на шкале $\ell\log(\cdot)$ (численно при $\phi=0.9$, $T=10^6$, $\lambda=2$: argmin полной кривой $b=38$ при $\ell\approx9.5$, $\ell\log T = 131$) и

$$\mathrm{MSE}^{*}_{LW} = O\!\Bigl(\sigma_\infty^4\,\frac{\ell\,\log T}{T}\Bigr)$$

— почти параметрическая скорость: асимптотически лучше и $T^{-2/3}$ у OBM, и $T^{-4/5}$ полиномиального шаблона $c_2/b^2$ (последний остаётся верхней огибающей при $\Gamma_2<\infty$ без геометрии: там $b^{*}_{LW}\asymp(c_2^2T/\sigma^4)^{1/5}$). На примере $\phi=0.9$, $T=10^6$: $\mathrm{MSE}^{*}_{LW}$ в 4.7 раза меньше OBM-ного $\mathrm{MSE}^{*}$. **Практическое следствие:** оптимумы живут на разных шкалах — lugsail при OBM-ном $b^{*}=512$ проигрывает своему же оптимуму в ~12 раз (душит ×4-дисперсия). Так что общий $b_n\sim T^{0.6}$ для обеих веток заведомо неоптимален для LW по MSE; для покрытия CI, впрочем, важнее малое смещение при недоборе — отдельный trade-off, который MSE не решает.

### 7.4. Сверка с arXiv:2505.08456 (Moulines–Naumov–Samsonov)

Их Theorem 1 при $p=2$ даёт $\mathrm{MSE}^{1/2} \lesssim \frac{t_{\mathrm{mix}}^3}{\sqrt{T}} + t_{\mathrm{mix}}^2\sqrt{\frac{b}{T}} + \frac{t_{\mathrm{mix}}^2}{\sqrt{b}}$ — те же две ветки ($b/T$ и убывание по $b$), но грубее точного MSE в обеих: $1/\sqrt{b}$ вместо $\Gamma_1/b \lesssim t_{\mathrm{mix}}^2/b$ (их $\bar R_n$-остаток, Lemma 5 — флуктуационная оценка через Минковского, а не смещение) и $t_{\mathrm{mix}}^4\,b/T$ вместо $\sigma_\infty^4\,b/T \lesssim t_{\mathrm{mix}}^2\,b/T$ (sup-норменные потери в двойном Беркхолдере, их $D_2$). Зато у них — все $p$-моменты и произвольное начальное распределение $\xi$, чего стационарное тождество §2 не даёт. Структурно их точное представление (eq. (12)) $\hat\sigma^2 = \frac{b}{N}X^\top B^\top BX + b(u^\top X)^2 - b(v^\top X)^2$ — это в точности наши «блочный член $A$ + центрирование $-2B+C$», а их тентовые диагональные веса (13)–(14) — наши $w_t/(bN)$ (проверено до машинной точности).

Скрипты: `tmp/check_obm_variance_const.py` (MC $4/3$), `tmp/check_lugsail_variance_const.py` (MC $\frac{4(\lambda+2)}{3}$, ковариация, корреляция), `tmp/check_mse_optimal_b.py` (полные кривые MSE, $b^{*}$, lugsail-оптимум).

## 8. Что меняется для нестационарных итератов LSA

Тождество §2 — про стационарный суррогат. Для реальных итератов добавляются два эффекта, и оба видно из той же квадратичной формы.

**(a) Дрейф среднего.** Если $Y_t = \mu_t + Z_t$ с центрированным $Z$, то точно

$$\mathbb{E}\bigl[\hat\sigma^2(b; Y)\bigr] = \underbrace{\frac{b}{N}\sum_j(\bar\mu_j - \bar\mu_T)^2}_{\ge 0} + \mathbb{E}\bigl[\hat\sigma^2(b; Z)\bigr]$$

(перекрёстный член линеен по $Z$ и имеет нулевое среднее). Оценщик инвариантен к сдвигу на константу, поэтому стационарное марковское смещение $\mathbb{E}\theta_\infty^{(\alpha)} - \theta^\star = O(\alpha)$ (то, что убирает RR по $\alpha$) сюда **не** попадает — вклад даёт только затухающий транзиент $\mu_t - \mu_\infty$. Но он **не** сосредоточен в первых $O(\ell)$ блоках (как утверждала первая версия заметки с порядком $O(\Delta_0^2\ell^3/(bT))$): транзиент сдвигает и глобальное среднее, $\bar\mu_T - \mu_\infty \approx \Delta_0\ell/T$, и этот сдвиг чувствуют **все** $\sim N$ блоков. Точный порядок для $\mu_t-\mu_\infty = \Delta_0 e^{-t/\ell}$, $1\ll\ell\ll b\ll T$ (проверено численно с константами, `tmp/check_obm_drift_term.py`):

$$\frac{b}{N}\sum_j(\bar\mu_j-\bar\mu_T)^2 \;=\; \Delta_0^2\Bigl[\frac{\ell^3}{2bT} + \frac{\ell^2 b}{T^2}\Bigr]\bigl(1+o(1)\bigr) \;=\; \Theta\!\Bigl(\Delta_0^2\,\ell^2\Bigl(\frac{\ell}{bT}+\frac{b}{T^2}\Bigr)\Bigr),$$

кроссовер при $b\approx\sqrt{\ell T/2}$; при кодовом $b\sim T^{0.6}$ второй (ранее пропущенный) член доминирует как раз в быстро-перемешивающем случае $\ell < T^{0.2}$, а при убывающем шаге с $\ell(T)\sim T^{0.65}$ — первый. $\Delta_0$ — начальное смещение. Положительный знак — единственный источник смещения вверх.

**(b) Медленно меняющиеся автоковариации** (убывающий шаг $\alpha_t$): подсчёт пар из шага 2 остаётся верным поблочно, но $\gamma$ становится локальной, и

$$c_1 \;\longrightarrow\; -\bar\Gamma_1(T) = -\frac{1}{N}\sum_j \Gamma_1(t_j), \qquad \Gamma_1(t) \asymp \sigma_\infty^2\,\ell(t), \quad \ell(t) \asymp \frac{(t+k_0)^{\gamma}}{c\,c_0}.$$

«Константа» $c_1$ растёт как $T^{\gamma}$ — это в точности механизм, из-за которого high-$T$ прогон показал нуль скорости смещения на $\eta \approx \gamma = 0.65$ и отсутствие левой ветки $\min$-формулы при $\eta<\gamma$ (`conversations/2026-07-29_obm-lw-rate-min-formula-assumptions.md`, постскриптум; `reports/2026-07-29_lugsail_highT_smalleta.md`).

## 9. Сверка с литературой

- **Vats–Flegal 2022** (Thm 4 для BM; для стандартного SV — Thm 2, а Cor 1 — это lugsail-версия): $\mathrm{Bias} = \Gamma/b + o(1/b)$ с их $\Gamma = -\sum_{k\ge1}k[R(k)+R(k)^\top]$, т.е. $= -\Gamma_1$ в нашей скалярной записи — совпадает; наш §2 уточняет $o(1/b)$ до явных членов для точного Бартлетта-OBM. Наша комбинация $\frac{\lambda}{\lambda-1}\hat\sigma^2(\lambda b)-\frac{1}{\lambda-1}\hat\sigma^2(b)$ при $\lambda=2$ — это в точности их zero lugsail $(r=2,\,c=1/2)$ на ширине $2b$.
- **Liu–Vats–Flegal 2022**, eq. (4): их ключевое ковариационное соотношение $\operatorname{Cov}[\bar Y_l(k)] - \operatorname{Cov}[\bar Y] = \frac{n-k}{kn}(\Sigma + \frac{n+k}{kn}\Gamma + o(k^{-2}))$ — это шаги 2–3 в свёрнутом виде.
- **Ng–Perron 1996**, (4.6)–(4.7): поправки unknown-mean $O(M_T/T)$ — это наши члены центрирования; их вывод объясняет, почему при $b$, сравнимом с $T$, эти члены становятся главными.
- **Samsonov et al. 2025**, Prop. 3: даёт **концентрацию** $\hat\sigma^2$ вокруг $\sigma^2(u)$ со скоростями (включая член $t_{\mathrm{mix}}^2\|\varepsilon\|_\infty^2/\sqrt{b_n}$ — грубее нашего $\Gamma_1/b$), но не разложение смещения; для итератов там сначала сводят $\hat\sigma^2_\theta$ к OBM по ненаблюдаемому шуму $\varepsilon(Z_\ell)$ (Prop. 2). Честная теорема для итератов LSA должна пройти тот же маршрут: наш §2 применяется к стационарному шумовому процессу, а переход «итераты $\to$ шум» оплачивается остатком типа $\mathcal{R}_{\mathrm{var}}$.
- **Moulines–Naumov–Samsonov 2025** (arXiv:2505.08456; `papers/Moulines_Naumov_Samsonov_2025_OBM_Concentration_Note.pdf`): концентрация OBM для равномерно геометрически эргодичных цепей через мартингейльное разложение квадратичной формы (метод Atchadé–Cattaneo); Theorem 1 — оценка всех $p$-моментов $|\hat\sigma^2_{OBM}-\sigma_\infty^2|$ с явной зависимостью от $p$ и $t_{\mathrm{mix}}$, при произвольном старте. Разложения смещения не содержит (подробная сверка при $p=2$ — §7.4); их представление (12) и тентовые веса (13)–(14) — прямой двойник нашей квадратичной формы $A - 2B + C$ и весов $w_t$.

## 10. Открытые хвосты

1. Для марковского шума $\gamma(k)$ имеет **два масштаба** — быстрый (перемешивание цепи, $\ell_{\mathrm{chain}}$) и медленный (AR-рекурсия, $1/(c\alpha)$); соответственно $\Gamma_1 = \Gamma_1^{\mathrm{fast}} + \Gamma_1^{\mathrm{slow}}$ с $\Gamma_1^{\mathrm{slow}}$ доминирующим при $\alpha \ell_{\mathrm{chain}} \ll 1$. Выписать обе константы через $\bar A$, $\Gamma_\varepsilon$ и переходное ядро — отдельная задача.
2. Строгая версия §8(b): нестационарное тождество с $\gamma_t(k)$ и контроль ошибки замороженного коэффициента на масштабе блока (большое $k_0$ этому помогает).
3. Вариационная сторона на уровне констант закрыта в §7 ($\frac43$, $\frac{4(\lambda+2)}{3}$, ковариация; MC-проверка на гауссовском AR(1)); открытым остаётся строгий вывод этих констант для марковского (негауссовского) случая — условия на четвёртые кумулянты, ср. заметку про $r(\eta)$.

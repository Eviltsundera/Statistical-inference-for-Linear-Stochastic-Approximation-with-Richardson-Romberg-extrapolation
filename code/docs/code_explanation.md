# Code Explanation

Подробное описание кода для сравнения стратегий статистического вывода (inference) в задаче линейной стохастической аппроксимации (LSA) с марковским шумом.

## Оглавление

1. [Архитектура](#1-архитектура)
2. [Генерация марковской цепи](#2-генерация-марковской-цепи)
3. [Генерация задачи LSA](#3-генерация-задачи-lsa)
4. [LSA Engine — итерации](#4-lsa-engine--итерации)
   - [4.1 Постоянный шаг](#41-run_lsa_const--постоянный-шаг-huo-et-al)
   - [4.2 Убывающий шаг с CLTZ20](#42-run_lsa_diminishing--убывающий-шаг-с-cltz20-батчингом)
   - [4.3 Polyak-Ruppert](#43-run_lsa_polyak_ruppert--polyak-ruppert-усреднение-samsonov-et-al)
   - [4.4 Richardson-Romberg](#44-run_rr--richardson-romberg-экстраполяция)
5. [Inference — построение CI](#5-inference--построение-доверительных-интервалов)
   - [5.1 Batch-mean CI](#51-batch_mean_ci--huo-et-al-2023)
   - [5.2 OBM CI](#52-obm_ci--samsonov-et-al-2025)
   - [5.3 MSB Bootstrap CI](#53-msb_ci--multiplier-subsample-bootstrap)
6. [Обработка дивергенций](#6-обработка-дивергенций)
7. [Оркестрация эксперимента](#7-оркестрация-эксперимента)
8. [Метрики и вывод](#8-метрики-и-вывод)

---

## 1. Архитектура

```
run_comparison.py               -- точка входа, параллелизация, вывод
lsa_inference/
  markov_chain.py               -- генерация цепи Маркова, симуляция
  lsa_problem.py                -- генерация A(x), b(x), theta*
  lsa_engine.py                 -- LSA итерации (4 режима + RR)
  inference.py                  -- построение CI (3 метода)
```

Поток данных одной задачи:

```
generate_problem()                          [run_comparison.py:36-43]
    -> P, pi, A_bar, theta_star, A_arr, b_arr

simulate_chains_batch(P, pi, T, n_traj)     [markov_chain.py:25-45]
    -> trajs: (n_traj, T) — индексы состояний

run_lsa_* / run_rr                          [lsa_engine.py]
    -> batch_means: (n_traj, K, d)    -- для Huo методов
    -> all_thetas:  (n_traj, T, d)    -- для Samsonov методов

batch_mean_ci / obm_ci / msb_ci            [inference.py]
    -> l2_errors, ci_widths, coverages — массивы (n_traj,)
```

Все 7 методов для одной задачи работают на **одних и тех же** траекториях марковской цепи — fair comparison.

---

## 2. Генерация марковской цепи

Файл: [`lsa_inference/markov_chain.py`](../lsa_inference/markov_chain.py)

### `generate_transition_matrix` (строка 7)

Генерирует случайную неприводимую апериодическую матрицу перехода:

```python
def generate_transition_matrix(n_states, rng):
    while True:
        M = rng.uniform(0, 1, (n_states, n_states))       # шаг 1: случайная матрица
        P = M / M.sum(axis=1, keepdims=True)               # шаг 2: нормализуем построчно
        vals, vecs = eig(P.T)                              # шаг 3: собственные значения P^T
        idx = np.argmin(np.abs(vals - 1.0))                # шаг 4: находим lambda=1
        pi = np.real(vecs[:, idx])                         # шаг 5: соответствующий собственный вектор
        pi /= pi.sum()                                     # шаг 6: нормализуем в распределение
        if np.all(pi > 0):                                 # шаг 7: проверяем, что все состояния достижимы
            return P, pi
```

**Что происходит:**
- Строим стохастическую матрицу из U[0,1] — строки нормализуются до 1
- Стационарное распределение `pi` — левый собственный вектор при собственном значении 1: `pi^T P = pi^T`
- Условие `pi > 0` гарантирует неприводимость (все состояния посещаются)
- Если условие нарушено — перегенерируем (на практике почти всегда выполнено с первого раза)

### `simulate_chains_batch` (строка 25)

Векторизованная симуляция `n_traj` цепей одновременно:

```python
def simulate_chains_batch(P, pi, T, n_traj, rng):
    cum_P = np.cumsum(P, axis=1)                           # кумулятивные вероятности переходов
    cum_pi = np.cumsum(pi)                                 # кумулятивное стационарное распределение

    trajs = np.empty((n_traj, T), dtype=np.int32)

    u = rng.uniform(size=n_traj)                           # n_traj случайных чисел
    trajs[:, 0] = np.searchsorted(cum_pi, u)               # начальные состояния из pi

    for t in range(1, T):
        u = rng.uniform(size=n_traj)
        prev = trajs[:, t - 1]                             # текущие состояния всех цепей
        trajs[:, t] = (u[:, None] < cum_P[prev]).argmax(axis=1)  # переход по P
```

**Трюк с `argmax`:**
- `cum_P[prev]` — матрица `(n_traj, n_states)` кумулятивных вероятностей
- `u[:, None] < cum_P[prev]` — boolean матрица `(n_traj, n_states)`
- `argmax(axis=1)` — индекс первого `True` в каждой строке = новое состояние
- Эквивалентно inverse CDF sampling, но без цикла по траекториям

---

## 3. Генерация задачи LSA

Файл: [`lsa_inference/lsa_problem.py`](../lsa_inference/lsa_problem.py)

### `generate_A` (строка 6) — матрицы A(x) с Hurwitz средним

```python
def generate_A(n_states, d, pi, rng):
    M_A = rng.standard_normal((d, d))                      # случайная d x d
    evals = np.linalg.eigvals(M_A)
    max_real = np.max(np.real(evals))
    if max_real >= 0:                                      # если не Hurwitz:
        M_A -= 2 * max_real * np.eye(d)                    #   сдвигаем спектр влево

    A_bar = M_A.copy()                                     # среднее E_pi[A(x)]
```

**Зачем Hurwitz?** Условие `Re(lambda_i(A_bar)) < 0` гарантирует стабильность LSA — без него итерации расходятся. Сдвиг `M -= 2*max_re*I` переносит все собственные значения в левую полуплоскость:
```
lambda_new = lambda_old - 2*max_re
если lambda_old = max_re:  lambda_new = -max_re < 0  ✓
если lambda_old < max_re:  lambda_new < -max_re < 0  ✓
```

Далее — создание state-dependent шума:

```python
    E = rng.uniform(-1, 1, (n_states - 1, d, d))          # шум для x=0..n-2
    A_list = [A_bar + E[x] for x in range(n_states - 1)]   # A(x) = A_bar + E(x)

    # Последнее состояние — коррекция, чтобы E_pi[A(x)] = A_bar точно
    A_last = A_bar - sum(pi[x] * E[x] for x in range(n_states - 1)) / pi[-1]
    A_list.append(A_last)
```

**Логика коррекции:** нужно `sum_x pi_x A(x) = A_bar`. Для `x < n-1`: `A(x) = A_bar + E(x)`, поэтому:
```
sum_{x=0}^{n-2} pi_x (A_bar + E(x)) + pi_{n-1} A(n-1) = A_bar
A_bar * (1 - pi_{n-1}) + sum pi_x E(x) + pi_{n-1} A(n-1) = A_bar
A(n-1) = (A_bar * pi_{n-1} - sum_{x<n-1} pi_x E(x)) / pi_{n-1}
        = A_bar - sum_{x<n-1} pi_x E(x) / pi_{n-1}
```

### `generate_b` (строка 29) — векторы b(x)

```python
def generate_b(n_states, d, rng):
    return [rng.uniform(-1, 1, d) for _ in range(n_states)]
```

Простой случай: каждый `b(x) ~ U[-1,1]^d` независимо. Коррекция не нужна — нас интересует `b_bar = sum pi_x b(x)`, а не `b_bar = конкретное значение`.

### `compute_theta_star` (строка 38) — целевой вектор

```python
def compute_theta_star(A_list, b_list, pi):
    A_bar = sum(pi[x] * A_list[x] for x in range(n_states))   # E_pi[A(x)]
    b_bar = sum(pi[x] * b_list[x] for x in range(n_states))   # E_pi[b(x)]
    return np.linalg.solve(A_bar, -b_bar)                      # theta* = -A_bar^{-1} b_bar
```

Решаем `A_bar * theta = -b_bar` (steady-state уравнение `A_bar * theta* + b_bar = 0`).

---

## 4. LSA Engine — итерации

Файл: [`lsa_inference/lsa_engine.py`](../lsa_inference/lsa_engine.py)

Все функции **векторизованы** — обрабатывают `n_traj` траекторий одновременно через numpy broadcasting. Центральная операция на каждом шаге (строка 67):

```python
thetas += alpha * (np.einsum('nij,nj->ni', A_t, thetas) + b_t)
```

Это формула `theta_{t+1} = theta_t + alpha * (A(x_t) * theta_t + b(x_t))` для всех n_traj сразу:
- `A_t = A_arr[trajs[:, t]]` — shape `(n_traj, d, d)`, матрица A(x) для каждой траектории
- `np.einsum('nij,nj->ni', A_t, thetas)` = батчевое матрично-векторное произведение `A_t @ thetas`
- `b_t = b_arr[trajs[:, t]]` — shape `(n_traj, d)`

### 4.1. `run_lsa_const` — Постоянный шаг (Huo et al.)

**Строка 45.** Формула: `theta_{t+1} = theta_t + alpha * (A(x_t) theta_t + b(x_t))`, alpha = const.

**Batching (non-overlapping):**

```
итерации:  [0 ... burn_in-1 | burn_in ... burn_in+n-1 | ... | ... burn_in+K*n-1]
            отбрасываем       batch 0                    ...   batch K-1
```

Размер батча: `n = (T - burn_in) // K`.

```python
    thetas = np.zeros((n_traj, d))                     # инициализация theta_0 = 0
    batch_sums = np.zeros((n_traj, K, d))              # аккумуляторы для batch means

    current_batch = 0
    batch_count = 0

    for t in range(T):
        # --- LSA update ---
        x_t = trajs[:, t]                              # (n_traj,) текущие состояния
        A_t = A_arr[x_t]                               # (n_traj, d, d)
        b_t = b_arr[x_t]                               # (n_traj, d)
        thetas += alpha * (np.einsum('nij,nj->ni', A_t, thetas) + b_t)

        if t % 100 == 99:                              # проверка дивергенции каждые 100 шагов
            _clamp_diverged(thetas)

        if t < burn_in:                                # пропускаем burn-in
            continue
        if current_batch >= K:                         # все батчи заполнены
            break

        if batch_count >= n0:                          # n0 — intra-batch discard (по дефолту 0)
            batch_sums[:, current_batch, :] += thetas  # аккумулируем

        batch_count += 1
        if batch_count == n:                           # батч заполнен
            current_batch += 1
            batch_count = 0

    effective = n - n0
    batch_means = batch_sums / effective               # (n_traj, K, d)
```

**Выход:** `batch_means (n_traj, K, d)` — среднее theta по каждому батчу для каждой траектории.

### 4.2. `run_lsa_diminishing` — Убывающий шаг с CLTZ20 батчингом

**Строка 95.** Формула: `alpha_t = alpha_0 / (t+1)^{0.5}`.

Ключевое отличие: **батчи неравной длины**. Раньше (в `run_lsa_const`) все батчи были размера `n`. Здесь батчи растут экспоненциально по формуле CLTZ20:

```python
    r = T ** (1 - alpha_exp) / (K + 1)                # alpha_exp = 0.5
    endpoints = [0]
    for k in range(1, K + 1):
        e_k = int(((k + 1) * r) ** (1 / (1 - alpha_exp)))
        endpoints.append(min(e_k, T))
```

Пример при T=100000, K=31:
```
batch 0:  итерации    0 -   101    (102 шт)   -- коротки, шаг большой, мало корреляции
batch 1:  итерации  102 -   422    (321 шт)
...
batch 30: итерации 82043 - 100000  (17957 шт) -- длинный, шаг мал, нужна декорреляция
```

Зачем: при убывающем шаге поздние итерации сильно коррелированы (шаг мал → цепь "прилипает"). Более длинные батчи нужны, чтобы batch means были примерно независимы.

```python
    batch_for_t = np.full(T, -1, dtype=np.int32)      # к какому батчу принадлежит каждый t
    for k in range(K):
        batch_for_t[endpoints[k]:endpoints[k + 1]] = k

    for t in range(T):
        step = alpha0 / (t + 1) ** alpha_exp           # убывающий шаг
        thetas += step * (np.einsum('nij,nj->ni', A_t, thetas) + b_t)

        k = batch_for_t[t]
        if k >= 0:
            batch_sums[:, k, :] += thetas              # аккумулируем в соответствующий батч
```

### 4.3. `run_lsa_polyak_ruppert` — Polyak-Ruppert усреднение (Samsonov et al.)

**Строка 151.** Формула:
```
alpha_k = c0 / (k + k0)^gamma
theta_bar_n = (1/n) sum_{k=0}^{n-1} theta_k
```

Главное отличие: хранит **все** итераты, а не только batch sums.

```python
    all_thetas = np.empty((n_traj, T, d))              # (n_traj, T, d) — ~400MB при 100x100000x5
    thetas = np.zeros((n_traj, d))

    for t in range(T):
        alpha_t = c0 / (t + k0) ** gamma               # убывающий шаг
        thetas += alpha_t * (np.einsum('nij,nj->ni', A_t, thetas) + b_t)

        if t % 100 == 99:
            _clamp_diverged(thetas)

        all_thetas[:, t, :] = thetas                   # сохраняем каждый итерат

    theta_bar = np.nanmean(all_thetas, axis=1)         # (n_traj, d) Polyak-Ruppert среднее
```

**Зачем хранить все итераты?** OBM и MSB bootstrap нуждаются в overlapping block averages — это скользящие средние по theta_t. Для их вычисления нужен доступ к каждому theta_t.

**Параметры по умолчанию:**
- `gamma = 0.75` — оптимально для Berry-Esseen (Remark 1, Samsonov: `gamma=3/4` дает `n^{-1/4}`)
- `c0 = 0.1` — начальный масштаб
- `k0 = max(100, log(T)^{1/gamma})` — сдвиг, чтобы первые шаги не были слишком большими

### 4.4. `run_rr` — Richardson-Romberg экстраполяция

**Строка 198.** Запускает `run_lsa_const` с M=2 разными шагами на **одних и тех же** траекториях.

```python
def run_rr(A_arr, b_arr, trajs, alphas, K, burn_in=100, n0=0):
    h = rr_coefficients(alphas)                        # Lagrange weights
    all_bm = []
    for m, alpha in enumerate(alphas):
        bm, n_m = run_lsa_const(A_arr, b_arr, trajs, alpha, K, burn_in, n0)
        all_bm.append(bm)                             # batch_means для каждого alpha

    rr_batch_means = sum(h[m] * all_bm[m] for m in range(len(alphas)))
    return rr_batch_means, n
```

**Коэффициенты RR** (строка 186) через Lagrange interpolation:

```python
def rr_coefficients(alphas):
    h = np.ones(M)
    for m in range(M):
        for l in range(M):
            if l != m:
                h[m] *= alphas[l] / (alphas[l] - alphas[m])
    return h
```

Для `alphas = [0.2, 0.02]`:
```
h[0] = 0.02 / (0.02 - 0.2)  = 0.02 / (-0.18) = -1/9  ≈ -0.111
h[1] = 0.2  / (0.2  - 0.02) = 0.2  / 0.18     = 10/9  ≈  1.111
```

Проверка: `h[0] + h[1] = -1/9 + 10/9 = 1` ✓
Bias cancellation: `h[0]*0.2 + h[1]*0.02 = -0.2/9 + 0.2/9 = 0` ✓

**Математика:** Asymptotic bias = `alpha * B^(1) + alpha^2 * B^(2) + ...` (Theorem A.4, Huo).
С двумя шагами RR убирает O(alpha) term: `h_1 * alpha_1 + h_2 * alpha_2 = 0`.
Остаточный bias = O(max(alpha_m)^2) вместо O(alpha).

---

## 5. Inference — построение доверительных интервалов

Файл: [`lsa_inference/inference.py`](../lsa_inference/inference.py)

### 5.1. `batch_mean_ci` — Huo et al. 2023

**Строка 17.** Вход: `batch_means (n_traj, K, d)`.

```python
def batch_mean_ci(batch_means, n, theta_star, n0=0, q=0.05, coord=0):
    z = stats.norm.ppf(1 - q / 2)                     # z_{0.975} = 1.96
    n_traj, K, d = batch_means.shape

    # Точечная оценка: среднее по батчам
    theta_bars = np.nanmean(batch_means, axis=1)       # (n_traj, d)

    # L2 ошибка
    l2_errors = np.linalg.norm(theta_bars - theta_star, axis=1)  # (n_traj,)

    # Оценка дисперсии для координаты coord (формула 4.2 из Huo)
    diffs = batch_means[:, :, coord] - theta_bars[:, coord:coord + 1]  # (n_traj, K)
    var_coord = (n - n0) / K * np.nansum(diffs ** 2, axis=1)          # (n_traj,)
```

**Формула дисперсии:**
```
Sigma_hat_{i,i} = ((n - n0) / K) * sum_{k=1}^K (theta_bar_k[i] - theta_bar[i])^2
```
Это оценка Section 4.2 из Huo et al. Множитель `(n-n0)` нормализует на размер батча.

```python
    # Standard error и CI
    se = np.sqrt(var_coord / (K * (n - n0)))           # sqrt(Sigma_hat / (K * (n-n0)))
    ci_widths = 2 * z * se                             # ширина CI

    lo = theta_bars[:, coord] - z * se                 # нижняя граница
    hi = theta_bars[:, coord] + z * se                 # верхняя граница
    coverages = ((lo <= theta_star[coord]) & (theta_star[coord] <= hi)).astype(float)
```

**CI:** `[theta_bar_i - 1.96 * se, theta_bar_i + 1.96 * se]` — стандартный 95% CI из CLT.

### 5.2. `obm_ci` — Samsonov et al. 2025

**Строка 97.** Вход: `all_thetas (n_traj, T, d)` — все итераты.

Сначала вычисляем OBM variance estimator (строка 61, функция `obm_variance`):

```python
def obm_variance(all_thetas, theta_bar, b_n, u):
    n_traj, T, d = all_thetas.shape
    n_blocks = T - b_n + 1                             # число перекрывающихся блоков

    # Проекция на направление u: theta_t^T u
    proj = np.einsum('ntd,d->nt', all_thetas, u)      # (n_traj, T)

    # Кумулятивная сумма для эффективного вычисления блочных средних
    cumsum_proj = np.concatenate(
        [np.zeros((n_traj, 1)), np.cumsum(proj, axis=1)], axis=1
    )                                                   # (n_traj, T+1)

    # Блочное среднее theta_{b_n,t}^T u = (cumsum[t+b_n] - cumsum[t]) / b_n
    block_avgs = (cumsum_proj[:, b_n:] - cumsum_proj[:, :n_blocks]) / b_n
    # shape: (n_traj, n_blocks)
```

**Трюк с cumsum:** вместо того чтобы считать `sum_{l=t}^{t+b_n-1} theta_l` для каждого `t` (O(T*b_n)), используем `cumsum[t+b_n] - cumsum[t]` — O(T) суммарно.

```python
    # Центрируем относительно общего среднего
    bar_proj = np.einsum('nd,d->n', theta_bar, u)      # (n_traj,)
    diffs = block_avgs - bar_proj[:, None]             # (n_traj, n_blocks)

    # OBM formula (eq. 14 from Samsonov)
    sigma_hat_sq = (b_n / n_blocks) * np.nansum(diffs ** 2, axis=1)
```

**Формула OBM:**
```
sigma_hat^2(u) = b_n / (T - b_n + 1) * sum_{t=0}^{T-b_n} (theta_{b_n,t}^T u - theta_bar^T u)^2
```

Затем в `obm_ci` строим CI:

```python
    se = np.sqrt(sigma_hat_sq / T)                     # standard error
    ci_widths = 2 * z * se
    lo = theta_bar[:, coord] - z * se
    hi = theta_bar[:, coord] + z * se
```

**Выбор `b_n`:** `b_n = T^{3/4}` (Corollary 2 из Samsonov: оптимальный rate O(n^{-1/8}) для оценки дисперсии). При T=100000 это `b_n = 5623`.

### 5.3. `msb_ci` — Multiplier Subsample Bootstrap

**Строка 139.** Вместо plug-in CI (как в OBM), используем bootstrap quantiles.

Предварительно: те же overlapping block averages, что и в OBM.

```python
    # Центрированные блочные средние для координаты coord
    proj = all_thetas[:, :, coord]                     # (n_traj, T)
    cumsum_proj = np.concatenate([np.zeros((n_traj, 1)), np.cumsum(proj, axis=1)], axis=1)
    block_avgs = (cumsum_proj[:, b_n:] - cumsum_proj[:, :n_blocks]) / b_n
    centered = block_avgs - bar_coord[:, None]         # (n_traj, n_blocks)
```

Для каждой траектории — bootstrap:

```python
    for i in range(n_traj):
        c_i = centered[i]                             # (n_blocks,)

        # Multiplier weights: w_t ~ N(0,1)
        weights = rng.standard_normal((n_bootstrap, n_blocks))

        # Bootstrap статистика (eq. 12 из Samsonov 2025):
        # theta_{n,b_n}(u) = sqrt(b_n) / sqrt(n_blocks) * sum_t w_t * c_t
        boot_stats = np.sqrt(b_n) / np.sqrt(n_blocks) * (weights * c_i[None, :]).sum(axis=1)
        # shape: (n_bootstrap,) — одно число на каждый bootstrap replicate

        # Квантили bootstrap распределения
        q_lo = np.percentile(boot_stats, 2.5)         # 2.5-й перцентиль
        q_hi = np.percentile(boot_stats, 97.5)        # 97.5-й перцентиль

        # CI: theta_bar +/- bootstrap quantiles / sqrt(T)
        ci_lo = bar_coord[i] - q_hi / np.sqrt(T)      # NB: знаки развёрнуты!
        ci_hi = bar_coord[i] - q_lo / np.sqrt(T)
```

**Почему знаки развёрнуты?** Bootstrap статистика аппроксимирует `sqrt(n)(theta_bar - theta*)`, а нам нужен CI для `theta*`. Если bootstrap говорит, что `sqrt(n)(theta_bar - theta*) ~ q`, то `theta* ~ theta_bar - q/sqrt(n)`. Upper quantile boot → lower CI bound.

**Идея MSB:** вместо оценки `sigma^2` и использования `theta_bar +/- 1.96*sigma/sqrt(n)`, MSB напрямую аппроксимирует распределение `sqrt(n)(theta_bar - theta*)` через перевзвешивание блочных средних. Это более гибко: работает даже если распределение не совсем нормальное.

---

## 6. Обработка дивергенций

Файл: [`lsa_inference/lsa_engine.py`](../lsa_inference/lsa_engine.py), строки 14-38.

LSA с постоянным шагом alpha=0.2 может расходиться, если спектральный радиус `||I + alpha * A_bar||` >= 1.

### Порог

```python
_DIVERG_THRESH = 1e6
```

Значения > 10^6 считаются дивергенцией. Порог 10^6 (а не 10^10 как раньше) — ловит расходимость до overflow float64.

### Раннее обнаружение

```python
def _clamp_diverged(thetas):
    """Вызывается каждые 100 шагов внутри LSA цикла."""
    bad = ~np.isfinite(thetas) | (np.abs(thetas) > _DIVERG_THRESH)
    if np.any(bad):
        bad_rows = np.any(bad, axis=1)                 # хотя бы одна координата плохая
        thetas[bad_rows] = np.nan                      # вся траектория → NaN
```

**In-place** — мутирует массив thetas напрямую. Расходившаяся траектория помечается NaN и дальше не влияет на batch sums (NaN + что-угодно = NaN).

Вызов в каждом LSA цикле:
```python
        if t % 100 == 99:
            _clamp_diverged(thetas)
```

Частота 100 — компромисс: чаще → больше overhead, реже → overflow может произойти между проверками.

### Подсчёт дивергенций

В [`run_comparison.py`](../run_comparison.py), строка 112:

```python
def _count_diverged(arr):
    """Считает NaN в массиве L2 ошибок."""
    return int(np.sum(np.isnan(arr)))
```

В воркере (строка 139):
```python
    diverged = {}
    for m in METHODS_ORDER:
        n_div = _count_diverged(results[m]['l2'])      # сколько траекторий расходились
        n_ok = n_traj - n_div
        diverged[m] = n_div
        summary[m] = {
            'l2':    float(np.nanmean(results[m]['l2'])) if n_ok > 0 else np.nan,
            ...
        }
```

Расходившиеся траектории:
- Coverage = 0 (CI не покрывает theta*, т.к. NaN)
- L2, width = NaN (исключаются из nanmean/nanmedian)

---

## 7. Оркестрация эксперимента

Файл: [`run_comparison.py`](../run_comparison.py)

### Параллелизация (строка 189)

```python
    with mp.Pool(n_workers) as pool:
        for summary, diverged in pool.imap_unordered(_solve_problem_worker, task_args):
            completed += 1
            for m in METHODS_ORDER:
                for metric in ('l2', 'width', 'cov'):
                    all_results[m][metric].append(summary[m][metric])
                all_diverged[m].append(diverged[m])
```

Каждый воркер (`_solve_problem_worker`, строка 117) получает seed и независимо:
1. Генерирует задачу LSA
2. Симулирует n_traj цепей Маркова
3. Прогоняет все 7 методов на тех же цепях
4. Возвращает `(summary, diverged)`

`imap_unordered` — результаты приходят по мере готовности (не в порядке отправки). Это ОК, т.к. мы просто аккумулируем.

### Воркер (строка 117)

```python
def _solve_problem_worker(args):
    (prob_seed, n_traj, T, n_states, d,
     K, burn_in, pr_c0, pr_k0, pr_gamma, b_n) = args

    rng = np.random.default_rng(prob_seed)             # полностью детерминирован seed'ом
    P, pi, A_bar, theta_star, A_arr, b_arr = generate_problem(n_states, d, rng)

    traj_rng = np.random.default_rng(rng.integers(0, 2**31))
    trajs = simulate_chains_batch(P, pi, T, n_traj, traj_rng)

    boot_rng = np.random.default_rng(rng.integers(0, 2**31))
    results = run_all_methods(
        A_arr, b_arr, trajs, K, burn_in, theta_star, T,
        pr_c0, pr_k0, pr_gamma, b_n, boot_rng
    )
```

**Три RNG:** `rng` → задача, `traj_rng` → цепи, `boot_rng` → bootstrap weights. Все порождены из одного seed → полная воспроизводимость.

### `run_all_methods` (строка 46)

Запускает все 7 методов последовательно:

```python
    # --- Huo et al. 2023: 5 методов ---
    # 1. const 0.2     → run_lsa_const(alpha=0.2) → batch_mean_ci
    # 2. const 0.02    → run_lsa_const(alpha=0.02) → batch_mean_ci
    # 3. RR            → run_rr([0.2, 0.02]) → batch_mean_ci
    # 4. dim 0.2/√k    → run_lsa_diminishing(alpha0=0.2) → batch_mean_ci
    # 5. dim 0.02/√k   → run_lsa_diminishing(alpha0=0.02) → batch_mean_ci

    # --- Samsonov et al. 2025: 2 метода ---
    all_thetas, theta_bar = run_lsa_polyak_ruppert(...)  # один прогон LSA
    # 6. PR + OBM      → obm_ci(all_thetas, ...)
    # 7. PR + MSB      → msb_ci(all_thetas, ...)
```

Методы 6 и 7 используют **один и тот же** LSA прогон — отличается только построение CI.

### Параметры, вычисляемые из T (строки 153-163)

```python
    K = max(int(T ** 0.3), 5)                          # ~31 при T=10^5
    burn_in = min(1000, T // 10)                       # 1000 при T=10^5

    pr_gamma = 0.75                                    # оптимальный для Berry-Esseen
    pr_c0 = 0.1                                        # масштаб шага PR
    pr_k0 = max(100, int(np.log(T) ** (1 / pr_gamma)))# сдвиг PR

    b_n = max(int(T ** 0.75), 10)                      # OBM block size: ~5623 при T=10^5
    b_n = min(b_n, T // 2)                             # safety cap
```

---

## 8. Метрики и вывод

### Три метрики (для каждой траектории)

**L2 error** = `||theta_bar - theta*||_2` (inference.py, строки 37, 118, 167)
- Измеряет суммарную ошибку оценки (bias + variance)
- Для const 0.2: доминирует bias → L2 большой
- Для RR/PR: bias мал → L2 определяется дисперсией

**CI width** = `2 * 1.96 * se` (inference.py, строки 43, 125, 203)
- Ширина 95% доверительного интервала для 1-й координаты
- Чем уже — тем информативнее, но нужен хороший coverage

**Coverage** = `1{theta*_1 in CI}` (inference.py, строки 47, 129, 204)
- 1 если истинное theta*[0] попало в CI, 0 иначе
- Целевой уровень = 95%
- Если coverage << 95% → CI слишком узкий или biased

### Агрегация

**Per-problem** (run_comparison.py, строка 144):
```python
    summary[m] = {
        'l2':    float(np.nanmean(results[m]['l2'])) if n_ok > 0 else np.nan,
        'width': float(np.nanmean(results[m]['width'])) if n_ok > 0 else np.nan,
        'cov':   float(np.nanmean(results[m]['cov'])) if n_ok > 0 else 0.0,
    }
```
`nanmean` по n_traj траекториям, игнорируя NaN (расходившиеся). Для coverage: среднее по 0/1 = доля покрытия.

**Across problems** (run_comparison.py, строки 238-253):
- **Median** — основная таблица (робастна к выбросам от расходимостей)
- **Mean coverage** — показывается рядом для сравнения
- **Percentiles** (10, 25, 50, 75, 90) — покрывают распределение по задачам

### Divergence report (строки 209-224)

```
Method                     Total div  Problems w/div   Max div/prob  Mean div/prob
--------------------------------------------------------------------------------
alpha=0.2 (const)                347              28            100           3.5
alpha=0.02 (const)                 0               0              0           0.0
RR (0.2+0.02)                    347              28            100           3.5
...
```

- `Total div` — суммарно расходившихся траекторий
- `Problems w/div` — сколько задач затронуто
- `Max/Mean div/prob` — максимум и среднее расходившихся на задачу

RR наследует дивергенции от const 0.2, т.к. запускает тот же LSA.

### CSV output (строка 278)

```python
    df = pd.DataFrame(rows)
    df.to_csv('results_comparison.csv', index=False)
```

Столбцы: `method, label, l2_median, width_median, cov_median, cov_mean, diverged_total`.

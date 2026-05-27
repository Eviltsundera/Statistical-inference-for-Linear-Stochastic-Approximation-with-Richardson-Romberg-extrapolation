# Переход в telescoping identity для $J^{(1)}$

## Вопрос

Объяснить переход в подпункте `Telescoping identity for $J^{(1)}$` из
`src/pr_weights/10_misadjustment_depth_two.typ`.

## Нотация

Фиксируем шаг $w$; в тексте сначала написано для $w=\alpha$. Обозначим

$$
B_w = I - w\overline A,
\qquad
\psi_k^{(w)}
  = \widetilde A(Z_k) J_{k-1}^{(0,w)}.
$$

Функция $\overline\psi_w$ определяется не в самом подпункте про telescoping, а
в imported lemma Levin Corollary 6:
`src/appendix/external_inputs.typ`, lemma `<lem:levin-cor-6>`.
Там записано

$$
\overline\psi_w(j,z)
  :=
  \widetilde A(z)j
  -
  \mathbb E_{\Pi_{J^{(0)},w}}
  [\widetilde A(Z_1)J_0^{(0,w)}].
$$

То есть это центрированная версия билинейного члена
$\widetilde A(z)j$.

Рекурсия для первого perturbation term имеет вид

$$
J_k^{(1,w)}
  = B_w J_{k-1}^{(1,w)}
    - w \widetilde A(Z_k)J_{k-1}^{(0,w)}
  = (I-w\overline A)J_{k-1}^{(1,w)} - w\psi_k^{(w)}.
$$

## Сам telescoping

Переносим члены:

$$
J_k^{(1,w)}
  = J_{k-1}^{(1,w)}
    - w\overline A J_{k-1}^{(1,w)}
    - w\psi_k^{(w)}.
$$

Отсюда

$$
w\overline A J_{k-1}^{(1,w)}
  = J_{k-1}^{(1,w)} - J_k^{(1,w)} - w\psi_k^{(w)}.
$$

Делим на $w$:

$$
\overline A J_{k-1}^{(1,w)}
  = -\psi_k^{(w)}
    + \frac{1}{w}\left(J_{k-1}^{(1,w)} - J_k^{(1,w)}\right).
$$

Теперь суммируем по $k=1,\dots,n$:

$$
\overline A \sum_{k=1}^n J_{k-1}^{(1,w)}
  =
  -\sum_{k=1}^n \psi_k^{(w)}
  + \frac{1}{w}\sum_{k=1}^n
      \left(J_{k-1}^{(1,w)} - J_k^{(1,w)}\right).
$$

Левая сумма просто меняет индекс:

$$
\sum_{k=1}^n J_{k-1}^{(1,w)}
  =
  \sum_{k=0}^{n-1} J_k^{(1,w)}.
$$

Правая разность телескопируется:

$$
\sum_{k=1}^n
  \left(J_{k-1}^{(1,w)} - J_k^{(1,w)}\right)
  =
  J_0^{(1,w)} - J_n^{(1,w)}.
$$

Поэтому

$$
\overline A \sum_{k=0}^{n-1} J_k^{(1,w)}
  =
  -\sum_{k=1}^n
      \widetilde A(Z_k)J_{k-1}^{(0,w)}
  + \frac{1}{w}
      \left(J_0^{(1,w)} - J_n^{(1,w)}\right).
$$

Это первая identity в тексте. Она верна для произвольного начального значения:
стационарность здесь еще не нужна.

## Откуда берется centered version

Дальше нужна стационарность. Пусть

$$
m_w := \mathbb E_\pi[J_\infty^{(1,w)}],
\qquad
\mu_w := \mathbb E_\pi[
  \widetilde A(Z_1)J_0^{(0,w)}
].
$$

В стационарности распределения $J_k^{(1,w)}$ и $J_{k-1}^{(1,w)}$ совпадают, так
что, взяв ожидание в рекурсии,

$$
m_w
  =
  (I-w\overline A)m_w - w\mu_w.
$$

Отсюда

$$
\overline A m_w = -\mu_w.
$$

Именно это записано в тексте как

$$
\mathbb E_\pi[
  \widetilde A(Z_1)J_0^{(0,w)}
]
  =
  -\overline A\,\mathbb E_\pi[J_\infty^{(1,w)}].
$$

Теперь вычитаем $n m_w$ из суммы $J^{(1,w)}$. Из уже полученной identity:

$$
\overline A
\sum_{k=0}^{n-1}(J_k^{(1,w)}-m_w)
 =
 -\sum_{k=1}^n \psi_k^{(w)}
 - n\overline A m_w
 + \frac{1}{w}(J_0^{(1,w)}-J_n^{(1,w)}).
$$

Так как $\mu_w=-\overline A m_w$, получаем

$$
-\sum_{k=1}^n \psi_k^{(w)}
 - n\overline A m_w
 =
 -\sum_{k=1}^n(\psi_k^{(w)}-\mu_w).
$$

А по определению Levin Corollary 6

$$
\overline\psi_w(j,z)
  =
  \widetilde A(z)j
  -
  \mathbb E_\pi[
    \widetilde A(Z_1)J_0^{(0,w)}
  ],
$$

то есть

$$
\psi_k^{(w)}-\mu_w
  =
  \overline\psi_w(J_{k-1}^{(0,w)},Z_k).
$$

Поэтому

$$
\overline A
\sum_{k=0}^{n-1}(J_k^{(1,w)}-m_w)
 =
 -\sum_{k=1}^n
    \overline\psi_w(J_{k-1}^{(0,w)},Z_k)
 + \frac{1}{w}(J_0^{(1,w)}-J_n^{(1,w)}).
$$

Наконец, умножаем слева на $\overline A^{-1}$:

$$
\sum_{k=0}^{n-1}(J_k^{(1,w)}-m_w)
 =
 -\overline A^{-1}\sum_{k=1}^n
    \overline\psi_w(J_{k-1}^{(0,w)},Z_k)
 + \frac{1}{w}\overline A^{-1}
    (J_0^{(1,w)}-J_n^{(1,w)}).
$$

При $w=\alpha$ это ровно формула `<eq:J1-telescope>`.

## Что важно не перепутать

1. Первый telescoping шаг не использует стационарность. Он работает для любого
   старта и дает boundary term $(J_0^{(1,w)}-J_n^{(1,w)})/w$.

2. Стационарность нужна только для центрирования:

   $$
   \mathbb E_\pi[\widetilde A(Z_1)J_0^{(0,w)}]
   =
   -\overline A\,\mathbb E_\pi[J_\infty^{(1,w)}].
   $$

3. Знак минус перед суммой $\overline\psi_w$ появляется потому, что в рекурсии
   forcing term стоит как $-w\widetilde A(Z_k)J_{k-1}^{(0,w)}$.

4. Индексация согласована с stationary augmented-chain convention:
   суммируется пара $(J_{k-1}^{(0,w)},Z_k)$, которая имеет тот же закон, что
   $(J_0^{(0,w)},Z_1)$.

#import "../defs.typ": *

== Norm Estimate for $H_j^((n))$

To bound $||H_j^((n))||$ we use the following standard result.

#lemma[
  *(Lyapunov contraction.)*
  Let $-overline(A)$ be a Hurwitz matrix. Then for any
  $P = P^top succ 0$ there exists a unique $Q = Q^top succ 0$ satisfying
  $overline(A)^top Q + Q overline(A) = P$. Let
  $a := lambda_"min"(P) / (2 ||Q||)$,
  $kappa_Q := lambda_"max"(Q) / lambda_"min"(Q)$, and
  $
  alpha_infinity
    := min (
      frac(lambda_"min"(P), 2 kappa_Q ||overline(A)||_Q^2),
      frac(||Q||, lambda_"min"(P))
    ).
  $
  Then for all $alpha in [0, alpha_infinity]$,
  $alpha a <= 1 slash 2$ and
  $||I - alpha overline(A)||_Q^2 <= 1 - alpha a$.
] <lem:lyapunov-contraction-local>

In the RR estimates below we assume explicitly that $0 < alpha$ and
$2 alpha <= alpha_infinity$. Hence the Lyapunov contraction applies at both
step sizes $alpha$ and $2 alpha$.

For $1 <= j <= n-1$, combine the triangle inequality, submultiplicativity,
the norm equivalence, and the Lyapunov contraction at step sizes $alpha$ and
$2 alpha$:
$
||H_j^((n))||
&<= kappa_Q sum_(i=1)^(n-j)
      (1 - alpha a)^((i-1) slash 2)
      (1 - 2 alpha a)^((n-j-i) slash 2) 
// &= kappa_Q (1 - alpha a)^((n-j-1) slash 2)
//     sum_(k=0)^(n-j-1)
//       lr((frac(1 - 2 alpha a, 1 - alpha a)))^(k slash 2) \
&<= kappa_Q (1 - alpha a)^((n-j-1) slash 2)
    frac(1, 1 - sqrt((1 - 2 alpha a) / (1 - alpha a))).
$
For $alpha a <= 1 slash 2$ the geometric rate is bounded below by an *elementary* inequality: combining $sqrt((1 - 2 alpha a) slash (1 - alpha a)) <= sqrt(1 - alpha a)$ (since $(1-2alpha a) <= (1-alpha a)^2$) with $1 - sqrt(1 - x) >= x slash 2$ on $[0, 1]$ gives
$
1 - sqrt(frac(1 - 2 alpha a, 1 - alpha a))
  >= frac(alpha a, 2),
quad
frac(1, 1 - sqrt((1 - 2 alpha a) / (1 - alpha a)))
  <= frac(2, alpha a).
$
Defining the chapter-local norm-equivalence constant $K_Q := kappa_Q$
(distinct from the assumption-2 sup-norm constant $C_A$, which enters
separately via $|| overline(A) ||$ in the next subsection), we arrive at the
final estimate
$
||H_j^((n))|| <= K_Q thin (1 - alpha a)^((n-j-1) slash 2) thin frac(2, alpha a).
$
The kernel decays geometrically in $n - j$ at rate $sqrt(1 - alpha a)$ but its summed weight is of order $1 slash (alpha a)$. The next subsection keeps the remaining powers of $a$ explicit when this kernel is multiplied by the prefactor $alpha^2$ in $Delta J_n^((0, alpha))$.
